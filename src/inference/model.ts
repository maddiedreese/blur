import * as ort from 'onnxruntime-web/webgpu';
import { aggregateCropScores, calibrateScore } from './scoring';

export type ModelOutput = {
  score: number;
  cropScores: number[];
  runtime: 'webgpu' | 'wasm';
  performance: { decodeMs: number; preprocessMs: number; inferenceMs: number; cropCount: number };
};

type Crop = { x: number; y: number; size: number };

const INPUT_SIZE = 384;
const RESIZE_SHORT_SIDE = 440;
const MAX_DECODED_PIXELS = 100_000_000;
const MAX_CROPS = 3;
let sessionPromise: Promise<ort.InferenceSession> | undefined;
let runtime: 'webgpu' | 'wasm' = 'wasm';

async function createSession(executionProvider: 'webgpu' | 'wasm'): Promise<ort.InferenceSession> {
  const modelUrl = chrome.runtime.getURL('assets/detector.onnx');
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: [executionProvider],
    graphOptimizationLevel: 'all',
  });
}

async function loadSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) sessionPromise = (async () => {
    ort.env.wasm.wasmPaths = chrome.runtime.getURL('assets/');
    // Keep the CPU fallback predictable in extension pages that are not
    // cross-origin isolated. WebGPU remains the primary execution provider.
    ort.env.wasm.numThreads = 1;
    try {
      const session = await createSession('webgpu');
      runtime = 'webgpu';
      return session;
    } catch {
      runtime = 'wasm';
      return createSession('wasm');
    }
  })().catch((error) => {
    sessionPromise = undefined;
    throw error;
  });
  return sessionPromise;
}

async function runWithFallback(session: ort.InferenceSession, tensor: ort.Tensor): Promise<ort.InferenceSession.OnnxValueMapType> {
  try {
    return await session.run({ [session.inputNames[0]]: tensor });
  } catch (error) {
    if (runtime !== 'webgpu') throw error;
    // A provider can initialize successfully and still fail on its first shader
    // or after device loss. Replace the cached session so later images do not
    // repeatedly retry a broken WebGPU path.
    runtime = 'wasm';
    const fallback = await createSession('wasm');
    sessionPromise = Promise.resolve(fallback);
    return fallback.run({ [fallback.inputNames[0]]: tensor });
  }
}

/** Select spatially distinct crops while bounding inference cost at three runs. */
export function selectCrops(width: number, height: number): Crop[] {
  if (!(width > 0 && height > 0)) throw new Error('Invalid image dimensions');
  const shortSide = Math.min(width, height);
  const cropSize = shortSide * INPUT_SIZE / RESIZE_SHORT_SIDE;
  const maxX = Math.max(0, width - cropSize);
  const maxY = Math.max(0, height - cropSize);
  const center = { x: maxX / 2, y: maxY / 2, size: cropSize };

  // Small images have little additional spatial information and are kept to a
  // single crop. Larger images receive end crops along their dominant axis.
  if (shortSide < 512) return [center];
  if (width / height >= 1.15) return [
    center,
    { x: maxX * 0.08, y: maxY / 2, size: cropSize },
    { x: maxX * 0.92, y: maxY / 2, size: cropSize },
  ].slice(0, MAX_CROPS);
  if (height / width >= 1.15) return [
    center,
    { x: maxX / 2, y: maxY * 0.08, size: cropSize },
    { x: maxX / 2, y: maxY * 0.92, size: cropSize },
  ].slice(0, MAX_CROPS);
  return [
    center,
    { x: maxX * 0.08, y: maxY * 0.08, size: cropSize },
    { x: maxX * 0.92, y: maxY * 0.92, size: cropSize },
  ].slice(0, MAX_CROPS);
}

function tensorForCrop(bitmap: ImageBitmap, crop: Crop, canvas: OffscreenCanvas): ort.Tensor {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas unavailable');
  context.clearRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  context.drawImage(bitmap, crop.x, crop.y, crop.size, crop.size, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const area = INPUT_SIZE * INPUT_SIZE;
  const data = new Float32Array(3 * area);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (let i = 0; i < area; i++) {
    data[i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
    data[area + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
    data[2 * area + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function softmaxAi(data: readonly number[]): number {
  if (data.length === 1) return 1 / (1 + Math.exp(-data[0]));
  const max = Math.max(data[0], data[1]);
  const real = Math.exp(data[0] - max);
  const ai = Math.exp(data[1] - max);
  return ai / (real + ai);
}

export async function inferVisual(bytes: ArrayBuffer): Promise<ModelOutput> {
  await loadSession();
  const decodeStarted = performance.now();
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const decodeMs = performance.now() - decodeStarted;
  if (bitmap.width * bitmap.height > MAX_DECODED_PIXELS) {
    bitmap.close();
    throw new Error('Decoded image is too large');
  }

  const crops = selectCrops(bitmap.width, bitmap.height);
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const cropScores: number[] = [];
  let preprocessMs = 0;
  let inferenceMs = 0;
  try {
    for (const crop of crops) {
      const preprocessStarted = performance.now();
      const tensor = tensorForCrop(bitmap, crop, canvas);
      preprocessMs += performance.now() - preprocessStarted;
      try {
        const inferenceStarted = performance.now();
        const outputs = await runWithFallback(await loadSession(), tensor);
        inferenceMs += performance.now() - inferenceStarted;
        const output = outputs[Object.keys(outputs)[0]];
        cropScores.push(softmaxAi(Array.from(output.data as Float32Array)));
      } finally {
        tensor.dispose();
      }
    }
  } finally {
    bitmap.close();
  }
  return {
    score: calibrateScore(aggregateCropScores(cropScores)),
    cropScores,
    runtime,
    performance: { decodeMs, preprocessMs, inferenceMs, cropCount: crops.length },
  };
}

export async function modelStatus(): Promise<{ ready: boolean; runtime?: string; error?: string }> {
  try { await loadSession(); return { ready: true, runtime }; }
  catch (error) { return { ready: false, error: error instanceof Error ? error.message : 'Model unavailable' }; }
}
