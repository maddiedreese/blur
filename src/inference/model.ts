import * as ort from 'onnxruntime-web/webgpu';

export type ModelOutput = { score: number; runtime: 'webgpu' | 'wasm' };
let sessionPromise: Promise<ort.InferenceSession> | undefined;
let runtime: 'webgpu' | 'wasm' = 'wasm';

async function loadSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) sessionPromise = (async () => {
    ort.env.wasm.wasmPaths = chrome.runtime.getURL('assets/');
    const modelUrl = chrome.runtime.getURL('assets/detector.onnx');
    try {
      const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
      runtime = 'webgpu';
      return session;
    } catch {
      runtime = 'wasm';
      return ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    }
  })();
  return sessionPromise;
}

async function preprocess(bytes: ArrayBuffer): Promise<ort.Tensor> {
  const blob = new Blob([bytes]);
  const bitmap = await createImageBitmap(blob);
  if (bitmap.width * bitmap.height > 100_000_000) { bitmap.close(); throw new Error('Decoded image is too large'); }
  const shortSide = Math.min(bitmap.width, bitmap.height);
  const scale = 440 / shortSide;
  const resizedWidth = Math.round(bitmap.width * scale);
  const resizedHeight = Math.round(bitmap.height * scale);
  const sourceX = Math.max(0, (resizedWidth - 384) / 2 / scale);
  const sourceY = Math.max(0, (resizedHeight - 384) / 2 / scale);
  const sourceSize = 384 / scale;
  const canvas = new OffscreenCanvas(384, 384);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas unavailable');
  context.drawImage(bitmap, sourceX, sourceY, sourceSize, sourceSize, 0, 0, 384, 384);
  bitmap.close();
  const pixels = context.getImageData(0, 0, 384, 384).data;
  const area = 384 * 384;
  const data = new Float32Array(3 * area);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (let i = 0; i < area; i++) {
    data[i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
    data[area + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
    data[2 * area + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor('float32', data, [1, 3, 384, 384]);
}

function softmaxAi(data: readonly number[]): number {
  if (data.length === 1) return 1 / (1 + Math.exp(-data[0]));
  const max = Math.max(data[0], data[1]);
  const real = Math.exp(data[0] - max);
  const ai = Math.exp(data[1] - max);
  return ai / (real + ai);
}

export async function inferVisual(bytes: ArrayBuffer): Promise<ModelOutput> {
  const session = await loadSession();
  const tensor = await preprocess(bytes);
  const inputName = session.inputNames[0];
  const outputs = await session.run({ [inputName]: tensor });
  const output = outputs[session.outputNames[0]];
  return { score: softmaxAi(Array.from(output.data as Float32Array)), runtime };
}

export async function modelStatus(): Promise<{ ready: boolean; runtime?: string; error?: string }> {
  try { await loadSession(); return { ready: true, runtime }; }
  catch (error) { return { ready: false, error: error instanceof Error ? error.message : 'Model unavailable' }; }
}
