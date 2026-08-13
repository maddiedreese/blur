import type { Detection, ExtensionMessage } from '../shared/messages';
import { firstSuccessfulCandidate } from './candidate-fetch';
import { inspectMetadata } from './metadata';
import { inferVisual, modelStatus } from './model';
import { verifyC2pa } from './provenance';
import { BoundedTaskQueue, LruCache } from './queue';
import { fuseEvidence, type ScoreEvidence } from './scoring';

const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const inferenceQueue = new BoundedTaskQueue(1, 64);
const resultCache = new LruCache<string, Detection>(256);
const urlInflight = new Map<string, Promise<Detection>>();

export type EvidenceProvider = (bytes: Uint8Array) => Promise<ScoreEvidence | undefined>;
const evidenceProviders: EvidenceProvider[] = [];

/** Future C2PA or watermark verifiers register here and remain independent of
 * image decoding, crop aggregation, and the model execution provider. */
export function registerEvidenceProvider(provider: EvidenceProvider): () => void {
  evidenceProviders.push(provider);
  return () => {
    const index = evidenceProviders.indexOf(provider);
    if (index >= 0) evidenceProviders.splice(index, 1);
  };
}

function imageMimeType(bytes: Uint8Array): string | undefined {
  if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'image/png';
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  if (bytes.length >= 12 && String.fromCharCode(...bytes.subarray(0, 4)) === 'RIFF' && String.fromCharCode(...bytes.subarray(8, 12)) === 'WEBP') return 'image/webp';
  return undefined;
}

registerEvidenceProvider(async (bytes) => {
  const mimeType = imageMimeType(bytes);
  if (!mimeType) return undefined;
  const result = await verifyC2pa(bytes, mimeType);
  if (!result.present) return undefined;
  return {
    score: result.scoreFloor,
    // A verified AI-created or AI-composited assertion is direct provenance,
    // so enforce the documented score floor rather than treating it as a weak
    // visual-model hint.
    decisive: result.aiCreated || result.aiModified,
    signals: result.signals,
    source: 'provenance',
  };
});

async function collectEvidence(bytes: Uint8Array): Promise<ScoreEvidence[]> {
  const metadata = inspectMetadata(bytes);
  const evidence: ScoreEvidence[] = [{
    score: metadata.score,
    decisive: metadata.decisive,
    signals: metadata.signals,
    source: 'metadata',
  }];
  const additional = await Promise.allSettled(evidenceProviders.map((provider) => provider(bytes)));
  for (const result of additional) if (result.status === 'fulfilled' && result.value) evidence.push(result.value);
  return evidence.sort((a, b) => Number(b.decisive) - Number(a.decisive) || b.score - a.score);
}

async function analyze(bytes: ArrayBuffer, queueDelayMs: number): Promise<Detection> {
  const started = performance.now();
  const evidence = await collectEvidence(new Uint8Array(bytes));
  const decisive = evidence.find((item) => item.decisive);
  const signals = [...new Set(evidence.flatMap((item) => item.signals))];
  if (decisive) return {
    score: decisive.score,
    label: decisive.score >= 0.65 ? 'ai' : 'real',
    source: 'provenance',
    signals,
    runtime: 'metadata-only',
    elapsedMs: performance.now() - started,
    performance: { fetchMs: 0, hashMs: 0, decodeMs: 0, preprocessMs: 0, inferenceMs: 0, cropCount: 0, cacheHit: false, queueDelayMs },
  };

  const visual = await inferVisual(bytes);
  const score = fuseEvidence(visual.score, evidence);
  return {
    score,
    label: score >= 0.65 ? 'ai' : 'real',
    source: signals.length ? 'hybrid' : 'model',
    signals,
    runtime: visual.runtime,
    elapsedMs: performance.now() - started,
    performance: {
      fetchMs: 0,
      hashMs: 0,
      ...visual.performance,
      cacheHit: false,
      queueDelayMs,
    },
  };
}

async function fetchImage(url: string): Promise<ArrayBuffer> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(url, { credentials: 'omit', cache: 'force-cache', referrerPolicy: 'no-referrer', signal: controller.signal });
    if (!response.ok) throw new Error(`Image request failed (${response.status})`);
    const length = Number(response.headers.get('content-length') || 0);
    if (length > MAX_IMAGE_BYTES) throw new Error('Image exceeds 32 MiB');
    const blob = await response.blob();
    if (blob.size > MAX_IMAGE_BYTES || (!blob.type.startsWith('image/') && !url.startsWith('data:image/'))) throw new Error('Response is not a supported image');
    return blob.arrayBuffer();
  } finally { clearTimeout(timer); }
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (value) => value.toString(16).padStart(2, '0')).join('');
}

function analyzeUrl(url: string, candidateUrls: readonly string[] = [url]): Promise<Detection> {
  const candidates = [...new Set([...candidateUrls.slice(0, 5), url])];
  const requestKey = candidates.join('\n');
  const existing = urlInflight.get(requestKey);
  if (existing) return existing;
  const promise = (async () => {
    const fetchStarted = performance.now();
    const bytes = await firstSuccessfulCandidate(candidates, fetchImage);
    const fetchMs = performance.now() - fetchStarted;
    const hashStarted = performance.now();
    const hash = await sha256(bytes);
    const hashMs = performance.now() - hashStarted;
    const cached = resultCache.get(hash);
    if (cached) return {
      ...cached,
      elapsedMs: fetchMs + hashMs,
      performance: cached.performance ? { ...cached.performance, fetchMs, hashMs, cacheHit: true, queueDelayMs: 0 } : undefined,
    };
    // Network I/O and hashing remain concurrent. Only memory-heavy decoding and
    // session.run are serialized so one slow origin cannot stall all fetching.
    const enqueuedAt = performance.now();
    const detection = await inferenceQueue.add(() => analyze(bytes, performance.now() - enqueuedAt));
    detection.elapsedMs += fetchMs + hashMs;
    if (detection.performance) Object.assign(detection.performance, { fetchMs, hashMs });
    resultCache.set(hash, detection);
    return detection;
  })().finally(() => urlInflight.delete(requestKey));
  urlInflight.set(requestKey, promise);
  return promise;
}

chrome.runtime.onMessage.addListener((message: ExtensionMessage, _sender, sendResponse) => {
  if (message.type === 'GET_RUNTIME_STATUS' && message.target === 'offscreen') return modelStatus().then(sendResponse), true;
  if (message.type !== 'INFER_URL') return;
  void analyzeUrl(message.url, message.candidates)
    .then((detection) => chrome.runtime.sendMessage({ type: 'INFERENCE_RESULT', requestId: message.requestId, detection } satisfies ExtensionMessage))
    .catch((error) => chrome.runtime.sendMessage({ type: 'INFERENCE_RESULT', requestId: message.requestId, error: error instanceof Error ? error.message : 'Inference failed' } satisfies ExtensionMessage));
});
