import type { Detection, ExtensionMessage } from '../shared/messages';
import { inspectMetadata } from './metadata';
import { inferVisual, modelStatus } from './model';

async function analyze(bytes: ArrayBuffer): Promise<Detection> {
  const started = performance.now();
  const view = new Uint8Array(bytes);
  const metadata = inspectMetadata(view);
  if (metadata.decisive) return { score: metadata.score, label: 'ai', source: 'provenance', signals: metadata.signals, runtime: 'metadata-only', elapsedMs: performance.now() - started };
  const visual = await inferVisual(bytes);
  const pixelLogOdds = Math.log(Math.max(1e-6, visual.score) / Math.max(1e-6, 1 - visual.score));
  const metadataBoost = metadata.score >= 0.85 ? 1.5 : metadata.score >= 0.8 ? 1.0 : 0;
  const score = metadataBoost ? 1 / (1 + Math.exp(-(pixelLogOdds + metadataBoost))) : visual.score;
  return { score, label: score >= 0.65 ? 'ai' : 'real', source: metadata.signals.length ? 'hybrid' : 'model', signals: metadata.signals, runtime: visual.runtime, elapsedMs: performance.now() - started };
}

async function fetchImage(url: string): Promise<ArrayBuffer> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(url, { credentials: 'omit', cache: 'force-cache', referrerPolicy: 'no-referrer', signal: controller.signal });
    if (!response.ok) throw new Error(`Image request failed (${response.status})`);
    const length = Number(response.headers.get('content-length') || 0);
    if (length > 32 * 1024 * 1024) throw new Error('Image exceeds 32 MiB');
    const blob = await response.blob();
    if (blob.size > 32 * 1024 * 1024 || (!blob.type.startsWith('image/') && !url.startsWith('data:image/'))) throw new Error('Response is not a supported image');
    return blob.arrayBuffer();
  } finally { clearTimeout(timer); }
}

chrome.runtime.onMessage.addListener((message: ExtensionMessage, _sender, sendResponse) => {
  if (message.type === 'GET_RUNTIME_STATUS') return modelStatus().then(sendResponse), true;
  if (message.type !== 'INFER_URL') return;
  void fetchImage(message.url).then(analyze)
    .then((detection) => chrome.runtime.sendMessage({ type: 'INFERENCE_RESULT', requestId: message.requestId, detection } satisfies ExtensionMessage))
    .catch((error) => chrome.runtime.sendMessage({ type: 'INFERENCE_RESULT', requestId: message.requestId, error: error instanceof Error ? error.message : 'Inference failed' } satisfies ExtensionMessage));
});
