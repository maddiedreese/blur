import { createC2pa, type C2paSdk, type ManifestStore } from '@contentauth/c2pa-web';

const TRAINED_MEDIA = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia';
const COMPOSITED_TRAINED_MEDIA = 'http://cv.iptc.org/newscodes/digitalsourcetype/compositedWithTrainedAlgorithmicMedia';

export type ProvenanceResult = {
  present: boolean;
  cryptographicallyValid: boolean;
  trustedSigner: boolean;
  aiCreated: boolean;
  aiModified: boolean;
  scoreFloor: number;
  signals: string[];
  warning?: string;
};

let sdkPromise: Promise<C2paSdk> | undefined;

function localSdk(): Promise<C2paSdk> {
  sdkPromise ??= createC2pa({
    wasmSrc: chrome.runtime.getURL('assets/c2pa_bg.wasm'),
    workerSrc: new URL(chrome.runtime.getURL('assets/c2pa_worker.js')),
    // Verify signatures and hard bindings without network-fetched trust material.
    // `Trusted` is therefore reported only when trust can be established locally.
    settings: { verify: { verifyAfterReading: true, verifyTrust: true } },
  });
  return sdkPromise;
}

function strings(value: unknown, output: string[] = [], depth = 0): string[] {
  if (depth > 12 || output.length > 10_000) return output;
  if (typeof value === 'string') output.push(value);
  else if (Array.isArray(value)) for (const item of value) strings(item, output, depth + 1);
  else if (value && typeof value === 'object') {
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (key === 'digitalSourceType' || key === 'digital_source_type') strings(item, output, depth + 1);
      else if (key === 'actions' || key === 'ingredients' || key === 'data') strings(item, output, depth + 1);
    }
  }
  return output;
}

export function classifyManifestStore(store: ManifestStore): ProvenanceResult {
  const valid = store.validation_state === 'Valid' || store.validation_state === 'Trusted';
  const trusted = store.validation_state === 'Trusted';
  const activeLabel = store.active_manifest;
  const active = activeLabel ? store.manifests?.[activeLabel] : undefined;
  const sourceTypes = active?.assertions
    ?.filter((assertion) => assertion.label === 'c2pa.actions' || assertion.label === 'c2pa.actions.v2')
    .flatMap((assertion) => strings(assertion.data)) ?? [];
  const aiCreated = valid && sourceTypes.includes(TRAINED_MEDIA);
  const aiModified = valid && sourceTypes.includes(COMPOSITED_TRAINED_MEDIA);
  const signals = [`provenance:${store.validation_state?.toLowerCase() ?? 'invalid'}`];
  if (aiCreated) signals.push('provenance:ai-created');
  if (aiModified) signals.push('provenance:ai-modified');
  return {
    present: true,
    cryptographicallyValid: valid,
    trustedSigner: trusted,
    aiCreated,
    aiModified,
    scoreFloor: aiCreated ? 0.995 : aiModified ? 0.9 : 0,
    signals,
    ...(valid ? {} : { warning: 'Content Credentials are present but did not validate.' }),
  };
}

export async function verifyC2pa(bytes: Uint8Array, mimeType: string): Promise<ProvenanceResult> {
  const sdk = await localSdk();
  const ownedBytes = bytes.slice().buffer as ArrayBuffer;
  const reader = await sdk.reader.fromBlob(mimeType, new Blob([ownedBytes], { type: mimeType }));
  if (!reader) return {
    present: false,
    cryptographicallyValid: false,
    trustedSigner: false,
    aiCreated: false,
    aiModified: false,
    scoreFloor: 0,
    signals: [],
  };
  try {
    return classifyManifestStore(await reader.manifestStore());
  } finally {
    await reader.free();
  }
}
