export type Detection = {
  score: number;
  label: 'ai' | 'real';
  source: 'model' | 'provenance' | 'hybrid' | 'unavailable';
  signals: string[];
  runtime: 'webgpu' | 'wasm' | 'metadata-only';
  elapsedMs: number;
};

export type Settings = { threshold: 0.65; disabledOrigins: string[] };

export type ExtensionMessage =
  | { type: 'ANALYZE_IMAGE'; requestId: string; url: string }
  | { type: 'INFER_URL'; requestId: string; url: string }
  | { type: 'INFERENCE_RESULT'; requestId: string; detection?: Detection; error?: string }
  | { type: 'GET_RUNTIME_STATUS' }
  | { type: 'GET_SETTINGS' }
  | { type: 'SET_SETTINGS'; settings: Partial<Settings> };

export const DEFAULT_SETTINGS: Settings = { threshold: 0.65, disabledOrigins: [] };
