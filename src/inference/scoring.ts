export type ScoreEvidence = {
  score: number;
  decisive?: boolean;
  signals: string[];
  source: 'metadata' | 'provenance' | 'forensic';
};

export type ScoreCalibration = {
  bias: number;
  temperature: number;
};

export const DEFAULT_VISUAL_CALIBRATION: ScoreCalibration = { bias: 0, temperature: 1 };

function clampProbability(score: number): number {
  return Math.min(1 - 1e-6, Math.max(1e-6, score));
}

export function logit(score: number): number {
  const bounded = clampProbability(score);
  return Math.log(bounded / (1 - bounded));
}

export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

/**
 * Fuse crop predictions without allowing one anomalous crop to dominate.
 * The small upper-tail contribution preserves localized forensic evidence,
 * while the log-odds mean supplies the stable whole-image prediction.
 */
export function aggregateCropScores(scores: readonly number[]): number {
  if (scores.length === 0) throw new Error('At least one crop score is required');
  if (scores.length === 1) return clampProbability(scores[0]);
  const logits = scores.map(logit);
  const mean = logits.reduce((sum, value) => sum + value, 0) / logits.length;
  const upper = Math.max(...logits);
  return sigmoid(mean * 0.9 + upper * 0.1);
}

/** Calibration is deliberately explicit so benchmark-fitted constants can be
 * frozen here without changing crop aggregation or metadata policy. */
export function calibrateScore(score: number, calibration = DEFAULT_VISUAL_CALIBRATION): number {
  if (!(calibration.temperature > 0)) throw new Error('Calibration temperature must be positive');
  return sigmoid((logit(score) + calibration.bias) / calibration.temperature);
}

export function fuseEvidence(visualScore: number, evidence: readonly ScoreEvidence[]): number {
  let fusedLogit = logit(visualScore);
  for (const item of evidence) {
    if (item.decisive) return clampProbability(item.score);
    // Only strong positive evidence moves the visual result. Missing or weak
    // metadata is intentionally not treated as evidence that an image is real.
    if (item.score >= 0.85) fusedLogit += 1.5;
    else if (item.score >= 0.8) fusedLogit += 1;
  }
  return sigmoid(fusedLogit);
}
