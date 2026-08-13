export const THRESHOLD: number;
export type ScoreRow = { id?: string; score: number; label: 0 | 1; group?: string };
export function metrics(rows: ScoreRow[]): { threshold: number; count: number; tp: number; tn: number; fp: number; fn: number; aiRecall: number; realRecall: number; balancedAccuracy: number };
