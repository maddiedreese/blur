export const THRESHOLD: number;
export type ScoreRow = { id?: string; score: number; label: 0 | 1; group?: string; source?: string; generatorFamily?: string; contentGroup?: string; transform?: string };
export type MetricReport = { threshold: number; count: number; tp: number; tn: number; fp: number; fn: number; aiRecall: number | null; realRecall: number | null; balancedAccuracy: number | null; falsePositiveRate: number | null; precision: number | null; brierScore: number | null };
export function metrics(rows: ScoreRow[]): MetricReport;
export function report(rows: ScoreRow[]): { contract: object; overall: MetricReport; byGroup: Record<string, MetricReport>; bySource: Record<string, MetricReport>; byGeneratorFamily: Record<string, MetricReport>; byContentGroup: Record<string, MetricReport>; byTransform: Record<string, MetricReport> };
