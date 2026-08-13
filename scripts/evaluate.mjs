import { readFile } from 'node:fs/promises';

export const THRESHOLD = 0.65;

export function metrics(rows) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const row of rows) {
    if (!Number.isFinite(row.score) || ![0, 1].includes(row.label)) throw new Error(`invalid score row ${row.id ?? ''}`);
    const predicted = row.score >= THRESHOLD ? 1 : 0;
    if (predicted && row.label) tp++; else if (predicted) fp++; else if (row.label) fn++; else tn++;
  }
  const aiRecall = tp / (tp + fn || 1);
  const realRecall = tn / (tn + fp || 1);
  return { threshold: THRESHOLD, count: rows.length, tp, tn, fp, fn, aiRecall, realRecall, balancedAccuracy: (aiRecall + realRecall) / 2 };
}

if (process.argv[1]?.endsWith('evaluate.mjs')) {
  const source = process.argv[2];
  if (!source) throw new Error('usage: node scripts/evaluate.mjs scores.jsonl');
  const rows = (await readFile(source, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const report = { overall: metrics(rows), byGroup: {} };
  for (const group of [...new Set(rows.map((row) => row.group).filter(Boolean))].sort()) report.byGroup[group] = metrics(rows.filter((row) => row.group === group));
  console.log(JSON.stringify(report, null, 2));
}
