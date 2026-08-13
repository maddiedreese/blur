import { readFile } from 'node:fs/promises';

export const THRESHOLD = 0.65;

function ratio(numerator, denominator) {
  return denominator === 0 ? null : numerator / denominator;
}

export function metrics(rows) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  let brierTotal = 0;
  for (const row of rows) {
    if (!Number.isFinite(row.score) || row.score < 0 || row.score > 1 || ![0, 1].includes(row.label)) {
      throw new Error(`invalid score row ${row.id ?? ''}`);
    }
    const predicted = row.score >= THRESHOLD ? 1 : 0;
    if (predicted && row.label) tp++; else if (predicted) fp++; else if (row.label) fn++; else tn++;
    brierTotal += (row.score - row.label) ** 2;
  }
  const aiRecall = ratio(tp, tp + fn);
  const realRecall = ratio(tn, tn + fp);
  const balancedAccuracy = aiRecall === null || realRecall === null ? null : (aiRecall + realRecall) / 2;
  return {
    threshold: THRESHOLD, count: rows.length, tp, tn, fp, fn,
    aiRecall, realRecall, balancedAccuracy,
    falsePositiveRate: ratio(fp, fp + tn),
    precision: ratio(tp, tp + fp),
    brierScore: rows.length ? brierTotal / rows.length : null,
  };
}

function groups(rows, field) {
  return Object.fromEntries([...new Set(rows.map((row) => row[field]).filter(Boolean))].sort()
    .map((value) => [value, metrics(rows.filter((row) => row[field] === value))]));
}

export function report(rows) {
  return {
    contract: { scoreMeaning: 'probability-like AI score', classification: 'score >= 0.65' },
    overall: metrics(rows),
    byGroup: groups(rows, 'group'),
    bySource: groups(rows, 'source'),
    byGeneratorFamily: groups(rows, 'generatorFamily'),
    byContentGroup: groups(rows, 'contentGroup'),
    byTransform: groups(rows, 'transform'),
  };
}

if (process.argv[1]?.endsWith('evaluate.mjs')) {
  const source = process.argv[2];
  if (!source) throw new Error('usage: node scripts/evaluate.mjs scores.jsonl');
  const rows = (await readFile(source, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  console.log(JSON.stringify(report(rows), null, 2));
}
