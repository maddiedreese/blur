import { readFile } from 'node:fs/promises';
import { metrics, THRESHOLD } from './evaluate.mjs';

const REQUIRED_RESOLUTION_GROUPS = ['full-res', 'thumbnail'];

function wilson(successes, total, direction) {
  if (total === 0) return null;
  const z = 1.959963984540054;
  const p = successes / total;
  const denominator = 1 + z * z / total;
  const center = (p + z * z / (2 * total)) / denominator;
  const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator;
  return direction === 'lower' ? Math.max(0, center - margin) : Math.min(1, center + margin);
}

export function releaseGate(rows) {
  const reasons = [];
  const ids = new Set();
  const modelHashes = new Set();
  const preprocessingVersions = new Set();
  for (const row of rows) {
    if (!row.id || ids.has(row.id)) reasons.push(`missing or duplicate id: ${row.id ?? '<missing>'}`);
    ids.add(row.id);
    if (row.split !== 'test') reasons.push(`row ${row.id ?? '<missing>'} is not frozen test data`);
    if (!row.baseId) reasons.push(`row ${row.id ?? '<missing>'} has no baseId`);
    if (!/^[a-f0-9]{64}$/.test(row.modelSha256 ?? '')) reasons.push(`row ${row.id ?? '<missing>'} has invalid modelSha256`); else modelHashes.add(row.modelSha256);
    if (!row.preprocessingVersion) reasons.push(`row ${row.id ?? '<missing>'} has no preprocessingVersion`); else preprocessingVersions.add(row.preprocessingVersion);
    if (!REQUIRED_RESOLUTION_GROUPS.includes(row.resolutionGroup)) reasons.push(`row ${row.id ?? '<missing>'} has invalid resolutionGroup`);
    if (row.evaluationMode !== 'deployed') reasons.push(`row ${row.id ?? '<missing>'} was not scored through the deployed pipeline`);
  }
  if (modelHashes.size !== 1) reasons.push('release evidence must cover exactly one model SHA-256');
  if (preprocessingVersions.size !== 1) reasons.push('release evidence must cover exactly one preprocessing version');

  let overall;
  try { overall = metrics(rows); } catch (error) { reasons.push(error instanceof Error ? error.message : 'invalid score rows'); }
  const realCount = new Set(rows.filter((row) => row.label === 0).map((row) => row.baseId)).size;
  const aiCount = new Set(rows.filter((row) => row.label === 1).map((row) => row.baseId)).size;
  const realSources = new Set(rows.filter((row) => row.label === 0).map((row) => row.source).filter(Boolean));
  const generatorFamilies = new Set(rows.filter((row) => row.label === 1).map((row) => row.generatorFamily).filter(Boolean));
  if (realCount < 500 || aiCount < 500) reasons.push('requires at least 500 real and 500 AI base evaluations');
  if (realSources.size < 3) reasons.push('requires at least three held-out real-image sources');
  if (generatorFamilies.size < 3) reasons.push('requires at least three held-out generator families');

  if (overall) {
    const realLower = wilson(overall.tn, overall.tn + overall.fp, 'lower');
    const fpUpper = wilson(overall.fp, overall.tn + overall.fp, 'upper');
    const aiLower = wilson(overall.tp, overall.tp + overall.fn, 'lower');
    if ((overall.balancedAccuracy ?? 0) < 0.9) reasons.push('balanced accuracy is below 0.90');
    if ((overall.realRecall ?? 0) < 0.98) reasons.push('real recall is below 0.98');
    if ((realLower ?? 0) < 0.95) reasons.push('95% Wilson lower bound for real recall is below 0.95');
    if ((fpUpper ?? 1) > 0.02) reasons.push('95% Wilson upper bound for false-positive rate exceeds 0.02');
    if ((aiLower ?? 0) < 0.8) reasons.push('95% Wilson lower bound for AI recall is below 0.80');
  }

  const resolutionGroups = {};
  for (const group of REQUIRED_RESOLUTION_GROUPS) {
    const groupRows = rows.filter((row) => row.resolutionGroup === group);
    const groupReal = new Set(groupRows.filter((row) => row.label === 0).map((row) => row.baseId)).size;
    const groupAi = new Set(groupRows.filter((row) => row.label === 1).map((row) => row.baseId)).size;
    const result = metrics(groupRows);
    resolutionGroups[group] = result;
    if (groupReal < 100 || groupAi < 100) reasons.push(`${group} requires at least 100 rows per class`);
    if ((result.balancedAccuracy ?? 0) < 0.85) reasons.push(`${group} balanced accuracy is below 0.85`);
    if ((result.realRecall ?? 0) < 0.97) reasons.push(`${group} real recall is below 0.97`);
  }

  for (const source of realSources) {
    const sourceRows = rows.filter((row) => row.label === 0 && row.source === source);
    const result = metrics(sourceRows);
    if (sourceRows.length < 50) reasons.push(`real source ${source} has fewer than 50 rows`);
    if ((result.falsePositiveRate ?? 1) > 0.05) reasons.push(`real source ${source} false-positive rate exceeds 0.05`);
  }

  return {
    approved: reasons.length === 0,
    policyVersion: 1,
    threshold: THRESHOLD,
    sampleCounts: { real: realCount, ai: aiCount },
    sourceCounts: { real: realSources.size, generatorFamilies: generatorFamilies.size },
    overall,
    resolutionGroups,
    reasons,
  };
}

export async function gateFile(source) {
  const rows = (await readFile(source, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const result = releaseGate(rows);
  if (!result.approved) throw new Error(`release blocked:\n- ${result.reasons.join('\n- ')}`);
  return result;
}

if (process.argv[1]?.endsWith('release-gate.mjs')) {
  const source = process.argv[2];
  if (!source) throw new Error('usage: node scripts/release-gate.mjs frozen-test-scores.jsonl');
  console.log(JSON.stringify(await gateFile(source), null, 2));
}
