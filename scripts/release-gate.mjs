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

export function independentBaseOutcomes(rows) {
  const groups = new Map();
  for (const row of rows) {
    if (!row.baseId || ![0, 1].includes(row.label) || !Number.isFinite(row.score)) continue;
    const key = `${row.label}:${row.baseId}`;
    if (!groups.has(key)) groups.set(key, { baseId: row.baseId, label: row.label, rows: [] });
    groups.get(key).rows.push(row);
  }
  return [...groups.values()].map((group) => ({
    baseId: group.baseId,
    label: group.label,
    // Conservative across deterministic derivatives: a real base succeeds only
    // if every view stays below threshold; an AI base succeeds only if every
    // required view reaches it. One base contributes one Bernoulli outcome.
    success: group.label === 0
      ? group.rows.every((row) => row.score < THRESHOLD)
      : group.rows.every((row) => row.score >= THRESHOLD),
  }));
}

function confidenceFor(rows) {
  const outcomes = independentBaseOutcomes(rows);
  const real = outcomes.filter((item) => item.label === 0);
  const ai = outcomes.filter((item) => item.label === 1);
  const realSuccesses = real.filter((item) => item.success).length;
  const aiSuccesses = ai.filter((item) => item.success).length;
  const realRecall = real.length ? realSuccesses / real.length : null;
  const aiRecall = ai.length ? aiSuccesses / ai.length : null;
  return {
    independentBases: { real: real.length, ai: ai.length },
    conservativeCounts: {
      realSuccesses,
      realFailures: real.length - realSuccesses,
      aiSuccesses,
      aiFailures: ai.length - aiSuccesses,
    },
    conservativeMetrics: {
      tp: aiSuccesses,
      fn: ai.length - aiSuccesses,
      tn: realSuccesses,
      fp: real.length - realSuccesses,
      aiRecall,
      realRecall,
      falsePositiveRate: realRecall === null ? null : 1 - realRecall,
      balancedAccuracy: aiRecall === null || realRecall === null ? null : (aiRecall + realRecall) / 2,
    },
    realRecallLower95: wilson(realSuccesses, real.length, 'lower'),
    falsePositiveRateUpper95: wilson(real.length - realSuccesses, real.length, 'upper'),
    aiRecallLower95: wilson(aiSuccesses, ai.length, 'lower'),
  };
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

  const confidence = confidenceFor(rows);
  if (overall) {
    const conservative = confidence.conservativeMetrics;
    const realLower = confidence.realRecallLower95;
    const fpUpper = confidence.falsePositiveRateUpper95;
    const aiLower = confidence.aiRecallLower95;
    if ((conservative.balancedAccuracy ?? 0) < 0.9) reasons.push('conservative independent-base balanced accuracy is below 0.90');
    if ((conservative.realRecall ?? 0) < 0.98) reasons.push('conservative independent-base real recall is below 0.98');
    if ((realLower ?? 0) < 0.95) reasons.push('95% Wilson lower bound for real recall is below 0.95');
    if ((fpUpper ?? 1) > 0.02) reasons.push('95% Wilson upper bound for false-positive rate exceeds 0.02');
    if ((aiLower ?? 0) < 0.8) reasons.push('95% Wilson lower bound for AI recall is below 0.80');
  }

  const resolutionGroups = {};
  const resolutionConfidence = {};
  for (const group of REQUIRED_RESOLUTION_GROUPS) {
    const groupRows = rows.filter((row) => row.resolutionGroup === group);
    const groupReal = new Set(groupRows.filter((row) => row.label === 0).map((row) => row.baseId)).size;
    const groupAi = new Set(groupRows.filter((row) => row.label === 1).map((row) => row.baseId)).size;
    const result = metrics(groupRows);
    resolutionConfidence[group] = confidenceFor(groupRows);
    resolutionGroups[group] = result;
    if (groupReal < 100 || groupAi < 100) reasons.push(`${group} requires at least 100 rows per class`);
    const conservative = resolutionConfidence[group].conservativeMetrics;
    if ((conservative.balancedAccuracy ?? 0) < 0.85) reasons.push(`${group} conservative independent-base balanced accuracy is below 0.85`);
    if ((conservative.realRecall ?? 0) < 0.97) reasons.push(`${group} conservative independent-base real recall is below 0.97`);
  }

  for (const source of realSources) {
    const sourceRows = rows.filter((row) => row.label === 0 && row.source === source);
    const outcomes = independentBaseOutcomes(sourceRows);
    const failures = outcomes.filter((item) => !item.success).length;
    if (outcomes.length < 50) reasons.push(`real source ${source} has fewer than 50 unique base images`);
    if (outcomes.length === 0 || failures / outcomes.length > 0.05) reasons.push(`real source ${source} conservative base false-positive rate exceeds 0.05`);
  }

  return {
    approved: reasons.length === 0,
    policyVersion: 1,
    threshold: THRESHOLD,
    sampleCounts: { real: realCount, ai: aiCount },
    sourceCounts: { real: realSources.size, generatorFamilies: generatorFamilies.size },
    overall,
    confidence,
    resolutionGroups,
    resolutionConfidence,
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
