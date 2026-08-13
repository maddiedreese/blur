import { readFile, writeFile } from 'node:fs/promises';

const EPSILON = 1e-6;
const MIN_BASES_PER_CLASS = 100;
const MIN_CLASS_DOMAINS = 3;
const MIN_BASES_PER_DOMAIN = 25;

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

function logit(score) {
  const bounded = Math.min(1 - EPSILON, Math.max(EPSILON, score));
  return Math.log(bounded / (1 - bounded));
}

export function fitPlatt(rows, { iterations = 4000, learningRate = 0.05, l2 = 1e-3 } = {}) {
  if (rows.length < 4 || !rows.some((row) => row.label === 0) || !rows.some((row) => row.label === 1)) {
    throw new Error('calibration requires at least four rows containing both labels');
  }
  for (const row of rows) {
    if (row.split !== 'calibration') throw new Error(`row ${row.id ?? ''} is not in the calibration split`);
    if (!Number.isFinite(row.rawScore) || row.rawScore < 0 || row.rawScore > 1 || ![0, 1].includes(row.label)) {
      throw new Error(`invalid calibration row ${row.id ?? ''}`);
    }
  }
  let slope = 1;
  let intercept = 0;
  for (let step = 0; step < iterations; step++) {
    let slopeGradient = l2 * slope;
    let interceptGradient = 0;
    for (const row of rows) {
      const x = logit(row.rawScore);
      const error = sigmoid(slope * x + intercept) - row.label;
      slopeGradient += error * x / rows.length;
      interceptGradient += error / rows.length;
    }
    const rate = learningRate / Math.sqrt(1 + step / 200);
    slope -= rate * slopeGradient;
    intercept -= rate * interceptGradient;
  }
  return { schemaVersion: 1, method: 'platt-logit', slope, intercept, threshold: 0.65 };
}

export function applyCalibration(rawScore, calibrator) {
  if (!Number.isFinite(rawScore) || rawScore < 0 || rawScore > 1) throw new Error('rawScore must be in [0,1]');
  return sigmoid(calibrator.slope * logit(rawScore) + calibrator.intercept);
}

export function validateCalibrationEvidence(rows) {
  const reasons = [];
  const ids = new Set();
  const baseLabels = new Map();
  const basesByLabel = [new Set(), new Set()];
  const realDomains = new Map();
  const aiDomains = new Map();
  for (const row of rows) {
    if (row.split !== 'calibration') reasons.push(`row ${row.id ?? '<missing>'} is not in the calibration split`);
    if (!row.id || ids.has(row.id)) reasons.push(`missing or duplicate id: ${row.id ?? '<missing>'}`);
    ids.add(row.id);
    if (!row.baseId) reasons.push(`row ${row.id ?? '<missing>'} has no baseId`);
    if (![0, 1].includes(row.label)) reasons.push(`row ${row.id ?? '<missing>'} has an invalid label`);
    if (!Number.isFinite(row.rawScore) || row.rawScore < 0 || row.rawScore > 1) reasons.push(`row ${row.id ?? '<missing>'} has an invalid rawScore`);
    if (!row.baseId || ![0, 1].includes(row.label)) continue;
    const priorLabel = baseLabels.get(row.baseId);
    if (priorLabel !== undefined && priorLabel !== row.label) reasons.push(`baseId ${row.baseId} has conflicting labels`);
    baseLabels.set(row.baseId, row.label);
    basesByLabel[row.label].add(row.baseId);
    const domain = row.label === 0 ? row.source : row.generatorFamily;
    if (!domain) {
      reasons.push(`row ${row.id} has no ${row.label === 0 ? 'real source' : 'generatorFamily'}`);
      continue;
    }
    const domains = row.label === 0 ? realDomains : aiDomains;
    if (!domains.has(domain)) domains.set(domain, new Set());
    domains.get(domain).add(row.baseId);
  }
  if (basesByLabel[0].size < MIN_BASES_PER_CLASS || basesByLabel[1].size < MIN_BASES_PER_CLASS) {
    reasons.push(`requires at least ${MIN_BASES_PER_CLASS} unique base images per class`);
  }
  if (realDomains.size < MIN_CLASS_DOMAINS) reasons.push(`requires at least ${MIN_CLASS_DOMAINS} real-image sources`);
  if (aiDomains.size < MIN_CLASS_DOMAINS) reasons.push(`requires at least ${MIN_CLASS_DOMAINS} generator families`);
  for (const [domain, bases] of [...realDomains, ...aiDomains]) {
    if (bases.size < MIN_BASES_PER_DOMAIN) reasons.push(`calibration domain ${domain} has fewer than ${MIN_BASES_PER_DOMAIN} unique base images`);
  }
  return {
    approved: reasons.length === 0,
    policyVersion: 1,
    uniqueBases: { real: basesByLabel[0].size, ai: basesByLabel[1].size },
    domainCounts: { realSources: realDomains.size, generatorFamilies: aiDomains.size },
    reasons,
  };
}

if (process.argv[1]?.endsWith('calibrate.mjs')) {
  const [source, destination] = process.argv.slice(2);
  if (!source || !destination) throw new Error('usage: node scripts/calibrate.mjs calibration-scores.jsonl calibrator.json');
  const rows = (await readFile(source, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const evidence = validateCalibrationEvidence(rows);
  if (!evidence.approved) throw new Error(`calibration evidence rejected:\n- ${evidence.reasons.join('\n- ')}`);
  const calibrator = fitPlatt(rows);
  await writeFile(destination, `${JSON.stringify({ ...calibrator, evidence }, null, 2)}\n`, { flag: 'wx' });
  console.log(`wrote ${destination} from ${rows.length} calibration-only rows`);
}
