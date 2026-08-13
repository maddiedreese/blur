import { readFile } from 'node:fs/promises';

const SPLITS = new Set(['train', 'calibration', 'test']);

export function validateDataset(rows) {
  const errors = [];
  const seenIds = new Set();
  const ownership = new Map();
  const own = (kind, value, split, id) => {
    if (!value) return;
    const key = `${kind}:${value}`;
    const previous = ownership.get(key);
    if (previous && previous !== split) errors.push(`${key} crosses ${previous}/${split} at ${id}`);
    else ownership.set(key, split);
  };
  for (const row of rows) {
    if (!row.id || seenIds.has(row.id)) errors.push(`missing or duplicate id: ${row.id ?? '<missing>'}`);
    seenIds.add(row.id);
    if (!SPLITS.has(row.split)) errors.push(`invalid split at ${row.id}`);
    if (![0, 1].includes(row.label)) errors.push(`invalid label at ${row.id}`);
    if (!row.baseId || !row.source || !row.sha256) errors.push(`baseId, source, and sha256 are required at ${row.id}`);
    own('base', row.baseId, row.split, row.id);
    own('sha256', row.sha256, row.split, row.id);
    // Real sources and AI generator families are held out as whole domains.
    if (row.label === 0) own('real-source', row.source, row.split, row.id);
    if (row.label === 1) {
      if (!row.generatorFamily) errors.push(`generatorFamily is required for AI row ${row.id}`);
      own('generator-family', row.generatorFamily, row.split, row.id);
    }
    if (row.splitGroup) own('split-group', row.splitGroup, row.split, row.id);
  }
  const counts = Object.fromEntries([...SPLITS].map((split) => [split, rows.filter((row) => row.split === split).length]));
  if (counts.calibration === 0 || counts.test === 0) errors.push('calibration and test splits must both be non-empty');
  return { valid: errors.length === 0, count: rows.length, counts, errors };
}

if (process.argv[1]?.endsWith('validate-dataset.mjs')) {
  const source = process.argv[2];
  if (!source) throw new Error('usage: node scripts/validate-dataset.mjs dataset.jsonl');
  const rows = (await readFile(source, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const result = validateDataset(rows);
  console.log(JSON.stringify(result, null, 2));
  if (!result.valid) process.exitCode = 1;
}
