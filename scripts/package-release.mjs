import { access } from 'node:fs/promises';
import { gateFile } from './release-gate.mjs';

try {
  await access('models/calibration.json');
  throw new Error('release blocked: models/calibration.json is an unapproved working artifact; move it to quarantine');
} catch (error) {
  if (error instanceof Error && !('code' in error && error.code === 'ENOENT')) throw error;
}

try {
  await access('artifacts/release-test-scores.jsonl');
} catch {
  throw new Error('release blocked: artifacts/release-test-scores.jsonl is absent');
}
await gateFile('artifacts/release-test-scores.jsonl');
await import('./package.mjs');
