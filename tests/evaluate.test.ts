import { describe, expect, it } from 'vitest';
// @ts-expect-error The CLI is intentionally plain ESM and exercised directly by Vitest.
import { metrics } from '../scripts/evaluate.mjs';

describe('fixed bounty threshold', () => {
  it('classifies exactly 0.65 as AI', () => {
    const result = metrics([{ score: 0.649999, label: 0 }, { score: 0.65, label: 1 }]);
    expect(result.balancedAccuracy).toBe(1);
  });
});
