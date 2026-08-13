import { describe, expect, it } from 'vitest';
// @ts-expect-error The CLI is intentionally plain ESM and exercised directly by Vitest.
import { metrics, report } from '../scripts/evaluate.mjs';
// @ts-expect-error The CLI is intentionally plain ESM and exercised directly by Vitest.
import { applyCalibration, fitPlatt, validateCalibrationEvidence } from '../scripts/calibrate.mjs';
// @ts-expect-error The CLI is intentionally plain ESM and exercised directly by Vitest.
import { validateDataset } from '../scripts/validate-dataset.mjs';
// @ts-expect-error The CLI is intentionally plain ESM and exercised directly by Vitest.
import { releaseGate } from '../scripts/release-gate.mjs';

describe('fixed bounty threshold', () => {
  it('classifies exactly 0.65 as AI', () => {
    const result = metrics([{ score: 0.649999, label: 0 }, { score: 0.65, label: 1 }]);
    expect(result.balancedAccuracy).toBe(1);
  });

  it('reports missing subgroup classes honestly instead of fabricating recall', () => {
    const result = metrics([{ score: 0.8, label: 1 }]);
    expect(result.realRecall).toBeNull();
    expect(result.balancedAccuracy).toBeNull();
  });

  it('reports source and corruption slices', () => {
    const result = report([
      { id: 'ai', score: 0.8, label: 1, source: 'gen-a', transform: 'jpeg-75' },
      { id: 'real', score: 0.2, label: 0, source: 'camera-a', transform: 'jpeg-75' },
    ]);
    expect(result.bySource['gen-a'].aiRecall).toBe(1);
    expect(result.byTransform['jpeg-75'].balancedAccuracy).toBe(1);
  });
});

describe('calibration split contract', () => {
  const rows = [
    { id: 'r1', split: 'calibration', rawScore: 0.1, label: 0 },
    { id: 'r2', split: 'calibration', rawScore: 0.3, label: 0 },
    { id: 'a1', split: 'calibration', rawScore: 0.7, label: 1 },
    { id: 'a2', split: 'calibration', rawScore: 0.9, label: 1 },
  ];

  it('fits deterministically and preserves ordering', () => {
    const calibrator = fitPlatt(rows);
    expect(calibrator.threshold).toBe(0.65);
    expect(applyCalibration(0.9, calibrator)).toBeGreaterThan(applyCalibration(0.1, calibrator));
    expect(fitPlatt(rows)).toEqual(calibrator);
  });

  it('refuses test rows during fitting', () => {
    expect(() => fitPlatt([...rows.slice(0, 3), { ...rows[3], split: 'test' }])).toThrow(/not in the calibration split/);
  });

  it('rejects tiny, single-source evidence for a deployable artifact', () => {
    const evidence = validateCalibrationEvidence(rows.map((row) => ({
      ...row,
      baseId: row.id,
      source: row.label === 0 ? 'one-real-source' : 'one-ai-source',
      generatorFamily: row.label === 1 ? 'one-generator' : undefined,
    })));
    expect(evidence.approved).toBe(false);
    expect(evidence.reasons.join('\n')).toMatch(/100 unique base images per class/);
    expect(evidence.reasons.join('\n')).toMatch(/three|3 real-image sources/);
  });
});

describe('source-separated dataset validation', () => {
  const row = (overrides: Record<string, unknown>) => ({
    id: 'x', label: 0, split: 'train', baseId: 'base-x', source: 'real-train', sha256: 'a'.repeat(64), ...overrides,
  });

  it('rejects base-image, byte, real-source, and generator-family leakage', () => {
    const result = validateDataset([
      row({ id: 'r-train' }),
      row({ id: 'r-test', split: 'test', baseId: 'base-x', source: 'real-train', sha256: 'a'.repeat(64) }),
      row({ id: 'a-cal', split: 'calibration', label: 1, baseId: 'a1', source: 'ai-a', sha256: 'b'.repeat(64), generatorFamily: 'family-a' }),
      row({ id: 'a-test', split: 'test', label: 1, baseId: 'a2', source: 'ai-b', sha256: 'c'.repeat(64), generatorFamily: 'family-a' }),
    ]);
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toMatch(/base:base-x/);
    expect(result.errors.join('\n')).toMatch(/real-source:real-train/);
    expect(result.errors.join('\n')).toMatch(/generator-family:family-a/);
  });
});

describe('release safety gate', () => {
  function releaseRows() {
    return Array.from({ length: 1000 }, (_, index) => {
      const label = index >= 500 ? 1 : 0;
      const classIndex = index % 500;
      return {
        id: `row-${index}`,
        baseId: `base-${index}`,
        split: 'test',
        label,
        score: label ? 0.9 : 0.1,
        source: label ? `ai-source-${classIndex % 3}` : `real-source-${classIndex % 3}`,
        generatorFamily: label ? `generator-${classIndex % 3}` : undefined,
        resolutionGroup: classIndex < 250 ? 'full-res' : 'thumbnail',
        modelSha256: 'a'.repeat(64),
        preprocessingVersion: 'browser-v1',
        evaluationMode: 'deployed',
      };
    });
  }

  it('approves only sufficiently broad, low-false-positive evidence', () => {
    expect(releaseGate(releaseRows()).approved).toBe(true);
  });

  it('blocks a small perfect-looking sample', () => {
    const result = releaseGate(releaseRows().slice(0, 60));
    expect(result.approved).toBe(false);
    expect(result.reasons.join('\n')).toMatch(/500 real and 500 AI/);
  });

  it('blocks a model with concentrated false positives', () => {
    const rows = releaseRows();
    for (const row of rows.filter((item) => item.label === 0 && item.source === 'real-source-0').slice(0, 20)) row.score = 0.9;
    const result = releaseGate(rows);
    expect(result.approved).toBe(false);
    expect(result.reasons.join('\n')).toMatch(/false-positive/);
  });
});
