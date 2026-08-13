import { describe, expect, it } from 'vitest';
// @ts-expect-error The deployed scorer helper is intentionally plain ESM.
import { activationExpression, batches, performanceSummary, terminal, withoutLocationFields } from '../tools/deployed-score-lib.mjs';

describe('deployed scoring batches', () => {
  it('splits manifests larger than the runtime queue into bounded waves', () => {
    const waves = batches(125, 48);
    expect(waves.map((wave: number[]) => wave.length)).toEqual([48, 48, 29]);
    expect(waves.flat()).toEqual(Array.from({ length: 125 }, (_, index) => index));
  });

  it('rejects a batch size that could saturate the 64-job queue', () => {
    expect(() => batches(100, 64)).toThrow('batchSize must be in [1, 60]');
  });

  it('recognizes scored, skipped and error outcomes as terminal', () => {
    expect(terminal({ score: 0.5 })).toBe(true);
    expect(terminal({ score: null, state: 'skipped-small' })).toBe(true);
    expect(terminal({ score: null, state: 'error' })).toBe(true);
    expect(terminal({ score: null, state: 'analyzing' })).toBe(false);
  });

  it('clears stale terminal state before activating a wave', () => {
    const expression = activationExpression([3]);
    expect(expression.indexOf('delete image.dataset.blurState')).toBeLessThan(expression.indexOf('image.src=image.dataset.source'));
  });

  it('removes manifest path and URL fields from deployed output', () => {
    expect(withoutLocationFields({ id: 'x', path: '/secret/image.png', originUrl: 'https://example.test', source: 'set' }))
      .toEqual({ id: 'x', source: 'set' });
  });
});

describe('deployed performance summary', () => {
  it('separates the cold case and reports warm latency/runtime/skips', () => {
    const summary = performanceSummary([
      { case: 0, score: 0.8, runtime: 'webgpu', elapsedMs: 100, wallMs: 120, performance: { inferenceMs: 50, cacheHit: false } },
      { case: 1, score: 0.7, runtime: 'webgpu', elapsedMs: 20, wallMs: 25, performance: { inferenceMs: 10, cacheHit: false } },
      { case: 2, score: 0.7, runtime: 'webgpu', elapsedMs: 30, wallMs: 40, performance: { inferenceMs: 15, cacheHit: false } },
      { case: 3, score: null, state: 'skipped-small' },
    ]);
    expect(summary.cold.case).toBe(0);
    expect(summary.warm.elapsedMs.p50).toBe(30);
    expect(summary.warm.wallMs.p50).toBe(40);
    expect(summary.runtimes).toEqual(['webgpu']);
    expect(summary.skipped).toBe(1);
  });
});
