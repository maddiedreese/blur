import { describe, expect, it } from 'vitest';
import { selectCrops } from '../src/inference/model';
import { BoundedTaskQueue, LruCache } from '../src/inference/queue';
import { aggregateCropScores, calibrateScore, fuseEvidence } from '../src/inference/scoring';

describe('multi-crop policy', () => {
  it('uses one center crop for small images', () => {
    expect(selectCrops(400, 300)).toHaveLength(1);
  });

  it('bounds large and panoramic images to three crops inside the image', () => {
    for (const [width, height] of [[1024, 1024], [2048, 768], [768, 2048]]) {
      const crops = selectCrops(width, height);
      expect(crops).toHaveLength(3);
      for (const crop of crops) {
        expect(crop.x).toBeGreaterThanOrEqual(0);
        expect(crop.y).toBeGreaterThanOrEqual(0);
        expect(crop.x + crop.size).toBeLessThanOrEqual(width);
        expect(crop.y + crop.size).toBeLessThanOrEqual(height);
      }
    }
  });
});

describe('calibrated score fusion', () => {
  it('preserves a single crop score with identity calibration', () => {
    expect(calibrateScore(aggregateCropScores([0.73]))).toBeCloseTo(0.73);
  });

  it('limits the influence of one anomalous crop', () => {
    const score = aggregateCropScores([0.1, 0.1, 0.99]);
    expect(score).toBeGreaterThan(0.1);
    expect(score).toBeLessThan(0.65);
  });

  it('adds only strong positive metadata evidence', () => {
    expect(fuseEvidence(0.5, [{ score: 0.5, signals: [], source: 'metadata' }])).toBeCloseTo(0.5);
    expect(fuseEvidence(0.5, [{ score: 0.9, signals: ['generator'], source: 'metadata' }])).toBeGreaterThan(0.65);
  });
});

describe('bounded runtime primitives', () => {
  it('serializes tasks and rejects overflow', async () => {
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const queue = new BoundedTaskQueue(1, 1);
    const first = queue.add(async () => { await gate; return 1; });
    const second = queue.add(async () => 2);
    await expect(queue.add(async () => 3)).rejects.toThrow('Inference queue is full');
    release();
    await expect(Promise.all([first, second])).resolves.toEqual([1, 2]);
  });

  it('evicts the least recently used cache entry', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.set('b', 2);
    expect(cache.get('a')).toBe(1);
    cache.set('c', 3);
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('a')).toBe(1);
  });
});
