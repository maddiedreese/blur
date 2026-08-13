import { describe, expect, it } from 'vitest';
import { inspectMetadata } from '../src/inference/metadata';

function pngText(value: string): Uint8Array {
  const payload = new TextEncoder().encode(`Software\0${value}`);
  const bytes = new Uint8Array(8 + 12 + payload.length);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  new DataView(bytes.buffer).setUint32(8, payload.length);
  bytes.set(new TextEncoder().encode('tEXt'), 12);
  bytes.set(payload, 16);
  return bytes;
}

describe('bounded metadata inspection', () => {
  it('recognizes generator evidence inside a real PNG metadata chunk', () => {
    const result = inspectMetadata(pngText('Stable Diffusion XL'));
    expect(result.signals).toContain('generator:Stable Diffusion');
    expect(result.score).toBe(0.85);
    expect(result.decisive).toBe(false);
  });
  it('does not scan arbitrary pixel bytes for marker text', () => {
    const result = inspectMetadata(new TextEncoder().encode('Stable Diffusion XL'));
    expect(result.signals).toEqual([]);
  });
  it('does not classify ordinary camera metadata as AI', () => {
    const result = inspectMetadata(pngText('Canon EOS R5 Adobe Lightroom'));
    expect(result.score).toBe(0);
  });
});
