export type MetadataSignals = { score: number; signals: string[]; decisive: boolean };

const GENERATOR_PATTERNS: Array<[RegExp, string]> = [
  [/\bstable[ _-]?diffusion\b/i, 'Stable Diffusion'],
  [/\bmidjourney\b/i, 'Midjourney'],
  [/\bdall(?:[·-]?e)?\b/i, 'DALL-E'],
  [/\bcomfyui\b/i, 'ComfyUI'],
  [/\bautomatic1111\b/i, 'AUTOMATIC1111'],
  [/\binvokeai\b/i, 'InvokeAI'],
  [/\bfooocus\b/i, 'Fooocus'],
  [/\bnovelai\b/i, 'NovelAI'],
  [/\badobe firefly\b/i, 'Adobe Firefly'],
  [/\bflux[ ._-]?1\b/i, 'FLUX.1'],
];

function u32be(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0;
}

function u32le(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24)) >>> 0;
}

function text(bytes: Uint8Array): string {
  return new TextDecoder('latin1').decode(bytes.subarray(0, Math.min(bytes.length, 512_000)));
}

function extractPng(bytes: Uint8Array): string[] {
  const fields: string[] = [];
  for (let offset = 8; offset + 12 <= bytes.length;) {
    const length = u32be(bytes, offset);
    if (length > 2_000_000 || offset + 12 + length > bytes.length) break;
    const type = text(bytes.subarray(offset + 4, offset + 8));
    if (['tEXt', 'iTXt', 'eXIf'].includes(type)) fields.push(text(bytes.subarray(offset + 8, offset + 8 + length)));
    offset += 12 + length;
  }
  return fields;
}

function extractJpeg(bytes: Uint8Array): string[] {
  const fields: string[] = [];
  for (let offset = 2; offset + 4 <= bytes.length;) {
    if (bytes[offset] !== 0xff) break;
    const marker = bytes[offset + 1];
    if (marker === 0xda || marker === 0xd9) break;
    if (marker === 0x00 || (marker >= 0xd0 && marker <= 0xd8)) { offset += 2; continue; }
    const length = (bytes[offset + 2] << 8) | bytes[offset + 3];
    if (length < 2 || length > 2_000_000 || offset + 2 + length > bytes.length) break;
    if ([0xe1, 0xeb, 0xed, 0xfe].includes(marker)) fields.push(text(bytes.subarray(offset + 4, offset + 2 + length)));
    offset += 2 + length;
  }
  return fields;
}

function extractWebp(bytes: Uint8Array): string[] {
  const fields: string[] = [];
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const type = text(bytes.subarray(offset, offset + 4));
    const length = u32le(bytes, offset + 4);
    if (length > 2_000_000 || offset + 8 + length > bytes.length) break;
    if (type === 'EXIF' || type === 'XMP ') fields.push(text(bytes.subarray(offset + 8, offset + 8 + length)));
    offset += 8 + length + (length & 1);
  }
  return fields;
}

export function inspectMetadata(bytes: Uint8Array): MetadataSignals {
  let fields: string[] = [];
  if (bytes.length >= 8 && bytes[0] === 0x89 && text(bytes.subarray(1, 4)) === 'PNG') fields = extractPng(bytes);
  else if (bytes[0] === 0xff && bytes[1] === 0xd8) fields = extractJpeg(bytes);
  else if (text(bytes.subarray(0, 4)) === 'RIFF' && text(bytes.subarray(8, 12)) === 'WEBP') fields = extractWebp(bytes);

  const joined = fields.join('\n');
  const signals: string[] = [];
  for (const [pattern, name] of GENERATOR_PATTERNS) if (pattern.test(joined)) signals.push(`generator:${name}`);
  if (/trainedAlgorithmicMedia/i.test(joined)) signals.push('metadata:trainedAlgorithmicMedia');
  if (/c2pa|content credentials|claim_generator/i.test(joined)) signals.push('provenance:present-unverified');
  const generatorEvidence = signals.some((signal) => signal.startsWith('generator:'));
  return { score: generatorEvidence ? 0.85 : signals.includes('metadata:trainedAlgorithmicMedia') ? 0.8 : 0, signals, decisive: false };
}
