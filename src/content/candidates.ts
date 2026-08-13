export type ImageCandidate = {
  url: string;
  priority: number;
  width?: number;
  density?: number;
};

const MAX_CANDIDATES = 6;
const MAX_GOOGLE_DOCID_LENGTH = 160;
const MAX_GOOGLE_SCRIPT_BYTES = 512 * 1024;
const MAX_GOOGLE_CONTAINER_BYTES = 48 * 1024;
const MAX_GOOGLE_URL_LENGTH = 4096;

function matchingArrayEnd(text: string, start: number, limit: number): number | undefined {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < Math.min(text.length, limit); index++) {
    const character = text[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') inString = true;
    else if (character === '[') depth++;
    else if (character === ']' && --depth === 0) return index + 1;
  }
  return;
}

function dimensionedImageUrls(container: string): Array<{ url: string; area: number }> {
  const results: Array<{ url: string; area: number }> = [];
  const tuples = /("(?:\\.|[^"\\])*")\s*,\s*(\d{2,5})\s*,\s*(\d{2,5})/g;
  for (const match of container.matchAll(tuples)) {
    let value: unknown;
    try { value = JSON.parse(match[1]); }
    catch { continue; }
    if (typeof value !== 'string' || value.length > MAX_GOOGLE_URL_LENGTH) continue;
    let url: URL;
    try { url = new URL(value); }
    catch { continue; }
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) continue;
    const width = Number(match[2]);
    const height = Number(match[3]);
    if (width < 64 || height < 64 || width > 20_000 || height > 20_000) continue;
    results.push({ url: url.href, area: width * height });
  }
  return results;
}

/** Parse a Google Images metadata entry without executing page script. The
 * exact docid JSON string must occur inside the same bounded array as a
 * dimensioned HTTP(S) image tuple. */
export function googleImageUrlForDocid(docid: string, scriptTexts: readonly string[]): string | undefined {
  if (!docid || docid.length > MAX_GOOGLE_DOCID_LENGTH || !/^[\w.+:/=-]+$/.test(docid)) return;
  const encodedDocid = JSON.stringify(docid);
  for (const originalText of scriptTexts) {
    if (!originalText || originalText.length > MAX_GOOGLE_SCRIPT_BYTES) continue;
    let tokenIndex = originalText.indexOf(encodedDocid);
    while (tokenIndex >= 0) {
      const valueStartMatch = originalText.slice(tokenIndex + encodedDocid.length, tokenIndex + encodedDocid.length + 32).match(/^\s*:\s*\[/);
      if (valueStartMatch) {
        const arrayStart = tokenIndex + encodedDocid.length + valueStartMatch[0].lastIndexOf('[');
        const arrayEnd = matchingArrayEnd(originalText, arrayStart, arrayStart + MAX_GOOGLE_CONTAINER_BYTES);
        if (arrayEnd && arrayEnd - arrayStart <= MAX_GOOGLE_CONTAINER_BYTES) {
          const candidates = dimensionedImageUrls(originalText.slice(arrayStart, arrayEnd));
          if (candidates.length) return candidates.sort((a, b) => b.area - a.area)[0].url;
          return;
        }
      }
      const searchStart = Math.max(0, tokenIndex - 8192);
      const starts: number[] = [];
      for (let index = searchStart; index < tokenIndex; index++) if (originalText[index] === '[') starts.push(index);
      for (const start of starts.reverse()) {
        const end = matchingArrayEnd(originalText, start, tokenIndex + MAX_GOOGLE_CONTAINER_BYTES);
        if (!end || end <= tokenIndex || end - start > MAX_GOOGLE_CONTAINER_BYTES) continue;
        const container = originalText.slice(start, end);
        if (!container.includes(encodedDocid)) continue;
        const candidates = dimensionedImageUrls(container);
        if (candidates.length) return candidates.sort((a, b) => b.area - a.area)[0].url;
        // The nearest enclosing array is the docid's record. Do not expand to
        // an outer collection where a neighboring result's image may appear.
        return;
      }
      tokenIndex = originalText.indexOf(encodedDocid, tokenIndex + encodedDocid.length);
    }
  }
  return;
}

export function parseSrcset(srcset: string): ImageCandidate[] {
  if (!srcset || srcset.trim().startsWith('data:')) return [];
  const candidates: ImageCandidate[] = [];
  for (const part of srcset.split(',')) {
    const match = part.trim().match(/^(\S+)(?:\s+(\d+(?:\.\d+)?)(w|x))?$/);
    if (!match) continue;
    const descriptor = Number(match[2] || 1);
    candidates.push({
      url: match[1],
      priority: 60,
      ...(match[3] === 'w' ? { width: descriptor } : { density: descriptor }),
    });
  }
  return candidates;
}

export function absoluteHttpImageUrl(value: string | null | undefined, baseUrl: string): string | undefined {
  if (!value) return;
  try {
    const url = new URL(value, baseUrl);
    if (!['http:', 'https:', 'data:'].includes(url.protocol)) return;
    return url.href;
  } catch { return; }
}

/** Extract only explicit image-target query parameters. This supports image
 * result links without following arbitrary page navigation or making requests. */
export function imageUrlFromLink(href: string | null | undefined, baseUrl: string): string | undefined {
  if (!href) return;
  try {
    const link = new URL(href, baseUrl);
    for (const key of ['imgurl', 'media', 'image_url']) {
      const candidate = absoluteHttpImageUrl(link.searchParams.get(key), baseUrl);
      if (candidate && /^https?:/i.test(candidate) && candidate !== link.href) return candidate;
    }
    if (/\.(?:avif|gif|jpe?g|png|webp)(?:$|[?#])/i.test(link.href)) return absoluteHttpImageUrl(link.href, baseUrl);
  } catch { /* Ignore malformed page-owned links. */ }
  return;
}

export function rankCandidates(candidates: readonly ImageCandidate[], baseUrl: string): string[] {
  const unique = new Map<string, ImageCandidate>();
  for (const candidate of candidates) {
    const url = absoluteHttpImageUrl(candidate.url, baseUrl);
    if (!url || url.startsWith('chrome-extension:')) continue;
    const previous = unique.get(url);
    if (!previous || candidate.priority > previous.priority) unique.set(url, { ...candidate, url });
  }
  return [...unique.values()]
    .sort((a, b) => b.priority - a.priority || (b.width || 0) - (a.width || 0) || (b.density || 0) - (a.density || 0))
    .slice(0, MAX_CANDIDATES)
    .map(({ url }) => url);
}
