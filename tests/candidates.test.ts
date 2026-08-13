import { describe, expect, it } from 'vitest';
import { googleImageUrlForDocid, imageUrlFromLink, parseSrcset, rankCandidates } from '../src/content/candidates';
import { firstSuccessfulCandidate } from '../src/inference/candidate-fetch';

describe('higher-resolution image candidates', () => {
  it('ranks the widest srcset candidate before the viewport thumbnail', () => {
    const candidates = parseSrcset('/small.webp 320w, /medium.webp 800w, /large.webp 1600w');
    expect(rankCandidates(candidates, 'https://images.example/results')[0]).toBe('https://images.example/large.webp');
  });

  it('extracts an explicit image result target but not an arbitrary destination', () => {
    const result = imageUrlFromLink(
      '/imgres?imgurl=https%3A%2F%2Fcdn.example%2Foriginal.jpg&imgrefurl=https%3A%2F%2Fexample.com',
      'https://search.example/results',
    );
    expect(result).toBe('https://cdn.example/original.jpg');
    expect(imageUrlFromLink('https://example.com/article', 'https://search.example/results')).toBeUndefined();
  });

  it('deduplicates candidates and caps remote targets at six', () => {
    const ranked = rankCandidates([
      { url: '/same.jpg', priority: 20 },
      { url: '/same.jpg', priority: 100 },
      ...Array.from({ length: 10 }, (_, index) => ({ url: `/image-${index}.jpg`, priority: 50, width: index })),
    ], 'https://example.com/');
    expect(ranked).toHaveLength(6);
    expect(ranked.filter((url) => url.endsWith('/same.jpg'))).toHaveLength(1);
    expect(ranked[0]).toBe('https://example.com/same.jpg');
  });

  it('does not parse data srcsets with embedded commas as multiple URLs', () => {
    expect(parseSrcset('data:image/svg+xml,<svg></svg> 1x')).toEqual([]);
  });
});

describe('candidate retrieval fallback', () => {
  it('tries ranked candidates in order and stops after the first success', async () => {
    const attempted: string[] = [];
    const result = await firstSuccessfulCandidate(['full', 'medium', 'thumbnail'], async (url) => {
      attempted.push(url);
      if (url === 'full') throw new Error('hotlink blocked');
      return url;
    });
    expect(result).toBe('medium');
    expect(attempted).toEqual(['full', 'medium']);
  });

  it('reports the final retrieval failure when every candidate fails', async () => {
    await expect(firstSuccessfulCandidate(['full', 'thumbnail'], async (url) => {
      throw new Error(`${url} failed`);
    })).rejects.toThrow('thumbnail failed');
  });
});

describe('Google Images inline metadata', () => {
  it('selects the largest image tuple tied to the exact docid and decodes escapes', () => {
    const script = String.raw`AF_initDataCallback({data:{"-0WdResult":[["https://encrypted-tbn0.gstatic.com/thumb.jpg",180,315],["https://image-generator.com/output.png?token\u003dabc",411,720],"https://publisher.example/article"]}});`;
    expect(googleImageUrlForDocid('-0WdResult', [script]))
      .toBe('https://image-generator.com/output.png?token=abc');
  });

  it('does not take a valid image tuple from a neighboring result', () => {
    const script = String.raw`var data=[["wanted-doc","https://publisher.example/page"],["other-doc",["https://cdn.example/other.png",1000,1000]]];`;
    expect(googleImageUrlForDocid('wanted-doc', [script])).toBeUndefined();
  });

  it('rejects javascript, data, credentialed, malformed, and non-dimensioned page URLs', () => {
    const script = String.raw`var data=[["doc-safe",["javascript:alert(1)",900,900],["data:image/png;base64,AAAA",900,900],["https://user:pass@cdn.example/private.png",900,900],"https://publisher.example/article",["https://cdn.example/broken.png","wide",900]]];`;
    expect(googleImageUrlForDocid('doc-safe', [script])).toBeUndefined();
  });

  it('rejects invalid docids and scripts above the scan bound', () => {
    const oversized = `${' '.repeat(512 * 1024)}["safe-doc",["https://cdn.example/image.png",800,800]]`;
    expect(googleImageUrlForDocid('safe doc', [oversized])).toBeUndefined();
    expect(googleImageUrlForDocid('safe-doc', [oversized])).toBeUndefined();
  });

  it('does not accept an image tuple unless the exact docid is present', () => {
    const script = String.raw`var data=[["doc-1234",["https://cdn.example/image.png",800,800]]];`;
    expect(googleImageUrlForDocid('doc-123', [script])).toBeUndefined();
  });
});
