import type { Detection, ExtensionMessage, Settings } from '../shared/messages';
import { DEFAULT_SETTINGS } from '../shared/messages';
import { googleImageUrlForDocid, imageUrlFromLink, parseSrcset, rankCandidates, type ImageCandidate } from './candidates';

const MIN_EDGE = 96;
const observed = new WeakMap<HTMLImageElement, string>();
const pending = new Map<string, { image: HTMLImageElement; fingerprint: string }>();
const badges = new Map<HTMLImageElement, HTMLElement>();
let settings: Settings = DEFAULT_SETTINGS;
let positionFrame = 0;

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) if (entry.isIntersecting && entry.target instanceof HTMLImageElement) void analyze(entry.target);
}, { rootMargin: '400px' });

function resolvedSource(image: HTMLImageElement): string {
  return image.currentSrc || image.src;
}

function googleImagesOriginal(image: HTMLImageElement): string | undefined {
  if (!/(^|\.)google\.[a-z.]+$/i.test(location.hostname) || !location.pathname.startsWith('/search')) return;
  let ancestor = image.parentElement;
  for (let depth = 0; ancestor && depth < 10; depth++, ancestor = ancestor.parentElement) {
    const docid = ancestor.getAttribute('data-docid');
    if (!docid || !ancestor.hasAttribute('data-lpage')) continue;
    const scriptTexts: string[] = [];
    let totalBytes = 0;
    for (const script of document.scripts) {
      const text = script.textContent || '';
      if (!text.includes(docid) || text.length > 512 * 1024) continue;
      totalBytes += text.length;
      if (totalBytes > 1536 * 1024 || scriptTexts.length >= 32) break;
      scriptTexts.push(text);
    }
    return googleImageUrlForDocid(docid, scriptTexts);
  }
  return;
}

function candidateSources(image: HTMLImageElement): string[] {
  const candidates: ImageCandidate[] = [];
  const add = (url: string | null | undefined, priority: number): void => {
    if (url) candidates.push({ url, priority });
  };

  add(googleImagesOriginal(image), 120);

  // Explicit full-resolution page metadata takes precedence. Keep the list
  // narrow to image-specific attributes; generic data-url/href values can be
  // navigation or tracking endpoints.
  const metadataElements: Element[] = [image];
  let ancestor = image.parentElement;
  for (let depth = 0; ancestor && depth < 10; depth++, ancestor = ancestor.parentElement) metadataElements.push(ancestor);
  for (const element of metadataElements) {
    for (const attribute of ['data-iurl', 'data-ou', 'data-full-src', 'data-original', 'data-image-url']) add(element.getAttribute(attribute), 100);
  }
  const link = image.closest('a[href]') as HTMLAnchorElement | null;
  add(imageUrlFromLink(link?.href, document.baseURI), 95);

  for (const source of image.closest('picture')?.querySelectorAll('source[srcset]') || []) {
    candidates.push(...parseSrcset(source.getAttribute('srcset') || '').map((item) => ({ ...item, priority: 80 })));
  }
  candidates.push(...parseSrcset(image.srcset));
  add(image.getAttribute('data-src'), 70);
  add(image.getAttribute('data-lazy-src'), 70);
  add(image.src, 30);
  add(image.currentSrc, 20);
  return rankCandidates(candidates, document.baseURI);
}

function eligible(image: HTMLImageElement): boolean {
  const source = resolvedSource(image);
  return Boolean(source && image.naturalWidth >= MIN_EDGE && image.naturalHeight >= MIN_EDGE && !source.startsWith('chrome-extension:'));
}

function watch(root: ParentNode): void {
  if (root instanceof HTMLImageElement) observer.observe(root);
  root.querySelectorAll?.('img').forEach((image) => observer.observe(image));
}

function positionBadges(): void {
  positionFrame = 0;
  for (const [image, badge] of badges) {
    if (!image.isConnected) { badge.remove(); badges.delete(image); continue; }
    const rect = image.getBoundingClientRect();
    badge.style.left = `${Math.max(0, rect.left + 6)}px`;
    badge.style.top = `${Math.max(0, rect.top + 6)}px`;
    badge.hidden = rect.bottom <= 0 || rect.top >= innerHeight || rect.right <= 0 || rect.left >= innerWidth;
  }
}

function schedulePositions(): void {
  if (!positionFrame) positionFrame = requestAnimationFrame(positionBadges);
}

async function analyze(image: HTMLImageElement): Promise<void> {
  if (!eligible(image) || settings.disabledOrigins.includes(location.origin)) return;
  const source = resolvedSource(image);
  const candidates = candidateSources(image);
  const fingerprint = candidates.join('\n') || source;
  if (observed.get(image) === fingerprint) return;
  observed.set(image, fingerprint);
  const requestId = crypto.randomUUID();
  pending.set(requestId, { image, fingerprint });
  delete image.dataset.blurResult;
  delete image.dataset.blurScore;
  delete image.dataset.blurRuntime;
  image.dataset.blurState = 'analyzing';
  const message: ExtensionMessage = { type: 'ANALYZE_IMAGE', requestId, url: source, candidates };
  try { await chrome.runtime.sendMessage(message); }
  catch { pending.delete(requestId); delete image.dataset.blurState; }
}

function display(image: HTMLImageElement, detection: Detection): void {
  const old = badges.get(image);
  old?.remove();
  delete image.dataset.blurState;
  image.dataset.blurResult = detection.label;
  image.dataset.blurScore = detection.score.toFixed(8);
  image.dataset.blurRuntime = detection.runtime;
  const badge = document.createElement('span');
  badge.className = `blur-score blur-score--${detection.label === 'ai' ? 'above-threshold' : 'below-threshold'}`;
  badge.textContent = `AI score ${Math.round(detection.score * 100)}`;
  badge.title = `Local ${detection.runtime} evidence score; AI at 65 or above. This is not a calibrated probability${detection.signals.length ? ` — ${detection.signals.join(', ')}` : ''}`;
  document.documentElement.append(badge);
  badges.set(image, badge);
  schedulePositions();
}

chrome.runtime.onMessage.addListener((message: ExtensionMessage) => {
  if (message.type !== 'INFERENCE_RESULT') return;
  const entry = pending.get(message.requestId);
  pending.delete(message.requestId);
  if (!entry) return;
  if (observed.get(entry.image) !== entry.fingerprint) return;
  if (message.detection) display(entry.image, message.detection);
  else delete entry.image.dataset.blurState;
});

chrome.storage.local.get(DEFAULT_SETTINGS).then((stored) => { settings = stored as Settings; watch(document); });
chrome.storage.onChanged.addListener((changes) => {
  for (const key of Object.keys(DEFAULT_SETTINGS) as (keyof Settings)[]) if (changes[key]) settings = { ...settings, [key]: changes[key].newValue };
});
new MutationObserver((records) => records.forEach((record) => {
  record.addedNodes.forEach((node) => { if (node instanceof Element) watch(node); });
  if (record.type === 'attributes') {
    const image = record.target instanceof HTMLImageElement
      ? record.target
      : record.target instanceof HTMLSourceElement
        ? record.target.closest('picture')?.querySelector('img')
        : record.target instanceof Element
          ? record.target.querySelector('img')
          : null;
    if (image) { observed.delete(image); observer.observe(image); }
  }
})).observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['src', 'srcset', 'sizes', 'data-src', 'data-lazy-src', 'data-iurl', 'data-ou', 'data-full-src', 'data-original', 'data-image-url'],
});
document.addEventListener('load', (event) => { if (event.target instanceof HTMLImageElement) { observed.delete(event.target); observer.observe(event.target); void analyze(event.target); } }, true);
addEventListener('scroll', schedulePositions, true);
addEventListener('resize', schedulePositions);
