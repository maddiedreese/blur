import type { Detection, ExtensionMessage, Settings } from '../shared/messages';
import { DEFAULT_SETTINGS } from '../shared/messages';

const MIN_EDGE = 96;
const observed = new WeakMap<HTMLImageElement, string>();
const pending = new Map<string, HTMLImageElement>();
const badges = new Map<HTMLImageElement, HTMLElement>();
let settings: Settings = DEFAULT_SETTINGS;
let positionFrame = 0;

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) if (entry.isIntersecting && entry.target instanceof HTMLImageElement) void analyze(entry.target);
}, { rootMargin: '400px' });

function resolvedSource(image: HTMLImageElement): string {
  return image.currentSrc || image.src;
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
  if (observed.get(image) === source) return;
  observed.set(image, source);
  const requestId = crypto.randomUUID();
  pending.set(requestId, image);
  image.dataset.blurState = 'analyzing';
  const message: ExtensionMessage = { type: 'ANALYZE_IMAGE', requestId, url: source };
  try { await chrome.runtime.sendMessage(message); }
  catch { pending.delete(requestId); delete image.dataset.blurState; }
}

function display(image: HTMLImageElement, detection: Detection): void {
  const old = badges.get(image);
  old?.remove();
  delete image.dataset.blurState;
  image.dataset.blurResult = detection.label;
  const badge = document.createElement('span');
  badge.className = `blur-score blur-score--${detection.label}`;
  badge.textContent = `${detection.label === 'ai' ? 'AI' : 'Real'} ${Math.round(detection.score * 100)}%`;
  badge.title = `Local ${detection.runtime} analysis${detection.signals.length ? ` — ${detection.signals.join(', ')}` : ''}`;
  document.documentElement.append(badge);
  badges.set(image, badge);
  schedulePositions();
}

chrome.runtime.onMessage.addListener((message: ExtensionMessage) => {
  if (message.type !== 'INFERENCE_RESULT') return;
  const image = pending.get(message.requestId);
  pending.delete(message.requestId);
  if (image && message.detection) display(image, message.detection);
  else if (image) delete image.dataset.blurState;
});

chrome.storage.local.get(DEFAULT_SETTINGS).then((stored) => { settings = stored as Settings; watch(document); });
chrome.storage.onChanged.addListener((changes) => {
  for (const key of Object.keys(DEFAULT_SETTINGS) as (keyof Settings)[]) if (changes[key]) settings = { ...settings, [key]: changes[key].newValue };
});
new MutationObserver((records) => records.forEach((record) => {
  record.addedNodes.forEach((node) => { if (node instanceof Element) watch(node); });
  if (record.type === 'attributes') {
    const image = record.target instanceof HTMLImageElement ? record.target : record.target instanceof HTMLSourceElement ? record.target.parentElement?.querySelector('img') : null;
    if (image) { observed.delete(image); observer.observe(image); }
  }
})).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'srcset', 'sizes'] });
document.addEventListener('load', (event) => { if (event.target instanceof HTMLImageElement) { observed.delete(event.target); observer.observe(event.target); void analyze(event.target); } }, true);
addEventListener('scroll', schedulePositions, true);
addEventListener('resize', schedulePositions);
