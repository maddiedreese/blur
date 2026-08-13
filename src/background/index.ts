import type { ExtensionMessage } from '../shared/messages';
import { DEFAULT_SETTINGS } from '../shared/messages';

type InflightTarget = {
  clientRequestId: string;
  tabId: number;
  frameId: number;
  documentId?: string;
  expiresAt: number;
  timeout?: ReturnType<typeof setTimeout>;
};

const MAX_INFLIGHT = 512;
const INFERENCE_TIMEOUT_MS = 180_000;
const inflight = new Map<string, InflightTarget>();
let creatingOffscreen: Promise<void> | undefined;

function sessionKey(internalRequestId: string): string {
  return `inference:${internalRequestId}`;
}

function persisted(target: InflightTarget): Omit<InflightTarget, 'timeout'> {
  return {
    clientRequestId: target.clientRequestId,
    tabId: target.tabId,
    frameId: target.frameId,
    documentId: target.documentId,
    expiresAt: target.expiresAt,
  };
}

async function ensureOffscreen(): Promise<void> {
  const url = chrome.runtime.getURL('offscreen.html');
  const contexts = await chrome.runtime.getContexts({ contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT], documentUrls: [url] });
  if (contexts.length) return;
  creatingOffscreen ??= chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: [chrome.offscreen.Reason.WORKERS, chrome.offscreen.Reason.BLOBS, chrome.offscreen.Reason.DOM_PARSER],
    justification: 'Decode images, inspect provenance, and run local inference.',
  }).finally(() => { creatingOffscreen = undefined; });
  await creatingOffscreen;
}

async function deliver(target: InflightTarget, message: ExtensionMessage): Promise<void> {
  const options = target.documentId ? { documentId: target.documentId } : { frameId: target.frameId };
  await chrome.tabs.sendMessage(target.tabId, message, options).catch(() => undefined);
}

function forget(internalRequestId: string): InflightTarget | undefined {
  const target = inflight.get(internalRequestId);
  if (target?.timeout) clearTimeout(target.timeout);
  inflight.delete(internalRequestId);
  return target;
}

async function recover(internalRequestId: string): Promise<InflightTarget | undefined> {
  const active = forget(internalRequestId);
  const key = sessionKey(internalRequestId);
  if (active) {
    await chrome.storage.session.remove(key);
    return active;
  }
  const stored = (await chrome.storage.session.get(key))[key] as Omit<InflightTarget, 'timeout'> | undefined;
  await chrome.storage.session.remove(key);
  if (!stored || stored.expiresAt < Date.now()) return;
  return stored;
}

chrome.runtime.onInstalled.addListener(() => chrome.storage.local.get(DEFAULT_SETTINGS).then((value) => chrome.storage.local.set({ ...DEFAULT_SETTINGS, ...value })));

chrome.runtime.onMessage.addListener((message: ExtensionMessage, sender, sendResponse) => {
  if (message.type === 'ANALYZE_IMAGE') {
    if (sender.tab?.id == null || sender.frameId == null) return;
    if (inflight.size >= MAX_INFLIGHT) {
      sendResponse({ accepted: false, error: 'Too many pending images' });
      return;
    }
    let parsed: URL;
    let candidates: string[];
    try {
      parsed = new URL(message.url);
      if (!['http:', 'https:', 'data:'].includes(parsed.protocol)) throw new Error('Unsupported image URL');
      if (message.requestId.length > 128) throw new Error('Invalid request identifier');
      candidates = [...new Set([...(message.candidates || []).slice(0, 5), message.url])].filter((value) => {
        try { return ['http:', 'https:', 'data:'].includes(new URL(value).protocol); }
        catch { return false; }
      });
    } catch (error) {
      sendResponse({ accepted: false, error: error instanceof Error ? error.message : 'Invalid image URL' });
      return;
    }

    const internalRequestId = crypto.randomUUID();
    const target: InflightTarget = {
      clientRequestId: message.requestId,
      tabId: sender.tab.id,
      frameId: sender.frameId,
      documentId: sender.documentId,
      expiresAt: Date.now() + INFERENCE_TIMEOUT_MS,
      timeout: setTimeout(() => {
        const expired = forget(internalRequestId);
        void chrome.storage.session.remove(sessionKey(internalRequestId));
        if (expired) void deliver(expired, { type: 'INFERENCE_RESULT', requestId: expired.clientRequestId, error: 'Analysis timed out' });
      }, INFERENCE_TIMEOUT_MS),
    };
    inflight.set(internalRequestId, target);
    void (async () => {
      try {
        await chrome.storage.session.set({ [sessionKey(internalRequestId)]: persisted(target) });
        await ensureOffscreen();
        await chrome.runtime.sendMessage({ type: 'INFER_URL', requestId: internalRequestId, url: message.url, candidates } satisfies ExtensionMessage);
      } catch (error) {
        const failed = await recover(internalRequestId);
        if (failed) await deliver(failed, { type: 'INFERENCE_RESULT', requestId: failed.clientRequestId, error: error instanceof Error ? error.message : 'Analysis failed' });
      }
    })();
    sendResponse({ accepted: true });
    return;
  }
  if (message.type === 'INFERENCE_RESULT') {
    void recover(message.requestId).then((target) => {
      if (target) return deliver(target, { ...message, requestId: target.clientRequestId });
    });
    return;
  }
  if (message.type === 'GET_SETTINGS') return chrome.storage.local.get(DEFAULT_SETTINGS).then(sendResponse), true;
  if (message.type === 'SET_SETTINGS') return chrome.storage.local.set(message.settings).then(() => sendResponse({ ok: true })), true;
  if (message.type === 'GET_RUNTIME_STATUS' && message.target !== 'offscreen') {
    return ensureOffscreen()
      .then(() => chrome.runtime.sendMessage({ type: 'GET_RUNTIME_STATUS', target: 'offscreen' } satisfies ExtensionMessage))
      .then(sendResponse), true;
  }
});
