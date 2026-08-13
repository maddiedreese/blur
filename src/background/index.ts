import type { ExtensionMessage } from '../shared/messages';
import { DEFAULT_SETTINGS } from '../shared/messages';

const inflight = new Map<string, { tabId: number; frameId: number }>();
let creatingOffscreen: Promise<void> | undefined;

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

chrome.runtime.onInstalled.addListener(() => chrome.storage.local.get(DEFAULT_SETTINGS).then((value) => chrome.storage.local.set({ ...DEFAULT_SETTINGS, ...value })));

chrome.runtime.onMessage.addListener((message: ExtensionMessage, sender, sendResponse) => {
  if (message.type === 'ANALYZE_IMAGE') {
    if (sender.tab?.id == null || sender.frameId == null) return;
    inflight.set(message.requestId, { tabId: sender.tab.id, frameId: sender.frameId });
    void (async () => {
      try {
        const parsed = new URL(message.url);
        if (!['http:', 'https:', 'data:'].includes(parsed.protocol)) throw new Error('Unsupported image URL');
        await ensureOffscreen();
        await chrome.runtime.sendMessage({ type: 'INFER_URL', requestId: message.requestId, url: message.url } satisfies ExtensionMessage);
      } catch (error) {
        const target = inflight.get(message.requestId);
        inflight.delete(message.requestId);
        if (target) await chrome.tabs.sendMessage(target.tabId, { type: 'INFERENCE_RESULT', requestId: message.requestId, error: error instanceof Error ? error.message : 'Analysis failed' } satisfies ExtensionMessage, { frameId: target.frameId }).catch(() => undefined);
      }
    })();
    sendResponse({ accepted: true });
    return;
  }
  if (message.type === 'INFERENCE_RESULT') {
    const target = inflight.get(message.requestId);
    inflight.delete(message.requestId);
    if (target) void chrome.tabs.sendMessage(target.tabId, message, { frameId: target.frameId }).catch(() => undefined);
    return;
  }
  if (message.type === 'GET_SETTINGS') return chrome.storage.local.get(DEFAULT_SETTINGS).then(sendResponse), true;
  if (message.type === 'SET_SETTINGS') return chrome.storage.local.set(message.settings).then(() => sendResponse({ ok: true })), true;
  if (message.type === 'GET_RUNTIME_STATUS') return ensureOffscreen().then(() => chrome.runtime.sendMessage({ type: 'GET_RUNTIME_STATUS' } satisfies ExtensionMessage)).then(sendResponse), true;
});
