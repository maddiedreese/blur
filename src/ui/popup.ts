import type { ExtensionMessage, Settings } from '../shared/messages';
import { DEFAULT_SETTINGS } from '../shared/messages';

const enabled = document.querySelector<HTMLInputElement>('#enabled')!;
const status = document.querySelector<HTMLElement>('#status')!;
let origin = '';
Promise.all([chrome.tabs.query({ active: true, currentWindow: true }), chrome.storage.local.get(DEFAULT_SETTINGS)]).then(([tabs, stored]) => {
  const settings = stored as Settings;
  try { origin = new URL(tabs[0]?.url || '').origin; } catch { origin = ''; }
  enabled.checked = Boolean(origin && !settings.disabledOrigins.includes(origin));
  status.textContent = enabled.checked ? 'Detection is active' : 'Detection is paused';
});
enabled.addEventListener('change', async () => {
  if (!origin) return;
  const stored = await chrome.storage.local.get(DEFAULT_SETTINGS) as Settings;
  const disabledOrigins = enabled.checked ? stored.disabledOrigins.filter((item) => item !== origin) : [...new Set([...stored.disabledOrigins, origin])];
  await chrome.runtime.sendMessage({ type: 'SET_SETTINGS', settings: { disabledOrigins } } satisfies ExtensionMessage);
  status.textContent = enabled.checked ? 'Detection is active' : 'Detection is paused';
});
