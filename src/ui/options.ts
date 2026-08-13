import type { ExtensionMessage } from '../shared/messages';

chrome.runtime.sendMessage({ type: 'GET_RUNTIME_STATUS' } satisfies ExtensionMessage).then((status) => { document.querySelector('#runtime')!.textContent = JSON.stringify(status, null, 2); });
