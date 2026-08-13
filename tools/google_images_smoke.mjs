#!/usr/bin/env node

import { writeFile } from 'node:fs/promises';

const endpoint = `http://127.0.0.1:${process.env.CDP_PORT || '9225'}`;
const targets = await (await fetch(`${endpoint}/json/list`)).json();
const page = targets.find((target) => target.type === 'page' && target.url.includes('google.'));
if (!page) throw new Error('No Google page target');
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let nextId = 1;
const pending = new Map();
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
};
function command(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve) => pending.set(id, resolve));
}

await command('Runtime.enable');
const deadline = Date.now() + Number(process.env.SMOKE_TIMEOUT_MS || 180_000);
let result;
while (Date.now() < deadline) {
  await new Promise((resolve) => setTimeout(resolve, 1_000));
  const response = await command('Runtime.evaluate', {
    expression: `JSON.stringify((() => {
      const visible = [...document.images].filter((image) => image.getBoundingClientRect().width >= 100 && image.getBoundingClientRect().height >= 100);
      const scored = visible.filter((image) => image.dataset.blurResult);
      return {
        visible: visible.length,
        scored: scored.length,
        ai: scored.filter((image) => image.dataset.blurResult === 'ai').length,
        real: scored.filter((image) => image.dataset.blurResult === 'real').length,
        pending: visible.filter((image) => image.dataset.blurState === 'pending' || image.dataset.blurState === 'analyzing').length,
        errors: visible.filter((image) => image.dataset.blurState === 'error').length,
        samples: scored.slice(0, 30).map((image) => ({
          result: image.dataset.blurResult,
          score: Number(image.dataset.blurScore),
          runtime: image.dataset.blurRuntime || null,
          displayedWidth: image.naturalWidth,
          displayedHeight: image.naturalHeight,
          src: image.currentSrc.slice(0, 180),
        })),
      };
    })())`,
    returnByValue: true,
  });
  result = JSON.parse(response.result.result.value);
  if (result.scored >= Math.min(12, result.visible)) break;
}
if (process.env.SCREENSHOT_PATH) {
  const capture = await command('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  await writeFile(process.env.SCREENSHOT_PATH, Buffer.from(capture.result.data, 'base64'));
}
socket.close();
console.log(JSON.stringify(result, null, 2));
if (!result?.scored) process.exitCode = 1;
