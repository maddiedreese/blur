#!/usr/bin/env node

const endpoint = `http://127.0.0.1:${process.env.CDP_PORT || '9223'}`;
const targets = await (await fetch(`${endpoint}/json/list`)).json();
const page = targets.find((target) => target.type === 'page');
if (!page) throw new Error('No Chrome page target');
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let nextId = 1;
const pending = new Map();
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.id && pending.has(message.id)) { pending.get(message.id)(message); pending.delete(message.id); }
};
function command(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve) => pending.set(id, resolve));
}
await command('Page.enable');
await command('Runtime.enable');
await command('Page.navigate', { url: 'http://127.0.0.1:4173/e2e/fixture.html' });

const deadline = Date.now() + 45_000;
let result;
while (Date.now() < deadline) {
  await new Promise((resolve) => setTimeout(resolve, 500));
  const response = await command('Runtime.evaluate', {
    expression: `JSON.stringify({badge:document.querySelector('.blur-score')?.textContent||null,title:document.querySelector('.blur-score')?.title||null,state:document.querySelector('#fixture')?.dataset.blurState||null,result:document.querySelector('#fixture')?.dataset.blurResult||null})`,
    returnByValue: true,
  });
  result = JSON.parse(response.result.result.value);
  if (result.badge) break;
}
socket.close();
console.log(JSON.stringify({ targets: targets.map(({ type, url }) => ({ type, url })), page: result }, null, 2));
if (!result?.badge) process.exitCode = 1;
