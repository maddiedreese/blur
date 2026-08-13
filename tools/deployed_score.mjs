#!/usr/bin/env node

/** Score a labeled JSONL manifest through the built Chrome extension.
 *
 * This is test tooling only: it serves the labeled images on loopback, opens a
 * compact page in an already-running clean Chrome test profile, and records the
 * score exposed by the content script after the complete deployed pipeline.
 */

import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const [manifestArg, outputArg] = process.argv.slice(2);
if (!manifestArg || !outputArg) {
  throw new Error('usage: node tools/deployed_score.mjs manifest.jsonl output.jsonl');
}
const manifestPath = path.resolve(manifestArg);
const manifestRoot = path.dirname(manifestPath);
let rows = (await readFile(manifestPath, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
if (process.env.SCORE_TRANSFORM) rows = rows.filter((row) => row.transform === process.env.SCORE_TRANSFORM);
if (!rows.length || rows.length > 64) throw new Error('deployed scorer accepts 1..64 rows per run');
const modelMetadata = JSON.parse(await readFile('models/model.json', 'utf8'));
const preprocessingVersion = 'blur-v1-resize440-crop384-spatial3-logodds-0.9-0.1';

const images = rows.map((row, index) => ({
  index,
  row,
  file: path.resolve(manifestRoot, row.path),
}));
const mime = (file) => file.toLowerCase().endsWith('.png') ? 'image/png'
  : file.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg';
const html = `<!doctype html><meta charset="utf-8"><title>Blur deployed benchmark</title>
<style>body{margin:0}.case{position:fixed;left:0;top:0;width:100px;height:100px;opacity:.01}</style>
${images.map(({ index }) => `<img class="case" data-case="${index}" src="/image/${index}">`).join('\n')}`;

const server = createServer(async (request, response) => {
  try {
    if (request.url === '/') {
      response.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
      response.end(html);
      return;
    }
    const match = request.url?.match(/^\/image\/(\d+)(?:\?.*)?$/);
    const item = match ? images[Number(match[1])] : undefined;
    if (!item) { response.writeHead(404); response.end(); return; }
    response.writeHead(200, { 'content-type': mime(item.file), 'cache-control': 'no-store' });
    response.end(await readFile(item.file));
  } catch (error) {
    response.writeHead(500); response.end(error instanceof Error ? error.message : 'server error');
  }
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});
const address = server.address();
if (!address || typeof address === 'string') throw new Error('loopback server did not bind');

const endpoint = `http://127.0.0.1:${process.env.CDP_PORT || '9226'}`;
try {
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
  const command = (method, params = {}) => {
    const id = nextId++;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve) => pending.set(id, resolve));
  };
  await command('Page.enable');
  await command('Runtime.enable');
  await command('Page.navigate', { url: `http://127.0.0.1:${address.port}/` });
  const deadline = Date.now() + Number(process.env.SCORE_TIMEOUT_MS || 300_000);
  let nextProgress = Date.now() + 10_000;
  let retries = 0;
  let deployed = [];
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const response = await command('Runtime.evaluate', {
      expression: `JSON.stringify([...document.querySelectorAll('img[data-case]')].map((image) => ({case:Number(image.dataset.case),score:image.dataset.blurScore?Number(image.dataset.blurScore):null,result:image.dataset.blurResult||null,runtime:image.dataset.blurRuntime||null,state:image.dataset.blurState||null})))`,
      returnByValue: true,
    });
    deployed = JSON.parse(response.result.result.value);
    if (deployed.length === rows.length && deployed.every((item) => item.score != null)) break;
    if (Date.now() >= nextProgress) {
      const completed = deployed.filter((item) => item.score != null).length;
      const states = deployed.filter((item) => item.score == null).reduce((counts, item) => {
        const key = item.state || 'idle';
        counts[key] = (counts[key] || 0) + 1;
        return counts;
      }, {});
      console.error(`deployed scoring progress: ${completed}/${rows.length}; pending states ${JSON.stringify(states)}`);
      if (states.idle && retries < 3) {
        retries++;
        await command('Runtime.evaluate', {
          expression: `[...document.querySelectorAll('img[data-case]')].filter((image)=>!image.dataset.blurScore&&!image.dataset.blurState).forEach((image)=>{const url=new URL(image.src);url.searchParams.set('retry','${retries}');image.src=url.href})`,
        });
      }
      nextProgress = Date.now() + 10_000;
    }
  }
  socket.close();
  if (deployed.length !== rows.length || deployed.some((item) => item.score == null)) {
    const pendingCases = deployed.filter((item) => item.score == null).map((item) => ({ case: item.case, state: item.state }));
    throw new Error(`deployed scoring timed out: ${JSON.stringify(pendingCases)}`);
  }
  const output = deployed.sort((a, b) => a.case - b.case).map((item, index) => ({
    ...rows[index],
    score: item.score,
    result: item.result,
    runtime: item.runtime,
    modelSha256: modelMetadata.onnx_sha256,
    preprocessingVersion,
    evaluationMode: 'deployed',
    resolutionGroup: rows[index].transform ? 'thumbnail' : 'full-res',
  }));
  await writeFile(outputArg, output.map((row) => JSON.stringify(row)).join('\n') + '\n');
  const digest = createHash('sha256').update(await readFile(outputArg)).digest('hex');
  console.log(JSON.stringify({ count: output.length, output: outputArg, sha256: digest, runtimes: [...new Set(output.map((row) => row.runtime))] }, null, 2));
} finally {
  await new Promise((resolve) => server.close(resolve));
}
