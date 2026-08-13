#!/usr/bin/env node

/** Score a labeled JSONL manifest through the built Chrome extension.
 *
 * Test tooling only. Images are served on loopback and activated in bounded
 * waves so the deployed 64-job inference limit is never exceeded. Output does
 * not retain local paths, origin URLs, or browser image URLs.
 */

import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { activationExpression, batches, performanceSummary, terminal, withoutLocationFields } from './deployed-score-lib.mjs';

const [manifestArg, outputArg] = process.argv.slice(2);
if (!manifestArg || !outputArg) throw new Error('usage: node tools/deployed_score.mjs manifest.jsonl output.jsonl');
const manifestPath = path.resolve(manifestArg);
const manifestRoot = path.dirname(manifestPath);
let rows = (await readFile(manifestPath, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
if (process.env.SCORE_TRANSFORM) rows = rows.filter((row) => row.transform === process.env.SCORE_TRANSFORM);
if (!rows.length) throw new Error('deployed scorer requires at least one row');
const batchSize = Number(process.env.SCORE_BATCH_SIZE || 48);
const waves = batches(rows.length, batchSize);
const expectedRuntime = process.env.SCORE_EXPECT_RUNTIME;
if (expectedRuntime && !['webgpu', 'wasm'].includes(expectedRuntime)) throw new Error('SCORE_EXPECT_RUNTIME must be webgpu or wasm');
const modelMetadata = JSON.parse(await readFile('models/model.json', 'utf8'));
const preprocessingVersion = 'blur-v1-resize440-crop384-spatial3-logodds-0.9-0.1';

const images = rows.map((row, index) => ({ index, row, file: path.resolve(manifestRoot, row.path) }));
const mime = (file) => file.toLowerCase().endsWith('.png') ? 'image/png'
  : file.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg';
const html = `<!doctype html><meta charset="utf-8"><title>Blur deployed benchmark</title>
<style>body{margin:0}.case{position:fixed;left:0;top:0;width:100px;height:100px;opacity:.01}</style>
${images.map(({ index }) => `<img class="case" data-case="${index}" data-source="/image/${index}" alt="benchmark case ${index}">`).join('\n')}`;

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
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Timed out connecting to the Chrome page target')), 10_000);
    socket.onopen = () => { clearTimeout(timeout); resolve(); };
    socket.onerror = (event) => { clearTimeout(timeout); reject(event.error || new Error('Chrome DevTools socket failed')); };
  });
  let nextId = 1;
  const pending = new Map();
  const forbiddenRequests = [];
  const rejectPending = (error) => {
    for (const entry of pending.values()) { clearTimeout(entry.timeout); entry.reject(error); }
    pending.clear();
  };
  socket.onmessage = ({ data }) => {
    let message;
    try { message = JSON.parse(data); }
    catch { return; }
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id);
      clearTimeout(entry.timeout);
      pending.delete(message.id);
      if (message.error) entry.reject(new Error(`CDP ${entry.method} failed: ${message.error.message || JSON.stringify(message.error)}`));
      else entry.resolve(message);
    }
    if (message.method === 'Network.requestWillBeSent') {
      const requestUrl = message.params?.request?.url || '';
      try {
        const request = new URL(requestUrl);
        const allowedLoopback = request.hostname === '127.0.0.1' && Number(request.port) === address.port;
        const allowedExtension = request.protocol === 'chrome-extension:';
        if (!allowedLoopback && !allowedExtension && !['data:', 'blob:'].includes(request.protocol)) forbiddenRequests.push(requestUrl.slice(0, 200));
      } catch { forbiddenRequests.push(requestUrl.slice(0, 200)); }
    }
  };
  socket.onclose = () => rejectPending(new Error('Chrome DevTools socket closed'));
  const command = (method, params = {}) => {
    const id = nextId++;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`CDP ${method} timed out`));
      }, Number(process.env.SCORE_CDP_TIMEOUT_MS || 15_000));
      pending.set(id, { resolve, reject, timeout, method });
    });
  };
  const inspect = async (cases) => {
    const response = await command('Runtime.evaluate', {
      expression: `JSON.stringify(${JSON.stringify(cases)}.map((caseId)=>{const image=document.querySelector('img[data-case="'+caseId+'"]');let performance=null;try{performance=JSON.parse(image?.dataset.blurPerformance||'null')}catch{}return {case:caseId,score:image?.dataset.blurScore?Number(image.dataset.blurScore):null,result:image?.dataset.blurResult||null,runtime:image?.dataset.blurRuntime||null,state:image?.dataset.blurState||null,error:image?.dataset.blurError||null,elapsedMs:image?.dataset.blurElapsedMs?Number(image.dataset.blurElapsedMs):null,wallMs:image?.dataset.blurWallMs?Number(image.dataset.blurWallMs):null,performance}}))`,
      returnByValue: true,
    });
    return JSON.parse(response.result.result.value);
  };

  await command('Page.enable');
  await command('Runtime.enable');
  await command('Network.enable');
  await command('Page.navigate', { url: `http://127.0.0.1:${address.port}/` });
  const allResults = [];
  for (let waveIndex = 0; waveIndex < waves.length; waveIndex++) {
    const cases = waves[waveIndex];
    await command('Runtime.evaluate', {
      expression: activationExpression(cases),
    });
    const deadline = Date.now() + Number(process.env.SCORE_BATCH_TIMEOUT_MS || 300_000);
    let nextProgress = Date.now() + 10_000;
    let results = [];
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 250));
      results = await inspect(cases);
      if (results.every(terminal)) break;
      if (Date.now() >= nextProgress) {
        const pendingStates = results.filter((item) => !terminal(item)).reduce((counts, item) => {
          const key = item.state || 'idle'; counts[key] = (counts[key] || 0) + 1; return counts;
        }, {});
        console.error(`wave ${waveIndex + 1}/${waves.length}: ${results.filter(terminal).length}/${cases.length}; pending ${JSON.stringify(pendingStates)}`);
        nextProgress = Date.now() + 10_000;
      }
    }
    if (!results.every(terminal)) throw new Error(`deployed scoring wave ${waveIndex + 1} timed out: ${JSON.stringify(results.filter((item) => !terminal(item)))}`);
    allResults.push(...results);
  }
  socket.close();

  const sorted = allResults.sort((a, b) => a.case - b.case);
  const output = sorted.map((item, index) => {
    const safeRow = withoutLocationFields(rows[index]);
    return {
      ...safeRow,
      score: item.score,
      result: item.result,
      runtime: item.runtime,
      elapsedMs: item.elapsedMs,
      wallMs: item.wallMs,
      performance: item.performance,
      evaluationStatus: item.score != null ? 'scored' : item.state,
      error: item.error,
      modelSha256: modelMetadata.onnx_sha256,
      preprocessingVersion,
      evaluationMode: 'deployed',
      resolutionGroup: rows[index].transform ? 'thumbnail' : 'full-res',
    };
  });
  await writeFile(outputArg, output.map((row) => JSON.stringify(row)).join('\n') + '\n');
  const summary = performanceSummary(sorted);
  const digest = createHash('sha256').update(await readFile(outputArg)).digest('hex');
  const report = { ...summary, batches: waves.length, batchSize, outputFile: path.basename(outputArg), sha256: digest, expectedRuntime: expectedRuntime || null, unexpectedNetworkRequestCount: forbiddenRequests.length };
  await writeFile(`${outputArg}.summary.json`, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
  if (expectedRuntime && summary.runtimes.some((runtime) => runtime !== expectedRuntime && runtime !== 'metadata-only')) {
    throw new Error(`runtime mismatch: expected ${expectedRuntime}, observed ${summary.runtimes.join(', ')}`);
  }
  if (forbiddenRequests.length) throw new Error(`unexpected network requests during deployed scoring: ${JSON.stringify(forbiddenRequests.slice(0, 10))}`);
  if (summary.skipped || summary.errors) throw new Error(`deployed scoring incomplete: ${summary.skipped} skipped, ${summary.errors} errors; see ${outputArg}.summary.json`);
} finally {
  await new Promise((resolve) => server.close(resolve));
}
