import { createHash } from 'node:crypto';
import { cp, mkdir, rm, access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { build } from 'esbuild';

const root = process.cwd();
const dist = path.join(root, 'dist');
const modelMetadata = JSON.parse(await readFile(path.join(root, 'models/model.json'), 'utf8'));

async function sha256(file) {
  return createHash('sha256').update(await readFile(file)).digest('hex');
}
await rm(dist, { recursive: true, force: true });
await mkdir(path.join(dist, 'assets'), { recursive: true });
await cp(path.join(root, 'static'), dist, { recursive: true });

for (const [entry, outfile] of [
  ['src/background/index.ts', 'background.js'], ['src/content/index.ts', 'content.js'],
  ['src/inference/index.ts', 'offscreen.js'], ['src/ui/popup.ts', 'popup.js'], ['src/ui/options.ts', 'options.js'],
]) await build({ entryPoints: [entry], bundle: true, outfile: path.join(dist, outfile), format: 'esm', target: 'chrome121', minify: false });

const ortDist = path.join(root, 'node_modules/onnxruntime-web/dist');
for (const file of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await cp(path.join(ortDist, file), path.join(dist, 'assets', file));
}
const c2paDist = path.join(root, 'node_modules/@contentauth/c2pa-web/dist');
await cp(path.join(c2paDist, 'resources/c2pa_bg.wasm'), path.join(dist, 'assets/c2pa_bg.wasm'));
await cp(path.join(c2paDist, 'c2pa_worker.js'), path.join(dist, 'assets/c2pa_worker.js'));
const modelPath = path.join(root, 'models/detector.onnx');
try { await access(modelPath); }
catch { throw new Error('models/detector.onnx is absent; run the documented model export before building'); }
const actualModelHash = await sha256(modelPath);
if (actualModelHash !== modelMetadata.onnx_sha256) {
  throw new Error(`models/detector.onnx SHA-256 mismatch: expected ${modelMetadata.onnx_sha256}, got ${actualModelHash}`);
}
await cp(modelPath, path.join(dist, 'assets/detector.onnx'));
console.log(`built ${dist}`);
