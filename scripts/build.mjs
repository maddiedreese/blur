import { cp, mkdir, rm, access } from 'node:fs/promises';
import path from 'node:path';
import { build } from 'esbuild';

const root = process.cwd();
const dist = path.join(root, 'dist');
await rm(dist, { recursive: true, force: true });
await mkdir(path.join(dist, 'assets'), { recursive: true });
await cp(path.join(root, 'static'), dist, { recursive: true });

for (const [entry, outfile] of [
  ['src/background/index.ts', 'background.js'], ['src/content/index.ts', 'content.js'],
  ['src/inference/index.ts', 'offscreen.js'], ['src/ui/popup.ts', 'popup.js'], ['src/ui/options.ts', 'options.js'],
]) await build({ entryPoints: [entry], bundle: true, outfile: path.join(dist, outfile), format: 'esm', target: 'chrome121', minify: false });

const ortDist = path.join(root, 'node_modules/onnxruntime-web/dist');
for (const file of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  try { await cp(path.join(ortDist, file), path.join(dist, 'assets', file)); } catch { /* version-dependent optional artifact */ }
}
try { await access(path.join(root, 'models/detector.onnx')); await cp(path.join(root, 'models/detector.onnx'), path.join(dist, 'assets/detector.onnx')); }
catch { throw new Error('models/detector.onnx is absent; run the documented model export before building'); }
console.log(`built ${dist}`);
