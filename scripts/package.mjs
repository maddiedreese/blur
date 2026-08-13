import archiver from 'archiver';
import { createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

await mkdir('release', { recursive: true });
const target = path.resolve('release/blur-chrome.zip');
const output = createWriteStream(target);
const archive = archiver('zip', { zlib: { level: 9 } });
archive.pipe(output); archive.directory('dist/', false); await archive.finalize();
await new Promise((resolve, reject) => { output.on('close', resolve); output.on('error', reject); });
console.log(`packaged ${target}`);
