import archiver from 'archiver';
import { createWriteStream } from 'node:fs';
import { mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';

const REPRODUCIBLE_DATE = new Date('1980-01-01T00:00:00.000Z');

async function filesUnder(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await filesUnder(absolute));
    else if (entry.isFile()) files.push(absolute);
  }
  return files.sort((left, right) => left.localeCompare(right));
}

await mkdir('release', { recursive: true });
const target = path.resolve('release/blur-chrome.zip');
const output = createWriteStream(target);
const archive = archiver('zip', { zlib: { level: 9 }, statConcurrency: 1 });
archive.pipe(output);
for (const file of await filesUnder('dist')) {
  archive.file(file, {
    name: path.relative('dist', file).split(path.sep).join('/'),
    date: REPRODUCIBLE_DATE,
    mode: 0o644,
  });
}
await archive.finalize();
await new Promise((resolve, reject) => { output.on('close', resolve); output.on('error', reject); });
console.log(`packaged ${target}`);
