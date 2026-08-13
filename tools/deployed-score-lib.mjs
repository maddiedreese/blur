export function batches(length, batchSize) {
  if (!Number.isInteger(length) || length < 0) throw new Error('length must be a non-negative integer');
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 60) throw new Error('batchSize must be in [1, 60]');
  const output = [];
  for (let start = 0; start < length; start += batchSize) {
    output.push(Array.from({ length: Math.min(batchSize, length - start) }, (_, offset) => start + offset));
  }
  return output;
}

export function terminal(item) {
  return item.score != null || item.state === 'error' || item.state?.startsWith('skipped-');
}

export function activationExpression(cases) {
  return `${JSON.stringify(cases)}.forEach((caseId)=>{const image=document.querySelector('img[data-case="'+caseId+'"]');delete image.dataset.blurState;delete image.dataset.blurError;delete image.dataset.blurScore;delete image.dataset.blurResult;delete image.dataset.blurRuntime;image.src=image.dataset.source})`;
}

export function withoutLocationFields(row) {
  return Object.fromEntries(Object.entries(row).filter(([key]) => !/(?:url|path)$/i.test(key)));
}

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

export function performanceSummary(items) {
  const scored = items.filter((item) => item.score != null);
  const cold = scored.find((item) => item.case === 0) ?? scored[0];
  const warm = scored.filter((item) => item !== cold && !item.performance?.cacheHit);
  const elapsed = warm.map((item) => Number(item.elapsedMs)).filter(Number.isFinite);
  const wall = warm.map((item) => Number(item.wallMs)).filter(Number.isFinite);
  const inference = warm.map((item) => Number(item.performance?.inferenceMs)).filter(Number.isFinite);
  const states = {};
  for (const item of items.filter((entry) => entry.score == null)) states[item.state || 'idle'] = (states[item.state || 'idle'] || 0) + 1;
  return {
    total: items.length,
    scored: scored.length,
    skipped: items.filter((item) => item.state?.startsWith('skipped-')).length,
    errors: items.filter((item) => item.state === 'error').length,
    terminalStates: states,
    runtimes: [...new Set(scored.map((item) => item.runtime))].sort(),
    cold: cold ? { case: cold.case, runtime: cold.runtime, elapsedMs: cold.elapsedMs, wallMs: cold.wallMs, performance: cold.performance } : null,
    warm: {
      count: warm.length,
      elapsedMs: { p50: percentile(elapsed, 0.5), p95: percentile(elapsed, 0.95), max: elapsed.length ? Math.max(...elapsed) : null },
      wallMs: { p50: percentile(wall, 0.5), p95: percentile(wall, 0.95), max: wall.length ? Math.max(...wall) : null },
      inferenceMs: { p50: percentile(inference, 0.5), p95: percentile(inference, 0.95), max: inference.length ? Math.max(...inference) : null },
    },
  };
}
