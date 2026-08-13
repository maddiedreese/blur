export async function firstSuccessfulCandidate<T>(
  candidates: readonly string[],
  fetchCandidate: (url: string) => Promise<T>,
): Promise<T> {
  let lastError: unknown;
  for (const url of candidates) {
    try { return await fetchCandidate(url); }
    catch (error) { lastError = error; }
  }
  throw lastError instanceof Error ? lastError : new Error('No image candidate could be retrieved');
}
