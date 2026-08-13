export class BoundedTaskQueue {
  readonly #pending: Array<{
    run: () => Promise<unknown>;
    resolve: (value: unknown) => void;
    reject: (reason?: unknown) => void;
  }> = [];
  #active = 0;

  constructor(readonly concurrency = 1, readonly maxPending = 64) {
    if (concurrency < 1 || maxPending < 1) throw new Error('Queue limits must be positive');
  }

  get active(): number { return this.#active; }
  get pending(): number { return this.#pending.length; }

  add<T>(run: () => Promise<T>): Promise<T> {
    if (this.#pending.length >= this.maxPending) return Promise.reject(new Error('Inference queue is full'));
    return new Promise<T>((resolve, reject) => {
      this.#pending.push({ run, resolve: resolve as (value: unknown) => void, reject });
      this.#drain();
    });
  }

  #drain(): void {
    while (this.#active < this.concurrency && this.#pending.length) {
      const task = this.#pending.shift();
      if (!task) return;
      this.#active++;
      void task.run().then(task.resolve, task.reject).finally(() => {
        this.#active--;
        this.#drain();
      });
    }
  }
}

export class LruCache<K, V> {
  readonly #values = new Map<K, V>();
  constructor(readonly capacity: number) {
    if (capacity < 1) throw new Error('Cache capacity must be positive');
  }

  get size(): number { return this.#values.size; }

  get(key: K): V | undefined {
    const value = this.#values.get(key);
    if (value === undefined) return undefined;
    this.#values.delete(key);
    this.#values.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    this.#values.delete(key);
    this.#values.set(key, value);
    if (this.#values.size > this.capacity) this.#values.delete(this.#values.keys().next().value as K);
  }
}
