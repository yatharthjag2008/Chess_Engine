// Rating estimates on several cores: the moves of a side are split between background workers (rating_worker.js),
// each with its own copy of the model, and the per-move results come back in order. If workers cannot start (an old
// browser, or a page opened from disk), the moves are scored on the page itself, as before.

import { moveBucketLL } from "./rating.js";

const WORKER_URL = new URL("./rating_worker.js", import.meta.url);

/** How many workers: one per core but one, at least one, at most eight. */
export function workerCount(cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 2) {
  return Math.max(1, Math.min(8, cores - 1));
}

/** The moves of one side as worker jobs: `parts` contiguous chunks, as even as possible (no empty ones). */
export function split(moves, parts) {
  const n = Math.min(parts, moves.length);
  const chunks = [];
  let start = 0;
  for (let i = 0; i < n; i++) {
    const size = Math.floor(moves.length / n) + (i < moves.length % n ? 1 : 0);
    chunks.push(moves.slice(start, start + size));
    start += size;
  }
  return chunks;
}

export class ScoringPool {
  constructor({ modelBase = "model/", size = workerCount(), makeWorker = (url) => new Worker(url, { type: "module" }) } = {}) {
    this.modelBase = new URL(modelBase, typeof location !== "undefined" ? location.href : WORKER_URL).href;
    this.size = size;
    this.makeWorker = makeWorker;
    this.workers = null;
    this.ready = null;
    this.jobs = 0;
  }

  /** Start the workers (once); rejects if they cannot start, so callers can fall back. */
  start() {
    if (!this.ready) {
      this.ready = (async () => {
        const workers = [];
        for (let i = 0; i < this.size; i++) workers.push(this.makeWorker(WORKER_URL));
        await Promise.all(workers.map((worker) => new Promise((resolve, reject) => {
          worker.onmessage = (event) => {
            if (event.data.type === "ready") resolve();
            else if (event.data.type === "error") reject(new Error(event.data.message));
          };
          worker.onerror = (event) => reject(new Error(event.message || "the worker could not start"));
          worker.postMessage({ type: "init", modelBase: this.modelBase });
        })));
        this.workers = workers;
      })();
      this.ready.catch(() => this.terminate());
    }
    return this.ready;
  }

  /** Per-move bucket log-likelihoods of one side (games.js), in move order; `onMove(done)` as moves finish.
   *  Moves are handed out one at a time, so faster cores take more of them (laptops mix fast and slow cores). */
  async score(side, onMove = () => {}) {
    await this.start();
    const moves = side.moves.map((m) => ({ fen: m.fen, uci: m.uci, oppRating: side.oppRating, clock: m.clock,
                                           oppClock: m.oppClock }));
    const rows = new Array(moves.length);
    let next = 0;
    let done = 0;
    await new Promise((resolve, reject) => {
      if (!moves.length) {
        resolve();
        return;
      }
      const give = (worker) => {
        if (next >= moves.length) return;
        const index = next++;
        const job = ++this.jobs;
        worker.onmessage = (event) => {
          const message = event.data;
          if (message.job !== job) return;
          if (message.type === "done") {
            rows[index] = message.rows[0];
            onMove(++done);
            if (done === moves.length) resolve();
            else give(worker);
          } else if (message.type === "error") {
            reject(new Error(message.message));
          }
        };
        worker.onerror = (event) => reject(new Error(event.message || "a worker failed"));
        worker.postMessage({ type: "score", job, moves: [moves[index]], base: side.base, increment: side.increment });
      };
      for (const worker of this.workers) give(worker);
    });
    return rows;
  }

  terminate() {
    for (const worker of this.workers || []) worker.terminate();
    this.workers = null;
    this.ready = null;
  }
}

/** One side's per-move likelihoods: on the pool's workers when they run, else on the page (yielding between moves). */
export async function scoreSide(pool, model, side, onMove = () => {}) {
  if (pool) {
    try {
      return await pool.score(side, onMove);
    } catch {
      pool.terminate();                         // workers unavailable here: the page does it
    }
  }
  const rows = [];
  for (const move of side.moves) {
    rows.push(moveBucketLL(model, { fen: move.fen, uci: move.uci, oppRating: side.oppRating, clock: move.clock,
                                    oppClock: move.oppClock }, side.base, side.increment));
    onMove(rows.length);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return rows;
}
