// Stockfish in the browser (vendor/stockfish/, the single-threaded WebAssembly build in a Web Worker), with the
// interface of src/chess_engine/review/engine.py's StockfishAnalyzer:
//
//   const engine = new StockfishAnalyzer({ depth: 14, multipv: 3 });
//   const a = await engine.analyse(fen);
//   a.lines[0].san, a.bestCp          // the engine's move and its evaluation (side to move's view)
//
// Evaluations are centipawns for the side to move; mate in n is +-(MATE_CP - n), so faster mates rank higher.
// Finished positions are scored without the engine (checkmated -MATE_CP, stalemate or no material 0), and positions
// already analysed come from a cache.

import { Chess } from "../vendor/chess.js";

export const MATE_CP = 10000;
export const ENGINE_URL = new URL("../vendor/stockfish/stockfish-18-lite-single.js", import.meta.url).href;

/** A UCI "info" line -> { depth, multipv, cp, pv } (cp for the side to move), or null if it has no score. */
export function parseInfo(line) {
  if (!line.startsWith("info ") || !line.includes(" score ") || !line.includes(" pv ")) return null;
  const tokens = line.split(/\s+/);
  const value = (name) => {
    const i = tokens.indexOf(name);
    return i >= 0 ? tokens[i + 1] : undefined;
  };
  const kind = tokens[tokens.indexOf("score") + 1];
  const amount = Number(tokens[tokens.indexOf("score") + 2]);
  if (tokens.includes("lowerbound") || tokens.includes("upperbound")) return null;   // not a final score
  let cp;
  if (kind === "cp") cp = amount;
  else if (kind === "mate") cp = amount > 0 ? MATE_CP - amount : -(MATE_CP - Math.abs(amount));
  else return null;
  return { depth: Number(value("depth")), multipv: Number(value("multipv") || 1), cp,
           pv: tokens.slice(tokens.indexOf("pv") + 1) };
}

/** The evaluation of a finished game for the side to move, or null if the game goes on. */
export function finishedEval(chess) {
  if (chess.isCheckmate()) return -MATE_CP;
  if (chess.isStalemate() || chess.isInsufficientMaterial()) return 0;
  return null;
}

/** A UCI move list -> SAN (as far as the moves are legal). */
function sanOf(fen, pv) {
  const chess = new Chess(fen);
  const out = [];
  for (const uci of pv) {
    try {
      out.push(chess.move(uci).san);
    } catch {
      break;
    }
  }
  return out;
}

export class StockfishAnalyzer {
  constructor({ depth = 14, multipv = 3, hashMb = 32, url = ENGINE_URL, makeWorker = (u) => new Worker(u) } = {}) {
    this.depth = depth;
    this.multipv = multipv;
    this.hashMb = hashMb;
    this.url = url;
    this.makeWorker = makeWorker;
    this.worker = null;
    this.listeners = new Set();
    this.queue = Promise.resolve();
    this.cache = new Map();
    this.ready = null;
  }

  send(command) {
    this.worker.postMessage(command);
  }

  /** The first engine line for which `done(line)` is true; every line is passed to `onLine` meanwhile. */
  until(done, onLine = () => {}, timeoutMs = 60000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.listeners.delete(listener);
        reject(new Error("Stockfish did not answer in time"));
      }, timeoutMs);
      const listener = (line) => {
        onLine(line);
        if (done(line)) {
          clearTimeout(timer);
          this.listeners.delete(listener);
          resolve(line);
        }
      };
      this.listeners.add(listener);
    });
  }

  /** Start the engine (once): UCI, options, ready. */
  start() {
    if (!this.ready) {
      this.ready = (async () => {
        this.worker = this.makeWorker(this.url);
        this.worker.onmessage = (event) => {
          const text = typeof event.data === "string" ? event.data : String(event.data);
          for (const line of text.split("\n")) for (const listener of [...this.listeners]) listener(line.trim());
        };
        const failed = new Promise((_, reject) => {
          this.worker.onerror = (event) => reject(new Error(`Stockfish could not start (${event.message || "error"})`));
        });
        const uciok = this.until((l) => l === "uciok", () => {}, 30000);
        this.send("uci");
        await Promise.race([uciok, failed]);
        this.send(`setoption name Hash value ${this.hashMb}`);
        this.send(`setoption name MultiPV value ${this.multipv}`);
        const ready = this.until((l) => l === "readyok");
        this.send("isready");
        await Promise.race([ready, failed]);
      })();
      this.ready.catch(() => { this.ready = null; });
    }
    return this.ready;
  }

  /** The engine's view of one position: { fen, lines: [{ uci, san, cp, pv, pvSan }] best first, bestCp }. */
  analyse(fen, { depth = this.depth } = {}) {
    const key = `${fen}|${depth}`;
    if (this.cache.has(key)) return Promise.resolve(this.cache.get(key));
    const job = this.queue.then(() => this.run(fen, depth));
    this.queue = job.catch(() => {});                          // one search at a time; a failure does not block
    return job;
  }

  async run(fen, depth) {
    const key = `${fen}|${depth}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const chess = new Chess(fen);
    const done = finishedEval(chess);
    let result;
    if (done !== null || !chess.moves().length) {
      result = { fen, lines: [], bestCp: done ?? 0 };
    } else {
      await this.start();
      const best = new Map();                                  // multipv -> the deepest line seen
      const finished = this.until((l) => l.startsWith("bestmove"), (line) => {
        const info = parseInfo(line);
        if (info && info.pv.length && (!best.has(info.multipv) || info.depth >= best.get(info.multipv).depth)) {
          best.set(info.multipv, info);
        }
      }, 120000);
      this.send(`position fen ${fen}`);
      this.send(`go depth ${depth}`);
      await finished;
      const legal = new Set(chess.moves({ verbose: true }).map((m) => m.lan));
      const lines = [...best.values()]
        .filter((info) => legal.has(info.pv[0]))
        .map((info) => {
          const pv = info.pv.slice(0, 8);
          const pvSan = sanOf(fen, pv);
          return { uci: pv[0], san: pvSan[0], cp: info.cp, pv, pvSan };
        })
        .sort((a, b) => b.cp - a.cp);
      result = { fen, lines, bestCp: lines.length ? lines[0].cp : 0 };
    }
    this.cache.set(key, result);
    return result;
  }

  terminate() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.ready = null;
    this.listeners.clear();
  }
}
