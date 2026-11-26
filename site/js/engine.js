// The human-like move model, running in the browser (no server).
//
// The network was trained in PyTorch; scripts/build_site.py exports its weights (model.bin) and everything else it
// needs (model.json). This file encodes a position exactly like src/chess_engine/encoding/board.py, recomputes the
// network (convolutions with batch norm folded in, FiLM conditioning on the ratings and clocks, the policy and
// win/draw/loss heads) and turns the move scores into probabilities over the legal moves.
//
//   const model = await loadModel("model/");
//   const { moves, wdl } = model.predict(chess, 1400);     // chess: a chess.js game; moves: likeliest first
//   const move = sampleMove(moves);                         // as often as players of that rating play it

const PIECES = ["p", "n", "b", "r", "q", "k"];
const NUM_PLANES = 20;

// --------------------------------------------------------------------------- encoding

/** "e2" -> 12 (rank * 8 + file), seen from White. */
export function squareIndex(name) {
  return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

/** A square name mirrored top to bottom (Black's view): "e7" -> "e2". */
export function mirror(name) {
  return name[0] + String(9 - Number(name[1]));
}

/** The 20 input planes of the position in ``chess`` (a chess.js game), from the side to move's view. */
export function encode(chess) {
  const planes = new Float32Array(NUM_PLANES * 64);
  const us = chess.turn();
  const flip = us === "b";
  const rows = chess.board();                       // rows[0] is rank 8
  for (let i = 0; i < 8; i++) {
    for (let file = 0; file < 8; file++) {
      const piece = rows[i][file];
      if (!piece) continue;
      const rank = 7 - i;
      const row = flip ? 7 - rank : rank;
      const plane = PIECES.indexOf(piece.type) + (piece.color === us ? 0 : 6);
      planes[plane * 64 + row * 8 + file] = 1;
    }
  }
  const fill = (plane, value) => planes.fill(value, plane * 64, plane * 64 + 64);
  if (us === "w") fill(12, 1);
  const them = us === "w" ? "b" : "w";
  const mine = chess.getCastlingRights(us);
  const theirs = chess.getCastlingRights(them);
  if (mine.k) fill(13, 1);
  if (mine.q) fill(14, 1);
  if (theirs.k) fill(15, 1);
  if (theirs.q) fill(16, 1);
  // The en passant square only counts when the capture is actually legal (as in the training data).
  const ep = chess.moves({ verbose: true }).find((m) => m.isEnPassant());
  if (ep) {
    const target = squareIndex(flip ? mirror(ep.to) : ep.to);
    planes[17 * 64 + target] = 1;
  }
  const halfmove = Number(chess.fen().split(" ")[4]) || 0;
  fill(18, Math.min(halfmove, 100) / 100);
  fill(19, 1);
  return planes;
}

/** The rating bucket of ``rating`` (ratings below the first edge use bucket 0). */
export function bucketOf(rating, edges) {
  let bucket = 0;
  edges.forEach((edge, i) => { if (rating >= edge) bucket = i; });
  return bucket;
}

/** The clock features (seconds left for the mover and the opponent, the time control). */
export function contextFeatures(clock, oppClock, base, increment, logScale) {
  const b = Math.max(base, 1);
  return Float32Array.from([
    Math.min(Math.max(clock, 0) / b, 2),
    Math.min(Math.max(oppClock, 0) / b, 2),
    Math.log1p(Math.max(clock, 0)) / logScale,
    Math.log1p(Math.max(oppClock, 0)) / logScale,
    Math.min(Math.max(increment / 10, 0), 3),
  ]);
}

// --------------------------------------------------------------------------- network pieces

function linear(x, w, b, outDim) {
  const inDim = x.length;
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = b ? b[o] : 0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) sum += w[row + i] * x[i];
    y[o] = sum;
  }
  return y;
}

/** A k x k convolution with zero padding on an 8x8 board: x is (cin, 8, 8), the result (cout, 8, 8). */
function conv(x, cin, w, b, cout, k) {
  const pad = k >> 1;
  const y = new Float32Array(cout * 64);
  for (let o = 0; o < cout; o++) {
    const out = o * 64;
    y.fill(b[o], out, out + 64);
    for (let c = 0; c < cin; c++) {
      const inp = c * 64;
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const weight = w[((o * cin + c) * k + ky) * k + kx];
          if (weight === 0) continue;
          const dy = ky - pad, dx = kx - pad;
          const r0 = Math.max(0, -dy), r1 = Math.min(8, 8 - dy);
          const c0 = Math.max(0, -dx), c1 = Math.min(8, 8 - dx);
          for (let r = r0; r < r1; r++) {
            const src = inp + (r + dy) * 8 + dx;
            const dst = out + r * 8;
            for (let col = c0; col < c1; col++) y[dst + col] += weight * x[src + col];
          }
        }
      }
    }
  }
  return y;
}

const relu = (x) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; };
const silu = (x) => { for (let i = 0; i < x.length; i++) x[i] = x[i] / (1 + Math.exp(-x[i])); return x; };

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

// --------------------------------------------------------------------------- the model

export class Model {
  constructor(manifest, buffer) {
    this.meta = manifest;
    this.data = new Float32Array(buffer);
    if (this.data.length !== manifest.floats) throw new Error("model.bin does not match model.json");
    this.channels = manifest.channels;
    this.vocab = new Map(manifest.vocabulary.map((uci, i) => [uci, i]));
  }

  t(name) {
    const entry = this.meta.tensors[name];
    if (!entry) throw new Error(`missing tensor ${name}`);
    const size = entry.shape.reduce((a, b) => a * b, 1);
    return this.data.subarray(entry.offset, entry.offset + size);
  }

  /** Move scores (1858) and win/draw/loss for encoded ``planes`` and the conditioning inputs; ``style`` (see
   *  loadStyle) adds a trained player's adapter: an offset to the conditioning, a shift to each block's modulation
   *  and a bias per move. */
  forward(planes, bucket, oppBucket, context, style = null) {
    const C = this.channels;
    const t = (n) => this.t(n);
    const emb = this.meta.tensors["cond.rating"].shape[1];
    const parts = new Float32Array(emb * 3);
    parts.set(t("cond.rating").subarray(bucket * emb, bucket * emb + emb), 0);
    parts.set(t("cond.opponent").subarray(oppBucket * emb, oppBucket * emb + emb), emb);
    parts.set(linear(context, t("cond.clock.w"), t("cond.clock.b"), emb), 2 * emb);
    const cond = linear(silu(linear(parts, t("cond.fc1.w"), t("cond.fc1.b"), C)), t("cond.fc2.w"), t("cond.fc2.b"), C);
    if (style) for (let c = 0; c < C; c++) cond[c] += style.offset[c];

    let x = relu(conv(planes, NUM_PLANES, t("stem.w"), t("stem.b"), C, 3));
    for (let i = 0; i < this.meta.blocks; i++) {
      const h = relu(conv(x, C, t(`block${i}.conv1.w`), t(`block${i}.conv1.b`), C, 3));
      const mod = linear(cond, t(`block${i}.mod.w`), t(`block${i}.mod.b`), 2 * C);
      if (style) for (let j = 0; j < 2 * C; j++) mod[j] += style.blockShift[i * 2 * C + j];
      for (let c = 0; c < C; c++) {
        const gamma = 1 + mod[c], beta = mod[C + c];
        for (let s = 0; s < 64; s++) h[c * 64 + s] = h[c * 64 + s] * gamma + beta;
      }
      const h2 = conv(h, C, t(`block${i}.conv2.w`), t(`block${i}.conv2.b`), C, 3);
      for (let j = 0; j < h2.length; j++) h2[j] += x[j];
      x = relu(h2);
    }

    const p = relu(conv(x, C, t("policy.conv.w"), t("policy.conv.b"), C, 3));
    const types = conv(p, C, t("policy.moves.w"), t("policy.moves.b"), 73, 1);
    const logits = new Float32Array(this.meta.gather.length);
    this.meta.gather.forEach((g, i) => { logits[i] = types[g]; });
    if (style) for (let i = 0; i < logits.length; i++) logits[i] += style.policyBias[i];

    const v = relu(conv(x, C, t("value.conv.w"), t("value.conv.b"), 8, 1));
    const joined = new Float32Array(v.length + C);
    joined.set(v, 0);
    joined.set(cond, v.length);
    const hidden = relu(linear(joined, t("value.fc1.w"), t("value.fc1.b"), 128));
    const [loss, draw, win] = softmax(Array.from(linear(hidden, t("value.fc2.w"), t("value.fc2.b"), 3)));
    return { logits, wdl: [win, draw, loss] };
  }

  /** The policy index of a chess.js move, seen from the side to move. */
  indexOf(move, turn) {
    const from = turn === "b" ? mirror(move.from) : move.from;
    const to = turn === "b" ? mirror(move.to) : move.to;
    const under = move.promotion && move.promotion !== "q" ? move.promotion : "";
    return this.vocab.get(from + to + under);
  }

  /** Every legal move with the probability that players rated ``rating`` play it (likeliest first). */
  predict(chess, rating, { oppRating = rating, clock = 180, oppClock = 180, base = 180, increment = 0, style = null } = {}) {
    const legal = chess.moves({ verbose: true });
    if (!legal.length) return { moves: [], wdl: null };
    const edges = this.meta.bucket_edges;
    const context = contextFeatures(clock, oppClock, base, increment, this.meta.log_clock_scale);
    const planes = encode(chess);
    const turn = chess.turn();
    const indices = legal.map((m) => this.indexOf(m, turn));
    const run = (s) => {
      const out = this.forward(planes, bucketOf(rating, edges), bucketOf(oppRating, edges), context, s);
      return { probs: softmax(indices.map((i) => out.logits[i])), wdl: out.wdl };
    };
    let { probs, wdl } = run(null);
    if (style) {                         // the style blended with the base model (StylePredictor in Python)
      const w = style.weight;
      const styled = run(style);
      probs = probs.map((p, i) => w * styled.probs[i] + (1 - w) * p);
      wdl = wdl.map((v, i) => w * styled.wdl[i] + (1 - w) * v);
    }
    const moves = legal.map((m, i) => ({ uci: m.lan, san: m.san, from: m.from, to: m.to,
                                         promotion: m.promotion || null, prob: probs[i] }));
    moves.sort((a, b) => b.prob - a.prob);
    return { moves, wdl };
  }

  /** Compare with the PyTorch numbers stored at export (the model's, or a style's): the largest difference. */
  selfTest(Chess, style = null) {
    let worst = 0;
    for (const test of (style ? style.tests : this.meta.tests)) {
      const { moves, wdl } = this.predict(new Chess(test.fen), test.rating, { oppRating: test.opp_rating, style });
      if (moves.length !== Object.keys(test.probs).length) return { ok: false, worst: Infinity, fen: test.fen };
      for (const m of moves) worst = Math.max(worst, Math.abs(m.prob - test.probs[m.uci]));
      wdl.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - test.wdl[i])); });
    }
    return { ok: worst < 1e-3, worst };
  }
}

export async function loadModel(base = "model/") {
  const [manifest, buffer] = await Promise.all([
    fetch(base + "model.json").then((r) => { if (!r.ok) throw new Error("model.json not found"); return r.json(); }),
    fetch(base + "model.bin").then((r) => { if (!r.ok) throw new Error("model.bin not found"); return r.arrayBuffer(); }),
  ]);
  return new Model(manifest, buffer);
}

/** The styles the site offers (styles/index.json): [{ name, kind, player, games, rating, until }]; none if absent. */
export async function loadStyles(base = "styles/") {
  try {
    const response = await fetch(base + "index.json");
    if (!response.ok) return [];
    const index = await response.json();
    return Array.isArray(index) ? index : [];
  } catch {
    return [];
  }
}

/** One style's adapter (styles/<name>.json), ready for Model.predict's `style` option. */
export async function loadStyle(name, base = "styles/") {
  const response = await fetch(`${base}${encodeURIComponent(name)}.json`);
  if (!response.ok) throw new Error(`style ${name} not found`);
  const data = await response.json();
  return { ...data, offset: Float32Array.from(data.offset), blockShift: Float32Array.from(data.block_shift.flat()),
           policyBias: Float32Array.from(data.policy_bias) };
}

/** Pick a move: ``temperature`` 1 plays each move as often as the players do; lower favours the likeliest. */
export function sampleMove(moves, temperature = 1, random = Math.random) {
  if (!moves.length) return null;
  const weights = moves.map((m) => Math.pow(m.prob, 1 / Math.max(temperature, 1e-3)));
  const total = weights.reduce((a, b) => a + b, 0);
  let r = random() * total;
  for (let i = 0; i < moves.length; i++) {
    r -= weights[i];
    if (r <= 0) return moves[i];
  }
  return moves[moves.length - 1];
}
