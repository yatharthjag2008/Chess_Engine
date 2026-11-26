// Estimating a player's rating from their moves, in the browser (the method of
// src/chess_engine/eval/rating_estimation.py):
//
//   log p(rating | moves) = sum over moves of log p(move | rating) / temper + log prior(rating) + const
//
// Each move's log-probability is computed at every rating bucket, placed at the bucket's centre rating and
// interpolated linearly to a grid of ratings; the estimate is the posterior mean, the range its central 80%.
// The prior (how ratings are spread among the training players) and the temper (fitted on held-out games) come
// from data/rating.json, written by scripts/build_rating_prior.py.

import { Chess } from "../vendor/chess.js";
import { bucketOf, contextFeatures, encode } from "./engine.js";

let cached = null;

/** data/rating.json (loaded once): { grid, log_prior, temper, interval, centers, ... }. */
export function loadRatingData(base = "data/") {
  if (!cached) {
    cached = fetch(base + "rating.json").then((r) => {
      if (!r.ok) throw new Error("rating.json not found");
      return r.json();
    });
    cached.catch(() => { cached = null; });                // a failed load may be tried again
  }
  return cached;
}

/** The ratings considered, from min to max in steps. */
export function ratingGrid({ min, max, step }) {
  const grid = [];
  for (let r = min; r <= max; r += step) grid.push(r);
  return grid;
}

/** numpy.interp: linear between the points `xs` (increasing), flat beyond them. */
export function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const last = xs.length - 1;
  if (x >= xs[last]) return ys[last];
  let i = 1;
  while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

/** The normalized posterior over the grid. */
export function posterior(logLikelihood, logPrior, temper = 1) {
  if (!(temper > 0)) throw new Error("temper must be positive");
  const logp = logLikelihood.map((ll, i) => ll / temper + logPrior[i]);
  const max = Math.max(...logp);
  const p = logp.map((v) => Math.exp(v - max));
  const total = p.reduce((a, b) => a + b, 0);
  return p.map((v) => v / total);
}

/** Mean, median and central `interval` of a posterior on `grid`. */
export function summarize(grid, probabilities, interval = 0.8, moves = 0) {
  const cdf = [];
  let sum = 0;
  for (const p of probabilities) cdf.push((sum += p));
  const tail = (1 - interval) / 2;
  const quantile = (q) => interp(q, cdf, grid);
  const estimate = grid.reduce((acc, r, i) => acc + r * probabilities[i], 0);
  return { estimate, median: quantile(0.5), low: quantile(tail), high: quantile(1 - tail), interval, moves };
}

/** A prior centred on the opponents' ratings (one per game; several games multiply), as opponent_log_prior. */
export function opponentLogPrior(grid, oppRatings, sigma) {
  return grid.map((r) => oppRatings.reduce((acc, opp) => acc - 0.5 * ((r - opp) / sigma) ** 2, 0));
}

/**
 * The estimate from per-move bucket log-likelihoods (moves x buckets), with the data prior and temper of `data`.
 * With `oppRatings` (the opponent's rating in each game) and a fitted opponent prior in `data`, the prior is centred
 * on them instead (and its own temper is used): far more accurate when players were paired by rating, but then
 * mostly the opponent's rating rather than the moves. Never for games against a bot of a chosen rating.
 */
export function estimateFromBucketLL(bucketLL, data, { temper = null, oppRatings = null } = {}) {
  const k = data.centers.length;
  const summed = new Array(k).fill(0);
  for (const row of bucketLL) row.forEach((v, b) => { summed[b] += v; });
  const grid = ratingGrid(data.grid);
  const total = grid.map((r) => interp(r, data.centers, summed));
  const useOpponent = Boolean(oppRatings && oppRatings.length && data.opponent);
  const prior = useOpponent ? opponentLogPrior(grid, oppRatings, data.opponent.sigma)
    : data.log_prior || new Array(grid.length).fill(0);
  const t = temper ?? (useOpponent ? data.opponent.temper : data.temper);
  return { ...summarize(grid, posterior(total, prior, t), data.interval, bucketLL.length), withOpponent: useOpponent };
}

/** "about 1430 (80% range 1250-1620, from 31 moves)". */
export function describeEstimate(est) {
  const r10 = (x) => Math.round(x / 10) * 10;
  return `about ${r10(est.estimate)} (${Math.round(est.interval * 100)}% range ${r10(est.low)}-${r10(est.high)}, `
    + `from ${est.moves} move${est.moves === 1 ? "" : "s"})`;
}

// --------------------------------------------------------------------------- per-move likelihoods

function logSoftmaxAt(values, index) {
  const max = Math.max(...values);
  let total = 0;
  for (const v of values) total += Math.exp(v - max);
  return values[index] - max - Math.log(total);
}

/**
 * The log-probability, at every rating bucket, of one move: `fen` the position before it, `uci` the move,
 * `oppRating` the opponent's rating (null: as strong as each candidate rating, since Lichess pairs players of
 * similar strength), clocks in seconds (default: the full base time).
 */
export function moveBucketLL(model, { fen, uci, oppRating = null, clock = null, oppClock = null }, base = 180,
                             increment = 0) {
  const chess = new Chess(fen);
  const legal = chess.moves({ verbose: true });
  const played = legal.findIndex((m) => m.lan === uci);
  if (played < 0) throw new Error(`${uci} is not legal in ${fen}`);
  const planes = encode(chess);
  const context = contextFeatures(clock ?? base, oppClock ?? base, base, increment, model.meta.log_clock_scale);
  const turn = chess.turn();
  const indices = legal.map((m) => model.indexOf(m, turn));
  const k = model.meta.bucket_edges.length;
  const known = oppRating !== null && oppRating !== undefined;
  const oppBucket = known ? bucketOf(oppRating, model.meta.bucket_edges) : null;
  const out = [];
  for (let b = 0; b < k; b++) {
    const { logits } = model.forward(planes, b, known ? oppBucket : b, context);
    out.push(logSoftmaxAt(indices.map((i) => logits[i]), played));
  }
  return out;
}
