// Adaptive difficulty in the browser: the same rules as src/chess_engine/bot/adaptive.py.
//
// * off:     the bot plays at the rating you choose.
// * between: your rating is kept like an Elo rating, updated after every finished game from the result against
//            the bot's rating; the next bot plays at your rating, so you should win about half the time.
// * live:    during a game the bot watches its own winning chances (the model's value head, averaged over its last
//            few moves) and becomes a step stronger when it is losing or a step weaker when it is winning, never
//            further than a cap from where the game started.

export const MODES = ["off", "between", "live"];

/** The defaults of configs/default.yaml (adaptive:). */
export const ADAPTIVE = {
  startRating: 1200,
  kFactor: 40,
  kNew: 80,          // bigger steps during the first games, so the bot finds your level quickly
  newGames: 5,
  minRating: 800,
  maxRating: 2400,
  roundTo: 50,
  liveStep: 50,
  liveCap: 200,
  liveWindow: 3,
  liveEvery: 4,
  liveAfter: 6,
  liveLow: 0.35,
  liveHigh: 0.65,
};

export const HISTORY = 50;   // games kept in the profile

const clamp = (x, low, high) => Math.min(Math.max(x, low), high);

/** Your expected score (0..1) against a bot rated `bot` (Elo formula). */
export function expectedScore(player, bot) {
  return 1 / (1 + 10 ** ((bot - player) / 400));
}

export function kFor(gamesPlayed, cfg = ADAPTIVE) {
  return gamesPlayed < cfg.newGames ? cfg.kNew : cfg.kFactor;
}

/** Your rating after scoring `score` (1 win, 0.5 draw, 0 loss) against a bot rated `bot`, kept in range. */
export function updateRating(player, bot, score, gamesPlayed, cfg = ADAPTIVE) {
  if (!(score >= 0 && score <= 1)) throw new Error(`score must be between 0 and 1, got ${score}`);
  const updated = player + kFor(gamesPlayed, cfg) * (score - expectedScore(player, bot));
  return clamp(updated, cfg.minRating, cfg.maxRating);
}

/** The bot's rating for your next game: your rating, rounded and kept within the bot's range. */
export function botRatingFor(player, cfg = ADAPTIVE) {
  return clamp(Math.round(player / cfg.roundTo) * cfg.roundTo, cfg.minRating, cfg.maxRating);
}

// --------------------------------------------------------------------------- your profile

/** A fresh profile: { rating, games, history }. */
export function freshProfile(rating = ADAPTIVE.startRating, cfg = ADAPTIVE) {
  return { rating: clamp(Number(rating), cfg.minRating, cfg.maxRating), games: 0, history: [] };
}

/** A profile read from storage, or a fresh one if it is missing or broken. */
export function readProfile(data, cfg = ADAPTIVE) {
  if (!data || !Number.isFinite(Number(data.rating))) return freshProfile(cfg.startRating, cfg);
  return {
    rating: clamp(Number(data.rating), cfg.minRating, cfg.maxRating),
    games: Math.max(0, Math.floor(Number(data.games) || 0)),
    history: Array.isArray(data.history) ? data.history.slice(-HISTORY) : [],
  };
}

/** The profile after one finished game (a new object; the history keeps the latest HISTORY games). */
export function recordGame(profile, botRating, score, mode, { cfg = ADAPTIVE, date = new Date() } = {}) {
  const before = profile.rating;
  const after = updateRating(before, botRating, score, profile.games, cfg);
  const entry = { date: date.toISOString().slice(0, 10), botRating, score, ratingBefore: Math.round(before * 10) / 10,
                  ratingAfter: Math.round(after * 10) / 10, mode };
  return { rating: after, games: profile.games + 1, history: [...profile.history, entry].slice(-HISTORY) };
}

// --------------------------------------------------------------------------- live, within a game

/** Moves the bot's rating during one game from its own view of the position (see adaptive.py's LiveAdjuster). */
export class LiveAdjuster {
  constructor(start, cfg = ADAPTIVE) {
    this.start = start;
    this.cfg = cfg;
    this.rating = start;
    this.expected = [];   // the bot's expected score, per bot move
    this.playedAt = [];   // the rating each bot move was played at
    this.changes = [];    // { botMove, rating, expected }
  }

  /** The rating the bot played at on average (what the game's result is measured against). */
  get meanRating() {
    return this.playedAt.length ? this.playedAt.reduce((a, b) => a + b, 0) / this.playedAt.length : this.start;
  }

  /** After each bot move, with the model's [win, draw, loss] for the bot; the new rating when it changes. */
  observe(wdl) {
    const cfg = this.cfg;
    this.playedAt.push(this.rating);
    if (!wdl) return null;
    const [win, draw] = wdl;
    this.expected.push(win + draw / 2);
    const moves = this.playedAt.length;
    const lastChange = this.changes.length ? this.changes[this.changes.length - 1].botMove : 0;
    if (moves <= cfg.liveAfter || (this.changes.length && moves - lastChange < cfg.liveEvery)) return null;
    const recent = this.expected.slice(-cfg.liveWindow);
    const average = recent.reduce((a, b) => a + b, 0) / recent.length;
    let target;
    if (average < cfg.liveLow) target = this.rating + cfg.liveStep;
    else if (average > cfg.liveHigh) target = this.rating - cfg.liveStep;
    else return null;
    const low = Math.max(this.start - cfg.liveCap, cfg.minRating);
    const high = Math.min(this.start + cfg.liveCap, cfg.maxRating);
    target = clamp(target, low, high);
    if (target === this.rating) return null;
    this.rating = target;
    this.changes.push({ botMove: moves, rating: target, expected: Math.round(average * 1000) / 1000 });
    return target;
  }

  toJSON() {
    return { start: this.start, rating: this.rating, expected: this.expected, playedAt: this.playedAt,
             changes: this.changes };
  }

  static fromJSON(data, cfg = ADAPTIVE) {
    const live = new LiveAdjuster(Number(data.start), cfg);
    live.rating = Number(data.rating) || live.start;
    live.expected = Array.isArray(data.expected) ? data.expected : [];
    live.playedAt = Array.isArray(data.playedAt) ? data.playedAt : [];
    live.changes = Array.isArray(data.changes) ? data.changes : [];
    return live;
  }
}

/** Your score in a finished game: 1 win, 0.5 draw, 0 loss (`winner` is "w", "b" or null). */
export function scoreFor(winner, human) {
  if (winner === null || winner === undefined) return 0.5;
  return winner === human ? 1 : 0;
}
