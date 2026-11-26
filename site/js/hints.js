// Tiered hints while playing the bot, in the browser: nudge -> point -> show (src/chess_engine/bot/hints.py).
//
// A hint is about Stockfish's best move in the position. The nudge says what kind of move to look for, from one-move
// board facts; the point says which piece; show gives the move. How vague the nudge is depends on how often players
// at your rating find the move (the model's probability for it): a move most players find gets the vaguest nudge,
// a rarely found one names the piece type too. Any hint makes the game unrated.

import { Chess } from "../vendor/chess.js";
import { PIECE_NAMES, PIECE_VALUES, hangingPieces } from "./coach.js";
import { winPercent } from "./review.js";

/** The defaults of configs/default.yaml (hints:). */
export const HINTS = { limit: 3, depth: 12, multipv: 2, easyFindRate: 0.35, hardFindRate: 0.10, onlyMoveGap: 15 };

export const THEMES = ["mate", "check", "win_material", "capture", "promotion", "save_piece", "defend", "threat",
                       "castle", "quiet"];

/** The themes of `uci` in `fen` (the side to move's move), most important first; at least one. */
export function moveThemes(fen, uci) {
  const board = new Chess(fen);
  const me = board.turn();
  const them = me === "w" ? "b" : "w";
  const next = new Chess(fen);
  const move = next.move(uci);
  const themes = [];
  if (next.isCheckmate()) themes.push("mate");
  else if (next.inCheck()) themes.push("check");
  if (move.captured) {
    const hangingBefore = new Set(hangingPieces(board, them).map((h) => h.square));
    const target = move.flags.includes("e") ? move.to[0] + move.from[1] : move.to;
    themes.push(hangingBefore.has(target) ? "win_material" : "capture");
  }
  if (move.promotion) themes.push("promotion");
  const mineBefore = new Set(hangingPieces(board, me).map((h) => h.square));
  const mineAfter = new Set(hangingPieces(next, me).map((h) => h.square));
  if (mineBefore.has(move.from) && !mineAfter.has(move.to)) themes.push("save_piece");
  else if ([...mineBefore].some((sq) => sq !== move.from && !mineAfter.has(sq)) && !next.isCheckmate()) themes.push("defend");
  const theirsBefore = new Set(hangingPieces(board, them).map((h) => h.square));
  const newTargets = hangingPieces(next, them).filter((h) => !theirsBefore.has(h.square) && h.square !== move.to
                                                           && next.attackers(h.square, me).includes(move.to));
  if (newTargets.length && !themes.includes("mate")) themes.push("threat");
  if (move.flags.includes("k") || move.flags.includes("q")) themes.push("castle");
  return themes.length ? themes : ["quiet"];
}

/** The most valuable opponent piece `uci` newly leaves hanging (for "attack their queen"), or null. */
export function threatenedPiece(fen, uci) {
  const board = new Chess(fen);
  const me = board.turn();
  const them = me === "w" ? "b" : "w";
  const next = new Chess(fen);
  const move = next.move(uci);
  const before = new Set(hangingPieces(board, them).map((h) => h.square));
  const targets = hangingPieces(next, them).filter((h) => !before.has(h.square)
                                                        && next.attackers(h.square, me).includes(move.to));
  if (!targets.length) return null;
  return targets.reduce((best, h) => (PIECE_VALUES[h.type] > PIECE_VALUES[best.type] ? h : best)).type;
}

/** How much the nudge gives away: "vague" (usually found), "theme", or "piece" (rarely found). */
export function detailFor(rate, cfg = HINTS) {
  if (rate >= cfg.easyFindRate) return "vague";
  if (rate < cfg.hardFindRate) return "piece";
  return "theme";
}

const FORCING = new Set(["mate", "check", "win_material", "capture", "promotion"]);

const THEME_NUDGE = {
  mate: "There is a checkmate!",
  check: "Look for a strong check.",
  win_material: "One of your opponent's pieces can be won.",
  capture: "Look at the captures.",
  promotion: "One of your pawns can promote.",
  save_piece: "One of your pieces is in danger: save it.",
  defend: "One of your pieces is loose: protect it.",
  threat: "You can attack something your opponent has left loose.",
  castle: "Think about your king's safety.",
  quiet: "Look for a quiet move that improves your position.",
};

const SHOW_REASON = {
  mate: "checkmate", check: "a check", win_material: "it wins material", capture: "a capture",
  promotion: "it promotes", save_piece: "it saves the piece", defend: "it protects your loose piece",
  threat: "it creates a threat", castle: "it gets your king to safety",
};

/** The first hint: what kind of move to look for, giving more away for rarely found moves. */
export function nudgeText(themes, pieceType, detail, { onlyMove = false, threatened = null, winning = false } = {}) {
  const main = themes[0];
  let text;
  if (detail === "vague") {
    if (FORCING.has(main)) text = "There is a forcing move here: look at checks, captures and threats.";
    else if (main === "save_piece" || main === "defend") text = "Before anything else, check that all your pieces are safe.";
    else if (main === "threat") text = "You can put your opponent under pressure.";
    else text = "No need for anything sharp: look for a calm, useful move.";
  } else {
    text = THEME_NUDGE[main];
    if (main === "threat" && threatened) text = `You can attack their ${PIECE_NAMES[threatened]}.`;
    if (detail === "piece") text += ` It's a ${PIECE_NAMES[pieceType]} move.`;
  }
  if (onlyMove && main !== "mate") {
    text += winning ? " Only one move makes the most of it." : " Careful: only one move keeps your position together.";
  }
  return text;
}

export function pointText(fen, uci) {
  const piece = new Chess(fen).get(uci.slice(0, 2));
  return `Move your ${PIECE_NAMES[piece.type]} on ${uci.slice(0, 2)}.`;
}

export function showText(fen, uci, themes) {
  let reasons = themes.filter((t) => SHOW_REASON[t] && (t !== "capture" || t === themes[0])).map((t) => SHOW_REASON[t]);
  if (themes.includes("mate")) reasons = [SHOW_REASON.mate];
  const san = new Chess(fen).move(uci).san;
  return `Play ${san}${reasons.length ? `: ${reasons.slice(0, 2).join(", and ")}.` : "."}`;
}

export const TIERS = ["nudge", "point", "show"];

/** The hint of `tier` about `uci` (the best move) in `fen`. */
export function makeHint(fen, uci, tier, rate, { cfg = HINTS, onlyMove = false, winning = false } = {}) {
  const themes = moveThemes(fen, uci);
  const detail = detailFor(rate, cfg);
  const piece = new Chess(fen).get(uci.slice(0, 2)).type;
  const nudge = nudgeText(themes, piece, detail, { onlyMove, threatened: threatenedPiece(fen, uci), winning });
  const base = { tier, themes, findRate: rate, detail, onlyMove };
  if (tier === "nudge") return { ...base, text: nudge, from: null, to: null, san: null };
  if (tier === "point") return { ...base, text: `${nudge} ${pointText(fen, uci)}`, from: uci.slice(0, 2), to: null, san: null };
  return { ...base, text: showText(fen, uci, themes), from: uci.slice(0, 2), to: uci.slice(2, 4),
           san: new Chess(fen).move(uci).san };
}

// --------------------------------------------------------------------------- the engine and the per-game limit

/** Hints from Stockfish (the best move) and the model (how often your level finds it); positions are remembered,
 *  so asking for the next tier is instant. */
export class HintEngine {
  constructor(model, analyzer, cfg = HINTS) {
    this.model = model;
    this.analyzer = analyzer;
    this.cfg = cfg;
    this.cache = new Map();
  }

  async best(fen, rating, options = {}) {
    const key = `${fen}|${rating}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const chess = new Chess(fen);
    if (chess.isGameOver()) throw new Error("the game is over");
    const analysis = await this.analyzer.analyse(fen);
    if (!analysis.lines.length) throw new Error("the engine found no move");
    const [top, second] = analysis.lines;
    const onlyMove = Boolean(second) && winPercent(top.cp) - winPercent(second.cp) >= this.cfg.onlyMoveGap;
    const prediction = this.model.predict(chess, rating, options);
    const found = prediction.moves.find((m) => m.uci === top.uci);
    const best = { uci: top.uci, findRate: found ? found.prob : 0, onlyMove, cp: top.cp,
                   winning: winPercent(top.cp) >= 60 };
    this.cache.set(key, best);
    return best;
  }

  async hint(fen, tier, rating, options = {}) {
    const best = await this.best(fen, rating, options);
    return makeHint(fen, best.uci, tier, best.findRate, { cfg: this.cfg, onlyMove: best.onlyMove, winning: best.winning });
  }
}

/** A position for hint counting: pieces, side to move, castling and en passant (move counters ignored). */
export const hintKey = (fen) => fen.split(" ").slice(0, 4).join(" ");

/** Hints in one game: `limit` positions may get hints; within a position each request goes one tier further at no
 *  extra cost. Any hint makes the game unrated. */
export class HintBudget {
  constructor(limit = HINTS.limit, { used = 0, given = {} } = {}) {
    this.limit = limit;
    this.used = used;
    this.given = { ...given };      // position -> highest tier index given there
  }

  get left() { return Math.max(0, this.limit - this.used); }

  get rated() { return this.used === 0; }

  canTake(fen) { return hintKey(fen) in this.given || this.used < this.limit; }

  nextTier(fen) {
    const key = hintKey(fen);
    return key in this.given ? TIERS[Math.min(this.given[key] + 1, TIERS.length - 1)] : "nudge";
  }

  /** Record a request and return its tier; throws for a new position when no hints are left. */
  take(fen) {
    const key = hintKey(fen);
    if (!(key in this.given)) {
      if (this.used >= this.limit) throw new Error(this.limit ? "No hints left in this game." : "Hints are off for this game.");
      this.used += 1;
    }
    const tier = this.nextTier(fen);
    this.given[key] = Math.max(this.given[key] ?? -1, TIERS.indexOf(tier));
    return tier;
  }

  toJSON() { return { limit: this.limit, used: this.used, given: this.given }; }

  static fromJSON(data) {
    return new HintBudget(Number(data.limit) || 0, { used: Number(data.used) || 0, given: data.given || {} });
  }
}
