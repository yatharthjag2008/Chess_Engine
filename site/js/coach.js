// The coach in the browser: your level vs. the next level, and mistakes explained in plain words, as
// src/chess_engine/review/ (level_compare.py, coach.py, features.py) does it.
//
// For each of your moves the model gives the likeliest move at your rating and at +200 / +400. Where the levels
// disagree and Stockfish says the stronger level's choice keeps clearly more winning chances (at least 8 win-%
// points), the position is a lesson. Mistakes from the review are explained by the first rule that applies: the best
// move mates; the move allows a forced mate; it leaves a piece hanging; the best move wins material; otherwise
// which move was stronger. Every sentence comes from facts checkable on the board and the engine's numbers.

import { Chess } from "../vendor/chess.js";
import { MATE_CP } from "./stockfish.js";
import { isError, winPercent } from "./review.js";

/** The defaults of configs/default.yaml (coach:). */
export const COACH = { levelOffsets: [200, 400], minWinGap: 8, topMoves: 3, maxMoments: 5 };

export const PIECE_VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
export const PIECE_NAMES = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };

// --------------------------------------------------------------------------- board facts (features.py)

/** The type of piece `uci` captures in `fen` (en passant counts as a pawn), or null. */
export function capturedPiece(fen, uci) {
  const move = new Chess(fen).move(uci);
  return move.captured || null;
}

export function givesMate(fen, uci) {
  const chess = new Chess(fen);
  chess.move(uci);
  return chess.isCheckmate();
}

/** `color`'s pieces (not the king) the opponent attacks that are undefended or attacked by a cheaper piece,
 *  most valuable first: [{ square, type }]. */
export function hangingPieces(chess, color) {
  const other = color === "w" ? "b" : "w";
  const out = [];
  for (const row of chess.board()) {
    for (const piece of row) {
      if (!piece || piece.color !== color || piece.type === "k") continue;
      const attackers = chess.attackers(piece.square, other);
      if (!attackers.length) continue;
      const defended = chess.attackers(piece.square, color).length > 0;
      const cheapest = Math.min(...attackers.map((sq) => PIECE_VALUES[chess.get(sq).type]));
      if (!defended || cheapest < PIECE_VALUES[piece.type]) out.push({ square: piece.square, type: piece.type });
    }
  }
  return out.sort((a, b) => PIECE_VALUES[b.type] - PIECE_VALUES[a.type]);
}

/** The mover's pieces hanging after `uci` that were not hanging before it (the moved piece included). */
export function newlyHanging(fen, uci) {
  const chess = new Chess(fen);
  const color = chess.turn();
  const before = new Set(hangingPieces(chess, color).map((h) => `${h.square}${h.type}`));
  chess.move(uci);
  return hangingPieces(chess, color).filter((h) => !before.has(`${h.square}${h.type}`));
}

// --------------------------------------------------------------------------- explanations (coach.py)

function pawns(cp) {
  if (Math.abs(cp) >= MATE_CP - 1000) return cp > 0 ? "a forced mate" : "getting mated";
  const value = cp / 100;
  return `${value >= 0 ? "+" : "-"}${Math.abs(value).toFixed(1)}`;
}

/** Why a reviewed move (inaccuracy or worse) was worse than the engine's choice. */
export function explainMistake(move) {
  const sentences = [];
  const otherBest = Boolean(move.bestUci) && move.bestUci !== move.uci;
  const won = otherBest ? capturedPiece(move.fenBefore, move.bestUci) : null;
  const hanging = newlyHanging(move.fenBefore, move.uci);
  if (otherBest && givesMate(move.fenBefore, move.bestUci)) {
    sentences.push(`There was mate in one: ${move.bestSan}.`);
  } else if (move.playedCp <= -(MATE_CP - 1000)) {
    sentences.push("This allows a forced mate.");
  } else if (hanging.length) {
    sentences.push(`This leaves your ${PIECE_NAMES[hanging[0].type]} on ${hanging[0].square} undefended.`);
  } else if (won && move.bestCp - move.playedCp >= 150) {
    sentences.push(`You could have won a ${PIECE_NAMES[won]} with ${move.bestSan}.`);
  } else if (move.bestSan && move.bestSan !== move.san) {
    sentences.push(`${move.bestSan} was stronger (${pawns(move.bestCp)} instead of ${pawns(move.playedCp)}).`);
  }
  sentences.push(`Your winning chances dropped from ${Math.round(move.winBefore)}% to ${Math.round(move.winAfter)}%.`);
  return sentences.join(" ");
}

const pct = (x) => `${Math.round(x * 100)}%`;

/** What players at your rating and at a higher rating play here, and what you played. */
export function explainLevel(moment) {
  const { yours, stronger } = moment.comparison;
  let text = `Players around ${yours.rating} usually play ${yours.san} here (${pct(yours.prob)}).`;
  if (!stronger) return `${text} Stronger players agree.`;
  text += ` Players around ${stronger.rating} prefer ${stronger.san} (${pct(stronger.prob)}), which keeps `
    + `${Math.round(stronger.win)}% winning chances instead of ${Math.round(yours.win)}%.`;
  if (moment.playedUci === stronger.uci) text += ` You found ${stronger.san}: that is the stronger players' move.`;
  else if (moment.playedUci === yours.uci) text += ` You played ${moment.playedSan}, like most players at your level.`;
  else text += ` You played ${moment.playedSan} (${Math.round(moment.playedWin)}%).`;
  return text;
}

// --------------------------------------------------------------------------- your level vs. the next (level_compare.py)

/** Stockfish's value (centipawns, the mover's view) of each move: from the review's lines when there, else by
 *  analysing the position after it. */
export async function moveValues(fen, ucis, analyzer, knownLines = []) {
  const values = {};
  for (const uci of ucis) {
    if (uci in values) continue;
    const line = knownLines.find((l) => l.uci === uci);
    if (line) {
      values[uci] = line.cp;
      continue;
    }
    const chess = new Chess(fen);
    chess.move(uci);
    values[uci] = -(await analyzer.analyse(chess.fen())).bestCp;
  }
  return values;
}

/** Your level against the higher levels in the position before a reviewed move. */
export async function comparePosition(model, move, rating, analyzer, { cfg = COACH, oppRating = null } = {}) {
  const chess = new Chess(move.fenBefore);
  const ratings = [rating, ...cfg.levelOffsets.map((o) => rating + o)];
  const choices = ratings.map((r) => ({
    rating: r, moves: model.predict(chess, r, { oppRating: oppRating ?? r }).moves.slice(0, cfg.topMoves),
  }));
  const values = await moveValues(move.fenBefore, choices.map((c) => c.moves[0].uci), analyzer, move.alternatives);
  const levels = choices.map((c) => {
    const top = c.moves[0];
    return { rating: c.rating, san: top.san, uci: top.uci, prob: top.prob, cp: values[top.uci],
             win: winPercent(values[top.uci]), alternatives: c.moves.slice(1).map((m) => [m.san, m.prob]) };
  });
  const yours = levels[0];
  const stronger = levels.slice(1).find((l) => l.uci !== yours.uci) || null;
  const winGap = stronger ? stronger.win - yours.win : 0;
  return { levels, yours, stronger, winGap, matters: Boolean(stronger) && winGap >= cfg.minWinGap };
}

/**
 * The coach's report on `color`'s moves in a review (see review.js): notes on errors and on the moments where the
 * next level plays clearly better, and how often you played your level's / the next level's favourite move.
 */
export async function coachGame(review, color, rating, model, analyzer, { cfg = COACH, oppRating = null,
                                                                        onProgress = () => {} } = {}) {
  const mine = review.moves.filter((m) => m.color === color);
  const moments = [];
  for (const move of mine) {
    const comparison = await comparePosition(model, move, rating, analyzer, { cfg, oppRating });
    moments.push({ ply: move.ply, moveNumber: move.moveNumber, playedSan: move.san, playedUci: move.uci,
                   playedWin: move.winAfter, comparison });
    onProgress(moments.length, mine.length);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const flagged = moments.filter((m) => m.comparison.matters)
    .sort((a, b) => b.comparison.winGap - a.comparison.winGap).slice(0, cfg.maxMoments);
  const levelByPly = new Map(flagged.map((m) => [m.ply, m]));
  const notes = [];
  for (const move of mine) {
    const kinds = [];
    const parts = [];
    if (isError(move.classification)) {
      kinds.push("mistake");
      parts.push(explainMistake(move));
    }
    if (levelByPly.has(move.ply)) {
      kinds.push("level");
      parts.push(explainLevel(levelByPly.get(move.ply)));
    }
    if (parts.length) notes.push({ ply: move.ply, moveNumber: move.moveNumber, san: move.san, kinds,
                                   classification: move.classification, text: parts.join(" ") });
  }
  const n = moments.length;
  const match = (i) => (n ? moments.filter((m) => m.playedUci === m.comparison.levels[i].uci).length / n : 0);
  return { color, rating, notes, moments, yourLevelMatch: match(0), nextLevelMatch: match(1),
           nextLevel: rating + cfg.levelOffsets[0] };
}
