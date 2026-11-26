// Game review in the browser: every move evaluated with Stockfish, classified and scored, as
// src/chess_engine/review/ (scoring.py, classify.py, analyzer.py) does it.
//
// How a move is judged: the position before it is analysed (the engine's best move and evaluation), and so is the
// position after it. The mover's win chance before the move (best play) against after it (the move played) gives
// the win % lost, the move's accuracy and its class. Centipawn loss is the same comparison in centipawns, each side
// capped at 1000.

import { Chess } from "../vendor/chess.js";
import { MATE_CP } from "./stockfish.js";

export const CP_CAP = 1000;

/** The defaults of configs/default.yaml (review:). */
export const REVIEW = { multipv: 3, inaccuracyDrop: 10, mistakeDrop: 20, blunderDrop: 30, goodDrop: 2, keyMoments: 5 };

export const CLASSES = ["best", "good", "inaccuracy", "mistake", "blunder"];
export const SYMBOLS = { best: "!", good: "", inaccuracy: "?!", mistake: "?", blunder: "??" };
const ERRORS = new Set(["inaccuracy", "mistake", "blunder"]);
export const isError = (c) => ERRORS.has(c);

// --------------------------------------------------------------------------- scoring (Lichess's formulas)

/** The side's win chance in % for an evaluation of `cp` centipawns (mates give 100 / 0). */
export function winPercent(cp) {
  if (cp >= MATE_CP - 1000) return 100;
  if (cp <= -(MATE_CP - 1000)) return 0;
  const capped = Math.max(-CP_CAP, Math.min(CP_CAP, cp));
  return 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * capped)) - 1);
}

/** Accuracy (0-100) of a move that took the mover's win chance from `before` to `after`. */
export function moveAccuracy(before, after) {
  const lost = Math.max(0, before - after);
  const raw = 103.1668100711649 * Math.exp(-0.04354415386753951 * lost) - 3.166924740191411;
  return Math.max(0, Math.min(100, raw + 1));
}

function harmonicMean(values) {
  const safe = values.map((v) => Math.max(v, 1e-6));
  return safe.length / safe.reduce((acc, v) => acc + 1 / v, 0);
}

function std(values) {                                       // numpy's default: the population deviation
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length);
}

/** A player's accuracy over a game (Lichess's method: volatility-weighted mean and harmonic mean, averaged). */
export function gameAccuracy(moveAccuracies, whiteWins, colorMoves) {
  if (!moveAccuracies.length) return NaN;
  const window = Math.max(2, Math.min(8, Math.floor(whiteWins.length / 10)));
  const weights = colorMoves.map((ply) => {
    const start = Math.max(0, Math.min(ply + 1 - window, whiteWins.length - window));
    return Math.max(0.5, Math.min(12, std(whiteWins.slice(start, start + window))));
  });
  const total = weights.reduce((a, b) => a + b, 0);
  const weighted = moveAccuracies.reduce((acc, a, i) => acc + a * weights[i], 0) / total;
  return (weighted + harmonicMean(moveAccuracies)) / 2;
}

/** The class of a move that lost `winLost` win-% points (negative = gained). */
export function classify(winLost, isEngineMove, cfg = REVIEW) {
  if (isEngineMove || winLost <= cfg.goodDrop) return "best";
  if (winLost >= cfg.blunderDrop) return "blunder";
  if (winLost >= cfg.mistakeDrop) return "mistake";
  if (winLost >= cfg.inaccuracyDrop) return "inaccuracy";
  return "good";
}

// --------------------------------------------------------------------------- reviewing a game

const cap = (cp) => Math.max(-CP_CAP, Math.min(CP_CAP, cp));

/** "12." for White's move at ply 22, "12..." for Black's at ply 23 (plies counted from the start position). */
export function moveNumber(ply, color, startFullmove = 1, blackFirst = false) {
  const offset = blackFirst ? 1 : 0;
  const number = startFullmove + Math.floor((ply + offset) / 2);
  return color === "white" ? `${number}.` : `${number}...`;
}

/**
 * Review `ucis` played from `startFen` with `analyzer` (anything with `analyse(fen) -> { lines, bestCp }`).
 * `onProgress(done, total)` is told as positions are analysed. Returns { headers, startFen, depth, moves }.
 */
export async function reviewMoves(startFen, ucis, analyzer, { cfg = REVIEW, headers = {}, depth = null,
                                                            onProgress = () => {} } = {}) {
  const chess = new Chess(startFen);
  const fens = [chess.fen()];
  const played = [];
  for (const uci of ucis) {
    let move;
    try {
      move = chess.move(uci);
    } catch {
      throw new Error(`illegal move ${uci} in ${chess.fen()}`);
    }
    played.push(move);
    fens.push(chess.fen());
  }
  const analyses = [];
  for (const fen of fens) {
    analyses.push(await analyzer.analyse(fen));
    onProgress(analyses.length, fens.length);
  }
  const fullmove = Number(startFen.split(" ")[5]) || 1;
  const blackFirst = startFen.split(" ")[1] === "b";
  const moves = played.map((move, ply) => {
    const analysis = analyses[ply];
    const after = analyses[ply + 1];
    const color = move.color === "w" ? "white" : "black";
    const line = analysis.lines.find((l) => l.uci === move.lan);
    const bestCp = analysis.bestCp;
    const best = analysis.lines[0] || null;
    const isEngineMove = Boolean(best && best.uci === move.lan);
    let playedCp = line ? line.cp : -after.bestCp;
    if (isEngineMove) playedCp = bestCp;
    const winBefore = winPercent(bestCp);
    const winAfter = winPercent(playedCp);
    const lost = winBefore - winAfter;
    const sign = color === "white" ? 1 : -1;
    return {
      ply, color, san: move.san, uci: move.lan, fenBefore: move.before, fenAfter: move.after,
      moveNumber: moveNumber(ply, color, fullmove, blackFirst),
      evalBefore: sign * bestCp, evalAfter: sign * playedCp,
      bestSan: best ? best.san : null, bestUci: best ? best.uci : null, bestCp, playedCp,
      cpLoss: Math.max(0, cap(bestCp) - cap(playedCp)),
      winBefore, winAfter, winLost: lost, accuracy: moveAccuracy(winBefore, winAfter),
      classification: classify(lost, isEngineMove, cfg),
      alternatives: analysis.lines.map((l) => ({ san: l.san, uci: l.uci, cp: l.cp, pvSan: l.pvSan || [] })),
    };
  });
  return { headers, startFen: fens[0], depth, moves };
}

/** White's win % before every move (best play) and after the last one. */
export function whiteWinCurve(review) {
  if (!review.moves.length) return [];
  const curve = review.moves.map((m) => winPercent(m.evalBefore));
  curve.push(winPercent(review.moves[review.moves.length - 1].evalAfter));
  return curve;
}

/** Per side: { color, moves, accuracy, averageCpLoss, counts }. */
export function summary(review) {
  const curve = whiteWinCurve(review);
  const out = {};
  for (const color of ["white", "black"]) {
    const moves = review.moves.filter((m) => m.color === color);
    const counts = Object.fromEntries(CLASSES.map((c) => [c, moves.filter((m) => m.classification === c).length]));
    out[color] = {
      color, moves: moves.length, counts,
      accuracy: moves.length ? gameAccuracy(moves.map((m) => m.accuracy), curve, moves.map((m) => m.ply)) : null,
      averageCpLoss: moves.length ? moves.reduce((a, m) => a + m.cpLoss, 0) / moves.length : null,
    };
  }
  return out;
}

const sideName = (color) => (color === "white" ? "White" : "Black");

/** "White: accuracy 87%, average loss 31 cp, 1 inaccuracy, 2 mistakes". */
export function describeSide(side) {
  if (!side.moves) return `${side.color}: no moves`;
  const plural = { inaccuracy: "inaccuracies", mistake: "mistakes", blunder: "blunders" };
  const errors = ["inaccuracy", "mistake", "blunder"].filter((k) => side.counts[k])
    .map((k) => `${side.counts[k]} ${side.counts[k] === 1 ? k : plural[k]}`).join(", ");
  return `${sideName(side.color)}: accuracy ${Math.round(side.accuracy)}%, average loss `
    + `${Math.round(side.averageCpLoss)} cp${errors ? `, ${errors}` : ", no errors"}`;
}

/** "3... Nf6?? dropped Black's chances from 50% to 4%. Better was g6." */
export function describeMoment(m) {
  let text = `${m.moveNumber} ${m.san}${SYMBOLS[m.classification]} dropped ${sideName(m.color)}'s chances from `
    + `${Math.round(m.winBefore)}% to ${Math.round(m.winAfter)}%.`;
  if (m.bestSan && m.bestSan !== m.san) text += ` Better was ${m.bestSan}.`;
  return text;
}

/** The `count` moves that lost the most win %, errors only, in game order, each with its explanation. */
export function keyMoments(review, count = REVIEW.keyMoments) {
  const worst = review.moves.filter((m) => isError(m.classification))
    .sort((a, b) => b.winLost - a.winLost).slice(0, count);
  return worst.sort((a, b) => a.ply - b.ply).map((m) => ({ ...m, text: describeMoment(m) }));
}

/** Per ply (0 = the start): White's win % and evaluation in centipawns (capped at +-1000). */
export function evalGraph(review) {
  if (!review.moves.length) return [];
  const points = review.moves.map((m) => ({ ply: m.ply, whiteWin: winPercent(m.evalBefore), eval: cap(m.evalBefore) }));
  const last = review.moves[review.moves.length - 1];
  points.push({ ply: last.ply + 1, whiteWin: winPercent(last.evalAfter), eval: cap(last.evalAfter) });
  return points;
}
