// Pasted or downloaded PGN -> the moves each player played, with the positions before them, ratings and clocks
// (the browser's version of src/chess_engine/eval/player_games.py).

import { Chess } from "../vendor/chess.js";

/** Split a PGN text holding any number of games into one text per game. */
export function splitPgn(text) {
  const games = [];
  let current = [];
  let inMoves = false;
  for (const line of String(text).replace(/^﻿/, "").split(/\r?\n/)) {
    const header = /^\s*\[[A-Za-z0-9_]+\s+".*"\]\s*$/.test(line);
    if (header && inMoves) {                                   // a header after moves: the next game
      games.push(current.join("\n"));
      current = [];
      inMoves = false;
    }
    if (!header && line.trim()) inMoves = true;
    current.push(line);
  }
  if (current.some((l) => l.trim())) games.push(current.join("\n"));
  return games.map((g) => g.trim()).filter(Boolean);
}

/** "180+2" -> { base: 180, increment: 2 }; Chess.com's "600" is 600+0; anything else (daily, "-") is null. */
export function parseTimeControl(value) {
  const match = /^\s*(\d+)(?:\+(\d+))?\s*$/.exec(String(value || ""));
  return match ? { base: Number(match[1]), increment: Number(match[2] || 0) } : null;
}

/** "[%clk 0:02:58.4]" in a comment -> seconds (null if there is none). */
export function clockOf(comment) {
  const match = /\[%clk\s+(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\]/.exec(comment || "");
  return match ? Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) : null;
}

const elo = (value) => (/^\d+$/.test(String(value || "").trim()) ? Number(value) : null);

/**
 * One game's PGN -> { headers, sides: { w, b }, error } where each side is
 * { color, name, rating, oppRating, result, base, increment, moves: [{ fen, uci, san, clock, oppClock }] }.
 * Clocks are the mover's and the opponent's seconds before each move (null when the PGN has none).
 */
export function readGame(pgn) {
  const chess = new Chess();
  try {
    chess.loadPgn(pgn);
  } catch (err) {
    return { error: `could not be read (${String(err.message || err).split("\n")[0]})` };
  }
  const headers = chess.getHeaders();
  const history = chess.history({ verbose: true });
  const tc = parseTimeControl(headers.TimeControl);
  const base = tc ? tc.base : 180;
  const increment = tc ? tc.increment : 0;
  const comments = new Map(chess.getComments().map((c) => [c.fen, c.comment]));
  const after = history.map((m) => clockOf(comments.get(m.after)));
  const haveClocks = history.length > 0 && after.every((c) => c !== null);
  const padded = [base, base, ...after];                   // before ply i: mover after i-2, opponent after i-1
  const side = (color) => {
    const [me, them] = color === "w" ? ["White", "Black"] : ["Black", "White"];
    return { color, name: headers[me] || "?", rating: elo(headers[`${me}Elo`]), oppRating: elo(headers[`${them}Elo`]),
             result: headers.Result || "*", base, increment, moves: [] };
  };
  const sides = { w: side("w"), b: side("b") };
  history.forEach((m, i) => {
    sides[m.color].moves.push({ fen: m.before, uci: m.lan, san: m.san,
                                clock: haveClocks ? padded[i] : null, oppClock: haveClocks ? padded[i + 1] : null });
  });
  return { headers, sides, plies: history.length };
}

/**
 * Every game in `text`, each player's side as an entry (sides without a move are left out). With `player`, only
 * that player's sides (matched on the White / Black header, ignoring case). Unreadable games are counted.
 */
export function playerGames(text, { player = "" } = {}) {
  const wanted = player.trim().toLowerCase();
  const out = [];
  let unreadable = 0;
  for (const pgn of splitPgn(text)) {
    const game = readGame(pgn);
    if (game.error) {
      unreadable += 1;
      continue;
    }
    for (const color of ["w", "b"]) {
      const side = game.sides[color];
      if (!side.moves.length) continue;
      if (wanted && side.name.trim().toLowerCase() !== wanted) continue;
      out.push({ ...side, headers: game.headers });
    }
  }
  return { games: out, unreadable };
}
