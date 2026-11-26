// "From a position": read what was pasted, a FEN or a PGN, into a chess.js game to carry on from.
//
//   const { chess, error } = parsePosition(text);
//   // chess: a game (a PGN keeps its moves, so they show in the move list); error: a message for the player.

import { Chess, validateFen } from "../vendor/chess.js";

/** A pasted FEN or PGN as a chess.js game, or { error } saying what is wrong. */
export function parsePosition(text) {
  const input = (text || "").trim();
  if (!input) return { error: "Paste a position (FEN) or a game (PGN) first." };
  const looksLikePgn = input.startsWith("[") || /^\d+\s*\./.test(input);
  let chess;
  if (looksLikePgn) {
    chess = new Chess();
    try {
      chess.loadPgn(input);
    } catch (err) {
      return { error: `That PGN could not be read (${err.message.replace(/^Invalid PGN: /, "")}).` };
    }
  } else {
    const check = validateFen(input);
    if (!check.ok) return { error: `That FEN is not valid: ${check.error.replace(/^Invalid FEN: /, "")}` };
    chess = new Chess(input);
  }
  if (chess.isGameOver()) return { error: "The game is already over in that position (mate, stalemate or a draw)." };
  return { chess };
}
