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

/** Positions to practise from (all checked legal and not over with python-chess). */
export const PRESETS = [
  { name: "Checkmate with king and queen", fen: "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1" },
  { name: "Checkmate with king and rook", fen: "8/8/8/4k3/8/8/4K3/R7 w - - 0 1" },
  { name: "Checkmate with two bishops", fen: "8/8/8/4k3/8/8/8/2B1KB2 w - - 0 1" },
  { name: "Checkmate with two rooks", fen: "8/8/8/4k3/8/8/8/R3K2R w - - 0 1" },
  { name: "King and pawn against king", fen: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" },
  { name: "Lucena position (rook endgame)", fen: "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1" },
  { name: "Italian Game, after 6...Bb4+", fen: "r1bqk2r/pppp1ppp/2n2n2/8/1bBPP3/5N2/PP3PPP/RNBQK2R w KQkq - 1 7" },
  { name: "Queen's Gambit Declined, after 4...Be7",
    fen: "rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR w KQkq - 4 5" },
  { name: "Sicilian Najdorf, after 5...a6", fen: "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6" },
];
