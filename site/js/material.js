// Captured pieces and the material balance, from two FEN positions (the game's start and now).
//
//   const { w, b, diff } = captured(START, chess.fen());
//   // w: the black pieces White has taken, e.g. ["p", "p", "n"]; b: the white pieces Black has taken;
//   // diff: White's material minus Black's (pawn 1, knight and bishop 3, rook 5, queen 9).

export const VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
export const ORDER = ["p", "n", "b", "r", "q"];
export const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/** How many of each piece each side has: { w: { p, n, b, r, q, k }, b: { ... } }. */
export function countPieces(fen) {
  const counts = { w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 } };
  for (const ch of fen.split(" ")[0]) {
    const type = ch.toLowerCase();
    if (type in VALUES) counts[ch === type ? "b" : "w"][type] += 1;
  }
  return counts;
}

/** Total material of one side's counts. */
export function materialOf(counts) {
  return Object.entries(counts).reduce((sum, [type, n]) => sum + VALUES[type] * n, 0);
}

/** The pieces of one side missing since the start (the ones the other side captured), cheapest first.
 *  A pawn that promoted was not captured: every piece beyond the start's count came from a pawn. */
function missing(start, now) {
  const promoted = ["n", "b", "r", "q"].reduce((sum, type) => sum + Math.max(0, now[type] - start[type]), 0);
  const pieces = [];
  for (const type of ORDER) {
    const gone = start[type] - now[type] - (type === "p" ? promoted : 0);
    for (let i = 0; i < gone; i++) pieces.push(type);
  }
  return pieces;
}

/** What each side has captured since the start position, and the material difference now. */
export function captured(startFen, fen) {
  const start = countPieces(startFen);
  const now = countPieces(fen);
  return {
    w: missing(start.b, now.b),
    b: missing(start.w, now.w),
    diff: materialOf(now.w) - materialOf(now.b),
  };
}
