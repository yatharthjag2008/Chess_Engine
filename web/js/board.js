// A small, dependency-free chessboard: renders a FEN position and (later) lets the player move.
//
//   const board = new Board(document.getElementById("board"), { orientation: "white" });
//   board.setPosition(fen, { lastMove: "e2e4", check: false });
//
// Pieces are Unicode chess symbols (the filled set for both colours, coloured with CSS),
// so there are no image files and no third-party code.

export const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
export const FILES = "abcdefgh";

// U+FE0E asks for the text (not emoji) presentation of the symbol.
const GLYPHS = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const NAMES = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };

export function glyph(type) {
  return GLYPHS[type] + "︎";
}

/** Parse the piece placement and side to move of a FEN.
 *  Returns { pieces: Map("e2" -> { color: "white" | "black", type: "p" }), turn: "white" | "black" }. */
export function parseFen(fen) {
  const [placement, turn = "w"] = fen.trim().split(/\s+/);
  const rows = placement.split("/");
  if (rows.length !== 8) throw new Error(`bad FEN placement: ${placement}`);
  const pieces = new Map();
  rows.forEach((row, i) => {
    const rank = 8 - i;
    let file = 0;
    for (const ch of row) {
      if (/[1-8]/.test(ch)) {
        file += Number(ch);
      } else {
        const type = ch.toLowerCase();
        if (!GLYPHS[type]) throw new Error(`bad FEN piece: ${ch}`);
        pieces.set(FILES[file] + rank, { color: ch === type ? "black" : "white", type });
        file += 1;
      }
    }
    if (file !== 8) throw new Error(`bad FEN rank: ${row}`);
  });
  return { pieces, turn: turn === "b" ? "black" : "white" };
}

/** Squares from the top-left corner as seen by `orientation`, row by row. */
export function squareOrder(orientation) {
  const squares = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = orientation === "white" ? col : 7 - col;
      const rank = orientation === "white" ? 8 - row : row + 1;
      squares.push(FILES[file] + rank);
    }
  }
  return squares;
}

export function isLightSquare(square) {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  return (file + rank) % 2 === 0; // a1: 0 + 1 is odd -> dark; h1: 7 + 1 is even -> light
}

export class Board {
  constructor(element, { orientation = "white" } = {}) {
    this.el = element;
    this.orientation = orientation;
    this.fen = START_FEN;
    this.lastMove = null;
    this.check = false;
    this.render();
  }

  setOrientation(orientation) {
    this.orientation = orientation;
    this.render();
  }

  flip() {
    this.setOrientation(this.orientation === "white" ? "black" : "white");
  }

  /** Show `fen`; `lastMove` (UCI) is highlighted, and the king of the side to move if `check`. */
  setPosition(fen, { lastMove = null, check = false } = {}) {
    this.fen = fen;
    this.lastMove = lastMove;
    this.check = check;
    this.render();
  }

  squareEl(square) {
    return this.el.querySelector(`[data-square="${square}"]`);
  }

  render() {
    const { pieces, turn } = parseFen(this.fen);
    const highlighted = this.lastMove ? [this.lastMove.slice(0, 2), this.lastMove.slice(2, 4)] : [];
    const order = squareOrder(this.orientation);
    const fragment = document.createDocumentFragment();
    order.forEach((square, index) => {
      const cell = document.createElement("div");
      cell.className = `square ${isLightSquare(square) ? "light" : "dark"}`;
      cell.dataset.square = square;
      cell.setAttribute("role", "gridcell");
      const piece = pieces.get(square);
      cell.setAttribute("aria-label", piece ? `${square} ${piece.color} ${NAMES[piece.type]}` : square);
      if (highlighted.includes(square)) cell.classList.add("last-move");
      if (this.check && piece && piece.type === "k" && piece.color === turn) cell.classList.add("in-check");
      if (index % 8 === 0) cell.append(coordinate("rank", square[1]));
      if (index >= 56) cell.append(coordinate("file", square[0]));
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece.color}`;
        span.dataset.piece = piece.color[0] + piece.type;
        span.textContent = glyph(piece.type);
        cell.append(span);
        cell.classList.add("occupied");
      }
      fragment.append(cell);
    });
    this.el.replaceChildren(fragment);
  }
}

function coordinate(kind, text) {
  const span = document.createElement("span");
  span.className = `coord ${kind}`;
  span.textContent = text;
  return span;
}
