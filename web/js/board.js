// A small, dependency-free chessboard: renders a FEN position and lets the player move.
//
//   const board = new Board(document.getElementById("board"), {
//     orientation: "white",
//     promotionEl: document.getElementById("promotion"),
//     onMove: (uci) => api.play(uci),
//   });
//   board.setPosition(fen, { lastMove: "e2e4", check: false });
//   board.setLegalMoves(state.legal_moves, "white");   // who may move now (null = nobody)
//
// Moving: click a piece to see where it can go, then click the target square, or drag the
// piece there (mouse, pen or touch). The board only offers legal moves (it is given the
// list by the server, it does not know the rules itself). Promotions ask which piece to take.
//
// Pieces are SVG images (the cburnett set in web/pieces/, see its LICENSE.txt), drawn by style.css from
// each piece's data-piece attribute ("wk", "bn", ...), so the board itself has no image paths.

export const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
export const FILES = "abcdefgh";

const NAMES = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };

/** A piece element: `color` "white" | "black", `type` "k" | "q" | ... (the image comes from style.css). */
export function pieceEl(color, type, tag = "span") {
  const el = document.createElement(tag);
  el.className = `piece ${color}`;
  el.dataset.piece = color[0] + type;
  return el;
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
        if (!NAMES[type]) throw new Error(`bad FEN piece: ${ch}`);
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
  constructor(element, { orientation = "white", promotionEl = null, onMove = null } = {}) {
    this.el = element;
    this.orientation = orientation;
    this.promotionEl = promotionEl;
    this.onMove = onMove;
    this.fen = START_FEN;
    this.lastMove = null;
    this.check = false;
    this.legal = [];          // [{ uci, from_square, to_square, promotion }]
    this.mover = null;        // the colour allowed to move, or null
    this.selected = null;     // square of the selected piece
    this.drag = null;         // { from, startX, startY, ghost, moved }
    this.suppressClick = false;
    this.marks = {};          // { square: "class names" } kept across renders (hints, better moves)
    this.el.addEventListener("click", (event) => this.handleClick(event));
    this.el.addEventListener("pointerdown", (event) => this.dragStart(event));
    this.render();
  }

  /** The square under viewport point (x, y), from the board's geometry (null outside the board). */
  squareAt(x, y) {
    const rect = this.el.getBoundingClientRect();
    const col = Math.floor(((x - rect.left) / rect.width) * 8);
    const row = Math.floor(((y - rect.top) / rect.height) * 8);
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    return squareOrder(this.orientation)[row * 8 + col];
  }

  dragStart(event) {
    if (!this.mover || this.promotionOpen || event.button > 0) return;
    const cell = event.target.closest("[data-square]");
    if (!cell) return;
    const piece = parseFen(this.fen).pieces.get(cell.dataset.square);
    if (!piece || piece.color !== this.mover || !this.targetsFrom(cell.dataset.square).length) return;
    event.preventDefault();
    this.drag = { from: cell.dataset.square, startX: event.clientX, startY: event.clientY, ghost: null, moved: false };
    const move = (e) => this.dragMove(e);
    const up = (e) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      this.dragEnd(e);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  }

  dragMove(event) {
    const drag = this.drag;
    if (!drag) return;
    if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 4) return;
    if (!drag.moved) {
      drag.moved = true;
      this.select(drag.from);                     // show the targets while dragging
      const source = this.squareEl(drag.from).querySelector(".piece");
      source.classList.add("dragging");
      const ghost = source.cloneNode(true);
      ghost.classList.remove("dragging");
      ghost.classList.add("drag-ghost");
      const size = this.el.getBoundingClientRect().width / 8;
      ghost.style.width = ghost.style.height = `${size}px`;
      document.body.append(ghost);
      drag.ghost = ghost;
    }
    drag.ghost.style.left = `${event.clientX}px`;
    drag.ghost.style.top = `${event.clientY}px`;
  }

  dragEnd(event) {
    const drag = this.drag;
    this.drag = null;
    if (!drag || !drag.moved) return;             // a plain click: the click handler deals with it
    drag.ghost.remove();
    this.suppressClick = true;                    // the browser also fires a click after a drag
    setTimeout(() => { this.suppressClick = false; }, 0);
    const to = this.squareAt(event.clientX, event.clientY);
    const moves = this.targetsFrom(drag.from).filter((m) => m.to_square === to);
    if (moves.length) {
      this.commit(moves);
    } else {
      this.select(null);                          // dropped off target: the piece goes back
    }
  }

  /** Moves the player may make now (from the API's legal_moves) and whose they are. */
  setLegalMoves(moves, mover) {
    this.legal = moves || [];
    this.mover = mover;
    this.selected = null;
    this.render();
  }

  targetsFrom(square) {
    return this.legal.filter((m) => m.from_square === square);
  }

  handleClick(event) {
    const cell = event.target.closest("[data-square]");
    if (!cell || !this.mover || this.promotionOpen || this.suppressClick) return;
    const square = cell.dataset.square;
    if (this.selected) {
      const moves = this.targetsFrom(this.selected).filter((m) => m.to_square === square);
      if (moves.length) {
        this.commit(moves);
        return;
      }
    }
    const piece = parseFen(this.fen).pieces.get(square);
    const ownPiece = piece && piece.color === this.mover && this.targetsFrom(square).length > 0;
    this.select(ownPiece && this.selected !== square ? square : null);
  }

  select(square) {
    this.selected = square;
    this.render();
  }

  /** Play one of `moves` (several only for a promotion: ask which piece). */
  async commit(moves) {
    let move = moves[0];
    if (moves.length > 1) {
      const type = await this.choosePromotion(moves.map((m) => m.promotion));
      if (!type) {            // cancelled
        this.select(null);
        return;
      }
      move = moves.find((m) => m.promotion === type);
    }
    this.selected = null;
    this.render();
    if (this.onMove) this.onMove(move.uci);
  }

  /** Show the promotion choices; resolves with "q" / "r" / "b" / "n", or null if cancelled. */
  choosePromotion(types) {
    const order = ["q", "r", "b", "n"].filter((t) => types.includes(t));
    if (!this.promotionEl) return Promise.resolve(order[0]);
    this.promotionOpen = true;
    return new Promise((resolve) => {
      const finish = (choice) => {
        this.promotionOpen = false;
        this.promotionEl.hidden = true;
        this.promotionEl.replaceChildren();
        resolve(choice);
      };
      const box = document.createElement("div");
      box.className = "choices";
      for (const type of order) {
        const button = pieceEl(this.mover, type, "button");
        button.type = "button";
        button.dataset.promotion = type;
        button.setAttribute("aria-label", `promote to ${NAMES[type]}`);
        button.addEventListener("click", (event) => { event.stopPropagation(); finish(type); });
        box.append(button);
      }
      this.promotionEl.replaceChildren(box);
      this.promotionEl.onclick = (event) => { if (event.target === this.promotionEl) finish(null); };
      this.promotionEl.hidden = false;
    });
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

  /** Extra classes on squares, e.g. { e2: "hint-from" }; {} clears them. */
  setMarks(marks) {
    this.marks = marks || {};
    this.render();
  }

  squareEl(square) {
    return this.el.querySelector(`[data-square="${square}"]`);
  }

  render() {
    const { pieces, turn } = parseFen(this.fen);
    const highlighted = this.lastMove ? [this.lastMove.slice(0, 2), this.lastMove.slice(2, 4)] : [];
    const order = squareOrder(this.orientation);
    const targets = new Set(this.selected ? this.targetsFrom(this.selected).map((m) => m.to_square) : []);
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
      if (square === this.selected) cell.classList.add("selected");
      if (targets.has(square)) cell.classList.add("target");
      if (this.marks[square]) cell.classList.add(...this.marks[square].split(" "));
      if (index % 8 === 0) cell.append(coordinate("rank", square[1]));
      if (index >= 56) cell.append(coordinate("file", square[0]));
      if (piece) {
        cell.append(pieceEl(piece.color, piece.type));
        cell.classList.add("occupied");
      }
      fragment.append(cell);
    });
    this.el.replaceChildren(fragment);
    this.el.classList.toggle("locked", !this.mover);
  }
}

function coordinate(kind, text) {
  const span = document.createElement("span");
  span.className = `coord ${kind}`;
  span.textContent = text;
  return span;
}
