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
// Premoves: while the other side is thinking, `setLegalMoves([], null, { premover: "white" })`
// lets White queue a move the same way. Any square the piece could reach on an empty board is
// offered (the position will change before it is played). Click elsewhere, right-click or
// press Escape to cancel it. When White's turn comes (`setLegalMoves(moves, "white")`), the
// premove is played through onMove if it is legal then, and dropped if not.
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

const KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
const KING = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
const RAYS = { b: [[1, 1], [1, -1], [-1, 1], [-1, -1]], r: [[1, 0], [-1, 0], [0, 1], [0, -1]] };
RAYS.q = [...RAYS.b, ...RAYS.r];

/** Where the piece on `square` could go on an empty board (premoves): its moves by shape, pawn
 *  captures on both diagonals, castling from the king's home square, promotions to a queen. */
export function premoveTargets(fen, square) {
  const { pieces } = parseFen(fen);
  const piece = pieces.get(square);
  if (!piece) return [];
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  const at = (f, r) => (f >= 0 && f < 8 && r >= 1 && r <= 8 ? FILES[f] + r : null);
  const targets = [];
  if (piece.type === "p") {
    const dir = piece.color === "white" ? 1 : -1;
    targets.push(at(file, rank + dir), at(file - 1, rank + dir), at(file + 1, rank + dir));
    if (rank === (dir === 1 ? 2 : 7)) targets.push(at(file, rank + 2 * dir));
  } else if (piece.type === "n" || piece.type === "k") {
    for (const [df, dr] of piece.type === "n" ? KNIGHT : KING) targets.push(at(file + df, rank + dr));
    const home = piece.color === "white" ? 1 : 8;
    const rook = (f) => { const p = pieces.get(FILES[f] + home); return p && p.type === "r" && p.color === piece.color; };
    if (piece.type === "k" && square === "e" + home) {
      if (rook(7)) targets.push("g" + home);
      if (rook(0)) targets.push("c" + home);
    }
  } else {
    for (const [df, dr] of RAYS[piece.type]) {
      for (let f = file + df, r = rank + dr; at(f, r); f += df, r += dr) targets.push(at(f, r));
    }
  }
  const last = piece.color === "white" ? 8 : 1;
  return [...new Set(targets.filter(Boolean))].map((to) => {
    const promotion = piece.type === "p" && Number(to[1]) === last ? "q" : null;
    return { uci: square + to + (promotion || ""), from_square: square, to_square: to, promotion };
  });
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

// Arrows and circles drawn on the board: { orig: "e2", dest: "e4" } is an arrow, { orig: "e4" } (no
// dest) a circle; `brush` is "green" (the default), "red", "blue" or "yellow".
export const BRUSHES = { green: "#15781b", red: "#882020", blue: "#003088", yellow: "#e68f00" };
const SVG = "http://www.w3.org/2000/svg";

/** The colour of a shape from the keys held while drawing: none green, Shift red, Alt blue, both yellow. */
export function brushFor(event) {
  if (event.shiftKey && event.altKey) return "yellow";
  if (event.shiftKey) return "red";
  if (event.altKey) return "blue";
  return "green";
}

function svgEl(name, attrs) {
  const el = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

/** The centre of `square` in board units (0-8 from the top-left corner) as seen by `orientation`. */
export function squareCentre(square, orientation) {
  const index = squareOrder(orientation).indexOf(square);
  return [(index % 8) + 0.5, Math.floor(index / 8) + 0.5];
}

/** The SVG layer with `shapes` drawn for `orientation`. */
export function shapesLayer(shapes, orientation) {
  const svg = svgEl("svg", { class: "shapes", viewBox: "0 0 8 8", "aria-hidden": "true" });
  const defs = svgEl("defs", {});
  for (const [brush, color] of Object.entries(BRUSHES)) {
    const marker = svgEl("marker", { id: `arrowhead-${brush}`, orient: "auto", markerWidth: 4, markerHeight: 4,
                                     refX: 2.05, refY: 2, markerUnits: "strokeWidth" });
    marker.append(svgEl("path", { d: "M0,0 V4 L3,2 Z", fill: color }));
    defs.append(marker);
  }
  svg.append(defs);
  for (const shape of shapes) {
    const brush = BRUSHES[shape.brush] ? shape.brush : "green";
    const [x1, y1] = squareCentre(shape.orig, orientation);
    if (!shape.dest || shape.dest === shape.orig) {
      svg.append(svgEl("circle", { class: `circle ${brush}`, cx: x1, cy: y1, r: 0.46, fill: "none",
                                   stroke: BRUSHES[brush], "stroke-width": 0.0625, opacity: 0.85 }));
      continue;
    }
    const [x2, y2] = squareCentre(shape.dest, orientation);
    const length = Math.hypot(x2 - x1, y2 - y1);
    const end = (length - 0.32) / length;                  // stop short so the head ends near the centre
    svg.append(svgEl("line", { class: `arrow ${brush}${shape.preview ? " preview" : ""}`, x1, y1, x2: x1 + (x2 - x1) * end, y2: y1 + (y2 - y1) * end,
                               stroke: BRUSHES[brush], "stroke-width": 0.16, "stroke-linecap": "round",
                               "marker-end": `url(#arrowhead-${brush})`, opacity: shape.preview ? 0.45 : 0.8 }));
  }
  return svg;
}

export function isLightSquare(square) {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  return (file + rank) % 2 === 0; // a1: 0 + 1 is odd -> dark; h1: 7 + 1 is even -> light
}

export class Board {
  constructor(element, { orientation = "white", promotionEl = null, onMove = null, onPremove = null } = {}) {
    this.el = element;
    this.orientation = orientation;
    this.promotionEl = promotionEl;
    this.onMove = onMove;
    this.onPremove = onPremove;  // called with "set" when a premove is queued, "dropped" when it became illegal
    this.fen = START_FEN;
    this.lastMove = null;
    this.check = false;
    this.legal = [];          // [{ uci, from_square, to_square, promotion }]
    this.mover = null;        // the colour allowed to move, or null
    this.premover = null;     // the colour that may queue a premove while it is not its turn, or null
    this.premove = null;      // the queued premove: { uci, from_square, to_square, promotion }
    this.selected = null;     // square of the selected piece
    this.drag = null;         // { from, startX, startY, ghost, moved }
    this.suppressClick = false;
    this.marks = {};          // { square: "class names" } kept across renders (hints, better moves)
    this.shapes = [];         // arrows and circles drawn on the board (see shapesLayer)
    this.el.addEventListener("click", (event) => this.handleClick(event));
    this.el.addEventListener("pointerdown", (event) => this.dragStart(event));
    this.el.addEventListener("pointerdown", (event) => this.annotateStart(event));
    this.el.addEventListener("contextmenu", (event) => {   // no browser menu on the board
      event.preventDefault();
      if (this.premove || (this.premover && this.selected)) this.cancelPremove();   // right-click cancels a premove
    });
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

  /** Right button down: start drawing, a circle if released on the same square, an arrow if dragged to
   *  another (shown faintly meanwhile). A right-click that cancels a premove draws nothing. */
  annotateStart(event) {
    if (event.button === 0 && this.shapes.length) {        // a left-click clears the drawing
      this.shapes = [];
      this.renderShapes();
      return;
    }
    if (event.button !== 2 || this.premove || (this.premover && this.selected)) return;
    const orig = this.squareAt(event.clientX, event.clientY);
    if (!orig) return;
    event.preventDefault();
    this.drawing = { orig, dest: orig, brush: brushFor(event) };
    const move = (e) => this.annotateMove(e);
    const up = (e) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      this.annotateEnd(e);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  annotateMove(event) {
    const drawing = this.drawing;
    const dest = drawing && this.squareAt(event.clientX, event.clientY);
    if (!dest || dest === drawing.dest) return;
    drawing.dest = dest;
    this.renderShapes();
  }

  /** Redraw only the shapes layer (cheap enough for every pointer move). */
  renderShapes() {
    const layer = this.el.querySelector(".shapes");
    if (layer) layer.replaceWith(this.shapesLayer());
  }

  /** The drawn shapes, plus the one being drawn (marked as a preview). */
  shapesLayer() {
    const drawing = this.drawing;
    const preview = drawing && drawing.dest !== drawing.orig ? [{ ...drawing, preview: true }] : [];
    return shapesLayer([...this.shapes, ...preview], this.orientation);
  }

  annotateEnd(event) {
    const drawing = this.drawing;
    this.drawing = null;
    if (!drawing) return;
    const dest = this.squareAt(event.clientX, event.clientY);
    if (!dest) {
      this.renderShapes();                                   // released off the board: nothing drawn
      return;
    }
    const shape = dest === drawing.orig ? { orig: drawing.orig } : { orig: drawing.orig, dest };
    if (drawing.brush !== "green") shape.brush = drawing.brush;
    this.toggleShape(shape);
  }

  /** Add a shape; the same one in the same colour is removed, in another colour recoloured. */
  toggleShape(shape) {
    const same = (s) => s.orig === shape.orig && (s.dest || null) === (shape.dest || null);
    const index = this.shapes.findIndex(same);
    const brush = (s) => s.brush || "green";
    if (index < 0) this.shapes = [...this.shapes, shape];
    else if (brush(this.shapes[index]) === brush(shape)) this.shapes = this.shapes.filter((_, i) => i !== index);
    else this.shapes = this.shapes.map((s, i) => (i === index ? shape : s));
    this.render();
  }

  /** Whose pieces can be picked up now: the side to move, or the side that may premove. */
  get active() {
    return this.mover || this.premover;
  }

  dragStart(event) {
    if (!this.active || this.promotionOpen || event.button > 0) return;
    const cell = event.target.closest("[data-square]");
    if (!cell) return;
    const piece = parseFen(this.fen).pieces.get(cell.dataset.square);
    if (!piece || piece.color !== this.active || !this.targetsFrom(cell.dataset.square).length) return;
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
    const over = this.squareAt(event.clientX, event.clientY);   // outline the square the piece would land on
    if (over !== drag.over) {
      if (drag.over) this.squareEl(drag.over)?.classList.remove("drop-over");
      if (over) this.squareEl(over)?.classList.add("drop-over");
      drag.over = over;
    }
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
      this.dragged = true;
      this.commit(moves);
    } else {
      this.select(null);                          // dropped off target: the piece goes back
    }
  }

  /** Moves the player may make now (from the API's legal_moves) and whose they are; while nobody may
   *  move, `premover` names the colour that may queue a premove (null: none, and a queued one is dropped). */
  setLegalMoves(moves, mover, { premover = null } = {}) {
    const queued = mover && this.premove && this.premove.color === mover ? this.premove : null;
    this.legal = moves || [];
    this.mover = mover;
    this.premover = mover ? null : premover;
    if (mover || !premover) this.premove = null;
    this.selected = null;
    this.render();
    const move = queued && this.legal.find((m) => m.from_square === queued.from_square &&
      m.to_square === queued.to_square && (m.promotion || null) === queued.promotion);
    if (queued && !move && this.onPremove) this.onPremove("dropped");
    if (move) {
      // After the caller has finished updating (this is often called while it renders).
      const legal = this.legal;
      setTimeout(() => {
        if (this.mover === mover && this.legal === legal && this.onMove) this.onMove(move.uci);
      }, 0);
    }
  }

  targetsFrom(square) {
    if (!this.mover && this.premover) return premoveTargets(this.fen, square);
    return this.legal.filter((m) => m.from_square === square);
  }

  /** Forget the queued premove (and any piece picked up for one). */
  cancelPremove() {
    this.premove = null;
    this.selected = null;
    this.render();
  }

  handleClick(event) {
    const cell = event.target.closest("[data-square]");
    if (!cell || !this.active || this.promotionOpen || this.suppressClick) return;
    const square = cell.dataset.square;
    if (this.selected) {
      const moves = this.targetsFrom(this.selected).filter((m) => m.to_square === square);
      if (moves.length) {
        this.commit(moves);
        return;
      }
    }
    const piece = parseFen(this.fen).pieces.get(square);
    const ownPiece = piece && piece.color === this.active && this.targetsFrom(square).length > 0;
    if (!this.mover && this.premove && !ownPiece) this.premove = null;   // a click elsewhere cancels it
    this.select(ownPiece && this.selected !== square ? square : null);
  }

  select(square) {
    this.selected = square;
    this.render();
  }

  /** Play one of `moves` (several only for a promotion: ask which piece). */
  async commit(moves) {
    if (!this.mover && this.premover) {           // not our turn: queue it (promotions become queens)
      this.premove = { ...(moves.find((m) => m.promotion === "q") || moves[0]), color: this.premover };
      this.selected = null;
      this.dragged = false;
      this.render();
      if (this.onPremove) this.onPremove("set");
      return;
    }
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
    if (this.dragged) this.dropped = move.uci;    // already where it was dropped: no slide
    this.dragged = false;
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

  /** Show `fen`; `lastMove` (UCI) is highlighted, and the king of the side to move if `check`.
   *  When `lastMove` leads from the position shown to this one, its piece slides there. */
  setPosition(fen, { lastMove = null, check = false } = {}) {
    const slide = lastMove && lastMove !== this.dropped && this.movedBy(lastMove, fen) ? lastMove : null;
    this.dropped = null;
    if (fen !== this.fen) this.shapes = [];                 // a new position: the old arrows no longer apply
    this.fen = fen;
    this.lastMove = lastMove;
    this.check = check;
    this.render();
    if (slide) this.slide(slide);
  }

  /** Did `uci` take the piece on its from-square in the shown position to its to-square in `fen`? */
  movedBy(uci, fen) {
    const before = parseFen(this.fen).pieces.get(uci.slice(0, 2));
    const after = parseFen(fen).pieces.get(uci.slice(2, 4));
    return Boolean(before && after && before.color === after.color && fen !== this.fen);
  }

  /** Animate the piece on `uci`'s to-square from its from-square. */
  slide(uci) {
    const from = this.squareEl(uci.slice(0, 2));
    const to = this.squareEl(uci.slice(2, 4));
    const piece = to && to.querySelector(".piece");
    if (!from || !piece || !piece.animate || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const a = from.getBoundingClientRect();
    const b = to.getBoundingClientRect();
    piece.style.zIndex = 5;                       // above the pieces it passes
    const animation = piece.animate(
      [{ transform: `translate(${a.left - b.left}px, ${a.top - b.top}px)` }, { transform: "none" }],
      { duration: 200, easing: "cubic-bezier(0.2, 0.7, 0.3, 1)" });
    animation.onfinish = () => { piece.style.zIndex = ""; };
  }

  /** Arrows and circles to draw, e.g. [{ orig: "e2", dest: "e4" }, { orig: "d5", brush: "red" }]; [] clears. */
  setShapes(shapes) {
    this.shapes = shapes || [];
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
      if (this.premove && (square === this.premove.from_square || square === this.premove.to_square)) {
        cell.classList.add("premove");
      }
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
    fragment.append(this.shapesLayer());
    this.el.replaceChildren(fragment);
    this.el.classList.toggle("locked", !this.active);
    this.el.classList.toggle("premoving", !this.mover && Boolean(this.premover));
  }
}

function coordinate(kind, text) {
  const span = document.createElement("span");
  span.className = `coord ${kind}`;
  span.textContent = text;
  return span;
}
