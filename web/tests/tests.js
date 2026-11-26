// Unit tests of board.js (and review.js, in review_unit.js), run in a real browser (see harness.js and tests/test_web_js.py).

import { Board, START_FEN, isLightSquare, parseFen, premoveTargets, squareOrder } from "../js/board.js";
import { assert, equal, run, test } from "./harness.js";
import "./review_unit.js";

function fixture() {
  // A fresh element per test, so listeners of earlier boards cannot react to this test's clicks.
  const old = document.getElementById("fixture");
  const el = document.createElement("div");
  el.id = "fixture";
  el.className = "board";
  el.style.cssText = "width: 400px; height: 400px";
  old.replaceWith(el);
  return el;
}

// ------------------------------------------------------------------ board.js

test("parseFen reads pieces and the side to move", () => {
  const { pieces, turn } = parseFen(START_FEN);
  equal(pieces.size, 32);
  equal(pieces.get("e1"), { color: "white", type: "k" });
  equal(pieces.get("d8"), { color: "black", type: "q" });
  equal(pieces.get("e4"), undefined);
  equal(turn, "white");
  equal(parseFen("8/8/8/8/8/8/8/K6k b - - 0 1").turn, "black");
});

test("parseFen rejects broken placements", () => {
  for (const bad of ["8/8/8", "9/8/8/8/8/8/8/8 w", "8/8/8/8/8/8/8/7X w"]) {
    let threw = false;
    try { parseFen(bad); } catch { threw = true; }
    assert(threw, `accepted ${bad}`);
  }
});

test("square order follows the orientation", () => {
  const white = squareOrder("white");
  equal([white[0], white[7], white[56], white[63]], ["a8", "h8", "a1", "h1"]);
  const black = squareOrder("black");
  equal([black[0], black[7], black[56], black[63]], ["h1", "a1", "h8", "a8"]);
  equal(new Set(white).size, 64);
  assert(isLightSquare("h1") && !isLightSquare("a1") && isLightSquare("a8"), "square colours");
});

test("board renders 64 squares with pieces and coordinates", () => {
  const board = new Board(fixture());
  const squares = board.el.querySelectorAll(".square");
  equal(squares.length, 64);
  equal(board.el.querySelectorAll(".piece").length, 32);
  equal(board.squareEl("e1").querySelector(".piece").dataset.piece, "wk");
  const king = getComputedStyle(board.squareEl("e1").querySelector(".piece")).backgroundImage;
  assert(king.includes("pieces/wK.svg"), `the white king's image: ${king}`);
  equal(board.el.querySelectorAll(".coord.rank").length, 8);
  equal(board.el.querySelectorAll(".coord.file").length, 8);
  equal(squares[0].dataset.square, "a8");
  assert(board.squareEl("a1").classList.contains("dark"), "a1 is dark");
});

test("flip, last move and check highlights", () => {
  const board = new Board(fixture());
  board.flip();
  equal(board.el.querySelector(".square").dataset.square, "h1");
  board.setPosition("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", { lastMove: "d8h4", check: true });
  assert(board.squareEl("d8").classList.contains("last-move") && board.squareEl("h4").classList.contains("last-move"));
  assert(board.squareEl("e1").classList.contains("in-check"), "white king in check");
  assert(!board.squareEl("e8").classList.contains("in-check"), "black king not in check");
  equal(board.squareEl("h4").getAttribute("aria-label"), "h4 black queen");
});

// ------------------------------------------------------------------ moving

const START_MOVES = [
  { uci: "e2e4", san: "e4", from_square: "e2", to_square: "e4", promotion: null },
  { uci: "e2e3", san: "e3", from_square: "e2", to_square: "e3", promotion: null },
  { uci: "g1f3", san: "Nf3", from_square: "g1", to_square: "f3", promotion: null },
];

function click(board, square) {
  board.squareEl(square).dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

test("click a piece, then a target, plays the move", () => {
  const played = [];
  const board = new Board(fixture(), { onMove: (uci) => played.push(uci) });
  board.setLegalMoves(START_MOVES, "white");
  click(board, "e2");
  assert(board.squareEl("e2").classList.contains("selected"), "e2 selected");
  equal([...board.el.querySelectorAll(".target")].map((c) => c.dataset.square).sort(), ["e3", "e4"]);
  click(board, "e4");
  equal(played, ["e2e4"]);
  equal(board.el.querySelectorAll(".selected, .target").length, 0, "selection cleared");
});

test("clicks that are not moves only change the selection", () => {
  const played = [];
  const board = new Board(fixture(), { onMove: (uci) => played.push(uci) });
  board.setLegalMoves(START_MOVES, "white");
  click(board, "e7");                     // opponent piece: nothing
  equal(board.selected, null);
  click(board, "a2");                     // own piece with no legal move in the list: nothing
  equal(board.selected, null);
  click(board, "e2");
  click(board, "g1");                     // another own piece: switch the selection
  equal(board.selected, "g1");
  click(board, "g1");                     // the same piece again: deselect
  equal(board.selected, null);
  click(board, "g1");
  click(board, "h5");                     // empty, unreachable square: deselect
  equal(board.selected, null);
  equal(played, []);
});

test("no moves while it is not the player's turn", () => {
  const played = [];
  const board = new Board(fixture(), { onMove: (uci) => played.push(uci) });
  board.setLegalMoves(START_MOVES, null);
  click(board, "e2");
  click(board, "e4");
  equal(played, []);
  assert(board.el.classList.contains("locked"), "board locked");
});

test("promotion asks which piece and plays the choice", async () => {
  const played = [];
  const promotionEl = document.createElement("div");
  promotionEl.hidden = true;
  document.body.append(promotionEl);
  const board = new Board(fixture(), { promotionEl, onMove: (uci) => played.push(uci) });
  board.setPosition("8/4P1k1/8/8/8/8/8/K7 w - - 0 1");
  board.setLegalMoves(["q", "r", "b", "n"].map((p) => (
    { uci: "e7e8" + p, san: "e8=" + p.toUpperCase(), from_square: "e7", to_square: "e8", promotion: p })), "white");
  click(board, "e7");
  click(board, "e8");
  await Promise.resolve();
  assert(!promotionEl.hidden, "chooser shown");
  equal([...promotionEl.querySelectorAll("button")].map((b) => b.dataset.promotion), ["q", "r", "b", "n"]);
  promotionEl.querySelector('[data-promotion="n"]').click();
  await new Promise((r) => setTimeout(r, 0));
  equal(played, ["e7e8n"]);
  assert(promotionEl.hidden, "chooser closed");

  click(board, "e7");                    // cancelling (click outside the choices) plays nothing
  click(board, "e8");
  await Promise.resolve();
  promotionEl.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 0));
  equal(played, ["e7e8n"]);
  promotionEl.remove();
});

// ------------------------------------------------------------------ dragging

function pointer(type, target, x, y) {
  target.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0, pointerId: 1 }));
}

function centre(board, square) {
  const rect = board.squareEl(square).getBoundingClientRect();
  return [rect.left + rect.width / 2, rect.top + rect.height / 2];
}

test("squareAt maps points to squares in both orientations", () => {
  const board = new Board(fixture());
  const rect = board.el.getBoundingClientRect();
  equal(board.squareAt(rect.left + 1, rect.top + 1), "a8");
  equal(board.squareAt(rect.right - 1, rect.bottom - 1), "h1");
  equal(board.squareAt(rect.left - 5, rect.top + 5), null);
  board.flip();
  equal(board.squareAt(rect.left + 1, rect.top + 1), "h1");
});

test("dragging a piece to a target plays the move", () => {
  const played = [];
  const board = new Board(fixture(), { onMove: (uci) => played.push(uci) });
  board.setLegalMoves(START_MOVES, "white");
  const [x0, y0] = centre(board, "g1");
  const [x1, y1] = centre(board, "f3");
  pointer("pointerdown", board.squareEl("g1").querySelector(".piece"), x0, y0);
  pointer("pointermove", window, (x0 + x1) / 2, (y0 + y1) / 2);
  assert(document.querySelector(".drag-ghost"), "a ghost piece follows the pointer");
  assert(board.squareEl("f3").classList.contains("target"), "targets shown while dragging");
  pointer("pointermove", window, x1, y1);
  pointer("pointerup", window, x1, y1);
  equal(played, ["g1f3"]);
  equal(document.querySelectorAll(".drag-ghost").length, 0, "ghost removed");
});

test("dropping off a target or a tiny drag plays nothing", () => {
  const played = [];
  const board = new Board(fixture(), { onMove: (uci) => played.push(uci) });
  board.setLegalMoves(START_MOVES, "white");
  const [x0, y0] = centre(board, "e2");
  const [x1, y1] = centre(board, "e5");
  pointer("pointerdown", board.squareEl("e2").querySelector(".piece"), x0, y0);
  pointer("pointermove", window, x1, y1);
  pointer("pointerup", window, x1, y1);
  equal(played, []);
  equal(board.selected, null, "the piece went back");
  pointer("pointerdown", board.squareEl("e2").querySelector(".piece"), x0, y0);
  pointer("pointermove", window, x0 + 2, y0 + 1);   // under the drag threshold: a click, not a drag
  pointer("pointerup", window, x0 + 2, y0 + 1);
  equal(document.querySelectorAll(".drag-ghost").length, 0);
  equal(played, []);
});

test("a move slides its piece; a dropped piece or a step back does not", () => {
  const board = new Board(fixture(), { onMove: () => {} });
  const sliding = (square) => board.squareEl(square).querySelector(".piece").getAnimations().length;
  board.setPosition("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", { lastMove: "e2e4" });
  equal(sliding("e4"), 1, "the pawn slides from e2 to e4");
  const after = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
  board.setPosition(after, { lastMove: "e7e5" });
  equal(sliding("e5"), 1);
  board.setPosition("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", { lastMove: "e2e4" });
  equal(sliding("e4"), 0, "stepping back shows the earlier position without replaying its move");
  board.setPosition(after, { lastMove: "e7e5" });
  board.setLegalMoves([{ uci: "g1f3", from_square: "g1", to_square: "f3", promotion: null }], "white");
  const [x0, y0] = centre(board, "g1");
  const [x1, y1] = centre(board, "f3");
  pointer("pointerdown", board.squareEl("g1").querySelector(".piece"), x0, y0);
  pointer("pointermove", window, x1, y1);
  pointer("pointerup", window, x1, y1);
  board.setPosition("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", { lastMove: "g1f3" });
  equal(sliding("f3"), 0, "a dragged piece is already there");
});

// ------------------------------------------------------------------ premoves

const targetsOf = (fen, square) => premoveTargets(fen, square).map((m) => m.to_square).sort().join(" ");

test("premove targets: every square the piece could reach on an empty board", () => {
  equal(targetsOf(START_FEN, "e2"), "d3 e3 e4 f3", "pawn: pushes and both captures");
  equal(targetsOf(START_FEN, "g1"), "e2 f3 h3", "knight");
  equal(targetsOf(START_FEN, "e1"), "c1 d1 d2 e2 f1 f2 g1", "king, with both castlings");
  equal(targetsOf(START_FEN, "a1").split(" ").length, 14, "rook: through its own pieces");
  equal(targetsOf(START_FEN, "d8").split(" ").length, 21, "queen");
  equal(targetsOf("8/8/8/8/8/8/8/4K3 w - - 0 1", "e1"), "d1 d2 e2 f1 f2", "no rooks: no castling");
  const promote = premoveTargets("8/1P6/8/8/8/8/8/k6K w - - 0 1", "b7");
  equal(promote.map((m) => m.uci).sort(), ["b7a8q", "b7b8q", "b7c8q"], "promotions are queens");
  equal(targetsOf("8/8/8/8/8/8/p7/7K b - - 0 1", "a2"), "a1 b1", "a black pawn goes down");
  equal(premoveTargets(START_FEN, "e4"), []);
});

/** A board where White waits for Black's reply. */
function premoveBoard(played = []) {
  const board = new Board(fixture(), { onMove: (uci) => played.push(uci) });
  board.setPosition("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", { lastMove: "e2e4" });
  board.setLegalMoves([], null, { premover: "white" });
  return board;
}

test("while the other side thinks, a click queues a premove", () => {
  const played = [];
  const board = premoveBoard(played);
  assert(board.el.classList.contains("premoving"), "the board shows premove mode");
  assert(!board.el.classList.contains("locked"), "your pieces can be picked up");
  board.squareEl("e7").click();
  equal(board.selected, null, "the opponent's pieces cannot");
  board.squareEl("g1").click();
  assert(board.squareEl("f3").classList.contains("target"), "targets shown");
  board.squareEl("f3").click();
  equal(board.premove.uci, "g1f3");
  equal(played, [], "nothing is played yet");
  assert(board.squareEl("g1").classList.contains("premove") && board.squareEl("f3").classList.contains("premove"),
         "the queued premove is highlighted");
  board.squareEl("a5").click();
  equal(board.premove, null, "a click elsewhere cancels it");
});

test("a premove by dragging, cancelled by a right-click or cancelPremove()", async () => {
  const board = premoveBoard();
  const [x0, y0] = centre(board, "d2");
  const [x1, y1] = centre(board, "d4");
  pointer("pointerdown", board.squareEl("d2").querySelector(".piece"), x0, y0);
  pointer("pointermove", window, x1, y1);
  pointer("pointerup", window, x1, y1);
  equal(board.premove.uci, "d2d4");
  await new Promise((resolve) => setTimeout(resolve, 0));   // the click that follows a drag is ignored
  board.el.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
  equal(board.premove, null, "right-click cancels");
  board.squareEl("b1").click();
  board.squareEl("c3").click();
  equal(board.premove.uci, "b1c3");
  board.cancelPremove();
  equal(board.premove, null);
  board.squareEl("b1").click();
  board.squareEl("c3").click();
  board.setLegalMoves([], null);
  equal(board.premove, null, "no premover (a new game, the game over): the premove is dropped");
  assert(board.el.classList.contains("locked"));
});

test("marks stay on squares across renders and can be cleared", () => {
  const board = new Board(fixture());
  board.setMarks({ e2: "hint-from", e4: "hint-to best-to" });
  assert(board.squareEl("e2").classList.contains("hint-from"), "from");
  assert(board.squareEl("e4").classList.contains("best-to"), "several classes");
  board.setPosition(START_FEN, { lastMove: "e2e4" });
  board.flip();
  assert(board.squareEl("e2").classList.contains("hint-from"), "kept after a new position and a flip");
  board.setMarks({});
  equal(board.el.querySelectorAll(".hint-from, .hint-to").length, 0);
});

run();
