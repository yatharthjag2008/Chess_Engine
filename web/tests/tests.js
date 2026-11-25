// Unit tests for the web UI, run in a real browser. Each test is a function that throws on failure.
// The summary line "RESULT passed=N failed=M" is what tests/test_web_js.py looks for.

import { Board, START_FEN, glyph, isLightSquare, parseFen, squareOrder } from "../js/board.js";

const tests = [];
export function test(name, fn) { tests.push({ name, fn }); }

function assert(condition, message) { if (!condition) throw new Error(message || "assertion failed"); }
function equal(actual, expected, message) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${message || "not equal"}: got ${a}, expected ${e}`);
}
function fixture() {
  const el = document.getElementById("fixture");
  el.replaceChildren();
  el.className = "board";
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
  equal(board.squareEl("e1").querySelector(".piece").textContent, glyph("k"));
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

// ------------------------------------------------------------------ runner

export async function run() {
  // Other test modules register themselves before the runner starts.
  await Promise.all((window.__extraTests || []).map((url) => import(url)));
  const lines = [];
  let passed = 0, failed = 0;
  for (const { name, fn } of tests) {
    try {
      await fn();
      passed += 1;
      lines.push(`ok   ${name}`);
    } catch (err) {
      failed += 1;
      lines.push(`FAIL ${name}: ${err.message}`);
    }
  }
  lines.push(`RESULT passed=${passed} failed=${failed}`);
  document.getElementById("results").textContent = lines.join("\n");
}

run();
