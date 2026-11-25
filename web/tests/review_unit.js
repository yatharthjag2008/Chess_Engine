// Unit tests of review.js on a hand-made review (no server): the edge cases the end-to-end tests do not reach.
// Imported by tests.js, so they run with the board tests on tests/index.html.

import { assert, equal, test } from "./harness.js";

// The review page's markup in a hidden box, so review.js has its elements.
const html = await (await fetch("../review.html")).text();
const doc = new DOMParser().parseFromString(html, "text/html");
doc.querySelectorAll("script").forEach((s) => s.remove());
const box = document.createElement("div");
box.id = "review-page";
box.hidden = true;
box.append(...doc.body.children);
document.body.append(box);
window.__noAutoInit = true;
const app = await import("../js/review.js");
app.init();
const $ = (id) => document.getElementById(id);

const AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
const AFTER_E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
const AFTER_NF3 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";

function move(fields) {
  return { eval_before: 20, eval_after: 20, best_cp: 20, played_cp: 20, cp_loss: 0, win_before: 52, win_after: 52,
           win_lost: 0, accuracy: 100, classification: "best", alternatives: [], ...fields };
}

/** A clean two-move review starting from a Black-to-move position. */
export function blackFirstReview() {
  const counts = { best: 1, good: 0, inaccuracy: 0, mistake: 0, blunder: 0 };
  return {
    headers: {}, start_fen: AFTER_E4, depth: 12,
    moves: [
      move({ ply: 1, color: "black", san: "e5", uci: "e7e5", fen_before: AFTER_E4, fen_after: AFTER_E5,
             best_san: "e5", best_uci: "e7e5", move_number: "1..." }),
      move({ ply: 2, color: "white", san: "Nf3", uci: "g1f3", fen_before: AFTER_E5, fen_after: AFTER_NF3,
             best_san: "Nf3", best_uci: "g1f3", move_number: "2." }),
    ],
    summary: { white: { color: "white", moves: 1, accuracy: 100, average_cp_loss: 0, counts },
               black: { color: "black", moves: 1, accuracy: 100, average_cp_loss: 0, counts } },
    key_moments: [],
    graph: [{ ply: 0, white_win: 52, eval: 20 }, { ply: 1, white_win: 52, eval: 20 }, { ply: 2, white_win: 52, eval: 20 }],
  };
}

function show(review, side = "white") {
  app.ui.review = review;
  app.ui.side = side;
  app.showReview();
}

test("review: a game starting with Black to move gets an empty White cell", () => {
  show(blackFirstReview());
  const rows = document.querySelectorAll("#move-list li");
  equal(rows.length, 2);
  const first = [...rows[0].children].map((c) => c.textContent);
  equal(first, ["1.", "", "e5"]);
  equal([...rows[1].children].map((c) => c.textContent), ["2.", "Nf3"]);
  equal($("ply-label").textContent, "Start");
});

test("review: the start position is the review's start FEN", () => {
  show(blackFirstReview());
  equal(app.fenAt(0), AFTER_E4);
  equal(app.fenAt(2), AFTER_NF3);
  app.goTo(-5);
  equal(app.ui.ply, 0);
  assert(app.ui.board.squareEl("e4").querySelector(".piece"), "the pawn already on e4");
  app.goTo(1);
  equal($("ply-label").textContent, "1... e5");
});

test("review: a clean game has no markers and says so in the summary", () => {
  show(blackFirstReview());
  equal(document.querySelectorAll("#graph circle.marker").length, 0);
  assert($("summary-panel").textContent.includes("a clean game"), $("summary-panel").textContent);
  equal(document.querySelectorAll("#summary-panel .moments").length, 0);
  assert(!$("summary-panel").textContent.includes("undefined"), $("summary-panel").textContent);
});

test("review: Stockfish's own move gets no 'better move' button", () => {
  show(blackFirstReview());
  app.goTo(2);
  equal($("show-best"), null);
  assert($("detail-panel").textContent.includes("This was Stockfish's choice."), $("detail-panel").textContent);
});

test("review: a review without moves hides the graph", () => {
  const empty = { ...blackFirstReview(), moves: [], graph: [], key_moments: [] };
  show(empty);
  assert($("graph").hidden, "no graph");
  equal(document.querySelectorAll("#move-list .san").length, 0);
  assert($("next").disabled && $("prev").disabled, "nothing to step through");
});

test("review: formatting and errors", () => {
  equal(app.formatEval(5), "+0.05");
  equal(app.formatEval(-1), "-0.01");
  equal(app.formatEval(9999), "#1");
  equal(app.errorText(new Error("boom")), "boom");
  equal(app.SYMBOLS.good, undefined);
  equal(app.sideOf({ white: "a", black: "b" }, null), null);
});
