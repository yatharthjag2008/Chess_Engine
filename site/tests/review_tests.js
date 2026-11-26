// End-to-end tests of the Review page: Stockfish's judgement of a game, shown move by move.

import { assert, equal, run, test, until } from "./harness.js";
import { Chess } from "../vendor/chess.js";
import { finishedEval, StockfishAnalyzer } from "../js/stockfish.js";

const html = await (await fetch("../review.html")).text();
const doc = new DOMParser().parseFromString(html, "text/html");
doc.querySelectorAll("script").forEach((s) => s.remove());
document.getElementById("page").replaceChildren(...doc.body.children);
const page = await import("../js/review_page.js");
await page.init();
const $ = (id) => document.getElementById(id);

/** Python's ScriptedAnalyzer (tests/review_helpers.py): scripted positions, else the first legal move at 0. */
function scripted(script) {
  return {
    depth: 12,
    async analyse(fen) {
      const chess = new Chess(fen);
      const done = finishedEval(chess);
      if (done !== null) return { fen, lines: [], bestCp: done };
      const legal = chess.moves({ verbose: true });
      const [bestUci, bestCp, others] = script[fen.split(" ").slice(0, 2).join(" ")]
        || [legal.map((m) => m.lan).sort()[0], 0, {}];
      const san = (uci) => legal.find((m) => m.lan === uci).san;
      const lines = [{ uci: bestUci, san: san(bestUci), cp: bestCp, pvSan: [san(bestUci)] },
                     ...Object.entries(others).map(([uci, cp]) => ({ uci, san: san(uci), cp, pvSan: [san(uci)] }))];
      lines.sort((a, b) => b.cp - a.cp);
      return { fen, lines, bestCp: lines[0].cp };
    },
  };
}

const keyOf = (...sans) => {
  const chess = new Chess();
  for (const san of sans) chess.move(san);
  return chess.fen().split(" ").slice(0, 2).join(" ");
};

const BLUNDER_GAME = `[White "Ann"]\n[Black "Bob"]\n[Result "1-0"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n`;

test("a reviewed game: summary, key moment, coloured moves, graph", async () => {
  page.ui.makeAnalyzer = () => scripted({
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4")]: ["g7g6", 0, { g8f6: -900 }],
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6")]: ["h5f7", 900, {}],
  });
  page.ui.analyzer = null;
  $("review-pgn").value = BLUNDER_GAME;
  const review = await page.startReview();
  equal(review.moves.length, 7);
  equal($("status").textContent, "Reviewed 7 moves at depth 12.");
  equal([$("top-name").textContent, $("bottom-name").textContent], ["Bob", "Ann"]);
  const cells = [...$("summary-panel").querySelectorAll(".summary-grid span")].map((s) => s.textContent);
  equal(cells.slice(0, 9), ["", "Ann", "Bob", "Accuracy", "100.0%", "49.2%", "Average loss", "0 cp", "300 cp"]);
  const moments = [...$("summary-panel").querySelectorAll(".moments button")];
  equal(moments.map((b) => b.textContent), ["3... Nf6?? dropped Black's chances from 50% to 4%. Better was g6."]);
  const blunder = $("move-list").querySelector('.san[data-ply="6"]');
  assert(blunder.classList.contains("blunder") && blunder.textContent === "Nf6??", blunder.outerHTML);
  equal($("graph").querySelectorAll("rect.hit").length, 8);
  equal($("graph").querySelectorAll("circle.marker.blunder").length, 1);
  equal(page.ui.ply, 0, "starts at the start");
  assert($("detail-panel").hidden);
});

test("stepping: a key moment, the move's detail, the better move on the board", async () => {
  $("summary-panel").querySelector(".moments button").click();
  equal(page.ui.ply, 6);
  equal($("ply-label").textContent, "3... Nf6");
  equal($("detail-panel").querySelector(".verdict").textContent, "3... Nf6??: Blunder");
  assert($("detail-panel").textContent.includes("win chance 50% \u2192 4% (\u221246%)"), $("detail-panel").textContent);
  assert($("detail-panel").textContent.includes("Best was g6 (0.00)."));
  $("show-best").click();
  equal(page.ui.board.marks, { g7: "best-from", g6: "best-to" });
  equal(page.ui.board.fen.split(" ")[0], keyOf("e4", "e5", "Qh5", "Nc6", "Bc4").split(" ")[0], "the position before");
  $("show-best").click();
  equal(page.ui.board.marks, {}, "back to the game");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
  equal(page.ui.ply, 7);
  assert($("next").disabled && !$("prev").disabled);
  assert($("detail-panel").textContent.includes("This was Stockfish's choice."));
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
  equal(page.ui.ply, 0);
  $("move-list").querySelector('.san[data-ply="3"]').click();
  equal($("ply-label").textContent, "2. Qh5");
  $("graph").querySelector('rect.hit[data-ply="5"]').dispatchEvent(new MouseEvent("click", { bubbles: true }));
  equal(page.ui.ply, 5, "the graph jumps too");
  $("flip").click();
  equal([page.ui.board.orientation, $("bottom-name").textContent], ["black", "Bob"]);
});

test("what cannot be reviewed says why", async () => {
  for (const [text, message] of [["", "Paste a game in PGN."],
                                 ["1. e4 e5 2. Qxf7 *", "That game could not be read"],
                                 ['[White "a"]\n[Black "b"]\n\n*', "That game has no moves to review."]]) {
    equal(await page.startReview(text, 12), null);
    assert($("review-error").textContent.startsWith(message), $("review-error").textContent);
  }
});

test("a game handed over by the play page: reviewed, from your side", async () => {
  page.ui.makeAnalyzer = () => scripted({});
  page.ui.analyzer = null;
  localStorage.setItem("human-chess-site.review", JSON.stringify(
    `[White "Bot 1500"]\n[Black "You"]\n[Result "*"]\n\n1. d4 d5 2. c4 *\n`));
  const review = await page.openHandedGame();
  equal(review.moves.length, 3);
  equal([page.ui.board.orientation, $("bottom-name").textContent], ["black", "You"]);
  equal(localStorage.getItem("human-chess-site.review"), null, "used once");
  equal(await page.openHandedGame(), null);
});

test("the coach: Black's blunder explained, with the stronger players' move", async () => {
  page.ui.makeAnalyzer = () => scripted({
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4")]: ["g7g6", 0, { g8f6: -900 }],
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6")]: ["h5f7", 900, {}],
  });
  page.ui.analyzer = null;
  await page.startReview(BLUNDER_GAME.replace('[Black "Bob"]', '[Black "Bob"]\n[BlackElo "1250"]'), 12);
  assert(!$("coach-panel").hidden);
  equal($("coach-start").textContent, "Coach White");
  $("flip").click();
  equal($("coach-start").textContent, "Coach Black", "the coach follows the side at the bottom");
  equal($("coach-rating").placeholder, "1250", "the game's rating");
  const before = keyOf("e4", "e5", "Qh5", "Nc6", "Bc4");
  page.ui.model = {
    predict(position, rating) {
      const fen = position.fen();
      const legal = position.moves({ verbose: true });
      const at = page.ui.review.moves.find((m) => m.fenBefore === fen);
      const favourite = fen.split(" ").slice(0, 2).join(" ") === before ? (rating < 1450 ? "g8f6" : "g7g6") : at.uci;
      const top = legal.find((m) => m.lan === favourite);
      return { moves: [{ uci: top.lan, san: top.san, prob: 0.6 }] };
    },
  };
  const result = await page.startCoach();
  equal([result.report.rating, result.ratingSource], [1250, "pgn"]);
  const stats = [...$("coach-panel").querySelectorAll(".coach-stats")].map((p) => p.textContent);
  equal(stats, ["Coaching Black, rated about 1250 (from the game).",
                "You played your level's favourite move 100% of the time, and the 1450-level move 67%."]);
  equal($("coach-note").textContent, "Step to one of your moves for its note.");
  $("next-note").click();
  equal(page.ui.ply, 6);
  assert($("coach-note").textContent.startsWith("g6 was stronger (+0.0 instead of -9.0)."), $("coach-note").textContent);
  assert($("coach-note").textContent.includes("Players around 1450 prefer g6 (60%)"), $("coach-note").textContent);
  page.goTo(4);
  equal($("coach-note").textContent, "Nothing to add about this move.");
  $("coach-again").click();
  assert($("coach-rating"), "ask again at another rating");
});

test("the coach estimates your rating when the game has none (the real model)", async () => {
  page.ui.makeAnalyzer = () => scripted({});
  page.ui.analyzer = null;
  page.ui.model = null;
  page.ui.modelBase = "../model/";
  await page.startReview(`[White "You"]\n[Black "Bot 1500"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 *\n`, 12);
  const result = await page.startCoach();
  equal(result.ratingSource, "estimated");
  assert(result.report.rating >= 800 && result.report.rating <= 2400 && result.report.rating % 50 === 0,
         String(result.report.rating));
  assert($("coach-panel").textContent.includes("(estimated from your moves)"));
  const typed = await page.startCoach(1700);
  equal([typed.report.rating, typed.ratingSource], [1700, "given"]);
});

test("with the real Stockfish: the blunder is found and the mate is the best move", async () => {
  page.ui.makeAnalyzer = (depth) => new StockfishAnalyzer({ depth, multipv: 3 });
  page.ui.analyzer = null;
  const review = await page.startReview(BLUNDER_GAME, 8);
  assert(review, $("review-error").textContent);
  equal(review.moves[5].classification, "blunder", "3...Nf6??");
  equal(review.moves[5].bestSan === "Nf6", false, "a better move was there");
  equal(review.moves[6].classification, "best", "4.Qxf7#");
  equal(review.moves[6].evalAfter, 9999, "mate in one, as the Python review scores it");
  equal($("status").textContent, "Reviewed 7 moves at depth 8.");
  page.ui.analyzer.terminate();
});

run();
