// End-to-end tests of the public play page: the bot runs in this browser.

import { assert, equal, run, test, until } from "./harness.js";

const html = await (await fetch("../index.html")).text();
const doc = new DOMParser().parseFromString(html, "text/html");
doc.querySelectorAll("script").forEach((s) => s.remove());
document.getElementById("page").replaceChildren(...doc.body.children);
const site = await import("../js/site.js");
await site.init("../model/");
site.ui.delay = 0;
const $ = (id) => document.getElementById(id);

test("the model loads and checks itself against PyTorch", () => {
  assert(!$("play").disabled, "Play is enabled once the model is ready");
  assert($("model-note").textContent.includes("checked against PyTorch"), $("model-note").textContent);
  equal($("status").textContent, "Choose a rating and press Play.");
});

test("you move, the bot answers like players of its rating", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1200, color: "white" });
  equal($("top-name").textContent, "Bot 1200");
  await site.play("e2e4");
  equal(site.ui.chess.history().length, 2, "the bot replied");
  assert(!$("thinking-panel").hidden, "what players at this rating play");
  const rows = $("candidates").querySelectorAll(".candidate");
  assert(rows.length >= 3, `${rows.length} candidates`);
  equal($("move-list").querySelectorAll("li").length, 1);
  equal($("status").textContent, "Your move.");
  assert(site.ui.board.mover === "white", "your turn on the board");
});

test("take back, then resign", async () => {
  site.takeback();
  equal(site.ui.chess.history().length, 0);
  assert($("takeback").disabled, "nothing more to take back");
  await site.play("d2d4");
  site.resign();
  assert(!$("result-banner").hidden, "result banner");
  equal($("result-banner").querySelector("h2").textContent, "The bot won");
  equal(site.ui.board.mover, null, "the board is locked");
});

test("playing Black: the bot opens", async () => {
  await site.newGame({ rating: 1800, color: "black" });
  equal(site.ui.chess.history().length, 1, "the bot moved first");
  equal(site.ui.board.orientation, "black");
  equal($("bottom-name").textContent, "You");
  assert($("result-banner").hidden, "a new game hides the old result");
});

test("checkmate ends the game", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("7k/8/6K1/8/8/8/R7/1R6 w - - 0 1");
  site.render();
  await site.play("a2a7");
  equal(site.ui.chess.history().slice(-1)[0], "Kg8", "the only move");
  await site.play("b1b8");
  equal(site.outcome().text, "White wins by checkmate");
  equal($("result-banner").querySelector("h2").textContent, "You won");
});

run();
