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

/** Click a square on the board, as a player would. */
const click = (square) => $("board").querySelector(`[data-square="${square}"]`).click();

test("castling: click the king, then its target square", async () => {
  for (const [fen, king, target, san, rook] of [
    ["r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "e1", "g1", "O-O", "f1"],
    ["r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "e1", "c1", "O-O-O", "d1"],
  ]) {
    await site.newGame({ rating: 1500, color: "white" });
    site.ui.chess.load(fen);
    site.render();
    click(king);
    click(target);
    await until(() => site.ui.chess.history().length === 2);
    equal(site.ui.chess.history()[0], san);
    equal(site.ui.chess.get(rook).type, "r", "the rook moved too");
    equal(site.ui.chess.get(target).type, "k");
  }
});

test("the bot castles too", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
  const { moves } = site.ui.model.predict(site.ui.chess, 1500);
  const castles = moves.filter((m) => m.san.startsWith("O-O"));
  equal(castles.length, 2, "both castling moves are among the bot's choices");
  assert(castles.every((m) => m.prob > 0), "with some probability");
  equal(castles.map((m) => m.uci).sort().join(" "), "e8c8 e8g8");
});

test("premove: queue a move while the bot thinks, it is played when your turn comes", async () => {
  site.ui.delay = 150;
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  const reply = site.play("e2e4");                 // the bot thinks for 150 ms
  assert(site.ui.busy && site.ui.board.el.classList.contains("premoving"), "premove mode while the bot thinks");
  click("g1");
  click("f3");
  equal(site.ui.board.premove.uci, "g1f3");
  await reply;
  await until(() => site.ui.chess.history().length === 4, "the premove was played and the bot answered it");
  equal(site.ui.chess.history({ verbose: true })[2].lan, "g1f3");
  equal(site.ui.board.premove, null);
  site.ui.delay = 0;
});

test("premove: one that is illegal when your turn comes is dropped", async () => {
  site.ui.delay = 100;
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("k7/2Q5/4p3/8/4P3/8/7P/7K w - - 0 1");   // after h3, Black's only move is ...e5
  site.render();
  const reply = site.play("h2h3");
  click("e4");
  click("e5");
  equal(site.ui.board.premove.uci, "e4e5");
  await reply;
  await new Promise((resolve) => setTimeout(resolve, 20));
  equal(site.ui.chess.history(), ["h3", "e5"], "blocked by ...e5: dropped, nothing played");
  assert(site.ui.sounds.log.includes("premove") && site.ui.sounds.log.includes("premove-dropped"),
         `a tick when queued, a blip when dropped: ${site.ui.sounds.log}`);
  equal(site.ui.board.premove, null);
  equal(site.ui.board.mover, "white", "your move");
  site.ui.delay = 0;
});

test("premove: none once the game is over", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  site.resign();
  assert(!site.ui.board.el.classList.contains("premoving"));
  assert(site.ui.board.el.classList.contains("locked"));
});

test("moves make their sounds: yours and the bot's", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1");
  site.render();
  site.ui.sounds.log = [];
  await site.play("e1g1");
  equal(site.ui.sounds.log.length, 2, "your move and the bot's reply");
  equal(site.ui.sounds.log[0], "castle");
  assert(["move", "capture", "check", "castle", "promote"].includes(site.ui.sounds.log[1]), site.ui.sounds.log[1]);
});

test("a game starts and ends with a sound: win, lose or draw", async () => {
  site.ui.sounds.log = [];
  await site.newGame({ rating: 1500, color: "white" });
  equal(site.ui.sounds.log, ["start"]);
  site.resign();
  equal(site.ui.sounds.log.slice(-1), ["lose"]);
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("7k/8/6K1/8/8/8/R7/1R6 w - - 0 1");
  site.render();
  await site.play("a2a7");
  await site.play("b1b8");
  equal(site.ui.sounds.log.slice(-2), ["check", "win"]);
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("k7/8/1Q6/8/8/8/8/7K w - - 0 1");
  site.render();
  await site.play("h1g1");                          // after Kg1 Black has no moves: stalemate
  equal(site.ui.sounds.log.slice(-1), ["draw"]);
});

test("the sound button (and the m key) mutes and unmutes, and is remembered", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  $("sound").click();
  equal($("sound").textContent, "Sound: off");
  equal($("sound").getAttribute("aria-pressed"), "false");
  equal(localStorage.getItem("human-chess-site.sound"), "off");
  site.ui.sounds.log = [];
  await site.play("e2e4");
  equal(site.ui.sounds.log, [], "silent");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "m" }));
  equal($("sound").textContent, "Sound: on");
  equal(localStorage.getItem("human-chess-site.sound"), "on");
});

test("the volume slider sets the volume and is remembered", () => {
  const slider = $("volume");
  slider.value = "30";
  slider.dispatchEvent(new Event("input"));
  equal(site.ui.sounds.volume, 0.3);
  equal(localStorage.getItem("human-chess-site.volume"), "30");
  site.setVolume(250);
  equal(site.ui.sounds.volume, 1, "kept within 0-100");
  site.setVolume(70);
});

run();
