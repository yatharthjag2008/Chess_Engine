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
  equal(localStorage.getItem("human-chess-site.sound"), '"off"');
  site.ui.sounds.log = [];
  await site.play("e2e4");
  equal(site.ui.sounds.log, [], "silent");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "m" }));
  equal($("sound").textContent, "Sound: on");
  equal(localStorage.getItem("human-chess-site.sound"), '"on"');
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

test("each player's captures are shown next to their name", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  site.render();
  equal($("bottom-captured").children.length, 0);
  site.ui.chess.move("exd5");
  site.render();
  const mine = [...$("bottom-captured").querySelectorAll(".piece")].map((el) => el.dataset.piece);
  equal(mine, ["bp"], "you (White, at the bottom) took a black pawn");
  equal($("bottom-captured").getAttribute("aria-label"), "Captured a pawn; 1 ahead.");
  equal($("bottom-captured").querySelector(".piece").getBoundingClientRect().width, 22, "small icons");
  site.ui.board.flip();
  site.render();
  equal([...$("top-captured").querySelectorAll(".piece")].map((el) => el.dataset.piece), ["bp"],
        "flipped: you are at the top");
  site.ui.chess.load("4k3/8/8/8/8/8/8/4K3 w - - 0 1");          // both sides took everything but the king
  site.render();
  const groups = [...$("top-captured").querySelectorAll(".group")].map((g) => `${g.dataset.type}${g.children.length}`);
  equal(groups, ["p8", "n2", "b2", "r2", "q1"], "grouped by kind, cheapest first");
  equal(document.querySelectorAll(".captured .score").length, 0, "kings only: level, no score");
  site.ui.chess.load("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  site.render();
  equal($("top-captured").querySelector(".score").textContent, "+9");
  site.ui.chess.load("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1");
  site.render();
  equal($("bottom-captured").querySelector(".score").textContent, "+3", "Black is ahead: shown with Black");
  site.ui.board.flip();
});

test("take back and a new game update the captures", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  site.render();
  await site.play("e4d5");
  assert($("bottom-captured").querySelector('[data-piece="bp"]'), "a pawn taken");
  site.takeback();
  equal($("bottom-captured").querySelectorAll(".piece").length, 0, "taken back");
  await site.play("e4d5");
  await site.newGame({ rating: 1500, color: "white" });
  equal(document.querySelectorAll(".captured .piece").length, 0, "a new game starts level");
});

test("stepping back shows an earlier position; the game is kept and you move only at its end", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  await site.play("g1f3");
  const now = site.ui.chess.fen();
  site.showPly(1);
  equal(site.ui.board.fen, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "after 1.e4");
  equal(site.ui.board.lastMove, "e2e4");
  equal(site.ui.board.mover, null, "no moves while looking back");
  equal(site.ui.board.premover, null);
  equal(site.ui.chess.fen(), now, "the game itself is unchanged");
  site.showPly(0);
  equal(site.ui.board.fen.split(" ")[0], "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR");
  site.showPly(null);
  equal(site.ui.board.fen, now);
  equal(site.ui.board.mover, "white", "back at the game: your move");
  site.showPly(99);
  equal(site.ui.view, null, "past the end is the game");
});

test("click a move in the list to look at it; the move shown is highlighted", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("d2d4");
  await site.play("c2c4");
  const cells = () => [...$("move-list").querySelectorAll(".san[data-ply]")];
  equal(cells().map((c) => c.dataset.ply), ["1", "2", "3", "4"]);
  equal($("move-list").querySelector(".latest").dataset.ply, "4", "the last move, while at the game");
  cells()[0].click();
  equal(site.ui.view, 1);
  equal($("move-list").querySelector(".latest").textContent, "d4", "the move looked at");
  equal(site.ui.board.lastMove, "d2d4");
  cells()[3].click();
  equal(site.ui.view, null, "the last move is the game itself");
});

const key = (name) => document.dispatchEvent(new KeyboardEvent("keydown", { key: name, cancelable: true }));

test("arrow keys step through the moves; Home and End jump to the start and back to the game", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  await site.play("b1c3");                         // 4 half-moves
  key("ArrowLeft");
  equal(site.ui.view, 3);
  key("ArrowLeft");
  key("ArrowLeft");
  equal(site.ui.view, 1);
  key("ArrowRight");
  equal(site.ui.view, 2);
  key("Home");
  equal(site.ui.view, 0);
  key("ArrowLeft");
  equal(site.ui.view, 0, "not before the start");
  key("End");
  equal(site.ui.view, null);
  key("ArrowRight");
  equal(site.ui.view, null, "not past the game");
  key("ArrowUp");
  equal(site.ui.view, 0);
  key("ArrowDown");
  equal(site.ui.view, null);
});

test("the step buttons, disabled at either end", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  assert($("next").disabled && $("last").disabled, "at the game: nothing ahead");
  assert(!$("prev").disabled && !$("first").disabled);
  $("prev").click();
  equal(site.ui.view, 1);
  assert(!$("next").disabled, "now there is");
  $("first").click();
  equal(site.ui.view, 0);
  assert($("prev").disabled && $("first").disabled, "at the start");
  $("next").click();
  equal(site.ui.view, 1);
  $("last").click();
  equal(site.ui.view, null);
});

test("while looking back the status says where you are; a click on the board returns", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  site.showPly(1);
  equal($("status").textContent, "Looking at the position after 1.e4. End, or a click on the board, returns to the game.");
  site.showPly(2);
  equal(site.ui.view, null, "ply 2 is the game");
  site.showPly(0);
  assert($("status").textContent.startsWith("Looking at the starting position."));
  assert(site.ui.board.el.classList.contains("viewing"), "the board is marked");
  click("e2");
  equal(site.ui.view, null, "back to the game");
  equal(site.ui.board.selected, null, "that click selected nothing");
  assert($("status").textContent.startsWith("Your move"), $("status").textContent);
});

test("looking back, the captures and the check are those of the position shown", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  site.ui.chess.load("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  site.ui.chess.move("exd5");
  site.ui.chess.move("Qxd5");
  site.ui.chess.move("Nc3");
  site.ui.chess.move("Qe5+");
  site.render();
  assert(site.ui.board.check, "in check now");
  equal(document.querySelectorAll(".captured .piece").length, 2, "a pawn each");
  site.showPly(1);
  equal(document.querySelectorAll(".captured .piece").length, 1, "after exd5: only White has taken");
  assert(!site.ui.board.check, "no check then");
  site.showPly(3);
  assert(!site.ui.board.check);
  site.showPly(null);
  assert(site.ui.board.check);
});

test("looking back at a bot move shows what players weighed there; take back returns to the game", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  const first = site.ui.last;
  await site.play("d2d4");
  site.showPly(2);
  assert(!$("thinking-panel").hidden, "shown for the bot's first move");
  assert($("thinking-caption").textContent.includes(`before ${first.san}`), $("thinking-caption").textContent);
  site.showPly(1);
  assert($("thinking-panel").hidden, "your move: nothing to show");
  site.takeback();
  equal(site.ui.view, null, "take back returns to the game");
  equal(site.ui.chess.history().length, 2);
});

const savedGame = () => JSON.parse(localStorage.getItem("human-chess-site.game"));

test("the game is saved in the browser after every move, and how it ended", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1700, color: "black" });
  let game = savedGame();
  equal(game.moves.length, 1, "the bot's first move");
  equal([game.human, game.rating, game.orientation, game.over], ["b", 1700, "black", null]);
  equal(game.start, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  assert(game.thoughts["1"].candidates.length <= 8, "the bot's thinking, trimmed");
  const reply = site.ui.chess.moves({ verbose: true })[0].lan;
  await site.play(reply);
  game = savedGame();
  equal(game.moves.length, 3);
  equal(game.moves[1], reply);
  site.resign();
  game = savedGame();
  equal(game.over, { text: "You resigned: White wins", winner: "w" });
});

test("a saved game is carried on: position, side, rating, the bot's thinking", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1300, color: "white" });
  await site.play("e2e4");
  await site.play("d2d4");
  const snap = site.snapshot();
  const fen = site.ui.chess.fen();
  const last = site.ui.last.san;
  await site.newGame({ rating: 2000, color: "black" });
  assert(site.restore(snap), "restored");
  equal(site.ui.chess.fen(), fen);
  equal([site.ui.human, site.ui.rating, site.ui.board.orientation], ["w", 1300, "white"]);
  equal(site.ui.chess.history().length, 4);
  equal(site.ui.last.san, last, "what the bot weighed on its last move");
  equal(site.ui.board.mover, "white", "your move");
  equal($("move-list").querySelectorAll(".san[data-ply]").length, 4);
  assert(!site.restore({ version: 1, start: "not a fen", moves: [] }), "a broken save is ignored");
  assert(!site.restore({ version: 99 }), "an unknown version too");
  assert(!site.restore(null));
});

test("reloaded while the bot was thinking: it moves after the restore", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  const snap = { ...site.snapshot(), moves: ["e2e4"] };    // your move made, the bot's not yet
  assert(site.restore(snap));
  await until(() => site.ui.chess.history().length === 2 && !site.ui.busy, "the bot's reply");
  equal(site.ui.board.mover, "white");
  assert(site.restore({ ...snap, moves: ["e2e4", "e7e5"] }));
  assert($("status").textContent.startsWith("Welcome back"), $("status").textContent);
});

run();
