// End-to-end tests of the public play page: the bot runs in this browser.

import { assert, equal, run, test, until } from "./harness.js";

const html = await (await fetch("../index.html")).text();
const doc = new DOMParser().parseFromString(html, "text/html");
doc.querySelectorAll("script").forEach((s) => s.remove());
document.getElementById("page").replaceChildren(...doc.body.children);
const site = await import("../js/site.js");
await site.init("../model/");
site.ui.delay = 0;
site.ui.thinkScale = 0.01;                       // timed games: the bot thinks 1% of its human-like time
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

test("a finished game comes back finished, and its PGN has the result", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  site.resign();
  const snap = site.snapshot();
  await site.newGame({ rating: 1500, color: "white" });
  assert(site.restore(snap));
  assert(site.ui.over, "over");
  equal($("status").textContent, "Your last game: You resigned: Black wins. Press Play for a new one.");
  equal(site.ui.board.mover, null);
  equal(site.ui.board.premover, null);
  assert($("resign").disabled && $("takeback").disabled);
  const text = site.pgn();
  assert(text.includes('[Result "0-1"]') && text.trim().endsWith("0-1"), text);
  assert(text.includes('[White "You"]') && text.includes('[Black "Bot 1500"]'), text);
});

test("the new-game choices are remembered (rating and side, random included)", async () => {
  $("rating").value = "1850";
  $("new-game").querySelector('input[value="random"]').checked = true;
  $("new-game").dispatchEvent(new Event("submit", { cancelable: true }));
  await until(() => !site.ui.busy, "the game started");
  equal(JSON.parse(localStorage.getItem("human-chess-site.settings")), { rating: 1850, color: "random", time: "none", difficulty: "off" });
  $("rating").value = "1000";
  $("new-game").querySelector('input[value="white"]').checked = true;
  site.applySettings({ rating: 1850, color: "random" });
  equal($("rating").value, "1850");
  equal($("rating-value").textContent, "1850");
  assert($("new-game").querySelector('input[value="random"]').checked);
  site.applySettings({ rating: 99999, color: "purple" });
  equal($("rating").value, "1850", "nonsense is ignored");
  $("new-game").querySelector('input[value="white"]').checked = true;
});

test("choose a time control: none, 1, 3, 5 or 10 minutes", async () => {
  const pick = (value) => { $("new-game").querySelector(`input[name="time"][value="${value}"]`).checked = true; };
  for (const [value, base] of [["1", 60], ["3", 180], ["5", 300], ["10", 600]]) {
    pick(value);
    const form = site.readForm();
    equal(form.time, value);
    await site.newGame({ ...form, color: "white" });
    equal(site.ui.clock.base, base);
  }
  pick("none");
  await site.newGame({ ...site.readForm(), color: "white" });
  equal(site.ui.clock, null, "untimed");
  site.applySettings({ rating: 1500, color: "white", time: "5" });
  assert($("new-game").querySelector('input[name="time"][value="5"]').checked, "remembered like the rest");
  pick("none");
});

test("the clocks: shown in timed games, your time runs on your move and the bot's on its move", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white", time: "3" });
  assert(!$("top-clock").hidden && !$("bottom-clock").hidden, "both clocks shown");
  equal([$("top-clock").textContent, $("bottom-clock").textContent], ["3:00", "3:00"]);
  equal(site.ui.clock.running, null, "nobody's time runs before White's first move");
  await site.play("e2e4");
  equal(site.ui.clock.running, "w", "after the bot's reply, your time runs");
  assert($("bottom-clock").classList.contains("running") && !$("top-clock").classList.contains("running"));
  assert(site.ui.clock.left("b") < 180000, "the bot used some time");
  site.ui.clock.remaining.w = 15000;
  site.updateClocks();
  assert($("bottom-clock").classList.contains("low"), "under 20 seconds: low");
  site.resign();
  equal(site.ui.clock.running, null, "stopped at the end");
  await site.newGame({ rating: 1500, color: "white" });
  assert($("top-clock").hidden && $("bottom-clock").hidden, "untimed: no clocks");
});

test("a flag falls: the side out of time loses, unless the other cannot checkmate", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white", time: "1" });
  await site.play("e2e4");
  site.ui.clock.remaining.w = 150;                // your time runs out
  await until(() => site.ui.over, "the flag");
  equal(site.ui.overText, "Black wins on time");
  equal($("result-banner").querySelector("h2").textContent, "The bot won");
  equal(site.ui.board.mover, null);
  await site.newGame({ rating: 1500, color: "white", time: "1" });
  site.ui.chess.load("4k3/8/8/8/8/8/8/3NK3 w - - 0 1");
  site.ui.clock.start("b");
  site.ui.clock.remaining.b = 100;                // the bot runs out, but you only have a knight
  await until(() => site.ui.over, "the flag");
  equal(site.ui.overText, "Black ran out of time, but White cannot checkmate: a draw");
});

test("timed games: the bot takes a human-like time, and it comes off its clock", async () => {
  site.ui.random = () => 0.5;
  site.ui.thinkScale = 1;
  await site.newGame({ rating: 1500, color: "white", time: "1" });
  await site.play("e2e4");                       // ply 1: 60 s / 80 = 0.75 s
  const used = 60000 - site.ui.clock.left("b");
  assert(used >= 700 && used < 2000, `the bot used ${used} ms`);
  site.ui.thinkScale = 0.01;
});

test("the network gets the clocks, so short time changes the bot's choices like a human's", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white", time: "3" });
  equal(site.clockContext(), { clock: 180, oppClock: 180, base: 180, increment: 0 });
  site.ui.clock.remaining.w = 5000;
  site.ui.clock.remaining.b = 170000;
  const context = site.clockContext();
  equal([context.clock, context.oppClock], [5, 170], "White to move with 5 seconds");
  const calm = site.ui.model.predict(site.ui.chess, 1500, { clock: 170, oppClock: 170, base: 180 }).moves[0].prob;
  const rushed = site.ui.model.predict(site.ui.chess, 1500, context).moves[0].prob;
  assert(Math.abs(calm - rushed) > 1e-4, `the clocks change the prediction (${calm} vs ${rushed})`);
  let seen = null;
  const predict = site.ui.model.predict.bind(site.ui.model);
  site.ui.model.predict = (chess, rating, options) => { seen = options; return predict(chess, rating, options); };
  site.botMove();
  site.ui.model.predict = predict;
  equal(seen.base, 180, "botMove passes them on");
  await site.newGame({ rating: 1500, color: "white" });
  equal(site.clockContext(), {}, "untimed: the model's default");
});

test("timed games: clocks survive a reload (time passes meanwhile), and there is no take back", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white", time: "5" });
  await site.play("e2e4");
  assert($("takeback").disabled, "no take back against the clock");
  site.takeback();
  equal(site.ui.chess.history().length, 2, "not even by calling it");
  const snap = site.snapshot();
  equal(snap.time, "5");
  equal(snap.clock.running, "w");
  const away = { ...snap, clock: { ...snap.clock, at: snap.clock.at - 30000 } };   // saved 30 s ago
  assert(site.restore(away));
  const left = site.ui.clock.left("w");
  assert(left <= snap.clock.remaining.w - 30000 && left > snap.clock.remaining.w - 31000, `${left}`);
  equal(site.ui.clock.left("b"), snap.clock.remaining.b, "the bot's clock was stopped");
  const gone = { ...snap, clock: { ...snap.clock, at: snap.clock.at - 400000 } };    // away longer than the time left
  assert(site.restore(gone));
  await until(() => site.ui.over, "the flag fell while away");
  equal(site.ui.overText, "Black wins on time");
});

test("draw arrows during a game; your next move clears them", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  const at = (square) => {
    const rect = site.ui.board.squareEl(square).getBoundingClientRect();
    return { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
  };
  const press = (type, target, square, extra = {}) => target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, cancelable: true, button: 2, pointerId: 1, ...at(square), ...extra }));
  press("pointerdown", site.ui.board.squareEl("e2"), "e2");
  press("pointermove", window, "e4");
  press("pointerup", window, "e4");
  press("pointerdown", site.ui.board.squareEl("f7"), "f7", { shiftKey: true });
  press("pointerup", window, "f7", { shiftKey: true });
  equal(site.ui.board.shapes, [{ orig: "e2", dest: "e4" }, { orig: "f7", brush: "red" }]);
  assert(site.ui.board.el.querySelector(".shapes line.green") && site.ui.board.el.querySelector(".shapes circle.red"));
  await site.play("e2e4");
  equal(site.ui.board.shapes, [], "a new position clears them");
});

test("the opening's name is shown above the moves", async () => {
  await until(() => site.ui.openings, "the name table");
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  assert($("opening").hidden, "no name at the start");
  site.ui.chess.move("e4");
  site.ui.chess.move("c5");
  site.render();
  assert(!$("opening").hidden);
  equal($("opening-name").textContent, "Sicilian Defense");
  equal($("opening-eco").textContent, "B20");
  site.ui.chess.move("Nf3");
  site.ui.chess.move("d6");
  site.render();
  equal($("opening-name").textContent, "Sicilian Defense: Modern Variations");
});

test("looking back, the opening is the one at that point of the game", async () => {
  await until(() => site.ui.openings, "the name table");
  await site.newGame({ rating: 1500, color: "white" });
  for (const san of ["e4", "c5", "Nf3", "d6"]) site.ui.chess.move(san);
  site.render();
  site.showPly(2);
  equal($("opening-name").textContent, "Sicilian Defense");
  site.showPly(1);
  equal($("opening-name").textContent, "King's Pawn Game");
  site.showPly(0);
  assert($("opening").hidden, "the start position");
  site.showPly(null);
  equal($("opening-name").textContent, "Sicilian Defense: Modern Variations");
});

test("the PGN names the opening (ECO and Opening headers)", async () => {
  await until(() => site.ui.openings, "the name table");
  await site.newGame({ rating: 1500, color: "white" });
  for (const san of ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]) site.ui.chess.move(san);
  site.render();
  const text = site.pgn();
  assert(text.includes('[Opening "Nimzo-Indian Defense"]'), text);
  assert(/\[ECO "E2\d"\]/.test(text), text);
});

test("the game modes: a standard game, or from a position (a box for a FEN or a PGN)", () => {
  const mode = (value) => {
    const radio = $("new-game").querySelector(`input[name="mode"][value="${value}"]`);
    radio.checked = true;
    radio.dispatchEvent(new Event("change"));
  };
  assert($("position-fields").hidden, "standard: no position box");
  mode("position");
  assert(!$("position-fields").hidden, "from a position: the box");
  assert($("position-text").placeholder.includes(" w "), "an example FEN as the placeholder");
  mode("standard");
  assert($("position-fields").hidden);
});

/** Fill in the new-game form for a game from a position. */
function positionForm(text, color = "white") {
  const radio = $("new-game").querySelector('input[name="mode"][value="position"]');
  radio.checked = true;
  radio.dispatchEvent(new Event("change"));
  $("position-text").value = text;
  $("new-game").querySelector(`input[name="color"][value="${color}"]`).checked = true;
}

function standardForm() {
  const radio = $("new-game").querySelector('input[name="mode"][value="standard"]');
  radio.checked = true;
  radio.dispatchEvent(new Event("change"));
  $("new-game").querySelector('input[name="color"][value="white"]').checked = true;
}

test("from a position: a FEN, and the bot moves first when it is its turn", async () => {
  site.ui.random = () => 0.5;
  positionForm("8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1");
  await site.startFromForm();
  equal(site.ui.chess.fen(), "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1");
  equal(site.ui.board.mover, "white", "your move: you are White and White is to move");
  positionForm("8/8/8/4k3/8/8/4K3/4Q3 b - - 0 1");
  await site.startFromForm();
  equal(site.ui.chess.history().length, 1, "Black to move and you are White: the bot moved");
  equal(site.ui.board.mover, "white");
  standardForm();
});

test("from a position: a PGN's moves are kept in the move list", async () => {
  site.ui.random = () => 0.5;
  positionForm('[Event "Club game"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *', "black");
  await site.startFromForm();
  equal(site.ui.chess.history().slice(0, 6), ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
  equal(site.ui.chess.history().length, 7, "White (the bot) played its 4th move");
  equal($("move-list").querySelectorAll(".san[data-ply]").length, 7);
  standardForm();
});

test("from a position: what cannot be played says so, and no game starts", async () => {
  await site.newGame({ rating: 1500, color: "white" });
  const before = site.ui.chess;
  positionForm("this is not a position");
  await site.startFromForm();
  assert(!$("position-error").hidden, "an error shown");
  assert($("position-error").textContent.startsWith("That FEN is not valid"), $("position-error").textContent);
  assert(site.ui.chess === before, "the game in progress is untouched");
  positionForm("8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1");
  await site.startFromForm();
  assert($("position-error").hidden, "a good one clears the error");
  standardForm();
});

test("from a position: pick a practice position", async () => {
  site.ui.random = () => 0.5;
  positionForm("");
  const select = $("preset");
  assert(select.options.length >= 9, "the presets are listed");
  select.value = [...select.options].find((o) => o.text.startsWith("Lucena")).value;
  select.dispatchEvent(new Event("change"));
  equal($("position-text").value, "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1");
  await site.startFromForm();
  equal(site.ui.chess.fen(), "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1");
  select.value = "";
  standardForm();
});

test("from a position: captures count from there, and the PGN records the position", async () => {
  site.ui.random = () => 0.5;
  positionForm("8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1");
  await site.startFromForm();
  equal(document.querySelectorAll(".captured .piece").length, 0, "nothing captured: the position had no black queen");
  equal($("bottom-captured").querySelector(".score").textContent, "+9", "but White is a queen up");
  await site.play("e1h4");
  const text = site.pgn();
  assert(text.includes('[SetUp "1"]') && text.includes('[FEN "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1"]'), text);
  assert(text.includes("from a position"), text);
  standardForm();
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  assert(!site.pgn().includes("[SetUp"), "a standard game has no SetUp header");
});

test("Copy FEN copies the position shown (the one looked at, when looking back)", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1500, color: "white" });
  await site.play("e2e4");
  let copied = null;
  const clipboard = navigator.clipboard;
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: async (t) => { copied = t; } } });
  try {
    $("copy-fen").click();
    await until(() => copied, "copied");
    equal(copied, site.ui.chess.fen());
    assert($("status").textContent.startsWith("Copied: "), $("status").textContent);
    site.showPly(0);
    equal(await site.copyFen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "the position looked at");
    Object.defineProperty(navigator, "clipboard", { configurable: true,
      value: { writeText: async () => { throw new Error("denied"); } } });
    await site.copyFen();
    assert($("status").textContent.startsWith("Your browser did not allow copying."), "says so, and shows the FEN");
  } finally {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: clipboard });
    site.showPly(null);
  }
});

test("Play from here: a new game from a position looked at, with the moves that led there", async () => {
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1400, color: "white" });
  await site.play("d2d4");
  await site.play("c2c4");
  assert($("play-from-here").hidden, "only while looking back");
  site.showPly(2);
  assert(!$("play-from-here").hidden);
  const fen = site.ui.board.fen;
  $("play-from-here").click();
  await until(() => site.ui.view === null && site.ui.chess.history().length === 2, "the new game");
  equal(site.ui.chess.fen(), fen);
  equal([site.ui.rating, site.ui.human], [1400, "w"], "same rating and side");
  equal(site.ui.board.mover, "white", "your move, as it was then");
});

test("a game from a position survives a reload", async () => {
  site.ui.random = () => 0.5;
  positionForm("1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1");
  await site.startFromForm();
  await site.play("c1d1");
  const snap = site.snapshot();
  const fen = site.ui.chess.fen();
  await site.newGame({ rating: 1500, color: "white" });
  assert(site.restore(snap));
  equal(site.ui.chess.fen(), fen);
  equal(site.ui.startFen, "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1");
  assert(site.pgn().includes('[FEN "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"]'));
  standardForm();
});

/** Choose a Difficulty on the new-game form. */
function difficulty(value) {
  $("new-game").querySelector(`input[name="difficulty"][value="${value}"]`).checked = true;
  $("new-game").querySelector(`input[name="difficulty"][value="${value}"]`).dispatchEvent(new Event("change"));
}

test("Follow my level: the bot plays at your rating, and a result moves it", async () => {
  site.ui.random = () => 0.5;
  site.resetProfile(1200);
  difficulty("between");
  assert($("rating").disabled, "the slider is not used");
  equal($("rating-value").textContent, "1200");
  assert($("adaptive-note").textContent.startsWith("Your first game starts at 1200"), $("adaptive-note").textContent);
  await site.startFromForm();
  equal(site.ui.rating, 1200);
  await site.play("e2e4");
  site.resign();
  equal(site.profile().games, 1);
  equal(Math.round(site.profile().rating), 1160, "a loss while new: K 80 x 0.5");
  equal($("result-banner").querySelector(".adaptive-result").textContent, "Your rating: 1200 \u2192 1160. Next bot: 1150.");
  site.resign();
  equal(site.profile().games, 1, "counted once");
  equal($("adaptive-note").textContent, "Your rating: 1160 after 1 game. The bot plays at 1150.");
  await site.startFromForm();
  equal(site.ui.rating, 1150, "the next bot");
  difficulty("off");
  assert(!$("rating").disabled);
  await site.newGame({ rating: 1500, color: "white" });
  site.resign();
  equal(site.profile().games, 1, "fixed-rating games do not count");
});

test("Follow my level: games from a position do not count, and a reload keeps the mode", async () => {
  site.ui.random = () => 0.5;
  site.resetProfile(1400);
  positionForm("8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1");
  difficulty("between");
  await site.startFromForm();
  site.resign();
  equal(site.profile().games, 0, "not counted");
  assert(!$("result-banner").querySelector(".adaptive-result"));
  standardForm();
  await site.startFromForm();
  const snap = site.snapshot();
  await site.newGame({ rating: 1500, color: "white" });
  assert(site.restore(JSON.parse(JSON.stringify(snap))));
  equal(site.ui.adaptive.mode, "between");
  site.resign();
  equal(site.profile().games, 1, "counted after the reload");
  difficulty("off");
});

run();
