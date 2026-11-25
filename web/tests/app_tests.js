// End-to-end tests of the play page (app.js + board.js + api.js) against the real API server.

import { assert, equal, run, test, until } from "./harness.js";

async function loadPage() {
  const html = await (await fetch("/index.html")).text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script").forEach((s) => s.remove());
  document.getElementById("page").replaceChildren(...doc.body.children);
}

await loadPage();
const app = await import("../js/app.js");
app.init();
const $ = (id) => document.getElementById(id);

function clickSquare(square) {
  app.ui.board.squareEl(square).dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

test("a new game as White shows the start position, ready to move", async () => {
  await app.startGame({ rating: 1200, color: "white", seed: 1 });
  equal(app.ui.game.status, "ongoing");
  equal(app.ui.board.el.querySelectorAll(".piece").length, 32);
  equal(app.ui.board.mover, "white");
  equal($("bottom-name").textContent, "You");
  equal($("top-name").textContent, "Bot 1200");
  assert($("status").textContent.startsWith("Your move"), $("status").textContent);
  assert(!$("game-panel").hidden, "game panel shown");
});

test("hidden overlays are really invisible (the promotion backdrop must not dim the board)", () => {
  equal(getComputedStyle($("promotion")).display, "none");
  equal(getComputedStyle($("result-banner")).display, "none");
});

test("clicking a move plays it and the bot answers", async () => {
  clickSquare("e2");
  clickSquare("e4");
  await until(() => app.ui.game.moves_san.length === 2 && !app.ui.busy, "bot reply");
  equal(app.ui.game.moves_san[0], "e4");
  equal(app.ui.board.mover, "white");
  assert(app.ui.board.squareEl(app.ui.game.last_move.slice(2, 4)).classList.contains("last-move"), "last move shown");
});

test("the move list and the bot's candidates are shown", async () => {
  const cells = [...document.querySelectorAll("#move-list .san")].map((c) => c.textContent);
  equal(cells, app.ui.game.moves_san);
  equal(document.querySelectorAll("#move-list .num")[0].textContent, "1.");
  assert(document.querySelector("#move-list .san.latest").textContent === app.ui.game.moves_san[1], "latest marked");
  assert(!$("thinking-panel").hidden, "candidates panel shown");
  const rows = document.querySelectorAll("#candidates .candidate");
  equal(rows.length, 5);
  assert($("thinking-caption").textContent.includes("about 1200"), $("thinking-caption").textContent);
  const played = app.ui.game.last_bot_move;
  if (played.candidates.some((c) => c.uci === played.uci)) {
    equal(document.querySelector("#candidates .played .san").textContent, played.san);
  }
  const width = parseFloat(rows[0].querySelector(".fill").style.width);
  assert(Math.abs(width - played.candidates[0].prob * 100) < 1.01, `bar width ${width}`);
});

test("move list rows start empty for White when Black moved first", () => {
  app.renderMoves({ fen: "x", turn: "white", moves_san: ["e5"] });
  const cells = [...document.querySelectorAll("#move-list .san")].map((c) => c.textContent);
  equal(cells, ["", "e5"]);
  app.renderMoves(app.ui.game);
});

test("as Black the board is flipped and the bot has opened", async () => {
  await app.startGame({ rating: 1800, color: "black", seed: 2 });
  equal(app.ui.board.orientation, "black");
  equal(app.ui.game.moves_san.length, 1);
  equal(app.ui.board.el.querySelector(".square").dataset.square, "h1");
  equal(app.ui.board.mover, "black");
  equal($("bottom-name").textContent, "You");
});

test("clock formatting and countdown", () => {
  equal(app.formatClock(185), "3:05");
  equal(app.formatClock(60), "1:00");
  equal(app.formatClock(9.46), "0:09.5");
  equal(app.formatClock(-2), "0:00.0");
  const clock = { white: 100, black: 50, running: "black", at: 1000 };
  equal(app.clockTimes(clock, 3500), { white: 100, black: 47.5 });
  equal(app.clockTimes({ ...clock, running: null }, 9000), { white: 100, black: 50 });
});

test("a timed game shows both clocks, the running one highlighted", async () => {
  await app.startGame({ rating: 1500, color: "white", time_control: "3+2", seed: 6 });
  assert(!$("top-clock").hidden && !$("bottom-clock").hidden, "clocks shown");
  equal($("bottom-clock").textContent, "3:00");
  equal($("top-clock").textContent, "3:00");
  await app.playMove("e2e4");
  await until(() => app.ui.game.moves_san.length === 2, "bot reply");
  assert($("bottom-clock").classList.contains("running"), "your clock runs after both first moves");
  assert(!$("top-clock").classList.contains("running"), "the bot's clock is stopped");
});

test("an untimed game hides the clocks", async () => {
  await app.startGame({ rating: 1500, color: "white", seed: 7 });
  assert($("top-clock").hidden && $("bottom-clock").hidden, "clocks hidden");
});

test("when the running clock reaches zero, the page asks the server", async () => {
  const real = { ...app.ui.game };
  const fakeClock = { white: 0.05, black: 100, running: "white", base_seconds: 60, increment: 0 };
  app.render({ ...real, clock: fakeClock });          // pretend White is about to flag
  await until(() => !app.ui.flagCheck && app.ui.game.clock === null, "state refreshed from the server", 3000);
  equal(app.ui.game.id, real.id);                      // the real (untimed) state came back
});

test("the form sends the chosen rating, side and time control", () => {
  $("rating").value = "1350";
  $("rating").dispatchEvent(new Event("input"));
  equal($("rating-value").textContent, "1350");
  document.querySelector('input[name="color"][value="black"]').checked = true;
  $("time-control").value = "5+3";
  equal(app.readForm($("new-game")), { rating: 1350, color: "black", time_control: "5+3" });
  $("time-control").value = "";
  equal(app.readForm($("new-game")).time_control, undefined);
});

test("server errors are shown, and the game resyncs", async () => {
  await app.startGame({ rating: 1500, color: "white", seed: 3 });
  await app.playMove("e2e5");           // illegal: the server says so
  assert($("status").textContent.includes("not legal"), $("status").textContent);
  equal(app.ui.game.moves_san.length, 0);
  equal(app.ui.busy, false);
});

run();
