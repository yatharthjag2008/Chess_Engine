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

test("clicking a move plays it and the bot answers", async () => {
  clickSquare("e2");
  clickSquare("e4");
  await until(() => app.ui.game.moves_san.length === 2 && !app.ui.busy, "bot reply");
  equal(app.ui.game.moves_san[0], "e4");
  equal(app.ui.board.mover, "white");
  assert(app.ui.board.squareEl(app.ui.game.last_move.slice(2, 4)).classList.contains("last-move"), "last move shown");
});

test("as Black the board is flipped and the bot has opened", async () => {
  await app.startGame({ rating: 1800, color: "black", seed: 2 });
  equal(app.ui.board.orientation, "black");
  equal(app.ui.game.moves_san.length, 1);
  equal(app.ui.board.el.querySelector(".square").dataset.square, "h1");
  equal(app.ui.board.mover, "black");
  equal($("bottom-name").textContent, "You");
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
