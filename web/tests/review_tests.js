// End-to-end tests of the review page (review.js) against the real API server.

import { assert, equal, run, test, until } from "./harness.js";

export const PGN = `[White "Ann"]
[Black "Ben"]
[WhiteElo "1400"]
[BlackElo "1350"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

[White "Ben"]
[Black "Cy"]
[Result "*"]

1. d4 d5 *
`;

async function loadPage() {
  const html = await (await fetch("/review.html")).text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script").forEach((s) => s.remove());
  document.getElementById("page").replaceChildren(...doc.body.children);
}

await loadPage();
export const app = await import("../js/review.js");
app.init();
export const $ = (id) => document.getElementById(id);

test("the source form switches between PGN and username", () => {
  assert(document.querySelector('#source-form textarea[name="pgn"]'), "PGN box first");
  const lichess = document.querySelector('input[name="source"][value="lichess"]');
  lichess.checked = true;
  lichess.dispatchEvent(new Event("change"));
  assert(document.querySelector('#source-form input[name="username"]'), "username box");
  equal(document.querySelector("#source-form button").textContent, "Fetch recent games");
  app.renderSourceForm("pgn");
});

test("pasted games are listed and can be chosen", async () => {
  await app.loadGames("pgn", PGN);
  equal(app.ui.games.length, 2);
  const buttons = document.querySelectorAll("#game-list button");
  equal(buttons.length, 2);
  assert(buttons[0].textContent.includes("Ann (1400) - Ben (1350)"), buttons[0].textContent);
  assert($("status").textContent.startsWith("2 games found"), $("status").textContent);
  const game = app.chooseGame(0, "black");
  equal(game.white, "Ann");
  equal(app.ui.side, "black");
});

test("the player's side comes from the username", () => {
  const game = { white: "Ann", black: "Ben" };
  equal(app.sideOf(game, "ben"), "black");
  equal(app.sideOf(game, "ANN"), "white");
  equal(app.sideOf(game, "zed"), null);
  app.ui.username = "Ben";
  app.ui.games = [game];
  app.chooseGame(0);
  equal(app.ui.side, "black");
  app.ui.username = null;
});

test("choosing a game offers your side and a Review button", async () => {
  await app.loadGames("pgn", PGN);
  app.chooseGame(0, "black");
  const radios = document.querySelectorAll('input[name="side"]');
  equal(radios.length, 2);
  assert(radios[1].checked && !radios[0].checked, "Black chosen");
  assert(radios[0].nextElementSibling.textContent.includes("Ann"), "names shown");
  equal(document.querySelector('#game-list button[data-index="0"]').getAttribute("aria-pressed"), "true");
  radios[0].checked = true;
  radios[0].dispatchEvent(new Event("change"));
  equal(app.ui.side, "white");
  assert($("review-button"), "Review button");
});

test("the review fills the move list with classes and symbols", async () => {
  app.chooseGame(0, "black");
  const review = await app.startReview();
  assert(review, $("status").textContent);
  equal(review.moves.length, 7);
  const cells = document.querySelectorAll("#move-list .san[data-ply]");
  equal(cells.length, 7);
  equal(document.querySelectorAll("#move-list li").length, 4);
  const blunder = cells[5];
  assert(blunder.classList.contains("blunder"), blunder.className);
  equal(blunder.textContent, "Nf6??");
  assert(!$("moves-panel").hidden, "moves shown");
  equal($("bottom-name").textContent, "Ben");
  equal(app.ui.board.orientation, "black");
  assert($("status").textContent.startsWith("Reviewed 7 moves"), $("status").textContent);
});

test("stepping with buttons, keys and clicks", () => {
  equal(app.ui.ply, 0);
  assert($("prev").disabled && !$("next").disabled, "at the start");
  $("next").click();
  equal(app.ui.ply, 1);
  equal($("ply-label").textContent, "1. e4");
  assert(app.ui.board.squareEl("e4").querySelector(".piece"), "pawn on e4");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
  equal(app.ui.ply, 2);
  equal($("ply-label").textContent, "1... e5");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true }));
  equal(app.ui.ply, 7);
  assert($("next").disabled, "at the end");
  assert(app.ui.board.squareEl("f7").querySelector(".piece"), "queen on f7 after mate");
  document.querySelector('#move-list .san[data-ply="3"]').click();
  equal(app.ui.ply, 3);
  const current = document.querySelectorAll("#move-list .san.current");
  equal(current.length, 1);
  equal(current[0].dataset.ply, "3");
  $("first").click();
  equal(app.ui.ply, 0);
  equal(document.querySelectorAll("#move-list .san.current").length, 0);
  app.goTo(99);
  equal(app.ui.ply, 7);
});

test("arrow keys typed into a text box do not step", () => {
  app.goTo(2);
  const box = document.querySelector("#source-form textarea, #source-form input");
  box.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
  equal(app.ui.ply, 2);
});

test("errors are shown", async () => {
  await app.loadGames("lichess", "x");
  assert($("status").textContent.includes("not a valid"), $("status").textContent);
});

run();
