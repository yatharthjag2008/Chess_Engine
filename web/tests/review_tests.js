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

test("errors are shown", async () => {
  await app.loadGames("lichess", "x");
  assert($("status").textContent.includes("not a valid"), $("status").textContent);
});

run();
