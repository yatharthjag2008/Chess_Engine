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

test("evaluations are formatted for people", () => {
  equal(app.formatEval(125), "+1.25");
  equal(app.formatEval(-40), "-0.40");
  equal(app.formatEval(0), "0.00");
  equal(app.formatEval(9997), "#3");
  equal(app.formatEval(-9998), "#-2");
  equal(app.formatEval(10000), "1-0");
  equal(app.formatEval(-10000), "0-1");
});

test("the detail panel explains the move and shows the better one", () => {
  app.goTo(0);
  assert($("detail-panel").hidden, "nothing to describe at the start");
  app.goTo(6);
  const panel = $("detail-panel");
  assert(!panel.hidden, "shown");
  const verdict = panel.querySelector(".verdict");
  assert(verdict.classList.contains("blunder"), verdict.className);
  equal(verdict.textContent, "3... Nf6??: Blunder");
  assert(panel.textContent.includes("Best was g6"), panel.textContent);
  assert(panel.querySelector(".detail-evals").textContent.includes("0.00 \u2192 +9.00"),
         panel.querySelector(".detail-evals").textContent);
  const show = $("show-best");
  show.click();
  assert(app.ui.board.squareEl("g7").classList.contains("best-from"), "from square marked");
  assert(app.ui.board.squareEl("g6").classList.contains("best-to"), "to square marked");
  assert(app.ui.board.squareEl("g8").querySelector(".piece"), "position before the move: knight still on g8");
  equal(show.getAttribute("aria-pressed"), "true");
  show.click();
  equal(app.ui.board.el.querySelectorAll(".best-from, .best-to").length, 0);
  assert(app.ui.board.squareEl("f6").querySelector(".piece"), "back after the move");
  show.click();
  app.step(1);
  equal(app.ui.board.el.querySelectorAll(".best-from, .best-to").length, 0, "stepping clears the marks");
  const mate = $("detail-panel").querySelector(".verdict");
  assert(mate.textContent.startsWith("4. Qxf7#"), mate.textContent);
});

test("the eval graph shows the game and jumps to a clicked move", () => {
  const graph = $("graph");
  assert(!graph.hidden, "graph shown");
  const hits = graph.querySelectorAll("rect.hit");
  equal(hits.length, 8, "start + 7 moves");
  const markers = graph.querySelectorAll("circle.marker");
  equal(markers.length, 1, "one error");
  assert(markers[0].classList.contains("blunder"), "the blunder");
  equal(markers[0].getAttribute("data-ply"), "6");
  assert(Number(markers[0].getAttribute("cy")) < app.GRAPH.height / 2, "White is winning after the blunder");
  assert(hits[6].querySelector("title").textContent.startsWith("3... Nf6??"), hits[6].querySelector("title").textContent);
  hits[3].dispatchEvent(new MouseEvent("click", { bubbles: true }));
  equal(app.ui.ply, 3);
  const cursor = graph.querySelector("line.current");
  equal(Number(cursor.getAttribute("x1")), (3 / 7) * app.GRAPH.width);
  app.goTo(7);
  equal(Number(cursor.getAttribute("x1")), app.GRAPH.width);
});

test("the summary shows both players and the key moments", () => {
  const panel = $("summary-panel");
  assert(!panel.hidden, "shown");
  const cells = [...panel.querySelectorAll(".summary-grid > span")].map((c) => c.textContent);
  equal(cells.slice(0, 3), ["", "Ann", "Ben"]);
  equal(cells[3], "Accuracy");
  assert(cells[4].endsWith("%") && cells[5].endsWith("%"), cells.slice(4, 6).join(" "));
  const blunderRow = cells.indexOf("Blunder ??");
  assert(blunderRow > 0, cells.join("|"));
  equal(cells.slice(blunderRow + 1, blunderRow + 3), ["0", "1"]);
  const moments = panel.querySelectorAll(".moments button");
  equal(moments.length, 1);
  assert(moments[0].textContent.startsWith("3... Nf6??"), moments[0].textContent);
  app.goTo(0);
  moments[0].click();
  equal(app.ui.ply, 6, "jumps to the position after the blunder");
});

test("the coach comments on your moves", async () => {
  const coach = await app.ui.coaching;
  assert(coach, $("coach-panel").textContent);
  const panel = $("coach-panel");
  assert(panel.textContent.includes("Coaching Black, rated about 1350 (from the game)"), panel.textContent);
  assert(panel.textContent.includes("1550-level move"), panel.textContent);
  const blunderNote = coach.report.notes.find((n) => n.san === "Nf6");
  assert(blunderNote, JSON.stringify(coach.report.notes));
  app.goTo(6);
  equal($("coach-note").textContent, blunderNote.text);
  app.goTo(1);
  equal($("coach-note").textContent, "Step to one of your moves for its note.");
  app.goTo(0);
  $("next-note").click();
  equal(app.ui.ply, coach.report.notes[0].ply + 1);
});

test("coaching at another rating and estimating the rating", async () => {
  $("coach-rating").value = "1800";
  const again = await app.loadCoach(1800);
  equal(again.report.rating, 1800);
  equal(again.rating_source, "given");
  assert($("coach-panel").textContent.includes("(as you set it)"), $("coach-panel").textContent);
  const estimate = await app.estimateRating();
  assert(estimate, $("estimate").textContent);
  assert(/look like a \d+ player/.test($("estimate").textContent), $("estimate").textContent);
});

test("a game against the bot opens from its id, from the side you played", async () => {
  const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                              body: JSON.stringify(body) }).then((r) => r.json());
  const game = await post("/api/games", { rating: 1500, color: "black", seed: 4 });
  if (!game.moves_san.length) await post(`/api/games/${game.id}/bot-move`);
  const after = (await post(`/api/games/${game.id}/moves`, { move: "g8f6", reply: true })).state;
  assert(after.moves_san.length >= 2, JSON.stringify(after.moves_san));
  const review = await app.reviewBotGame(game.id);
  assert(review, $("status").textContent);
  equal(review.moves.length, after.moves_san.length);
  equal(app.ui.side, "black");
  equal(app.ui.board.orientation, "black");
  await app.ui.coaching;
  assert($("coach-panel").textContent.includes("Coaching Black"), $("coach-panel").textContent);
  history.replaceState(null, "", `#game=${game.id}`);
  equal(app.gameFromUrl(), game.id);
  history.replaceState(null, "", location.pathname);
  equal(app.gameFromUrl(), null);
});

test("an unknown bot game says so", async () => {
  equal(await app.reviewBotGame("no-such-game"), null);
  assert($("status").textContent.includes("no longer on the server"), $("status").textContent);
});

test("errors are shown", async () => {
  await app.loadGames("lichess", "x");
  assert($("status").textContent.includes("not a valid"), $("status").textContent);
});

run();
