// End-to-end tests of the play page (app.js + board.js + api.js) against the real API server.

import { assert, equal, run, test, until } from "./harness.js";
import { api } from "../js/api.js";

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
  const fill = rows[0].querySelector(".fill").getBoundingClientRect().width;
  const track = rows[0].querySelector(".track").getBoundingClientRect().width;
  assert(track > 0 && Math.abs(fill / track - Math.max(0.01, played.candidates[0].prob)) < 0.02,
         `the bar is drawn to scale (fill ${fill}px of ${track}px)`);
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
  equal(app.readForm($("new-game")), { rating: 1350, color: "black", time_control: "5+3", hint_limit: 3,
                                        adaptive: "off" });
  $("time-control").value = "";
  equal(app.readForm($("new-game")).time_control, undefined);
});

test("resign needs a second click, then the result banner offers the PGN", async () => {
  await app.startGame({ rating: 1500, color: "white", seed: 8 });
  await app.playMove("d2d4");
  await until(() => app.ui.game.moves_san.length === 2, "bot reply");
  $("resign").click();
  equal(app.ui.game.status, "ongoing");
  equal($("resign").textContent, "Click again to resign");
  $("resign").click();
  await until(() => app.ui.game.status === "over", "resigned");
  assert(!$("result-banner").hidden, "banner shown");
  equal($("result-banner").querySelector("h2").textContent, "The bot won");
  const link = $("result-banner").querySelector("a");
  assert(link.href.endsWith(`/api/games/${app.ui.game.id}/pgn`), link.href);
  const pgn = await (await fetch(link.href)).text();
  assert(pgn.includes('[Termination "resignation"]'), pgn);
  const review = $("result-banner").querySelector("a.review-link");
  assert(review && review.getAttribute("href") === `review.html#game=${app.ui.game.id}`, review && review.href);
  assert($("resign").disabled && $("takeback").disabled, "buttons disabled after the game");
  equal(app.ui.board.mover, null);
});

test("take back undoes your move and the bot's reply", async () => {
  await app.startGame({ rating: 1500, color: "white", seed: 9 });
  assert($("takeback").disabled, "nothing to take back yet");
  await app.playMove("e2e4");
  await until(() => app.ui.game.moves_san.length === 2, "bot reply");
  assert(!$("takeback").disabled, "take back enabled");
  await app.takeback();
  equal(app.ui.game.moves_san, []);
  equal(app.ui.board.mover, "white");
});

test("timed games have no take back", async () => {
  await app.startGame({ rating: 1500, color: "white", time_control: "5+0", seed: 10 });
  await app.playMove("e2e4");
  await until(() => app.ui.game.moves_san.length === 2, "bot reply");
  assert($("takeback").disabled, "take back disabled");
});

test("the game id is kept in the URL and a reload continues the game", async () => {
  const id = app.ui.game.id;
  equal(location.hash, `#game=${id}`);
  app.ui.game = null;
  assert(await app.resumeFromUrl(), "resumed");
  equal(app.ui.game.id, id);
  history.replaceState(null, "", "#game=unknown-id");
  assert(!(await app.resumeFromUrl()), "unknown game");
  equal(location.hash, "");
  assert($("status").textContent.includes("no longer on the server"), $("status").textContent);
});

test("a checkmate ends the game with the right banner", async () => {
  await app.startGame({ rating: 1500, color: "white", fen: "7k/8/6K1/8/8/8/R7/1R6 w - - 0 1", seed: 11 });
  await app.playMove("a2a7");
  await until(() => app.ui.game.moves_san.length === 2, "forced reply");
  await app.playMove("b1b8");
  equal(app.ui.game.outcome.termination, "checkmate");
  equal($("result-banner").querySelector("h2").textContent, "You won");
  $("result-banner").querySelector("button:last-child").click();
  assert($("result-banner").hidden, "banner closed");
});

test("hints go nudge -> piece -> move, marked on the board, and make the game unrated", async () => {
  await app.startGame({ rating: 1500, color: "white", hint_limit: 2, seed: 12 });
  const button = $("hint");
  assert(!button.hidden && !button.disabled, "hint button available");
  equal(button.textContent, "Hint (2 left)");
  assert($("unrated").hidden, "rated before any hint");
  const nudge = await app.askHint();
  equal(nudge.tier, "nudge");
  assert(!$("hint-panel").hidden, "panel shown");
  equal($("hint-text").textContent, nudge.text);
  equal($("hint-tier").textContent, "1 of 3: a nudge");
  equal($("hint-more").textContent, "Which piece?");
  equal(button.textContent, "Hint (1 left)");
  assert(!$("unrated").hidden, "Unrated badge");
  equal(app.ui.board.el.querySelectorAll(".hint-from, .hint-to").length, 0, "a nudge marks nothing");
  $("hint-more").click();
  await until(() => app.ui.hint.data.tier === "point", "point");
  assert(app.ui.board.squareEl("a2").classList.contains("hint-from"), "the piece is marked");
  await app.askHint();
  equal(app.ui.hint.data.tier, "show");
  assert(app.ui.board.squareEl("a3").classList.contains("hint-to"), "the target is marked");
  assert($("hint-more").hidden, "nothing more to show");
  equal(app.ui.hint.data.hints.used, 1, "one position, one hint");
  $("flip").click();
  assert(app.ui.board.squareEl("a2").classList.contains("hint-from"), "marks survive a flip");
  $("flip").click();
  await app.playMove("a2a3");
  await until(() => app.ui.game.moves_san.length === 2, "bot reply");
  assert($("hint-panel").hidden, "the hint belongs to the old position");
  equal(app.ui.board.el.querySelectorAll(".hint-from, .hint-to").length, 0);
  const pgn = await (await fetch(`/api/games/${app.ui.game.id}/pgn`)).text();
  assert(pgn.includes('[Hints "1"]'), pgn);
});

test("the hint button runs out, and is hidden when hints are off", async () => {
  await app.startGame({ rating: 1500, color: "white", hint_limit: 1, seed: 13 });
  await app.askHint();
  await app.playMove("a2a3");
  await until(() => app.ui.game.moves_san.length === 2, "bot reply");
  assert($("hint").disabled, "no hints left for a new position");
  equal($("hint").textContent, "Hint (0 left)");
  await app.startGame({ rating: 1500, color: "white", hint_limit: 3, seed: 14 });
  await app.askHint();
  await app.startGame({ rating: 1500, color: "white", hint_limit: 0, seed: 15 });
  assert($("hint-panel").hidden, "the last game's hint (same start position) is not shown in a new game");
  equal(app.ui.board.el.querySelectorAll(".hint-from, .hint-to").length, 0);
  assert($("hint").hidden, "hints off");
  assert($("unrated").hidden, "a new game is rated");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "h", bubbles: true }));
  equal(app.ui.hint.id === app.ui.game.id, false, "the h key does nothing when hints are off");
});

test("adapt between games: the bot plays at your rating and the result moves it", async () => {
  await api.resetAdaptive(1630);
  await app.loadAdaptive();
  const form = $("new-game");
  const between = form.querySelector('input[name="adaptive"][value="between"]');
  between.checked = true;
  between.dispatchEvent(new Event("change"));
  assert($("rating").disabled, "the slider is not used");
  equal($("rating").value, "1650");
  assert($("adaptive-note").textContent.includes("Your rating: 1630 (0 rated games)"), $("adaptive-note").textContent);
  const options = app.readForm(form);
  equal(options.rating, 1650, "a disabled slider is still read");
  equal(options.adaptive, "between");
  await app.startGame({ ...options, color: "white", hint_limit: 0, seed: 21 });
  equal(app.ui.game.bot_rating, 1650);
  equal($("top-name").textContent, "Bot 1650");
  $("resign").click();
  $("resign").click();
  await until(() => app.ui.game.status === "over", "resigned");
  await until(() => app.ui.adaptiveLoading, "profile reload started");
  await app.ui.adaptiveLoading;
  const line = $("result-banner").querySelector(".adaptive-result");
  assert(line && line.textContent.startsWith("Your rating: 1630 \u2192 1592. Next bot: 1600."), line && line.textContent);
  equal(app.ui.adaptive.games, 1);
  equal($("rating").value, "1600", "the form is ready for the next game");
  const before = app.ui.adaptiveLoading;
  $("flip").click();
  equal(app.ui.adaptiveLoading, before, "the profile is loaded once per game, not on every re-render");
  const off = form.querySelector('input[name="adaptive"][value="off"]');
  off.checked = true;
  off.dispatchEvent(new Event("change"));
  assert(!$("rating").disabled, "fixed difficulty uses the slider again");
  await api.resetAdaptive();
});

test("the bot's label and the adaptive summary", () => {
  const live = { mode: "live", start_rating: 1500, changes: [], result: null };
  equal(app.botLabel({ bot_rating: 1500, adaptive: live }), "Bot 1500");
  equal(app.botLabel({ bot_rating: 1600, adaptive: live }), "Bot 1600 \u2191");
  equal(app.botLabel({ bot_rating: 1450, adaptive: live }), "Bot 1450 \u2193");
  equal(app.botLabel({ bot_rating: 1450, adaptive: { ...live, mode: "off", start_rating: 1500 } }), "Bot 1450");
  const over = { status: "over", hints: { rated: false }, adaptive: { mode: "between", result: null } };
  equal(app.adaptiveSummary(over), "Hints were used, so your adaptive rating is unchanged.");
  equal(app.adaptiveSummary({ ...over, status: "ongoing" }), "");
  equal(app.adaptiveSummary({ ...over, adaptive: { mode: "off", result: null } }), "");
});

test("the bot can play in a trained style", async () => {
  const styles = await app.loadStyles();
  equal(styles.map((st) => st.name), ["ann"]);
  assert(!$("style-field").hidden, "the Style choice is shown when styles exist");
  const option = $("style").querySelector('option[value="ann"]');
  equal(option.textContent, "Ann: style, 200 games");
  $("style").value = "ann";
  $("style").dispatchEvent(new Event("change"));
  equal($("style-note").textContent, "Imitates Ann's choices fully.");
  const options = app.readForm($("new-game"));
  equal(options.style, "ann");
  await app.startGame({ ...options, color: "black", temperature: 0, hint_limit: 0 });
  equal(app.ui.game.moves_san, ["a3"], "Ann's favourite first move");
  equal(app.ui.game.style.player, "Ann");
  equal($("top-name").textContent, `Bot ${options.rating} (Ann)`);
  $("style").value = "";
  $("style").dispatchEvent(new Event("change"));
  equal($("style-note").textContent, "");
  equal(app.readForm($("new-game")).style, undefined, "no style: the rating model");
});

test("style labels and blending notes", () => {
  equal(app.styleLabel({ player: "me", kind: "ghost", until: "2025-05-31", games: 40 }), "me: ghost, games up to 2025-05-31");
  equal(app.styleLabel({ player: "Ann", kind: "style", games: 12 }), "Ann: style, 12 games");
  equal(app.botLabel({ bot_rating: 1600, style: { player: "Ann" },
                       adaptive: { mode: "live", start_rating: 1500 } }), "Bot 1600 (Ann) \u2191");
});

test("new-game settings are remembered in this browser", () => {
  app.saveSettings({ rating: 1750, color: "black", time_control: "10+0", hint_limit: 0, adaptive: "live" });
  const form = $("new-game");
  form.elements.rating.value = "1500";
  app.loadSettings(form);
  equal(app.readForm(form), { rating: 1750, color: "black", time_control: "10+0", hint_limit: 0, adaptive: "live" });
  app.saveSettings({ rating: 1750, color: "black", time_control: "10+0", hint_limit: 7 });
  app.loadSettings(form);
  equal(form.elements.hint_limit.value, "0", "an unknown hint setting is ignored");
  form.elements.hint_limit.value = "3";
  form.querySelector('input[name="adaptive"][value="off"]').checked = true;
  localStorage.setItem("human-chess-bot.settings", "{broken");
  app.loadSettings(form);                             // ignored, no error
  localStorage.removeItem("human-chess-bot.settings");
});

test("the f key flips the board, but not while typing in the form", () => {
  const before = app.ui.board.orientation;
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "f", bubbles: true }));
  assert(app.ui.board.orientation !== before, "flipped");
  $("rating").dispatchEvent(new KeyboardEvent("keydown", { key: "f", bubbles: true }));
  assert(app.ui.board.orientation !== before, "not flipped back by a key typed in the form");
  $("flip").click();
});

test("server errors are shown, and the game resyncs", async () => {
  await app.startGame({ rating: 1500, color: "white", seed: 3 });
  await app.playMove("e2e5");           // illegal: the server says so
  assert($("status").textContent.includes("not legal"), $("status").textContent);
  equal(app.ui.game.moves_san.length, 0);
  equal(app.ui.busy, false);
});

run();
