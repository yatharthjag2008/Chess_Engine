// The play page across a real reload (tests/test_site.py): play, reload, and the game is still there.

import { assert, equal, run, test, until } from "./harness.js";

const html = await (await fetch("../index.html")).text();
const doc = new DOMParser().parseFromString(html, "text/html");
doc.querySelectorAll("script").forEach((s) => s.remove());
document.getElementById("page").replaceChildren(...doc.body.children);
const site = await import("../js/site.js");
const PHASE = "reload-test-phase";

if (sessionStorage.getItem(PHASE) !== "after") {
  // First visit: a fresh browser, no saved game. Play 1.e4, let the bot answer, then reload.
  localStorage.clear();
  await site.init("../model/");
  site.ui.delay = 0;
  site.ui.random = () => 0.5;
  await site.newGame({ rating: 1650, color: "white" });
  await site.play("e2e4");
  sessionStorage.setItem(PHASE, "after");
  sessionStorage.setItem("reload-test-fen", site.ui.chess.fen());
  location.reload();
} else {
  await site.init("../model/");
  test("after a reload the game carries on where it was", async () => {
    await until(() => site.ui.chess, "restored");
    equal(site.ui.chess.fen(), sessionStorage.getItem("reload-test-fen"));
    equal(site.ui.chess.history().length, 2);
    equal(site.ui.rating, 1650);
    equal(site.ui.board.mover, "white", "your move");
    assert(document.getElementById("status").textContent.startsWith("Welcome back"));
    assert(!document.getElementById("thinking-panel").hidden, "the bot's thinking on its last move too");
  });
  run();
}
