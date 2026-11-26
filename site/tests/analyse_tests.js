// End-to-end tests of the Analyse page: rating estimates computed in this browser.

import { assert, equal, run, test } from "./harness.js";

const html = await (await fetch("../analyse.html")).text();
const doc = new DOMParser().parseFromString(html, "text/html");
doc.querySelectorAll("script").forEach((s) => s.remove());
document.getElementById("page").replaceChildren(...doc.body.children);
const analyse = await import("../js/analyse.js");
const { ScoringPool, split, workerCount } = await import("../js/pool.js");
await analyse.init("../model/");
const $ = (id) => document.getElementById(id);

// The first game of tests/fixtures/lichess_real.pgn (a real, anonymized Lichess game).
const REAL_GAME = "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/game/fixt0001\"]\n[Date \"2026.10.01\"]\n[Round \"-\"]\n[White \"player01\"]\n[Black \"player02\"]\n[Result \"1/2-1/2\"]\n[UTCDate \"2026.10.01\"]\n[UTCTime \"00:00:25\"]\n[WhiteElo \"1825\"]\n[BlackElo \"1790\"]\n[WhiteRatingDiff \"-1\"]\n[BlackRatingDiff \"+1\"]\n[ECO \"A46\"]\n[Opening \"Indian Defense: Wade-Tartakower Defense\"]\n[TimeControl \"300+0\"]\n[Termination \"Normal\"]\n\n1. d4 { [%clk 0:05:00] } 1... Nf6 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:05:00] } 2... d6 { [%clk 0:04:58] } 3. e3 { [%clk 0:04:59] } 3... g6 { [%clk 0:04:57] } 4. c4 { [%clk 0:04:59] } 4... Bg7 { [%clk 0:04:56] } 5. Be2 { [%clk 0:04:58] } 5... Kf8 { [%clk 0:04:55] } 6. O-O { [%clk 0:04:58] } 6... Kg8 { [%clk 0:04:51] } 7. Nc3 { [%clk 0:04:58] } 7... h5 { [%clk 0:04:50] } 8. Qc2 { [%clk 0:04:56] } 8... Kh7 { [%clk 0:04:48] } 9. a3 { [%clk 0:04:55] } 9... Re8 { [%clk 0:04:46] } 10. e4 { [%clk 0:04:53] } 10... Kg8 { [%clk 0:04:45] } 11. Be3 { [%clk 0:04:46] } 11... e5 { [%clk 0:04:41] } 12. dxe5 { [%clk 0:04:43] } 12... dxe5 { [%clk 0:04:39] } 13. Rfd1 { [%clk 0:04:41] } 13... Qe7 { [%clk 0:04:37] } 14. Nd5 { [%clk 0:04:40] } 14... Nxd5 { [%clk 0:04:35] } 15. cxd5 { [%clk 0:04:39] } 15... Bg4 { [%clk 0:04:29] } 16. Rac1 { [%clk 0:04:36] } 16... Nd7 { [%clk 0:04:23] } 17. Qxc7 { [%clk 0:04:32] } 17... Rac8 { [%clk 0:04:18] } 18. Qxb7 { [%clk 0:04:29] } 18... Rxc1 { [%clk 0:04:16] } 19. Rxc1 { [%clk 0:04:26] } 19... Rb8 { [%clk 0:04:03] } 20. Rc8+ { [%clk 0:04:20] } 20... Rxc8 { [%clk 0:04:00] } 21. Qxc8+ { [%clk 0:04:20] } 21... Kh7 { [%clk 0:03:59] } 22. h3 { [%clk 0:04:19] } 22... Bxf3 { [%clk 0:03:53] } 23. Bxf3 { [%clk 0:04:17] } 23... Nf6 { [%clk 0:03:43] } 24. Qc6 { [%clk 0:04:08] } 24... Bf8 { [%clk 0:03:35] } 25. Bg5 { [%clk 0:04:06] } 25... Kg7 { [%clk 0:03:32] } 26. Bxf6+ { [%clk 0:04:03] } 26... Qxf6 { [%clk 0:03:26] } 27. Qxf6+ { [%clk 0:04:02] } 27... Kxf6 { [%clk 0:03:25] } 28. Kf1 { [%clk 0:04:01] } 28... Bc5 { [%clk 0:03:23] } 29. Be2 { [%clk 0:03:59] } 29... Kg5 { [%clk 0:03:21] } 30. g3 { [%clk 0:03:50] } 30... a5 { [%clk 0:03:19] } 31. Bb5 { [%clk 0:03:47] } 31... f5 { [%clk 0:03:13] } 32. Bd3 { [%clk 0:03:36] } 32... f4 { [%clk 0:03:10] } 33. Ke2 { [%clk 0:03:33] } 33... Kf6 { [%clk 0:03:05] } 34. Kf3 { [%clk 0:03:27] } 34... g5 { [%clk 0:03:03] } 35. gxf4 { [%clk 0:03:22] } 35... gxf4 { [%clk 0:02:58] } 36. Bb5 { [%clk 0:03:16] } 36... Kg5 { [%clk 0:02:55] } 37. Be8 { [%clk 0:03:08] } 37... Kh4 { [%clk 0:02:53] } 38. Bf7 { [%clk 0:03:00] } 38... Kxh3 { [%clk 0:02:42] } 39. Bxh5 { [%clk 0:02:56] } 39... Kh2 { [%clk 0:02:41] } 40. Kg4 { [%clk 0:02:48] } 40... Bxf2 { [%clk 0:02:36] } 41. Kf5 { [%clk 0:02:44] } 41... Bd4 { [%clk 0:02:34] } 42. b4 { [%clk 0:02:43] } 42... axb4 { [%clk 0:02:32] } 43. axb4 { [%clk 0:02:41] } 43... Kg3 { [%clk 0:02:30] } 44. b5 { [%clk 0:02:40] } 44... f3 { [%clk 0:02:29] } 45. b6 { [%clk 0:02:34] } 45... Bxb6 { [%clk 0:02:22] } 46. Kxe5 { [%clk 0:02:32] } 46... f2 { [%clk 0:02:21] } 47. Be2 { [%clk 0:02:30] } 47... Kg2 { [%clk 0:02:18] } 48. d6 { [%clk 0:02:26] } 48... Kg3 { [%clk 0:02:15] } 49. Ke6 { [%clk 0:02:24] } 49... Kf4 { [%clk 0:02:13] } 50. e5 { [%clk 0:02:23] } 50... Bd8 { [%clk 0:02:08] } 51. Bf1 { [%clk 0:02:20] } 51... Ke4 { [%clk 0:02:05] } 52. d7 { [%clk 0:02:17] } 52... Kd4 { [%clk 0:02:03] } 53. Kd6 { [%clk 0:02:14] } 53... Bb6 { [%clk 0:01:49] } 54. e6 { [%clk 0:02:08] } 54... Bc5+ { [%clk 0:01:46] } 55. Kc6 { [%clk 0:02:07] } 55... Be7 { [%clk 0:01:44] } 56. Kc7 { [%clk 0:02:05] } 56... Ke5 { [%clk 0:01:42] } 57. Bh3 { [%clk 0:02:01] } 57... Kf6 { [%clk 0:01:27] } 58. d8=Q { [%clk 0:01:56] } 58... Bxd8+ { [%clk 0:01:25] } 59. Kxd8 { [%clk 0:01:55] } 59... f1=Q { [%clk 0:01:22] } 60. Bxf1 { [%clk 0:01:52] } 60... Kxe6 { [%clk 0:01:20] } 1/2-1/2";

test("the page is ready once the model and the prior are loaded", () => {
  assert(!$("estimate").disabled);
  equal($("estimate-status").textContent, "Paste games and press Estimate.");
});

test("a real game: the estimates match Python's (estimate_rating with the same model, prior and temper)", async () => {
  $("pgn-text").value = REAL_GAME;
  $("player-name").value = "";
  const result = await analyse.estimate();
  equal(result.results.length, 2);
  // Python, moves alone: player01 1753.78 (1421.44-2060.46), player02 2023.71 (1672.91-2351.50), 60 moves each;
  // with the opponent prior: player01 1790.22 (1696.72-1874.11), player02 1835.29 (1740.93-1919.47).
  const alone = [[1753.78, 1421.44, 2060.46], [2023.71, 1672.91, 2351.5]];
  const paired = [[1790.22, 1696.72, 1874.11], [1835.29, 1740.93, 1919.47]];
  result.results.forEach(({ estimate, movesOnly }, i) => {
    for (const [e, expected] of [[movesOnly, alone[i]], [estimate, paired[i]]]) {
      const got = [e.estimate, e.low, e.high];
      got.forEach((v, j) => assert(Math.abs(v - expected[j]) < 2, `${got} vs ${expected}`));
      equal(e.moves, 60);
    }
    assert(estimate.withOpponent && !movesOnly.withOpponent);
  });
  const rows = [...$("estimate-list").querySelectorAll("li")].map((li) => li.textContent);
  assert(rows[0].startsWith("player01 (White, rated 1825 in the game, against 1790): about 1790 (80% range 1700-1870, "
    + "from 60 moves), with the opponent's rating. From the moves alone: about 1750 (80% range 1420-2060, from 60 moves)."), rows[0]);
  assert(rows[1].includes("vs. player01, 2026.10.01, 1/2-1/2"), rows[1]);
  assert($("overall").hidden, "no combined estimate without a player");
  equal($("estimate-status").textContent, "Done: 120 moves scored at every rating.");
});

test("one player across games: their sides only, and one estimate from all of them", async () => {
  const short = `[White "me"]\n[Black "you"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0\n\n`
    + `[White "you"]\n[Black "Me"]\n[Result "0-1"]\n\n1. d4 d5 2. c4 e6 0-1\n`;
  $("pgn-text").value = short;
  $("player-name").value = "me";
  const result = await analyse.estimate();
  equal(result.results.map((r) => r.side.color), ["w", "b"]);
  equal(result.overall.moves, 5);
  assert($("overall").textContent.startsWith("All 2 games of me: about "), $("overall").textContent);
});

test("nothing to estimate says why", async () => {
  $("pgn-text").value = "";
  $("player-name").value = "";
  await analyse.estimate();
  equal($("estimate-error").textContent, "Paste at least one game in PGN.");
  $("pgn-text").value = `[White "a"]\n[Black "b"]\n\n1. e4 e5 *`;
  $("player-name").value = "nobody";
  await analyse.estimate();
  assert($("estimate-error").textContent.startsWith('No game of "nobody" was found'), $("estimate-error").textContent);
  $("pgn-text").value = "1. e4 e5 2. Qxf7 *";
  $("player-name").value = "";
  await analyse.estimate();
  equal($("estimate-error").textContent, "1 game could not be read, and none could.");
});

/** A stand-in for fetch: `routes` maps a URL prefix to [status, body]; every URL asked is recorded. */
function fakeFetch(routes) {
  const asked = [];
  const impl = async (url, options = {}) => {
    asked.push({ url, accept: options.headers && options.headers.Accept });
    const key = Object.keys(routes).find((prefix) => url.startsWith(prefix));
    if (!key) throw new TypeError("Failed to fetch");
    const [status, body] = routes[key];
    return { status, ok: status >= 200 && status < 300, text: async () => body,
             json: async () => (typeof body === "string" ? JSON.parse(body) : body) };
  };
  return { impl, asked };
}

const LICHESS_PGN = `[Event "Rated blitz game"]\n[White "SomePlayer"]\n[Black "other"]\n[WhiteElo "1600"]\n[BlackElo "1580"]\n`
  + `[Result "1-0"]\n[TimeControl "180+0"]\n\n1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:57] } 1-0\n\n`
  + `[Event "Rated blitz game"]\n[White "other"]\n[Black "SomePlayer"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n`;

test("Lichess: recent games by username fill the box, with the player's name", async () => {
  const fake = fakeFetch({ "https://lichess.org/api/games/user/someplayer": [200, LICHESS_PGN] });
  analyse.page.fetchImpl = fake.impl;
  $("source").value = "lichess";
  $("username").value = " someplayer ";
  equal(await analyse.fetchRecent(), 2);
  assert(fake.asked[0].url.startsWith("https://lichess.org/api/games/user/someplayer?max=10&clocks=true"), fake.asked[0].url);
  equal(fake.asked[0].accept, "application/x-chess-pgn");
  equal($("pgn-text").value, LICHESS_PGN.trim());
  equal($("player-name").value, "SomePlayer", "as Lichess spells it");
  equal($("estimate-status").textContent, "Fetched 2 games of SomePlayer from Lichess. Press Estimate.");
  const result = await analyse.estimate();
  equal(result.results.length, 2);
  assert(!$("overall").hidden, "one estimate from both games");
});

test("Chess.com: the newest archives first, standard chess only, never another host", async () => {
  const game = (white, black, n) => `[White "${white}"]\n[Black "${black}"]\n[Result "*"]\n[TimeControl "600"]\n\n1. e4 e5 ${n} *`;
  const fake = fakeFetch({
    "https://api.chess.com/pub/player/someplayer/games/archives": [200, { archives: [
      "https://api.chess.com/pub/player/someplayer/games/2026/10",
      "https://elsewhere.example/pub/player/someplayer/games/2026/11",
      "https://api.chess.com/pub/player/someplayer/games/2026/11"] }],
    "https://api.chess.com/pub/player/someplayer/games/2026/11": [200, { games: [
      { rules: "chess", pgn: game("SomePlayer", "x", "2. Nf3") },
      { rules: "chess960", pgn: game("SomePlayer", "y", "2. Nc3") }] }],
    "https://api.chess.com/pub/player/someplayer/games/2026/10": [200, { games: [
      { rules: "chess", pgn: game("z", "SomePlayer", "2. Bc4") }] }],
  });
  analyse.page.fetchImpl = fake.impl;
  $("source").value = "chess.com";
  $("username").value = "SomePlayer";
  equal(await analyse.fetchRecent(), 2);
  equal(fake.asked.map((a) => a.url.split("/").slice(-2).join("/")), ["games/archives", "2026/11", "2026/10"]);
  assert(!fake.asked.some((a) => a.url.includes("elsewhere")), "the other host is not asked");
  assert($("pgn-text").value.includes("2. Nf3") && !$("pgn-text").value.includes("2. Nc3"), "no Chess960");
  assert($("pgn-text").value.indexOf("2. Nf3") < $("pgn-text").value.indexOf("2. Bc4"), "newest first");
});

test("fetching: bad names, unknown players and unreachable sites say why", async () => {
  analyse.page.fetchImpl = fakeFetch({ "https://lichess.org/api/games/user/ghost": [404, "<!DOCTYPE html>"],
                                       "https://lichess.org/api/user/ghost": [404, { error: "Not found" }] }).impl;
  $("source").value = "lichess";
  const cases = [
    ["a", '"a" is not a valid Lichess username.'],
    ["bad name!", '"bad name!" is not a valid Lichess username.'],
    ["ghost", "Lichess has no player of that name."],
    ["offline", "Lichess could not be reached (offline, or the browser blocked the request). You can paste the games\u2019 PGN instead."],
  ];
  for (const [name, message] of cases) {
    $("username").value = name;
    equal(await analyse.fetchRecent(), 0, name);
    equal($("estimate-error").textContent, message);
  }
  analyse.page.fetchImpl = fakeFetch({ "https://lichess.org/api/games/user/busy": [429, ""] }).impl;
  $("username").value = "busy";
  await analyse.fetchRecent();
  equal($("estimate-error").textContent, "Lichess asks us to slow down: try again in a minute.");
  analyse.page.fetchImpl = fakeFetch({ "https://lichess.org/api/games/user/quiet": [200, ""] }).impl;
  $("username").value = "quiet";
  await analyse.fetchRecent();
  equal($("estimate-error").textContent, "No standard games of quiet were found on Lichess.");
  assert(!$("fetch").disabled, "the button works again");
});

test("a game handed over by the play page: your moves are estimated", async () => {
  localStorage.setItem("human-chess-site.analyse", JSON.stringify(
    `[White "You"]\n[Black "Bot 1500"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *\n`));
  const result = await analyse.openHandedGame();
  equal($("player-name").value, "You");
  equal(result.results.map((r) => [r.side.name, r.estimate.moves]), [["You", 3]]);
  equal(localStorage.getItem("human-chess-site.analyse"), null, "used once");
  equal(await analyse.openHandedGame(), null, "nothing handed over");
});

test("workers: the moves are shared out, and the results equal the page's own, move for move", async () => {
  equal([workerCount(8), workerCount(2), workerCount(1), workerCount(64)], [7, 1, 1, 8]);
  equal(split([1, 2, 3, 4, 5], 2), [[1, 2, 3], [4, 5]]);
  equal(split([1, 2], 4), [[1], [2]], "no empty chunks");
  const text = `[White "a"]\n[Black "b"]\n[WhiteElo "1400"]\n[BlackElo "1500"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 *\n`;
  const pool = analyse.page.pool;
  assert(pool, "the page started its workers");
  $("player-name").value = "";
  const withWorkers = await analyse.estimateText(text);
  analyse.page.pool = null;
  try {
    const onPage = await analyse.estimateText(text);
    equal(withWorkers.results.map((r) => r.estimate), onPage.results.map((r) => r.estimate), "identical estimates");
  } finally {
    analyse.page.pool = pool;
  }
  const broken = new ScoringPool({ modelBase: "../no-such-model/", size: 1 });
  let failed = false;
  try { await broken.start(); } catch { failed = true; }
  assert(failed, "a pool that cannot start says so (the page then scores the moves itself)");
});

test("Lichess: a refused request is not mistaken for an unknown player", async () => {
  $("source").value = "lichess";
  // Lichess answers an unknown player and a refused request with the same HTML 404: its user endpoint decides.
  analyse.page.fetchImpl = fakeFetch({ "https://lichess.org/api/games/user/thibault": [404, "<!DOCTYPE html>"],
                                       "https://lichess.org/api/user/thibault": [200, { username: "thibault" }] }).impl;
  $("username").value = "thibault";
  equal(await analyse.fetchRecent(), 0);
  assert($("estimate-error").textContent.startsWith("Lichess has thibault, but refused to send their games"),
         $("estimate-error").textContent);
  analyse.page.fetchImpl = fakeFetch({ "https://lichess.org/api/games/user/blocked": [404, "<!DOCTYPE html>"],
                                       "https://lichess.org/api/user/blocked": [404, "<!DOCTYPE html>"] }).impl;
  $("username").value = "blocked";
  await analyse.fetchRecent();
  assert($("estimate-error").textContent.startsWith("Lichess refused to send the games"), $("estimate-error").textContent);
});

test("an opponent whose rating does not fit the moves is flagged, and left out of the combined estimate", async () => {
  const game = (white, black, whiteElo, blackElo, moves) => `[White "${white}"]\n[Black "${black}"]\n`
    + `[WhiteElo "${whiteElo}"]\n[BlackElo "${blackElo}"]\n[Result "*"]\n\n${moves} *\n`;
  const text = [game("me", "a", 1500, 1480, "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"),
                game("b", "me", 610, 1500, "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O"),
                game("me", "c", 1500, 1530, "1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Nb6")].join("\n");
  $("pgn-text").value = text;
  $("player-name").value = "me";
  const result = await analyse.estimate();
  const fits = result.results.map((r) => r.fits);
  equal(fits[1], false, "an opponent rated 610 against these moves");
  const flagged = [...$("estimate-list").querySelectorAll(".misfit")];
  equal(flagged.length, fits.filter((f) => !f).length);
  assert(flagged[0].textContent.startsWith("These disagree: the opponent's rating (610) is far from what the moves alone suggest. Either"),
         flagged[0].textContent);
  const left = fits.filter((f) => !f).length;
  assert($("overall").textContent.includes(`(${left} game${left === 1 ? "" : "s"} left out where the opponent's rating did not fit the moves)`),
         $("overall").textContent);
  equal(result.overall.opponentsUsed, 3 - left);
});

run();
