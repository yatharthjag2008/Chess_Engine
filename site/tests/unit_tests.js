// Unit tests of the website's modules, in a real browser (tests/test_site.py).

import { Chess } from "../vendor/chess.js";
import { assert, equal, run, test } from "./harness.js";
import { RECIPES, Sounds, soundFor } from "../js/sounds.js";
import { stored } from "../js/storage.js";
import { loadOpenings, openingOf, positionKey } from "../js/openings.js";
import { PRESETS, parsePosition } from "../js/position.js";
import { Clock, TIME_CONTROLS, baseSeconds, formatClock, thinkTime } from "../js/clock.js";
import { START, canMate, captured, countPieces, describeCaptures, materialOf } from "../js/material.js";
import { describeEstimate, estimateFromBucketLL, interp, loadRatingData, moveBucketLL, opponentFits, posterior,
         summarize } from "../js/rating.js";
import { bucketOf, loadModel } from "../js/engine.js";
import { clockOf, parseTimeControl, playerGames, readGame, splitPgn } from "../js/games.js";
import { classify, describeSide, evalGraph, gameAccuracy, keyMoments, moveAccuracy, reviewMoves, summary,
         winPercent } from "../js/review.js";
import { MATE_CP, finishedEval } from "../js/stockfish.js";
import { coachGame, explainLevel, explainMistake, hangingPieces, newlyHanging } from "../js/coach.js";
import { HintBudget, HintEngine, detailFor, makeHint, moveThemes, threatenedPiece } from "../js/hints.js";
import { ADAPTIVE, LiveAdjuster, botRatingFor, expectedScore, readProfile, recordGame, scoreFor,
         updateRating } from "../js/adaptive.js";

// ------------------------------------------------------------------ sounds

/** The sound of each move in a game given in SAN. */
function soundsOf(...sans) {
  const chess = new Chess();
  return sans.map((san) => {
    const move = chess.move(san);
    return soundFor(move, { check: chess.inCheck() });
  });
}

test("each kind of move has its sound", () => {
  equal(soundsOf("e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+"), ["move", "move", "capture", "capture", "move", "check"]);
  equal(soundsOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"), ["move", "move", "move", "move", "move", "move", "castle"]);
  const chess = new Chess("8/1P6/8/8/8/8/8/k6K w - - 0 1");
  equal(soundFor(chess.move("b8=Q")), "promote");
});

test("sounds play through Web Audio, and quietly do nothing without it", () => {
  const sounds = new Sounds();
  for (const name of Object.keys(RECIPES)) assert(sounds.play(name), name);
  equal(sounds.log, Object.keys(RECIPES));
  assert(!sounds.play("no-such-sound"), "unknown names are ignored");
  const silent = new Sounds({ makeContext: () => null });
  assert(silent.play("move"), "no audio: still counted as played");
  const broken = new Sounds({ makeContext: () => { throw new Error("no audio here"); } });
  assert(broken.play("capture"), "a failing audio context does not throw");
});

test("unlock resumes a suspended audio context (the first click or key press)", () => {
  let resumed = 0;
  const fake = { state: "suspended", resume: () => { resumed += 1; fake.state = "running"; return Promise.resolve(); } };
  const sounds = new Sounds({ makeContext: () => fake });
  assert(sounds.unlock());
  equal(resumed, 1);
  sounds.unlock();
  equal(resumed, 1, "already running");
  assert(!new Sounds({ makeContext: () => null }).unlock(), "no audio at all");
});

test("muted: nothing plays", () => {
  const sounds = new Sounds({ enabled: false });
  assert(!sounds.play("move"));
  equal(sounds.log, []);
});

// ------------------------------------------------------------------ material

/** The FEN after a game given in SAN. */
function fenAfter(...sans) {
  const chess = new Chess();
  for (const san of sans) chess.move(san);
  return chess.fen();
}

test("piece counts and material", () => {
  const counts = countPieces(START);
  equal(counts.w, { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 });
  equal(counts.b, counts.w);
  equal(materialOf(counts.w), 39);
});

test("captured pieces and the material balance", () => {
  equal(captured(START, START), { w: [], b: [], diff: 0 });
  equal(captured(START, fenAfter("e4", "d5", "exd5", "Qxd5")), { w: ["p"], b: ["p"], diff: 0 });
  const scholar = fenAfter("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#");
  equal(captured(START, scholar), { w: ["p"], b: [], diff: 1 });
  const trade = fenAfter("e4", "d5", "exd5", "Qxd5", "Nc3", "Qxa2", "Rxa2");
  equal(captured(START, trade), { w: ["p", "q"], b: ["p", "p"], diff: 8 });
});

test("a promoted pawn is not a captured one", () => {
  const promoted = "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNQ w Qkq - 0 1";   // a pawn fewer, a rook fewer, a queen more
  equal(captured(START, promoted), { w: [], b: ["r"], diff: 3 }, "a pawn became a queen, a rook was taken");
  const twoQueens = "rnbqkbnr/pppppppp/8/8/8/8/2PPPPPP/RNBQKBNR w KQkq - 0 1".replace("RNBQKBNR w", "RNBQKQQR w");
  equal(captured(START, twoQueens), { w: [], b: ["n", "b"], diff: 10 },
        "two pawns became queens; the knight and bishop they replace were taken");
});

test("who could still checkmate (for a flag fall)", () => {
  assert(canMate(START, "w") && canMate(START, "b"));
  assert(!canMate("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "w"), "a lone king");
  assert(!canMate("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", "w"), "king and knight against a lone king");
  assert(!canMate("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", "w"), "king and bishop against a lone king");
  assert(canMate("4k3/4p3/8/8/8/8/8/3NK3 w - - 0 1", "w"), "with a black pawn about, a knight can mate");
  assert(canMate("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "w"), "a pawn can queen");
  assert(canMate("4k3/8/8/8/8/8/8/2NNK3 w - - 0 1", "w"), "two knights: not counted as impossible");
});

test("captures in words, for screen readers", () => {
  equal(describeCaptures([]), "Nothing captured.");
  equal(describeCaptures(["p"], 1), "Captured a pawn; 1 ahead.");
  equal(describeCaptures(["p", "p", "n"], 5), "Captured 2 pawns and a knight; 5 ahead.");
  equal(describeCaptures(["p", "b", "r", "q"]), "Captured a pawn, a bishop, a rook and a queen.");
});

// ------------------------------------------------------------------ storage

/** An in-memory stand-in for localStorage. */
function memoryStorage() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, String(v)),
    removeItem: (k) => data.delete(k),
    data,
  };
}

test("stored values round-trip as JSON", () => {
  const storage = memoryStorage();
  const game = stored("game", storage);
  equal(game.load(), null);
  equal(game.load({ none: true }), { none: true }, "the fallback when nothing is saved");
  assert(game.save({ moves: ["e2e4"], rating: 1500 }));
  equal(game.load(), { moves: ["e2e4"], rating: 1500 });
  game.clear();
  equal(game.load(), null);
  storage.setItem("game", "{not json");
  equal(game.load("fallback"), "fallback", "unreadable: as if nothing were saved");
});

test("storage that throws (private windows, a full quota) never breaks the page", () => {
  const angry = { getItem() { throw new Error("denied"); }, setItem() { throw new Error("full"); },
                  removeItem() { throw new Error("denied"); } };
  const value = stored("x", angry);
  equal(value.load(7), 7);
  assert(!value.save(1));
  value.clear();
  const none = stored("x", null);
  equal(none.load(3), 3);
  assert(!none.save(1));
});

// ------------------------------------------------------------------ clocks

/** A clock on a fake time line: `t.now` is moved by hand. */
function fakeClock(base) {
  const t = { now: 1000 };
  return { clock: new Clock(base, { now: () => t.now }), t };
}

test("the time controls: no clock, 1, 3, 5 and 10 minutes", () => {
  equal(TIME_CONTROLS.map((c) => c.id), ["none", "1", "3", "5", "10"]);
  equal([baseSeconds("none"), baseSeconds("1"), baseSeconds(3), baseSeconds("5"), baseSeconds("10")],
        [null, 60, 180, 300, 600]);
  equal(baseSeconds("7"), null);
});

test("a clock charges only the side whose time runs", () => {
  const { clock, t } = fakeClock(60);
  equal([clock.left("w"), clock.left("b")], [60000, 60000]);
  clock.start("b");
  t.now += 5000;
  equal([clock.left("w"), clock.left("b")], [60000, 55000]);
  clock.start("w");
  t.now += 2500;
  equal([clock.left("w"), clock.left("b")], [57500, 55000]);
  clock.stop();
  t.now += 99999;
  equal([clock.left("w"), clock.left("b")], [57500, 55000], "stopped: nobody's time runs");
  equal(clock.flagged(), null);
});

test("running out of time, and never below zero", () => {
  const { clock, t } = fakeClock(1);
  clock.start("w");
  t.now += 1500;
  equal(clock.left("w"), 0);
  equal(clock.flagged(), "w");
});

test("a saved clock carries on, the running side charged for the time since", () => {
  const { clock, t } = fakeClock(180);
  clock.start("b");
  t.now += 10000;
  const saved = JSON.parse(JSON.stringify(clock));
  equal(saved, { base: 180, remaining: { w: 180000, b: 170000 }, running: "b", at: t.now });
  t.now += 4000;                                   // the page was closed for 4 seconds
  const back = Clock.fromJSON(saved, { now: () => t.now });
  equal([back.left("w"), back.left("b")], [180000, 166000]);
  equal(Clock.fromJSON({ ...saved, running: null }).left("b"), 170000);
});

test("clock faces: minutes and seconds, tenths under ten seconds", () => {
  equal(formatClock(600000), "10:00");
  equal(formatClock(179001), "3:00", "a part-second counts as a whole one");
  equal(formatClock(61000), "1:01");
  equal(formatClock(10000), "0:10");
  equal(formatClock(9400), "9.4");
  equal(formatClock(50), "0.0");
  equal(formatClock(-5), "0.0");
});

test("the bot's thinking time: quick in the opening, a share of its time later, hurried when short", () => {
  const mid = () => 0.5;                         // spread factor 1
  equal(thinkTime(180000, 2, mid), 2250, "opening: 1/80 of three minutes");
  equal(thinkTime(180000, 30, mid), 6000, "later: 1/30");
  equal(thinkTime(600000, 40, () => 0.99), 30000, "never more than 30 seconds");
  equal(thinkTime(2000, 40, mid), 150, "two seconds left: a quick move");
  assert(thinkTime(5000, 40, mid) <= 1000, "at most a fifth of what is left");
  assert(thinkTime(180000, 30, () => 0) < thinkTime(180000, 30, () => 0.99), "the spread");
});

// ------------------------------------------------------------------ openings

/** The positions of a game given in SAN, the start included. */
function fensOf(...sans) {
  const chess = new Chess();
  const fens = [chess.fen()];
  for (const san of sans) {
    chess.move(san);
    fens.push(chess.fen());
  }
  return fens;
}

test("the opening of a game is its latest named position", () => {
  const table = {
    [positionKey(fensOf("e4").pop())]: ["B00", "King's Pawn Game"],
    [positionKey(fensOf("e4", "c5").pop())]: ["B20", "Sicilian Defense"],
  };
  equal(openingOf(table, fensOf()), null, "the start position has no name");
  equal(openingOf(table, fensOf("e4")), { eco: "B00", name: "King's Pawn Game", ply: 1 });
  equal(openingOf(table, fensOf("e4", "c5", "Nf3", "d6", "a3")), { eco: "B20", name: "Sicilian Defense", ply: 2 },
        "out of the book: the name stays");
  equal(openingOf(null, fensOf("e4")), null);
  equal(positionKey("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq");
});

test("the real table: names, transpositions, fetched once", async () => {
  const table = await loadOpenings("../data/");
  assert(Object.keys(table).length > 3000);
  assert((await loadOpenings("../data/")) === table, "cached");
  equal(openingOf(table, fensOf("e4", "e5", "Nf3", "Nc6", "Bb5")).name, "Ruy Lopez");
  equal(openingOf(table, fensOf("d4", "Nf6", "c4", "e6", "Nc3", "Bb4")).name, "Nimzo-Indian Defense");
  const transposed = openingOf(table, fensOf("c4", "e6", "d4", "Nf6", "Nc3", "Bb4"));
  equal(transposed.name, "Nimzo-Indian Defense", "another move order, the same position");
});

// ------------------------------------------------------------------ from a position

test("a pasted FEN becomes the position to play from", () => {
  const { chess, error } = parsePosition("  8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1 \n");
  equal(error, undefined);
  equal(chess.fen(), "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1");
  equal(chess.history(), []);
});

test("a pasted PGN keeps its moves", () => {
  const { chess } = parsePosition('[Event "Casual"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 *');
  equal(chess.history(), ["e4", "e5", "Nf3", "Nc6", "Bb5"]);
  equal(chess.turn(), "b");
  equal(parsePosition("1. d4 d5 2. c4").chess.history().length, 3, "movetext without headers");
});

test("every practice position can be played from", () => {
  assert(PRESETS.length >= 8);
  for (const preset of PRESETS) {
    const { chess, error } = parsePosition(preset.fen);
    equal(error, undefined, preset.name);
    equal(chess.fen(), preset.fen, preset.name);
  }
});

test("what cannot be played from says why", () => {
  assert(parsePosition("").error.startsWith("Paste a position"));
  assert(parsePosition("rnbqkbnr/pppppppp/8/8 w KQkq - 0 1").error.startsWith("That FEN is not valid"),
         parsePosition("rnbqkbnr/pppppppp/8/8 w KQkq - 0 1").error);
  assert(parsePosition("1. e4 e5 2. Ke3 Ke6 3. Qxh8").error.startsWith("That PGN could not be read"));
  equal(parsePosition("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").error,
        "The game is already over in that position (mate, stalemate or a draw).");
});

// ------------------------------------------------------------------ adaptive difficulty

const near = (a, b, tol = 1e-9) => Math.abs(a - b) < tol;

test("adaptive: the Elo update matches the Python numbers", () => {
  assert(near(expectedScore(1500, 1500), 0.5));
  assert(near(updateRating(1200, 1400, 1, 0), 1260.7797541318366), "a win while new: K 80");
  equal(updateRating(1500, 1500, 0.5, 9), 1500, "a draw at your level changes nothing");
  assert(near(updateRating(1500, 1300, 0, 7), 1469.6101229340816), "a loss later: K 40");
  equal(updateRating(2390, 2400, 1, 9), 2400, "kept in range");
  equal([botRatingFor(1437), botRatingFor(3000), botRatingFor(100)], [1450, 2400, 800]);
  let threw = false;
  try { updateRating(1500, 1500, 2, 0); } catch { threw = true; }
  assert(threw, "a score above 1 is refused");
  equal([scoreFor("w", "w"), scoreFor("b", "w"), scoreFor(null, "b")], [1, 0, 0.5]);
});

test("adaptive: the profile records games and survives broken storage", () => {
  equal(readProfile(null), { rating: ADAPTIVE.startRating, games: 0, history: [] });
  equal(readProfile({ rating: "nonsense" }).rating, ADAPTIVE.startRating);
  equal(readProfile({ rating: 5000, games: 3 }).rating, 2400);
  let profile = readProfile(null);
  profile = recordGame(profile, 1200, 1, "between", { date: new Date("2026-11-26T12:00:00Z") });
  equal(profile.games, 1);
  equal(profile.history[0], { date: "2026-11-26", botRating: 1200, score: 1, ratingBefore: 1200, ratingAfter: 1240,
                              mode: "between" });
  for (let i = 0; i < 60; i++) profile = recordGame(profile, 1200, 0.5, "between");
  equal(profile.history.length, 50, "the latest 50 games are kept");
});

test("adaptive: live changes follow the bot's chances, with a cap", () => {
  const losing = new LiveAdjuster(1400);
  const changes = [];
  for (let i = 0; i < 12; i++) changes.push(losing.observe([0.2, 0.1, 0.7]));
  equal(changes, [null, null, null, null, null, null, 1450, null, null, null, 1500, null], "the Python sequence");
  equal(losing.meanRating, 1425);
  const winningAtTheBottom = new LiveAdjuster(800);
  for (let i = 0; i < 12; i++) equal(winningAtTheBottom.observe([0.9, 0.05, 0.05]), null, "already as weak as it gets");
  const capped = new LiveAdjuster(1400);
  for (let i = 0; i < 60; i++) capped.observe([0, 0, 1]);
  equal(capped.rating, 1600, "never more than 200 from the start");
  equal(new LiveAdjuster(1400).observe(null), null, "no value head: no change");
  const copy = LiveAdjuster.fromJSON(JSON.parse(JSON.stringify(losing)));
  equal([copy.rating, copy.meanRating, copy.changes.length], [1500, 1425, 2], "saved and restored");
});

// ------------------------------------------------------------------ rating estimation

test("rating: interpolation, posterior and summary behave like numpy's", () => {
  equal([interp(5, [0, 10], [0, 1]), interp(-3, [0, 10], [2, 4]), interp(99, [0, 10], [2, 4])], [0.5, 2, 4]);
  const p = posterior([0, Math.log(3)], [0, 0]);
  assert(near(p[0], 0.25) && near(p[1], 0.75));
  const s = summarize([1000, 1100, 1200], [0.25, 0.5, 0.25], 0.5, 3);
  equal(s.estimate, 1100);
  equal(describeEstimate({ estimate: 1434, low: 1251, high: 1618, interval: 0.8, moves: 31 }),
        "about 1430 (80% range 1250-1620, from 31 moves)");
});

test("rating: the ends of the scale, and whether an opponent's rating fits the moves", () => {
  const grid = [600, 700, 800, 900, 1000];
  const top = summarize(grid, [0, 0, 0, 0.05, 0.95], 0.8, 12);
  assert(top.atTop && !top.atBottom);
  equal(describeEstimate(top), "1000 or more (the estimate's scale ends at 1000; 80% range from 910, from 12 moves)");
  const bottom = summarize(grid, [0.95, 0.05, 0, 0, 0], 0.8, 1);
  equal(describeEstimate(bottom), "600 or less (the estimate's scale starts at 600; 80% range up to 600, from 1 move)");
  const middle = summarize(grid, [0, 0, 1, 0, 0], 0.8, 3);
  assert(!middle.atTop && !middle.atBottom && describeEstimate(middle).startsWith("about 800"));
  const alone = { wide: { low: 1280, high: 2190 } };
  equal([opponentFits(alone, 1500), opponentFits(alone, 578), opponentFits(alone, 2500)], [true, false, false]);
  const strongest = { wide: { low: 1900, high: 2400 } };
  const centers = [900, 1100, 1300, 1500, 1700, 1900, 2100, 2300];
  equal([opponentFits(strongest, 3120), opponentFits(strongest, 3120, centers), opponentFits(alone, 578, centers)],
        [false, true, false], "beyond the model's top bucket the moves cannot tell: 3120 counts as 2300");
});

test("rating: the estimate matches Python's for the same move likelihoods", async () => {
  const data = await loadRatingData("../data/");
  equal(data.log_prior.length, data.points);
  const est = estimateFromBucketLL(data.test.bucket_ll, data);
  for (const key of ["estimate", "median", "low", "high"]) assert(near(est[key], data.test[key], 1e-6), `${key}: ${est[key]} vs ${data.test[key]}`);
  equal(est.moves, data.test.moves);
});

test("rating: with the opponents' ratings, the estimate matches Python's opponent prior", async () => {
  const data = await loadRatingData("../data/");
  const v = data.opponent.test_vector;
  const est = estimateFromBucketLL(v.bucket_ll, data, { oppRatings: v.opp_ratings });
  assert(est.withOpponent);
  for (const key of ["estimate", "low", "high"]) assert(near(est[key], v[key], 1e-6), `${key}: ${est[key]} vs ${v[key]}`);
  equal(estimateFromBucketLL(v.bucket_ll, data, { oppRatings: [] }).withOpponent, false, "no opponents: the data prior");
  equal(estimateFromBucketLL(v.bucket_ll, { ...data, opponent: undefined }, { oppRatings: [1500] }).withOpponent, false,
        "no fitted opponent prior: the data prior");
});

test("rating: a move's likelihood at each rating is PyTorch's probability", async () => {
  const model = await loadModel("../model/");
  for (const t of model.meta.tests.slice(0, 2)) {
    const [uci, prob] = Object.entries(t.probs).sort((a, b) => b[1] - a[1])[0];
    const ll = moveBucketLL(model, { fen: t.fen, uci, oppRating: t.opp_rating });
    equal(ll.length, model.meta.bucket_edges.length);
    const b = bucketOf(t.rating, model.meta.bucket_edges);
    assert(Math.abs(Math.exp(ll[b]) - prob) < 1e-3, `${Math.exp(ll[b])} vs ${prob}`);
  }
  let threw = false;
  try { moveBucketLL(model, { fen: model.meta.tests[0].fen, uci: "a1a8" }); } catch { threw = true; }
  assert(threw, "an illegal move is refused");
});

// ------------------------------------------------------------------ games from PGN

const TWO_GAMES = `[Event "Rated blitz game"]
[White "Ann"]
[Black "Bob"]
[WhiteElo "1520"]
[BlackElo "1488"]
[Result "1-0"]
[TimeControl "180+2"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:58.4] } 2... Nc6 { [%clk 0:02:55] } 3. Bb5 { [%clk 0:02:50] } 1-0

[Event "Casual"]
[White "Bob"]
[Black "ann"]
[Result "0-1"]

1. d4 d5 2. c4 0-1
`;

test("games: PGN text is split into games, and time controls and clocks are read", () => {
  equal(splitPgn(TWO_GAMES).length, 2);
  equal(splitPgn("1. e4 e5 *").length, 1, "moves without headers");
  equal(splitPgn("  \n "), []);
  equal([parseTimeControl("180+2"), parseTimeControl("600"), parseTimeControl("-"), parseTimeControl("1/86400")],
        [{ base: 180, increment: 2 }, { base: 600, increment: 0 }, null, null]);
  equal([clockOf("[%clk 0:02:58.4]"), clockOf("[%clk 1:00:00]"), clockOf("a comment")], [178.4, 3600, null]);
});

test("games: each player's moves, with ratings and clocks before each move, as in Python", () => {
  const { games, unreadable } = playerGames(TWO_GAMES);
  equal(unreadable, 0);
  equal(games.map((g) => [g.name, g.color, g.rating, g.oppRating, g.base, g.increment, g.moves.map((m) => m.uci)]), [
    ["Ann", "w", 1520, 1488, 180, 2, ["e2e4", "g1f3", "f1b5"]],
    ["Bob", "b", 1488, 1520, 180, 2, ["e7e5", "b8c6"]],
    ["Bob", "w", null, null, 180, 0, ["d2d4", "c2c4"]],
    ["ann", "b", null, null, 180, 0, ["d7d5"]],
  ]);
  equal(games[0].moves.map((m) => [m.clock, m.oppClock]), [[180, 180], [180, 180], [178.4, 175]]);
  equal(games[1].moves.map((m) => [m.clock, m.oppClock]), [[180, 180], [180, 178.4]]);
  equal(games[2].moves[0].clock, null, "no clocks in the PGN");
  equal(games[0].moves[1].fen, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "the position before");
  equal(playerGames(TWO_GAMES, { player: "ANN" }).games.map((g) => g.color), ["w", "b"], "one player, any case");
});

test("games: an unreadable game is counted, and the others are still read", () => {
  const { games, unreadable } = playerGames(`[White "A"]\n[Black "B"]\n\n1. e4 e5 2. Ke3 Ke6 3. Qxh8 *\n\n${TWO_GAMES}`);
  equal([unreadable, games.length], [1, 4]);
  assert(readGame("1. e4 e5 2. Qxf7").error.startsWith("could not be read"));
});

// ------------------------------------------------------------------ game review

test("review: win %, move and game accuracy are Lichess's (the Python numbers)", () => {
  const wins = [0, 100, -250, 1500, 9997, -9990].map(winPercent);
  const expected = [50, 59.10258971916128, 28.485195256375974, 97.54474363414323, 100, 0];
  wins.forEach((w, i) => assert(near(w, expected[i], 1e-9), `${w} vs ${expected[i]}`));
  assert(near(moveAccuracy(60, 40), 41.016818741417126, 1e-9));
  equal(moveAccuracy(50, 55), 100, "gaining is perfect");
  assert(near(moveAccuracy(90, 10), 1.000170640274321, 1e-9));
  assert(near(gameAccuracy([100, 80, 30, 95], [50, 52, 55, 40, 20, 30, 35, 34, 33], [0, 2, 4, 6]), 55.38879709187046, 1e-9));
  equal([classify(1, false), classify(5, false), classify(12, false), classify(25, false), classify(31, false),
         classify(50, true)], ["best", "good", "inaccuracy", "mistake", "blunder", "best"]);
});

/** Python's ScriptedAnalyzer: { "placement side": [bestUci, bestCp, { uci: cp }] }, else the first legal move at 0. */
function scripted(script) {
  return {
    async analyse(fen) {
      const chess = new Chess(fen);
      const done = finishedEval(chess);
      if (done !== null) return { fen, lines: [], bestCp: done };
      const key = fen.split(" ").slice(0, 2).join(" ");
      const legal = chess.moves({ verbose: true });
      const first = legal.map((m) => m.lan).sort()[0];
      const [bestUci, bestCp, others] = script[key] || [first, 0, {}];
      const san = (uci) => legal.find((m) => m.lan === uci).san;
      const lines = [{ uci: bestUci, san: san(bestUci), cp: bestCp },
                     ...Object.entries(others).map(([uci, cp]) => ({ uci, san: san(uci), cp }))];
      lines.sort((a, b) => b.cp - a.cp);
      return { fen, lines, bestCp: lines[0].cp };
    },
  };
}

const keyOf = (...sans) => {
  const chess = new Chess();
  for (const san of sans) chess.move(san);
  return chess.fen().split(" ").slice(0, 2).join(" ");
};

test("review: a scripted game is judged exactly as the Python review judges it", async () => {
  const analyzer = scripted({
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4")]: ["g7g6", 0, { g8f6: -900 }],
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6")]: ["h5f7", 900, {}],
  });
  const chess = new Chess();
  for (const san of ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]) chess.move(san);
  const progress = [];
  const review = await reviewMoves(new Chess().fen(), chess.history({ verbose: true }).map((m) => m.lan), analyzer,
                                   { onProgress: (done, total) => progress.push(`${done}/${total}`) });
  equal(progress.length, 8);
  equal(progress[7], "8/8");
  equal(review.moves.map((m) => m.classification), ["best", "best", "best", "best", "best", "blunder", "best"]);
  const blunder = review.moves[5];
  equal([blunder.moveNumber, blunder.cpLoss, blunder.evalBefore, blunder.evalAfter, blunder.bestSan],
        ["3...", 900, 0, 900, "g6"]);
  assert(near(blunder.winLost, 46.490164802806575, 1e-9) && near(blunder.accuracy, 11.45897404122671, 1e-9));
  const s = summary(review);
  equal([s.white.accuracy, s.white.averageCpLoss, s.white.counts.best], [100, 0, 4]);
  assert(near(s.black.accuracy, 49.22684928745974, 1e-9), String(s.black.accuracy));
  equal([s.black.averageCpLoss, s.black.counts.blunder], [300, 1]);
  equal(describeSide(s.black), "Black: accuracy 49%, average loss 300 cp, 1 blunder");
  equal(describeSide(s.white), "White: accuracy 100%, average loss 0 cp, no errors");
  equal(keyMoments(review).map((k) => k.text), ["3... Nf6?? dropped Black's chances from 50% to 4%. Better was g6."]);
  const graph = evalGraph(review);
  equal(graph.length, 8);
  equal([graph[6].eval, graph[7].ply], [900, 7]);
  assert(near(graph[6].whiteWin, 96.49016480280658, 1e-9));
  let threw = false;
  try { await reviewMoves(new Chess().fen(), ["e2e5"], analyzer); } catch { threw = true; }
  assert(threw, "an illegal move is refused");
  equal(MATE_CP, 10000);
});

// ------------------------------------------------------------------ the coach

test("coach: hanging pieces, as features.py finds them", () => {
  equal(hangingPieces(new Chess("4k3/8/8/3p4/4N3/8/8/4K2R w - - 0 1"), "w"), [{ square: "e4", type: "n" }],
        "attacked by a cheaper piece");
  equal(newlyHanging("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3g5"),
        [{ square: "g5", type: "n" }]);
  equal(newlyHanging("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1c4"), []);
});

test("coach: mistakes explained with the Python sentences, rule by rule", () => {
  const move = (fen, uci, bestUci, bestCp, playedCp, winBefore, winAfter) => {
    const chess = new Chess(fen);
    const san = chess.move(uci).san;
    const bestSan = new Chess(fen).move(bestUci).san;
    return { fenBefore: fen, uci, san, bestUci, bestSan, bestCp, playedCp, winBefore, winAfter };
  };
  const cases = [
    [move("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "h2h3", "a1a8", 9999, 0, 100, 50),
     "There was mate in one: Ra8#. Your winning chances dropped from 100% to 50%."],
    [move("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "g1f1", "h2h3", 0, -9998, 50, 0),
     "This allows a forced mate. Your winning chances dropped from 50% to 0%."],
    [move("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3g5", "f1c4", 40, -250, 54, 29),
     "This leaves your knight on g5 undefended. Your winning chances dropped from 54% to 29%."],
    [move("rnb1kbnr/pppp1ppp/8/4p1q1/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "a2a3", "f3g5", 900, 0, 96, 50),
     "You could have won a queen with Nxg5. Your winning chances dropped from 96% to 50%."],
    [move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g2g4", "e2e4", 30, -120, 53, 39),
     "e4 was stronger (+0.3 instead of -1.2). Your winning chances dropped from 53% to 39%."],
    [move("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "d1g4", "g1f3", -4, -150, 50, 36),
     "Nf3 was stronger (-0.0 instead of -1.5). Your winning chances dropped from 50% to 36%."],
  ];
  for (const [m, text] of cases) equal(explainMistake(m), text);
});

test("coach: level differences in words", () => {
  const yours = { rating: 1200, san: "Nf6", uci: "g8f6", prob: 0.5, win: 3.5 };
  const stronger = { rating: 1400, san: "g6", uci: "g7g6", prob: 0.6, win: 50 };
  const moment = (playedUci, playedSan, playedWin) => ({ playedUci, playedSan, playedWin, comparison: { yours, stronger } });
  equal(explainLevel(moment("g8f6", "Nf6", 3.5)), "Players around 1200 usually play Nf6 here (50%). Players around 1400 "
    + "prefer g6 (60%), which keeps 50% winning chances instead of 4%. You played Nf6, like most players at your level.");
  assert(explainLevel(moment("g7g6", "g6", 50)).endsWith("You found g6: that is the stronger players' move."));
  assert(explainLevel(moment("a7a6", "a6", 20)).endsWith("You played a6 (20%)."));
  equal(explainLevel({ comparison: { yours, stronger: null } }), "Players around 1200 usually play Nf6 here (50%). Stronger players agree.");
});

test("coach: a whole game, with the levels disagreeing where it matters", async () => {
  const analyzer = scripted({
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4")]: ["g7g6", 0, { g8f6: -900 }],
    [keyOf("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6")]: ["h5f7", 900, {}],
  });
  const chess = new Chess();
  for (const san of ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]) chess.move(san);
  const review = await reviewMoves(new Chess().fen(), chess.history({ verbose: true }).map((m) => m.lan), analyzer);
  const before = keyOf("e4", "e5", "Qh5", "Nc6", "Bc4");
  const model = {                        // players below 1400 play Nf6 there, stronger ones g6; elsewhere the game's move
    predict(position, rating) {
      const fen = position.fen();
      const legal = position.moves({ verbose: true });
      const at = review.moves.find((m) => m.fenBefore === fen);
      const favourite = fen.split(" ").slice(0, 2).join(" ") === before ? (rating < 1400 ? "g8f6" : "g7g6") : at.uci;
      const top = legal.find((m) => m.lan === favourite);
      const rest = legal.filter((m) => m.lan !== favourite).slice(0, 2);
      return { moves: [{ uci: top.lan, san: top.san, prob: 0.6 }, ...rest.map((m) => ({ uci: m.lan, san: m.san, prob: 0.1 }))] };
    },
  };
  const progress = [];
  const report = await coachGame(review, "black", 1200, model, analyzer, { onProgress: (d, t) => progress.push(`${d}/${t}`) });
  equal(progress, ["1/3", "2/3", "3/3"]);
  equal(report.notes.map((n) => [n.moveNumber, n.san, n.kinds]), [["3...", "Nf6", ["mistake", "level"]]]);
  equal(report.notes[0].text, "g6 was stronger (+0.0 instead of -9.0). Your winning chances dropped from 50% to 4%. "
    + "Players around 1200 usually play Nf6 here (60%). Players around 1400 prefer g6 (60%), which keeps 50% winning "
    + "chances instead of 4%. You played Nf6, like most players at your level.");
  equal([report.yourLevelMatch, report.nextLevelMatch, report.nextLevel], [1, 2 / 3, 1400]);
});

// ------------------------------------------------------------------ hints

// From src/chess_engine/bot/hints.py: [fen, move, themes, threatened piece, [nudge at 50%, nudge at 20%,
// point at 5% (only move), show (only move, winning)]].
const HINT_CASES = [
[
"6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
"a1a8",
[
"mate"
],
null,
[
"There is a forcing move here: look at checks, captures and threats.",
"There is a checkmate!",
"There is a checkmate! It's a rook move. Move your rook on a1.",
"Play Ra8#: checkmate."
]
],
[
"rnb1kbnr/pppp1ppp/8/4p1q1/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
"f3g5",
[
"win_material"
],
null,
[
"There is a forcing move here: look at checks, captures and threats.",
"One of your opponent's pieces can be won.",
"One of your opponent's pieces can be won. It's a knight move. Careful: only one move keeps your position together. Move your knight on f3.",
"Play Nxg5: it wins material."
]
],
[
"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
"e4d5",
[
"capture"
],
null,
[
"There is a forcing move here: look at checks, captures and threats.",
"Look at the captures.",
"Look at the captures. It's a pawn move. Careful: only one move keeps your position together. Move your pawn on e4.",
"Play exd5: a capture."
]
],
[
"4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1",
"e4c3",
[
"save_piece",
"threat"
],
"p",
[
"Before anything else, check that all your pieces are safe.",
"One of your pieces is in danger: save it.",
"One of your pieces is in danger: save it. It's a knight move. Careful: only one move keeps your position together. Move your knight on e4.",
"Play Nc3: it saves the piece, and it creates a threat."
]
],
[
"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
"e1g1",
[
"defend",
"castle"
],
null,
[
"Before anything else, check that all your pieces are safe.",
"One of your pieces is loose: protect it.",
"One of your pieces is loose: protect it. It's a king move. Careful: only one move keeps your position together. Move your king on e1.",
"Play O-O: it protects your loose piece, and it gets your king to safety."
]
],
[
"rnb1kbnr/pppp1ppp/8/4p3/7q/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
"g2g3",
[
"threat"
],
"q",
[
"You can put your opponent under pressure.",
"You can attack their queen.",
"You can attack their queen. It's a pawn move. Careful: only one move keeps your position together. Move your pawn on g2.",
"Play g3: it creates a threat."
]
],
[
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
"e2e4",
[
"quiet"
],
null,
[
"No need for anything sharp: look for a calm, useful move.",
"Look for a quiet move that improves your position.",
"Look for a quiet move that improves your position. It's a pawn move. Careful: only one move keeps your position together. Move your pawn on e2.",
"Play e4."
]
],
[
"8/1P6/8/8/8/8/k7/7K w - - 0 1",
"b7b8q",
[
"promotion"
],
null,
[
"There is a forcing move here: look at checks, captures and threats.",
"One of your pawns can promote.",
"One of your pawns can promote. It's a pawn move. Careful: only one move keeps your position together. Move your pawn on b7.",
"Play b8=Q: it promotes."
]
],
[
"4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
"a1a8",
[
"check"
],
null,
[
"There is a forcing move here: look at checks, captures and threats.",
"Look for a strong check.",
"Look for a strong check. It's a rook move. Careful: only one move keeps your position together. Move your rook on a1.",
"Play Ra8+: a check."
]
],
[
"4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
"e5d6",
[
"capture"
],
null,
[
"There is a forcing move here: look at checks, captures and threats.",
"Look at the captures.",
"Look at the captures. It's a pawn move. Careful: only one move keeps your position together. Move your pawn on e5.",
"Play exd6: a capture."
]
]
];

test("hints: themes and texts are the Python ones, case by case", () => {
  for (const [fen, uci, themes, threatened, texts] of HINT_CASES) {
    equal(moveThemes(fen, uci), themes, `${fen} ${uci}`);
    equal(threatenedPiece(fen, uci), threatened, `${fen} ${uci}`);
    equal([makeHint(fen, uci, "nudge", 0.5).text, makeHint(fen, uci, "nudge", 0.2).text,
           makeHint(fen, uci, "point", 0.05, { onlyMove: true }).text,
           makeHint(fen, uci, "show", 0.05, { onlyMove: true, winning: true }).text], texts, `${fen} ${uci}`);
  }
  equal([detailFor(0.35), detailFor(0.2), detailFor(0.09)], ["vague", "theme", "piece"]);
  const shown = makeHint(HINT_CASES[0][0], "a1a8", "show", 0.3);
  equal([shown.from, shown.to, shown.san, shown.tier], ["a1", "a8", "Ra8#", "show"]);
  const pointed = makeHint(HINT_CASES[0][0], "a1a8", "point", 0.3);
  equal([pointed.from, pointed.to, pointed.san], ["a1", null, null]);
});

test("hints: the engine's best move, how often your level finds it, and whether it is the only move", async () => {
  const fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1";
  let searches = 0;
  const analyzer = { async analyse() { searches += 1; return { lines: [{ uci: "a1a8", cp: 9999 }, { uci: "h2h3", cp: 10 }], bestCp: 9999 }; } };
  const model = { predict: () => ({ moves: [{ uci: "h2h3", prob: 0.7 }, { uci: "a1a8", prob: 0.2 }] }) };
  const engine = new HintEngine(model, analyzer);
  const nudge = await engine.hint(fen, "nudge", 1200);
  equal([nudge.findRate, nudge.detail, nudge.onlyMove, nudge.text], [0.2, "theme", true, "There is a checkmate!"]);
  equal((await engine.hint(fen, "show", 1200)).text, "Play Ra8#: checkmate.");
  equal(searches, 1, "the position is remembered");
  let threw = false;
  try { await engine.best("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 1200); } catch { threw = true; }
  assert(threw, "no hint once the game is over");
});

test("hints: the per-game budget counts positions, and each request goes one tier further", () => {
  const budget = new HintBudget(2);
  const a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
  const c = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
  assert(budget.rated);
  equal([budget.take(a), budget.take(a), budget.take(a), budget.take(a)], ["nudge", "point", "show", "show"]);
  equal([budget.used, budget.left, budget.rated], [1, 1, false]);
  equal(budget.take(b.replace(" 0 1", " 5 9")), "nudge", "move counters do not matter");
  assert(!budget.canTake(c) && budget.canTake(a));
  let message = null;
  try { budget.take(c); } catch (err) { message = err.message; }
  equal(message, "No hints left in this game.");
  const copy = HintBudget.fromJSON(JSON.parse(JSON.stringify(budget)));
  equal([copy.used, copy.nextTier(a), copy.nextTier(c)], [2, "show", "nudge"]);
  message = null;
  try { new HintBudget(0).take(a); } catch (err) { message = err.message; }
  equal(message, "Hints are off for this game.");
});

run();
