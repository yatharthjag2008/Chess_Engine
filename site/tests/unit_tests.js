// Unit tests of the website's modules, in a real browser (tests/test_site.py).

import { Chess } from "../vendor/chess.js";
import { assert, equal, run, test } from "./harness.js";
import { RECIPES, Sounds, soundFor } from "../js/sounds.js";
import { stored } from "../js/storage.js";
import { loadOpenings, openingOf, positionKey } from "../js/openings.js";
import { Clock, TIME_CONTROLS, baseSeconds, formatClock, thinkTime } from "../js/clock.js";
import { START, canMate, captured, countPieces, describeCaptures, materialOf } from "../js/material.js";

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

run();
