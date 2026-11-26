// Unit tests of the website's modules, in a real browser (tests/test_site.py).

import { Chess } from "../vendor/chess.js";
import { assert, equal, run, test } from "./harness.js";
import { RECIPES, Sounds, soundFor } from "../js/sounds.js";
import { stored } from "../js/storage.js";
import { START, captured, countPieces, describeCaptures, materialOf } from "../js/material.js";

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

run();
