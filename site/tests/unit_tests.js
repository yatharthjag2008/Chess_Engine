// Unit tests of the website's modules, in a real browser (tests/test_site.py).

import { Chess } from "../vendor/chess.js";
import { assert, equal, run, test } from "./harness.js";
import { RECIPES, Sounds, soundFor } from "../js/sounds.js";
import { START, captured, countPieces, materialOf } from "../js/material.js";

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

run();
