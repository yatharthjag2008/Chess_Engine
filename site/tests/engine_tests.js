// Tests of site/js/engine.js: encoding helpers, move indices, sampling, and the whole network against PyTorch.

import { Chess } from "../vendor/chess.js";
import { bucketOf, contextFeatures, encode, loadModel, mirror, sampleMove, squareIndex } from "../js/engine.js";
import { assert, equal, run, test } from "./harness.js";

const model = await loadModel("../model/");

test("squares, mirroring and rating buckets", () => {
  equal(squareIndex("a1"), 0);
  equal(squareIndex("e2"), 12);
  equal(squareIndex("h8"), 63);
  equal(mirror("e7"), "e2");
  const edges = [800, 1000, 1200];
  equal([500, 800, 999, 1000, 2500].map((r) => bucketOf(r, edges)), [0, 0, 0, 1, 2]);
});

test("clock features match the training data's", () => {
  const f = contextFeatures(180, 90, 180, 2, Math.log1p(600));
  equal(Array.from(f).map((v) => Math.round(v * 1e4) / 1e4),
        [1, 0.5, Math.round(Math.log1p(180) / Math.log1p(600) * 1e4) / 1e4,
         Math.round(Math.log1p(90) / Math.log1p(600) * 1e4) / 1e4, 0.2]);
});

test("encoding: the start position from each side", () => {
  const white = encode(new Chess());
  equal(white.length, 20 * 64);
  equal(white[0 * 64 + 8], 1, "my pawn on a2 (row 1)");
  equal(white[11 * 64 + 60], 1, "their king on e8 (row 7, file 4)");
  equal(white[12 * 64], 1, "White to move");
  const black = new Chess();
  black.move("e4");
  const b = encode(black);
  equal(b[12 * 64], 0, "Black to move");
  equal(b[0 * 64 + 8 + 4], 1, "my (Black's) pawn e7 seen on row 1");
  equal(b[6 * 64 + 4 * 8 + 4], 1, "their pawn e4 (rank 4) seen on row 4 from Black's side");
  equal(b[19 * 64 + 63], 1, "the ones plane");
});

test("en passant counts only when the capture is legal", () => {
  const ep = new Chess("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
  const planes = encode(ep);
  equal(planes[17 * 64 + squareIndex("f6")], 1);
  const none = new Chess("rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq d6 0 2");
  equal(encode(none).slice(17 * 64, 18 * 64).reduce((a, v) => a + v, 0), 0);
});

test("move indices: promotions and Black's moves", () => {
  const promo = new Chess("8/P6k/8/8/8/8/6K1/8 w - - 0 60");
  const moves = promo.moves({ verbose: true }).filter((m) => m.from === "a7");
  const indices = moves.map((m) => model.indexOf(m, "w"));
  assert(indices.every((i) => i !== undefined), "every promotion has an index");
  equal(new Set(indices).size, 4, "queen, rook, bishop and knight are different outputs");
  const start = new Chess("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
  const e5 = start.moves({ verbose: true }).find((m) => m.san === "e5");
  const e4 = new Chess().moves({ verbose: true }).find((m) => m.san === "e4");
  equal(model.indexOf(e5, "b"), model.indexOf(e4, "w"), "...e5 is Black's e4");
});

test("the browser computes what PyTorch computed", () => {
  const result = model.selfTest(Chess);
  assert(result.ok, `largest difference ${result.worst}`);
});

test("sampling follows the probabilities", () => {
  const moves = [{ uci: "a", prob: 0.7 }, { uci: "b", prob: 0.3 }];
  equal(sampleMove(moves, 1, () => 0.5).uci, "a");
  equal(sampleMove(moves, 1, () => 0.9).uci, "b");
  equal(sampleMove(moves, 0.01, () => 0.9).uci, "a", "a low temperature keeps to the likeliest");
  equal(sampleMove([], 1), null);
});

run();
