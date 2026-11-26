// Stockfish in this browser (site/js/stockfish.js with vendor/stockfish/): tests/test_site.py runs this page.

import { assert, equal, run, test } from "./harness.js";
import { MATE_CP, StockfishAnalyzer, finishedEval, parseInfo } from "../js/stockfish.js";
import { Chess } from "../vendor/chess.js";

test("UCI info lines: centipawns, mates, bounds", () => {
  equal(parseInfo("info depth 12 seldepth 18 multipv 2 score cp -35 nodes 1 nps 1 pv e7e5 g1f3"),
        { depth: 12, multipv: 2, cp: -35, pv: ["e7e5", "g1f3"] });
  equal(parseInfo("info depth 5 score mate 3 pv h5f7").cp, MATE_CP - 3, "mate in 3 for the side to move");
  equal(parseInfo("info depth 5 score mate -2 pv a2a3").cp, -(MATE_CP - 2), "mated in 2");
  equal(parseInfo("info depth 9 score cp 20 lowerbound pv e2e4"), null, "not a final score");
  equal(parseInfo("info string NNUE evaluation enabled"), null);
  equal(parseInfo("info depth 3 currmove e2e4 currmovenumber 1"), null);
});

test("finished games are scored without the engine", () => {
  equal(finishedEval(new Chess("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")), -MATE_CP, "checkmated");
  equal(finishedEval(new Chess("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")), 0, "stalemate");
  equal(finishedEval(new Chess("8/8/4k3/8/8/4K3/8/8 w - - 0 1")), 0, "kings only");
  equal(finishedEval(new Chess()), null);
});

const engine = new StockfishAnalyzer({ depth: 10, multipv: 3 });

test("Stockfish starts in a worker and finds a mate in one", async () => {
  const a = await engine.analyse("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
  equal(a.lines[0].san, "Ra8#");
  equal(a.bestCp, MATE_CP - 1);
  equal(a.lines[0].pv[0], "a1a8");
});

test("three lines, best first, with a sensible opening evaluation", async () => {
  const a = await engine.analyse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  equal(a.lines.length, 3);
  assert(a.lines[0].cp >= a.lines[1].cp && a.lines[1].cp >= a.lines[2].cp, "sorted");
  assert(Math.abs(a.bestCp) < 100, `about equal: ${a.bestCp}`);
  assert(new Set(a.lines.map((l) => l.uci)).size === 3, "different moves");
  assert(a.lines[0].pvSan.length >= 2, "a main line in SAN");
  equal(await engine.analyse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), a, "from the cache");
});

test("Black to move: the evaluation is Black's", async () => {
  const a = await engine.analyse("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
  equal(a.bestCp, -MATE_CP, "White is already mated: no engine needed");
  const b = await engine.analyse("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2");
  equal(b.lines[0].san, "Qh4#");
  equal(b.bestCp, MATE_CP - 1, "a mate for Black, the side to move");
});

test("searches queue one at a time", async () => {
  const fens = ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2"];
  const results = await Promise.all(fens.map((fen) => engine.analyse(fen, { depth: 8 })));
  equal(results.map((r) => r.fen), fens);
  assert(results.every((r) => r.lines.length === 3));
  engine.terminate();
});

run();
