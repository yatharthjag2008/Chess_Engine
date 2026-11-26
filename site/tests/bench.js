// Timing of the in-browser network: one pass, and one side of a real game's rating estimate.

import { Chess } from "../vendor/chess.js";
import { encode, contextFeatures, loadModel } from "../js/engine.js";
import { playerGames } from "../js/games.js";
import { loadRatingData } from "../js/rating.js";
import { sideLikelihoods } from "../js/analyse.js";

const out = [];
const model = await loadModel("../model/");
await loadRatingData("../data/");
const chess = new Chess("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
const planes = encode(chess);
const context = contextFeatures(180, 180, 180, 0, model.meta.log_clock_scale);
for (let i = 0; i < 5; i++) model.forward(planes, 3, 3, context);     // warm up
let t = performance.now();
const passes = 40;
for (let i = 0; i < passes; i++) model.forward(planes, i % 8, 3, context);
out.push(`ms_per_pass=${((performance.now() - t) / passes).toFixed(1)}`);

const pgn = await (await fetch("bench_game.pgn")).text();
const side = playerGames(pgn).games[0];
t = performance.now();
await sideLikelihoods(model, side);
out.push(`side_moves=${side.moves.length} side_seconds=${((performance.now() - t) / 1000).toFixed(2)}`);
out.push(`cores=${navigator.hardwareConcurrency}`);
document.getElementById("results").textContent = out.join("\n") + "\nRESULT passed=1 failed=0";
fetch("/__results", { method: "POST", body: document.getElementById("results").textContent });
