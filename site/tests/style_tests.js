// An exported style against the PyTorch styled predictor's numbers (written into the style file at export).

import { Chess } from "../vendor/chess.js";
import { loadModel, loadStyle, loadStyles } from "../js/engine.js";
import { assert, equal, run, test } from "./harness.js";

const model = await loadModel("../model/");
const styles = await loadStyles("../styles/");

test("the index lists the exported style", () => {
  equal(styles.map((s) => [s.name, s.player, s.kind, s.games]), [["ann", "Ann", "style", 200]]);
});

test("the style's probabilities are PyTorch's (blended with the base model)", async () => {
  const style = await loadStyle("ann", "../styles/");
  const check = model.selfTest(Chess, style);
  assert(check.ok, `worst difference ${check.worst}`);
  equal(style.weight, 1, "200 games: the style counts fully");
});

test("Ann always opens 1. a3: the styled bot does, the base bot does not", async () => {
  const style = await loadStyle("ann", "../styles/");
  const start = new Chess();
  equal(model.predict(start, 1500, { style }).moves[0].uci, "a2a3");
  assert(model.predict(start, 1500).moves[0].uci !== "a2a3" || model.predict(start, 1500).moves[0].prob < 0.5);
  const half = { ...style, weight: 0.5 };
  const blended = model.predict(start, 1500, { style: half }).moves;
  const total = blended.reduce((a, m) => a + m.prob, 0);
  assert(Math.abs(total - 1) < 1e-5, "still a distribution");
});

test("no styles: an empty list, not an error", async () => {
  equal(await loadStyles("../no-such-folder/"), []);
});

run();
