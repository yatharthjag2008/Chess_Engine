"""An honest rating dial: what each setting really plays like, from the Elo-dial validation (step 11).

The model is told a rating (its *input*: "play like Lichess players rated 1400"), but it changes its play less than
the rating says: in the validation games the settings 800 to 2300 measured only about 670 Elo apart, 800 and 1100
played alike, and 2000 measured no stronger than 1700. The dial's *labels* should mean strength, so:

1. the measured strengths are made non-decreasing (weighted isotonic regression: a setting is never shown as
   weaker than a lower one, which the measurement cannot tell apart within its noise anyway);
2. a label differs from another by exactly the measured Elo difference of their inputs, with one **anchor**: the
   input 1500 (the middle of the training data) keeps the label 1500. The measurements are on Stockfish's Elo
   scale, not Lichess's, so differences are meaningful and the anchor is an assumption, not a measurement;
3. a label is turned back into the input that plays at that strength (the lowest one, where several play alike).

So the dial only offers the strengths the bot can actually play, and steps on it are real Elo steps.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def isotonic(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """The non-decreasing sequence closest to ``values`` in weighted least squares (pool adjacent violators)."""
    if len(values) != len(weights) or not len(values):
        raise ValueError("values and weights must be non-empty and of the same length")
    blocks: list[list[float]] = []            # [weighted mean, total weight, count]
    for v, w in zip(values, weights):
        if w <= 0:
            raise ValueError("weights must be positive")
        blocks.append([float(v), float(w), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out: list[float] = []
    for mean, _, count in blocks:
        out.extend([mean] * count)
    return out


@dataclass(frozen=True)
class DialPoint:
    input: int            # the rating the model is told
    measured: float       # its measured strength (Stockfish's Elo scale)
    low: float            # 95% interval of the measurement
    high: float
    fitted: float         # after isotonic regression
    label: float          # what the dial shows for it


@dataclass(frozen=True)
class DialCalibration:
    points: list[DialPoint]               # by input
    anchor: int = 1500                    # this input keeps this label

    @property
    def inputs(self) -> np.ndarray:
        return np.array([p.input for p in self.points], dtype=np.float64)

    @property
    def labels(self) -> np.ndarray:
        return np.array([p.label for p in self.points], dtype=np.float64)

    def label_for(self, rating_input: float) -> float:
        """The honest label of an input (flat beyond the measured settings)."""
        return float(np.interp(rating_input, self.inputs, self.labels))

    def input_for(self, label: float) -> int:
        """The lowest input that plays at ``label`` (labels outside the range give the end settings)."""
        inputs, labels = self.inputs, self.labels
        if label <= labels[0]:
            return int(inputs[0])
        if label >= labels[-1]:
            return int(inputs[-1])
        i = int(np.searchsorted(labels, label, side="left"))   # first point with label >= target
        if labels[i] == label:
            return int(inputs[i])
        t = (label - labels[i - 1]) / (labels[i] - labels[i - 1])
        return int(round(inputs[i - 1] + t * (inputs[i] - inputs[i - 1])))

    def range(self, step: int = 50) -> tuple[int, int]:
        """The labels the dial offers: the measured range, rounded inwards to ``step``."""
        low = int(np.ceil(self.labels[0] / step) * step)
        high = int(np.floor(self.labels[-1] / step) * step)
        return low, high

    def to_dict(self, step: int = 50) -> dict:
        low, high = self.range(step)
        return {"anchor": self.anchor, "step": step, "min": low, "max": high,
                "points": [asdict(p) for p in self.points]}


def calibrate(bots: Sequence[dict], anchor: int = 1500) -> DialCalibration:
    """The calibration from the validation's per-setting results (``results.json``'s ``bots``: dial, measured,
    interval); each measurement weighs by its precision (from its 95% interval)."""
    rows = sorted(bots, key=lambda b: b["dial"])
    if len(rows) < 2:
        raise ValueError("need at least two measured settings")
    measured = [float(b["measured"]) for b in rows]
    sds = [max((b["interval"][1] - b["interval"][0]) / 3.92, 1.0) for b in rows]
    fitted = isotonic(measured, [1 / sd**2 for sd in sds])
    inputs = np.array([b["dial"] for b in rows], dtype=np.float64)
    if not inputs[0] <= anchor <= inputs[-1]:
        raise ValueError(f"the anchor {anchor} is outside the measured settings")
    at_anchor = float(np.interp(anchor, inputs, fitted))
    points = [DialPoint(int(b["dial"]), m, float(b["interval"][0]), float(b["interval"][1]), f,
                        anchor + f - at_anchor) for b, m, f in zip(rows, measured, fitted)]
    return DialCalibration(points, anchor)


def load_calibration(results_path: str | Path, anchor: int = 1500) -> DialCalibration:
    """The calibration from an Elo-dial validation's ``results.json``."""
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return calibrate(data["bots"], anchor)
