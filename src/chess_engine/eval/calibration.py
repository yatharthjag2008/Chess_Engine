"""Calibration: when the model gives a move probability p, is it played about p of the time?

A human-like bot *samples* moves from the model, so the probabilities themselves
must be right, not just their order. Two reliability views:

* **all legal moves**: every legal move of every position, binned by its predicted
  probability; the observed frequency is how often the moves in a bin were the one
  played (the histogram is accumulated during the prediction pass);
* **top-1**: the model's favourite move, binned by its probability, against how
  often it was played (the "confidence vs. accuracy" diagram).

The **expected calibration error** (ECE) is the average gap between predicted and
observed frequency, weighted by how many predictions fall in each bin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chess_engine.eval.predictions import PredictionTable


def reliability(counts: np.ndarray, sum_pred: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
    """Reliability table and ECE from per-bin counts, summed predictions and summed outcomes (equal-width bins)."""
    counts = np.asarray(counts, dtype=np.float64)
    bins = len(counts)
    edges = np.linspace(0, 1, bins + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.where(counts > 0, sum_pred / np.maximum(counts, 1), np.nan)
        observed = np.where(counts > 0, hits / np.maximum(counts, 1), np.nan)
    total = counts.sum()
    gaps = np.abs(np.nan_to_num(mean_pred) - np.nan_to_num(observed))
    ece = float((counts * gaps).sum() / total) if total else float("nan")
    return {
        "bins": [
            {"low": float(edges[i]), "high": float(edges[i + 1]), "count": int(counts[i]),
             "mean_predicted": None if np.isnan(mean_pred[i]) else float(mean_pred[i]),
             "observed": None if np.isnan(observed[i]) else float(observed[i])}
            for i in range(bins)
        ],
        "ece": ece,
        "predictions": int(total),
    }


def binned_reliability(predicted: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """:func:`reliability` for per-prediction probabilities in [0, 1] and 0/1 (or fractional) outcomes."""
    predicted = np.clip(np.asarray(predicted, dtype=np.float64), 0, 1)
    outcome = np.asarray(outcome, dtype=np.float64)
    which = np.minimum((predicted * bins).astype(np.int64), bins - 1)
    return reliability(np.bincount(which, minlength=bins), np.bincount(which, weights=predicted, minlength=bins),
                       np.bincount(which, weights=outcome, minlength=bins))


def move_calibration(table: PredictionTable) -> dict[str, Any]:
    """``{"all_moves": ..., "top1": ...}`` reliability tables of the module docstring."""
    hist = table.legal_calibration
    bins = len(hist)
    return {
        "all_moves": reliability(hist[:, 0], hist[:, 1], hist[:, 2]),
        "top1": binned_reliability(table["top_probs"][:, 0], table.top1_hit, bins),
    }
