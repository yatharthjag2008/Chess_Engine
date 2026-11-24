"""Move-matching accuracy: how often the model's prediction is the move the human actually played.

    result = move_matching(table, labels=cfg.ratings.bucket_labels())
    result["overall"]["top1"], result["buckets"]["1400-1599"]["top1"]

Per group (all positions, each rating bucket, ...):

* ``top1`` / ``top3`` / ``top5``: the human move is the model's favourite / in its 3 or 5 likeliest;
* ``log_likelihood``: mean log-probability of the human move (higher is better; this is
  what training optimizes, and unlike top-1 it rewards putting probability on the
  second-choice moves humans also play);
* ``perplexity``: ``exp(-log_likelihood)``, "the model is as unsure as a uniform choice
  between this many moves";
* ``mean_legal_moves`` for context (a random legal move matches ``mean(1 / legal)`` of the time).

``bucket_mean_top1`` averages the bucket accuracies, so every rating level counts
equally whatever its number of positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chess_engine.eval.predictions import PredictionTable

TOP_K: tuple[int, ...] = (1, 3, 5)


def group_means(values: np.ndarray, groups: np.ndarray, num_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` and number of entries in each group ``0 .. num_groups-1`` (NaN if empty)."""
    groups = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(groups, minlength=num_groups)[:num_groups]
    sums = np.bincount(groups, weights=np.asarray(values, dtype=np.float64), minlength=num_groups)[:num_groups]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), counts


def match_stats(table: PredictionTable, rows: np.ndarray | None = None) -> dict[str, float]:
    """The metrics of the module docstring over ``rows`` (a bool mask or indices; all rows if None)."""
    rank = table["target_rank"] if rows is None else table["target_rank"][rows]
    n = len(rank)
    if n == 0:
        return {"positions": 0}
    prob = table["target_prob"] if rows is None else table["target_prob"][rows]
    legal = table["legal_count"] if rows is None else table["legal_count"][rows]
    log_likelihood = float(np.log(np.maximum(prob.astype(np.float64), 1e-12)).mean())
    out: dict[str, float] = {"positions": int(n)}
    out.update({f"top{k}": float((rank <= k).mean()) for k in TOP_K})
    out.update(log_likelihood=log_likelihood, perplexity=float(np.exp(-log_likelihood)),
               mean_legal_moves=float(legal.mean()), random_top1=float((1.0 / legal).mean()))
    return out


def move_matching(table: PredictionTable, labels: Sequence[str] | None = None) -> dict[str, Any]:
    """``{"overall": stats, "buckets": {label: stats}, "bucket_mean_top1": float}`` (see the module docstring)."""
    labels = list(labels if labels is not None else table.meta.get("bucket_labels", []))
    if not labels:
        labels = [str(b) for b in range(int(table["bucket"].max()) + 1)]
    buckets = {label: match_stats(table, table["bucket"] == b) for b, label in enumerate(labels)}
    present = [s["top1"] for s in buckets.values() if s["positions"]]
    return {
        "overall": match_stats(table),
        "buckets": buckets,
        "bucket_mean_top1": float(np.mean(present)) if present else float("nan"),
    }


def format_move_matching(result: dict[str, Any]) -> str:
    """A per-bucket text table of :func:`move_matching`."""
    lines = [f"{'bucket':<10} {'positions':>9} {'top1':>6} {'top3':>6} {'top5':>6} {'loglik':>7} {'perpl':>6}"]

    def row(label: str, s: dict[str, float]) -> str:
        if not s["positions"]:
            return f"{label:<10} {0:>9}"
        return (f"{label:<10} {s['positions']:>9,} {s['top1']:>6.1%} {s['top3']:>6.1%} {s['top5']:>6.1%} "
                f"{s['log_likelihood']:>7.3f} {s['perplexity']:>6.2f}")

    lines.extend(row(label, s) for label, s in result["buckets"].items())
    lines.append(row("all", result["overall"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- breakdowns

Band = tuple[str, float, float]  # label, lower bound (inclusive), upper bound (exclusive)

#: Game phase by move number (plies 0-19 are moves 1-10).
PHASES: tuple[Band, ...] = (("moves 1-10", 0, 20), ("moves 11-25", 20, 50), ("moves 26-40", 50, 80),
                            ("moves 41+", 80, float("inf")))
#: Time pressure by the mover's clock.
CLOCK_BANDS: tuple[Band, ...] = (("under 15 s", 0, 15), ("15-60 s", 15, 60), ("1-2 min", 60, 120),
                                 ("over 2 min", 120, float("inf")))
#: How many legal moves the position has.
LEGAL_BANDS: tuple[Band, ...] = (("1-10 moves", 1, 11), ("11-25 moves", 11, 26), ("26-40 moves", 26, 41),
                                 ("41+ moves", 41, float("inf")))


def breakdown(table: PredictionTable, column: str, bands: Sequence[Band],
              labels: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Match statistics per band of ``column``, plus top-1 per rating bucket within each band."""
    labels = list(labels if labels is not None else table.meta.get("bucket_labels", []))
    values = table[column].astype(np.float64)
    out = {}
    for label, low, high in bands:
        rows = (values >= low) & (values < high)
        stats: dict[str, Any] = match_stats(table, rows)
        if labels:
            top1, _ = group_means(table.top1_hit[rows], table["bucket"][rows], len(labels))
            stats["bucket_top1"] = [None if np.isnan(v) else float(v) for v in top1]
        out[label] = stats
    return out


def breakdowns(table: PredictionTable, labels: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Accuracy by game phase, by the mover's clock and by the number of legal moves."""
    return {
        "phase": breakdown(table, "ply", PHASES, labels),
        "clock": breakdown(table, "clock", CLOCK_BANDS, labels),
        "legal_moves": breakdown(table, "legal_count", LEGAL_BANDS, labels),
    }
