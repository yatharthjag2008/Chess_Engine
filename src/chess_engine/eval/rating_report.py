"""Charts and Markdown report for rating estimation (inputs: :mod:`chess_engine.eval.rating_eval` results)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chess_engine import plotting
from chess_engine.eval.rating_eval import EvalData, Estimates

CHARTS = {"scatter": "estimated_vs_true.png", "error": "error_by_rating.png", "convergence": "convergence.png"}

#: Colours by role: the estimate is the model (slot 1); the baselines take the next slots in order.
ESTIMATE, OPPONENT, AVERAGE = plotting.SERIES[0], plotting.SERIES[1], plotting.SERIES[2]


def plot_scatter(est: Estimates, data: EvalData, path: str | Path) -> Path:
    """Estimated against true rating, one dot per player-game."""
    fig, ax = plotting.new_figure(width=6.4, height=6.0)
    lo = min(data.rating.min(), est.estimate.min()) - 50
    hi = max(data.rating.max(), est.estimate.max()) + 50
    ax.plot([lo, hi], [lo, hi], color=plotting.REFERENCE, lw=1.5, ls="--", label="estimate = true rating", zorder=1)
    ax.scatter(data.rating, est.estimate, s=7, color=ESTIMATE, alpha=0.25, linewidths=0, label="player-game",
               zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    plotting.style_axes(ax, "Estimated vs. true rating (one game each)", "true rating", "estimated rating")
    plotting.legend(ax, loc="upper left")
    return plotting.save_figure(fig, path)


def plot_error_by_rating(scores: dict[str, Any], path: str | Path) -> Path:
    """Mean absolute error per true-rating bucket: the estimate and the two baselines."""
    labels = [label for label, b in scores["buckets"].items() if b.get("players")]
    x = np.arange(len(labels))
    fig, ax = plotting.new_figure(width=9.0, height=4.4)
    series = [("estimate from the moves", ESTIMATE, lambda b: b["mae"]),
              ("opponent's rating", OPPONENT, lambda b: b["baseline_opponent"]["mae"]),
              ("everyone is average", AVERAGE, lambda b: b["baseline_average"]["mae"])]
    for name, color, value in series:
        y = [value(scores["buckets"][label]) for label in labels]
        ax.plot(x, y, color=color, lw=plotting.LINE_WIDTH, marker="o", ms=plotting.MARKER_SIZE, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, None)
    plotting.style_axes(ax, "How far off is the estimate? (one game per estimate)", "true rating of the player",
                        "mean absolute error (Elo)")
    plotting.legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return plotting.save_figure(fig, path)


def plot_convergence(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Error and range width against the number of moves seen (two panels, one unit each)."""
    steps = [r for r in rows if r["max_moves"] is not None]
    x = [r["max_moves"] for r in steps]
    fig, axes = plotting.new_figure(1, 2, width=10.0, height=4.0)
    error_ax, width_ax = axes[0]
    error_ax.plot(x, [r["mae"] for r in steps], color=ESTIMATE, lw=plotting.LINE_WIDTH, marker="o",
                  ms=plotting.MARKER_SIZE)
    plotting.style_axes(error_ax, "Error after the first n moves", "moves seen (per player)",
                        "mean absolute error (Elo)")
    width_ax.plot(x, [r["mean_width"] for r in steps], color=ESTIMATE, lw=plotting.LINE_WIDTH, marker="o",
                  ms=plotting.MARKER_SIZE)
    plotting.style_axes(width_ax, "Width of the rating range", "moves seen (per player)", "range width (Elo)")
    for ax in (error_ax, width_ax):
        ax.set_ylim(0, None)
    return plotting.save_figure(fig, path)
