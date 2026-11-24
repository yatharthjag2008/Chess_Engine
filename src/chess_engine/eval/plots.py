"""Evaluation charts saved as PNGs (style: :mod:`chess_engine.plotting`).

Every series keeps one colour across all charts: the model is always slot 1 (blue),
then the baselines in a fixed order. The random-move baseline is a grey dashed
reference line, not a series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chess_engine import plotting

#: Fixed colour slot per predictor, so a predictor never changes colour between charts.
SERIES_SLOTS = {
    "model": 0,
    "most common move": 1,
    "Stockfish": 2,
    "model, one rating for all": 3,
    "most common move (same rating)": 4,
}


def series_color(name: str) -> str:
    """The predictor's fixed colour (matched on the start of its name, e.g. ``Stockfish depth 10``)."""
    for prefix, slot in sorted(SERIES_SLOTS.items(), key=lambda item: -len(item[0])):
        if name.startswith(prefix):
            return plotting.SERIES[slot]
    raise KeyError(f"no colour slot for series {name!r}")


@dataclass(frozen=True)
class Series:
    """One predictor's values per rating bucket (None where it has no data), with optional CI bounds."""

    name: str
    values: Sequence[float | None]
    low: Sequence[float | None] | None = None
    high: Sequence[float | None] | None = None


def _array(values: Sequence[float | None] | None) -> np.ndarray | None:
    return None if values is None else np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def plot_accuracy_by_rating(
    labels: Sequence[str],
    series: Sequence[Series],
    path: str | Path,
    random: Sequence[float | None] | None = None,
    title: str = "Top-1 move match by rating",
    ylabel: str = "positions where the human move was predicted",
) -> Path:
    """Lines of top-1 accuracy per rating bucket, one per predictor, with shaded confidence bands."""
    fig, ax = plotting.new_figure(width=9.0, height=4.4)
    x = np.arange(len(labels))
    for s in series:
        color = series_color(s.name)
        values = _array(s.values)
        ax.plot(x, values, color=color, lw=plotting.LINE_WIDTH, marker="o", ms=plotting.MARKER_SIZE,
                label=s.name, zorder=3)
        low, high = _array(s.low), _array(s.high)
        if low is not None and high is not None:
            ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0, zorder=2)
        last = int(np.flatnonzero(np.isfinite(values))[-1]) if np.isfinite(values).any() else None
        if last is not None and s.name == "model":  # selective direct label: the model only
            ax.annotate(f"{values[last]:.0%}", (x[last], values[last]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=plotting.INK_SECONDARY)
    if random is not None:
        ax.plot(x, _array(random), color=plotting.REFERENCE, lw=1.5, ls="--", label="random legal move", zorder=2)
    ax.set_xticks(x, labels, rotation=0)
    ax.set_xlim(-0.3, len(labels) - 0.4)
    ax.set_ylim(0, None)
    plotting.style_axes(ax, title, "rating of the player to move", ylabel)
    plotting.percent_axis(ax)
    plotting.legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return plotting.save_figure(fig, path)
