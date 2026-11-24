"""Evaluation charts saved as PNGs (style: :mod:`chess_engine.plotting`).

Every series keeps one colour across all charts: the model is always slot 1 (blue),
then the baselines in a fixed order. The random-move baseline is a grey dashed
reference line, not a series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

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


def plot_calibration(calibration: dict[str, Any], path: str | Path) -> Path:
    """Reliability diagrams side by side: every legal move, and the model's top-1 move.

    ``calibration`` is :func:`chess_engine.eval.calibration.move_calibration`'s result.
    Points on the dashed diagonal are perfectly calibrated; marker area shows how
    many predictions fall in each bin.
    """
    panels = (("all_moves", "Every legal move"), ("top1", "The model's top move"))
    fig, axes = plotting.new_figure(1, 2, width=10.5, height=5.2)
    color = series_color("model")
    for ax, (key, title) in zip(axes[0], panels):
        result = calibration[key]
        bins = [b for b in result["bins"] if b["count"]]
        pred = np.array([b["mean_predicted"] for b in bins])
        obs = np.array([b["observed"] for b in bins])
        counts = np.array([b["count"] for b in bins], dtype=np.float64)
        ax.plot([0, 1], [0, 1], color=plotting.REFERENCE, lw=1.5, ls="--", label="perfect calibration")
        ax.plot(pred, obs, color=color, lw=plotting.LINE_WIDTH, label="model", zorder=3)
        if len(counts):
            sizes = 30 + 250 * np.sqrt(counts / counts.max())
            ax.scatter(pred, obs, s=sizes, color=color, edgecolor=plotting.SURFACE, linewidth=2, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plotting.style_axes(ax, f"{title}\nexpected calibration error {result['ece']:.3f}",
                            "predicted probability", "how often it was played")
        plotting.percent_axis(ax, "x")
        plotting.percent_axis(ax, "y")
        plotting.legend(ax, loc="upper left")
    return plotting.save_figure(fig, path)


#: Sequential blue ramp (steps 100 -> 700) for magnitudes: light = low, dark = high.
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def plot_cross_rating(cross: dict[str, Any], path: str | Path) -> Path:
    """Heatmap of top-1 accuracy: rows = rating the model was told, columns = player's real rating.

    Each column's best row is outlined; if rating conditioning works, the outlines
    sit on (or next to) the diagonal.
    """
    labels = cross["labels"]
    matrix = np.array([[np.nan if v is None else v for v in row] for row in cross["top1"]], dtype=np.float64)
    k = len(labels)
    fig, ax = plotting.new_figure(width=7.6, height=6.4)
    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    image = ax.imshow(matrix, cmap=_SEQUENTIAL, vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9,
                      origin="lower")
    mid = (vmin + vmax) / 2
    for m in range(k):
        for p in range(k):
            if np.isfinite(matrix[m, p]):
                ink = "#ffffff" if matrix[m, p] > mid else plotting.INK
                ax.text(p, m, f"{matrix[m, p]:.1%}", ha="center", va="center", fontsize=7.5, color=ink)
    for p in range(k):
        column = matrix[:, p]
        if np.isfinite(column).any():
            best = int(np.nanargmax(column))
            ax.add_patch(Rectangle((p - 0.5, best - 0.5), 1, 1, fill=False, edgecolor=plotting.INK, lw=2))
    ax.set_xticks(range(k), labels, rotation=45, ha="right")
    ax.set_yticks(range(k), labels)
    ax.grid(False)
    plotting.style_axes(ax, "Top-1 match: rating the model is told vs. the player's real rating",
                        "player's real rating", "rating the model is told")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, format=lambda v, _: f"{v:.0%}")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=plotting.INK_MUTED, labelcolor=plotting.INK_SECONDARY, labelsize=8)
    return plotting.save_figure(fig, path)


def plot_breakdowns(breakdowns: dict[str, Any], labels: Sequence[str], path: str | Path) -> Path:
    """Top-1 accuracy by game phase and by the mover's clock, one line per rating bucket (light = low rating).

    ``breakdowns`` is :func:`chess_engine.eval.move_matching.breakdowns`'s result.
    """
    panels = (("phase", "By game phase", "move number"), ("clock", "By time left on the mover's clock", "time left"))
    fig, axes = plotting.new_figure(1, 2, width=11.0, height=4.4)
    colors = plotting.bucket_colors(len(labels))
    for ax, (key, title, xlabel) in zip(axes[0], panels):
        bands = list(breakdowns[key])
        x = np.arange(len(bands))
        for b, (label, color) in enumerate(zip(labels, colors)):
            values = _array([breakdowns[key][band].get("bucket_top1", [None] * len(labels))[b] for band in bands])
            if np.isfinite(values).any():
                ax.plot(x, values, color=color, lw=plotting.LINE_WIDTH, marker="o", ms=plotting.MARKER_SIZE - 1,
                        label=label)
        ax.set_xticks(x, bands)
        ax.set_ylim(0, None)
        plotting.style_axes(ax, title, xlabel, "top-1 move match")
        plotting.percent_axis(ax)
    plotting.legend(axes[0][1], loc="upper left", bbox_to_anchor=(1.02, 1.0), title="rating")
    return plotting.save_figure(fig, path)
