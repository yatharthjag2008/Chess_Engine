"""Shared matplotlib style for the report charts (training curves, evaluation plots).

Charts are static PNGs saved to ``reports/``, so they use one light theme:

* categorical series take the palette slots **in fixed order** (``SERIES``): the
  model is always slot 1 (blue), the next series slot 2 (orange), and so on, so a
  series keeps its colour from chart to chart;
* rating buckets are ordered, so they use one hue from light to dark
  (``bucket_colors``): low ratings light, high ratings dark;
* thin 2 px lines, recessive hairline grid, text in neutral ink (never in a series colour),
  one y-axis per chart, a legend whenever there are two or more series.

The hex values are the reference data-viz palette (light mode).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; never needs a display

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Categorical slots in their fixed order (blue, orange, aqua, yellow, magenta, green, violet, red).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
#: Neutral grey for reference lines (chance level, perfect calibration).
REFERENCE = "#898781"

#: Blue ramp for ordered marks, lightest usable step on a light surface (250) to darkest (700).
_BLUE_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
              "#104281", "#0d366b")

LINE_WIDTH = 2.0
MARKER_SIZE = 6.0


def bucket_colors(n: int) -> list[str]:
    """``n`` ordered colours from light (lowest rating) to dark (highest), evenly spread over the ramp."""
    if n <= 0:
        return []
    if n == 1:
        return [_BLUE_RAMP[len(_BLUE_RAMP) // 2]]
    last = len(_BLUE_RAMP) - 1
    return [_BLUE_RAMP[round(i * last / (n - 1))] for i in range(n)]


def style_axes(ax: Axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> Axes:
    """Recessive grid and spines, neutral ink for every text element."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    return ax


def legend(ax: Axes, **kwargs: object) -> None:
    """A frameless legend in secondary ink."""
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, **kwargs)
    if leg is not None:
        leg.set_zorder(5)


def new_figure(rows: int = 1, cols: int = 1, width: float = 7.0, height: float = 4.0) -> tuple[Figure, object]:
    """A figure (``width`` x ``height`` inches per panel row) with every panel styled."""
    fig, axes = plt.subplots(rows, cols, figsize=(width, height * rows), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes if rows * cols > 1 else axes[0, 0]


def percent_axis(ax: Axes, axis: str = "y", decimals: int = 0) -> None:
    """Show a 0..1 axis as percentages."""
    from matplotlib.ticker import PercentFormatter

    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save as PNG (parent directories created) and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
