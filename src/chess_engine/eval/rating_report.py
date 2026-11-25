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


def _row(label: str, b: dict[str, Any]) -> str:
    return (f"| {label} | {b['players']:,} | {b['mae']:.0f} | {b['bias']:+.0f} | {b['coverage']:.0%} | "
            f"{b['mean_width']:.0f} | {b['baseline_opponent']['mae']:.0f} | {b['baseline_average']['mae']:.0f} |")


def render_report(scores: dict[str, Any], rows: list[dict[str, Any]], temper_fit: dict[str, Any],
                  setup: dict[str, Any]) -> str:
    """The Markdown report."""
    o = scores["overall"]
    interval = f"{scores['interval']:.0%}"
    beats_opp = o["mae"] < o["baseline_opponent"]["mae"]
    out = [f"# Rating estimation: {setup['model_name']}", "", "## Summary", ""]
    out.append(f"- From **one game** ({o['mean_moves']:.0f} moves on average), the estimate is off by "
               f"**{o['mae']:.0f} Elo** on average (median {o['median_abs_error']:.0f}, bias {o['bias']:+.0f}), "
               f"over {o['players']:,} held-out player-games.")
    out.append(f"- Baselines: guessing the opponent's rating is off by {o['baseline_opponent']['mae']:.0f}, "
               f"guessing the average rating ({scores['mean_rating']:.0f}) by {o['baseline_average']['mae']:.0f}. "
               + ("The estimate beats both." if beats_opp else
                  "The estimate beats the average but **not** the opponent's rating: when the opponent's rating is "
                  "known, it is the better single guess."))
    out.append(f"- The {interval} ranges contain the true rating {o['coverage']:.0%} of the time "
               f"(average width {o['mean_width']:.0f} Elo), after tempering the evidence by "
               f"{temper_fit['temper']:.1f} (fitted on {setup['calibration_split']} games).")
    first = rows[0]
    last = rows[-1]
    out.append(f"- More moves help: after {first['max_moves']} moves the error is {first['mae']:.0f} and the range "
               f"{first['mean_width']:.0f} wide; with every move of the game, {last['mae']:.0f} and "
               f"{last['mean_width']:.0f}.")
    out += ["", f"![Estimated vs. true]({CHARTS['scatter']})", "", "## By true rating", "",
            f"| true rating | player-games | error | bias | in {interval} range | range width | opponent's rating | average |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, b in scores["buckets"].items():
        if b.get("players"):
            out.append(_row(label, b))
    out.append(_row("**all**", o))
    out += ["", "Errors are mean absolute errors in Elo; bias is estimate minus truth (negative = estimates too low).",
            "", f"![Error by rating]({CHARTS['error']})", "", "## More moves, better estimates", "",
            "| moves seen | error | in range | range width |", "|---|---:|---:|---:|"]
    for r in rows:
        label = "all" if r["max_moves"] is None else f"first {r['max_moves']}"
        out.append(f"| {label} (avg. {r['mean_moves']:.0f}) | {r['mae']:.0f} | {r['coverage']:.0%} | "
                   f"{r['mean_width']:.0f} |")
    out += ["", f"![Convergence]({CHARTS['convergence']})", "", "## Setup", "",
            f"- Model `{setup['model_path']}`, scored on the {setup['split']} split; each estimate uses one side of one "
            "game, with the opponent's real rating and both clocks given to the model.",
            f"- Prior: {setup['prior']}; rating grid {setup['rating_min']}-{setup['rating_max']}; "
            f"temper {temper_fit['temper']:.2f} (moves within a game are correlated evidence, so their summed "
            "log-likelihood is divided by it to keep the ranges honest).", ""]
    return "\n".join(out)
