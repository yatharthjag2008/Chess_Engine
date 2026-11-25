"""Charts and Markdown report for the Elo-dial validation (input: :func:`elo_validation.analyze`'s result)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from chess_engine import plotting

CHARTS = {"dial": "dial_vs_measured.png", "scores": "score_matrix.png"}

#: Diverging blue (bot scores well) <-> red (bot scores badly), neutral grey at 50%.
_DIVERGING = LinearSegmentedColormap.from_list(
    "score", ["#d03b3b", "#ec835a", "#f0efec", "#6da7ec", "#256abf"])


def _label(key: str) -> str:
    kind, value = key.split(":", 1)
    return f"Stockfish {value}" if kind == "sf" else f"bot {value}"


def plot_dial(analysis: dict[str, Any], path: str | Path) -> Path:
    """Measured Elo (with intervals) against the dial setting; Stockfish levels as reference lines."""
    bots = analysis["bots"]
    dial = np.array([b["dial"] for b in bots], dtype=float)
    measured = np.array([b["measured"] for b in bots])
    low = measured - np.array([b["interval"][0] for b in bots])
    high = np.array([b["interval"][1] for b in bots]) - measured
    fig, ax = plotting.new_figure(width=8.5, height=5.0)
    lo, hi = min(dial.min(), measured.min() - low.max()), max(dial.max(), measured.max() + high.max())
    span = np.array([lo - 100, hi + 100])
    ax.plot(span, span, color=plotting.REFERENCE, lw=1.5, ls="--", label="measured = dial setting", zorder=1)
    for elo in analysis["stockfish"]:
        ax.axhline(elo, color=plotting.GRID, lw=1.2, zorder=0)
        ax.annotate(f"Stockfish {elo:.0f}", (span[0], elo), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=plotting.INK_MUTED)
    color = plotting.SERIES[0]
    ax.errorbar(dial, measured, yerr=[low, high], color=color, lw=plotting.LINE_WIDTH, marker="o",
                ms=plotting.MARKER_SIZE + 1, capsize=4, label="bot (95% interval)", zorder=3)
    for x, y in zip(dial, measured):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(8, -4), textcoords="offset points", fontsize=8.5,
                    color=plotting.INK_SECONDARY)
    ax.set_xlim(span)
    ax.set_ylim(span)
    scale = "Stockfish's UCI_Elo scale" if analysis["anchored"] else "relative scale (mean = mean dial)"
    plotting.style_axes(ax, "Does the rating dial change the bot's strength?", "dial setting (rating the bot imitates)",
                        f"measured strength ({scale})")
    plotting.legend(ax, loc="upper left")
    return plotting.save_figure(fig, path)


def plot_scores(analysis: dict[str, Any], path: str | Path) -> Path:
    """Heatmap of each bot's score against each opponent it played."""
    bots = analysis["bots"]
    opponents: list[str] = []
    for b in bots:
        for o in b["scores"]:
            if o not in opponents:
                opponents.append(o)
    opponents.sort(key=lambda k: (k.startswith("bot:"), int(k.split(":")[1])))
    matrix = np.full((len(bots), len(opponents)), np.nan)
    for i, b in enumerate(bots):
        for j, o in enumerate(opponents):
            if o in b["scores"]:
                matrix[i, j] = b["scores"][o]
    fig, ax = plotting.new_figure(width=1.2 * len(opponents) + 3, height=0.55 * len(bots) + 2)
    ax.imshow(matrix, cmap=_DIVERGING, vmin=0, vmax=1, aspect="auto")
    for i in range(len(bots)):
        for j in range(len(opponents)):
            if np.isfinite(matrix[i, j]):
                v = matrix[i, j]
                ink = "#ffffff" if abs(v - 0.5) > 0.3 else plotting.INK
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color=ink)
    ax.set_xticks(range(len(opponents)), [_label(o) for o in opponents], rotation=30, ha="right")
    ax.set_yticks(range(len(bots)), [_label(b["key"]) for b in bots])
    plotting.style_axes(ax, "Bot score against each opponent (blue: bot ahead, red: behind)")
    ax.grid(False)
    return plotting.save_figure(fig, path)


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def render_report(analysis: dict[str, Any], setup: dict[str, Any]) -> str:
    """The Markdown report."""
    bots = analysis["bots"]
    dial = analysis["dial"]
    sf_keys = [f"sf:{e:.0f}" for e in analysis["stockfish"]]
    out = [f"# Elo-dial validation: {setup['model_name']}", ""]
    out += ["## Summary", ""]
    out.append(f"- {analysis['games']:,} games: every bot setting against "
               + (", ".join(f"Stockfish {e:.0f}" for e in analysis["stockfish"]) or "no Stockfish level")
               + f", and bot against bot ({setup['self_play']}).")
    out.append(f"- Rank correlation between the dial and measured strength: **{dial['spearman']:.2f}** "
               f"(1 = strength always rises with the dial). A linear fit gives **{dial['slope'] * 100:.0f} Elo "
               f"of measured strength per 100 points on the dial**.")
    first, last = bots[0], bots[-1]
    out.append(f"- From {first['dial']} to {last['dial']} on the dial, measured strength goes from "
               f"{first['measured']:.0f} to {last['measured']:.0f} "
               f"({last['measured'] - first['measured']:+.0f} Elo for {last['dial'] - first['dial']:+d} on the dial).")
    out.append(f"- {_pct(analysis['adjudicated_share'])} of games were decided by the judge (the bots never resign).")
    out += ["", f"![Dial vs. measured strength]({CHARTS['dial']})", ""]

    out += ["## Measured strength per setting", ""]
    header = ["dial", "measured Elo", "95% interval"] + [f"vs Stockfish {k.split(':')[1]}" for k in sf_keys]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for b in bots:
        cells = [str(b["dial"]), f"{b['measured']:.0f}", f"{b['interval'][0]:.0f} to {b['interval'][1]:.0f}"]
        cells += [_pct(b["scores"][k]) if k in b["scores"] else "-" for k in sf_keys]
        out.append("| " + " | ".join(cells) + " |")
    out += ["", "**Each step up the dial**", "", "| step | measured gap | chance the higher setting is stronger |",
            "|---|---:|---:|"]
    for step in dial["steps"]:
        out.append(f"| {_label(step['from'])} -> {_label(step['to'])} | {step['measured_gap']:+.0f} | "
                   f"{_pct(step['prob_stronger'])} |")
    out += ["", f"![Scores]({CHARTS['scores']})", ""]

    out += ["## Every pairing", "", "| pairing | games | +/=/- (first player) | score | judged | avg. plies |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in analysis["pairings"]:
        out.append(f"| {_label(row['a'])} vs {_label(row['b'])} | {row['games']} | "
                   f"{row['wins']}/{row['draws']}/{row['losses']} | {_pct(row['score'])} | {row['adjudicated']} | "
                   f"{row['average_plies']:.0f} |")
    out += ["", "**How games ended:** " + ", ".join(f"{k} {v}" for k, v in analysis["terminations"].items()), ""]

    out += ["## Setup and caveats", "",
            f"- Model `{setup['model_path']}`; bots sample with temperature {setup['temperature']:g}.",
            f"- Stockfish {setup.get('stockfish_version', '')} limited with `UCI_LimitStrength` / `UCI_Elo`, "
            f"{setup['stockfish_move_seconds'] * 1000:.0f} ms per move. Its Elo scale was calibrated by the Stockfish "
            "team at longer time controls against engines; it is **not** the Lichess blitz scale the bot was trained "
            "on, so measured numbers are comparable with each other, not with Lichess ratings.",
            f"- A judge (full Stockfish, {setup['judge_nodes']:,} nodes) ends a game when one side has been ahead by "
            f"{setup['adjudicate_cp']} centipawns for {setup['adjudicate_plies']} plies in a row; after "
            f"{setup['max_plies']} plies its last evaluation decides.",
            f"- Intervals: {setup['bootstrap']} bootstrap resamples of the games within each pairing.",
            f"- Run time: {setup['seconds'] / 60:.0f} minutes.", ""]
    return "\n".join(out)
