"""Read a run's metrics log back as arrays, and plot its training curves.

    history = load_history("checkpoints/laptop_64x6")
    history.train["step"], history.train["loss"], history.val["top1"]
    plot_training_curves(history, "reports/training/laptop_64x6.png")

The figure has four panels, each with a single y-axis: loss (train and
validation), top-1 move match (train and validation), validation top-1 per
rating bucket, and the learning rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_engine import plotting
from chess_engine.training.run_log import MetricsLog, RunDirectory


@dataclass(frozen=True)
class History:
    """Logged metrics of one run as numpy arrays (NaN where a value was not logged)."""

    run: str
    train: dict[str, np.ndarray]                   # step, loss, policy_loss, value_loss, top1, lr, ...
    val: dict[str, np.ndarray]                     # step, loss, top1, top5, value_accuracy, ...
    val_buckets: dict[str, dict[str, np.ndarray]]  # bucket label -> {"top1": ..., "nll": ...} at val steps

    @property
    def empty(self) -> bool:
        return len(self.train.get("step", [])) == 0


def _column(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r.get(key) if r.get(key) is not None else np.nan for r in records], dtype=np.float64)


def load_history(run_dir: str | Path) -> History:
    """Everything in ``<run_dir>/metrics.jsonl`` (a run in progress can be read too)."""
    run = RunDirectory(run_dir)
    log = MetricsLog(run.metrics.path)
    train = log.records("train")
    val = log.records("val")
    train_keys = ("step", "loss", "policy_loss", "value_loss", "top1", "lr", "grad_norm", "positions_per_second")
    val_columns = {key: _column(val, key) for key in ("step", "loss", "policy_loss", "value_loss")}
    for key in ("top1", "top3", "top5", "nll", "value_accuracy"):
        val_columns[key] = np.array([r.get("overall", {}).get(key, np.nan) for r in val], dtype=np.float64)
    labels = list(val[-1]["buckets"]) if val else []
    buckets = {
        label: {key: np.array([r["buckets"].get(label, {}).get(key, np.nan) for r in val], dtype=np.float64)
                for key in ("top1", "nll")}
        for label in labels
    }
    return History(run.name, {key: _column(train, key) for key in train_keys}, val_columns, buckets)


def plot_training_curves(history: History, path: str | Path) -> Path:
    """Save the four-panel training-curve figure as a PNG."""
    if history.empty:
        raise ValueError(f"run {history.run} has no training records to plot")
    fig, axes = plotting.new_figure(2, 2, width=11.0, height=3.8)
    (loss_ax, top1_ax), (bucket_ax, lr_ax) = axes
    train_color, val_color = plotting.SERIES[0], plotting.SERIES[1]
    t, v = history.train, history.val
    lw = plotting.LINE_WIDTH

    loss_ax.plot(t["step"], t["loss"], color=train_color, lw=lw, label="train")
    if len(v["step"]):
        loss_ax.plot(v["step"], v["loss"], color=val_color, lw=lw, marker="o", ms=plotting.MARKER_SIZE,
                     label="validation")
    plotting.style_axes(loss_ax, "Loss", "step", "cross-entropy")
    plotting.legend(loss_ax)

    top1_ax.plot(t["step"], t["top1"], color=train_color, lw=lw, label="train")
    if len(v["step"]):
        top1_ax.plot(v["step"], v["top1"], color=val_color, lw=lw, marker="o", ms=plotting.MARKER_SIZE,
                     label="validation")
    plotting.style_axes(top1_ax, "Top-1 move match", "step", "positions")
    plotting.percent_axis(top1_ax)
    plotting.legend(top1_ax)

    labels = list(history.val_buckets)
    for color, label in zip(plotting.bucket_colors(len(labels)), labels):
        bucket_ax.plot(v["step"], history.val_buckets[label]["top1"], color=color, lw=lw, marker="o",
                       ms=plotting.MARKER_SIZE - 1, label=label)
    plotting.style_axes(bucket_ax, "Validation top-1 by rating", "step", "positions")
    plotting.percent_axis(bucket_ax)
    if labels:
        plotting.legend(bucket_ax, ncol=2, loc="lower right")

    lr_ax.plot(t["step"], t["lr"], color=train_color, lw=lw)
    plotting.style_axes(lr_ax, "Learning rate", "step", "learning rate")
    lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.suptitle(f"Training run {history.run}", x=0.02, ha="left", color=plotting.INK, fontsize=12)
    return plotting.save_figure(fig, path)


def summarize_history(history: History) -> dict[str, Any]:
    """Headline numbers of a run: steps, last train loss, best validation loss/top-1 and where."""
    out: dict[str, Any] = {"run": history.run, "steps": int(np.nanmax(history.train["step"]))
                           if not history.empty else 0}
    if not history.empty:
        out["last_train_loss"] = float(history.train["loss"][-1])
    if len(history.val["step"]):
        best = int(np.nanargmin(history.val["loss"]))
        out.update(best_val_step=int(history.val["step"][best]), best_val_loss=float(history.val["loss"][best]),
                   best_val_top1=float(history.val["top1"][best]))
    return out
