"""Tests for reading a run's history back and plotting its curves (and the shared plot style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_engine import plotting
from chess_engine.training.history import load_history, plot_training_curves, summarize_history
from chess_engine.training.run_log import MetricsLog
from chess_engine.training.train import Trainer
from training_helpers import train_config


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory, tiny_dataset: Path) -> Path:
    tmp_path = tmp_path_factory.mktemp("history")
    cfg = train_config(tmp_path, max_steps=30, eval_every_steps=10, log_every_steps=5)
    Trainer(cfg, tmp_path / "run", tiny_dataset, log=lambda _: None).train()
    return tmp_path / "run"


def test_history_arrays(trained_run: Path) -> None:
    history = load_history(trained_run)
    assert history.train["step"].tolist() == [5, 10, 15, 20, 25, 30]
    assert history.val["step"].tolist() == [10, 20, 30]
    assert np.isfinite(history.train["loss"]).all() and np.isfinite(history.val["top1"]).all()
    assert len(history.val_buckets) == 8
    assert all(len(b["top1"]) == 3 for b in history.val_buckets.values())
    summary = summarize_history(history)
    assert summary["steps"] == 30 and summary["best_val_step"] in (10, 20, 30)


def test_plot_is_written(trained_run: Path, tmp_path: Path) -> None:
    out = plot_training_curves(load_history(trained_run), tmp_path / "plots" / "curves.png")
    assert out.is_file() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_empty_or_partial_runs(tmp_path: Path) -> None:
    history = load_history(tmp_path / "nothing")
    assert history.empty
    with pytest.raises(ValueError, match="no training records"):
        plot_training_curves(history, tmp_path / "x.png")
    MetricsLog(tmp_path / "early" / "metrics.jsonl").write("train", 5, loss=3.0, top1=0.1, lr=1e-4)
    early = load_history(tmp_path / "early")
    assert not early.empty and len(early.val["step"]) == 0
    plot_training_curves(early, tmp_path / "early.png")  # no validation yet: still plots
    assert "best_val_step" not in summarize_history(early)


def test_bucket_colors_are_ordered_light_to_dark() -> None:
    from matplotlib.colors import to_rgb

    colors = plotting.bucket_colors(8)
    assert len(set(colors)) == 8
    lightness = [sum(to_rgb(c)) for c in colors]
    assert lightness == sorted(lightness, reverse=True)
    assert plotting.bucket_colors(0) == [] and len(plotting.bucket_colors(1)) == 1
