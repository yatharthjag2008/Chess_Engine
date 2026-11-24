"""Tests for the evaluation charts (they are written, readable PNGs; colours follow the series)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine import plotting
from chess_engine.eval.plots import Series, plot_accuracy_by_rating, series_color

PNG = b"\x89PNG\r\n\x1a\n"
LABELS = ["800-999", "1000-1199", "1200-1399"]


def test_series_colours_are_fixed_per_predictor() -> None:
    assert series_color("model") == plotting.SERIES[0]
    assert series_color("Stockfish depth 10") == plotting.SERIES[2]
    assert series_color("most common move") == plotting.SERIES[1]
    assert series_color("most common move (same rating)") == plotting.SERIES[4]
    with pytest.raises(KeyError):
        series_color("mystery")


def test_accuracy_chart(tmp_path: Path) -> None:
    series = [
        Series("model", [0.40, 0.42, 0.45], [0.38, 0.40, 0.43], [0.42, 0.44, 0.47]),
        Series("most common move", [0.20, None, 0.22]),
        Series("Stockfish depth 10", [0.30, 0.33, 0.36]),
    ]
    out = plot_accuracy_by_rating(LABELS, series, tmp_path / "a" / "accuracy.png", random=[0.05, 0.05, 0.05])
    assert out.read_bytes()[:8] == PNG
    empty = plot_accuracy_by_rating(LABELS, [Series("model", [None, None, None])], tmp_path / "empty.png")
    assert empty.is_file()


def test_calibration_chart(tmp_path: Path) -> None:
    import numpy as np

    from chess_engine.eval.calibration import binned_reliability
    from chess_engine.eval.plots import plot_calibration

    rng = np.random.default_rng(0)
    p = rng.random(5000)
    calibration = {"all_moves": binned_reliability(p, rng.random(5000) < p),
                   "top1": binned_reliability(p[:100] * 0.5 + 0.5, rng.random(100) < 0.6)}
    assert plot_calibration(calibration, tmp_path / "cal.png").read_bytes()[:8] == PNG


def test_cross_rating_heatmap(tmp_path: Path) -> None:
    from chess_engine.eval.plots import plot_cross_rating

    cross = {"labels": LABELS, "top1": [[0.40, 0.35, None], [0.38, 0.41, None], [0.30, 0.39, None]]}
    assert plot_cross_rating(cross, tmp_path / "cross.png").read_bytes()[:8] == PNG
    flat = {"labels": LABELS[:1], "top1": [[0.4]]}
    assert plot_cross_rating(flat, tmp_path / "flat.png").is_file()


def test_breakdown_chart(tmp_path: Path) -> None:
    from chess_engine.eval.move_matching import breakdowns
    from chess_engine.eval.plots import plot_breakdowns
    from eval_helpers import LABELS as BUCKETS
    from eval_helpers import make_table

    table = make_table(rank=[1, 2, 1, 1, 3, 1], prob=[0.5] * 6, bucket=[0, 0, 1, 1, 2, 2],
                       ply=[0, 30, 60, 90, 10, 100], clock=[170.0, 100.0, 50.0, 10.0, 175.0, 5.0])
    out = plot_breakdowns(breakdowns(table), BUCKETS, tmp_path / "b.png")
    assert out.read_bytes()[:8] == PNG
