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
