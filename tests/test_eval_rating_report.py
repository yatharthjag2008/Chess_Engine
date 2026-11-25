"""Tests for the rating-estimation charts and report."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chess_engine.config import Config, RatingEstimationConfig
from chess_engine.eval.rating_eval import convergence, estimate_all, score
from chess_engine.eval.rating_report import plot_convergence, plot_error_by_rating, plot_scatter
from test_eval_rating_eval import _synthetic

PNG = b"\x89PNG\r\n\x1a\n"


def _results(default_config: Config):
    cfg = RatingEstimationConfig(prior="flat")
    data = _synthetic()
    est = estimate_all(data, default_config.ratings, cfg, np.zeros(221), temper=3.0)
    rows = convergence(data, default_config.ratings, cfg, np.zeros(221), 3.0, steps=(5, 20))
    return data, est, score(est, data, default_config.ratings), rows


def test_charts(tmp_path: Path, default_config: Config) -> None:
    data, est, scores, rows = _results(default_config)
    assert plot_scatter(est, data, tmp_path / "s.png").read_bytes()[:8] == PNG
    assert plot_error_by_rating(scores, tmp_path / "e.png").read_bytes()[:8] == PNG
    assert plot_convergence(rows, tmp_path / "c.png").read_bytes()[:8] == PNG
