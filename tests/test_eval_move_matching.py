"""Tests for move-matching accuracy per rating bucket."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chess_engine.eval.move_matching import format_move_matching, group_means, match_stats, move_matching
from eval_helpers import make_table


def test_group_means() -> None:
    means, counts = group_means(np.array([1.0, 0.0, 1.0, 1.0]), np.array([0, 0, 2, 2]), 3)
    assert counts.tolist() == [2, 0, 2]
    assert means[0] == 0.5 and np.isnan(means[1]) and means[2] == 1.0


def test_match_stats() -> None:
    table = make_table(rank=[1, 2, 4, 9], prob=[0.5, 0.25, 0.1, 0.05], bucket=[0, 0, 0, 0], legal=[10, 20, 40, 10])
    s = match_stats(table)
    assert s["positions"] == 4
    assert s["top1"] == 0.25 and s["top3"] == 0.5 and s["top5"] == 0.75
    expected = np.mean(np.log([0.5, 0.25, 0.1, 0.05]))
    assert s["log_likelihood"] == pytest.approx(expected, rel=1e-5)
    assert s["perplexity"] == pytest.approx(math.exp(-expected), rel=1e-5)
    assert s["mean_legal_moves"] == 20 and s["random_top1"] == pytest.approx((0.1 + 0.05 + 0.025 + 0.1) / 4)
    assert match_stats(table, np.zeros(4, dtype=bool)) == {"positions": 0}


def test_per_bucket_and_bucket_mean() -> None:
    table = make_table(rank=[1, 1, 1, 2, 1, 3], prob=[0.5] * 6, bucket=[0, 0, 0, 0, 2, 2])
    result = move_matching(table)
    assert result["buckets"]["low"]["top1"] == 0.75 and result["buckets"]["high"]["top1"] == 0.5
    assert result["buckets"]["mid"] == {"positions": 0}
    assert result["overall"]["top1"] == pytest.approx(4 / 6)
    assert result["bucket_mean_top1"] == pytest.approx(0.625)  # every bucket with data counts equally
    table_text = format_move_matching(result)
    assert "low" in table_text and "75.0%" in table_text and "all" in table_text


def test_zero_probability_does_not_give_minus_infinity() -> None:
    s = match_stats(make_table(rank=[30], prob=[0.0], bucket=[0]))
    assert np.isfinite(s["log_likelihood"])
