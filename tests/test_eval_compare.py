"""Tests for comparing models on the same positions."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.eval.compare import compare_tables, format_comparison
from eval_helpers import LABELS, make_table


def _pair(n: int = 400, better_by: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    base_hit = rng.random(n) < 0.4
    improved = base_hit | (rng.random(n) < better_by)
    buckets = rng.integers(0, 2, n).tolist()  # bucket "high" stays empty
    games = (np.arange(n) // 10).tolist()
    a = make_table(rank=np.where(base_hit, 1, 2).tolist(), prob=[0.5] * n, bucket=buckets, game=games)
    b = make_table(rank=np.where(improved, 1, 2).tolist(), prob=[0.5] * n, bucket=buckets, game=games)
    return a, b, improved.mean() - base_hit.mean()


def test_paired_difference_and_table() -> None:
    a, b, expected = _pair()
    result = compare_tables({"old": a, "new": b}, LABELS, samples=300)
    diff = result["differences"]["new"]["overall"]
    assert diff["difference"] == pytest.approx(expected)
    assert 0 < diff["ci"][0] <= diff["difference"] <= diff["ci"][1]
    assert "high" not in result["differences"]["new"]["buckets"]
    assert result["models"]["old"]["overall"]["top1"] == pytest.approx(a.top1_hit.mean())
    text = format_comparison(result)
    assert "new - old" in text and "pts" in text and "all" in text and "paired" in text


def test_identical_models_differ_by_zero() -> None:
    a, _, _ = _pair()
    result = compare_tables({"a": a, "b": a}, LABELS, samples=100)
    assert result["differences"]["b"]["overall"] == {"difference": 0.0, "ci": [0.0, 0.0]}


def test_errors() -> None:
    a, b, _ = _pair()
    with pytest.raises(ValueError, match="at least two"):
        compare_tables({"a": a}, LABELS)
    with pytest.raises(ValueError, match="different positions"):
        compare_tables({"a": a, "b": b.take(np.arange(10))}, LABELS)
