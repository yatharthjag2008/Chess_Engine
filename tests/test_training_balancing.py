"""Tests for rating-bucket balancing weights."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.training.balancing import bucket_weights, effective_shares, format_balance, position_weights

# Positions per bucket in the real 50k-game training split (step 4).
REAL_COUNTS = [113058, 253301, 405719, 599078, 677653, 624911, 365906, 151907]


def test_power_zero_keeps_the_data_as_it_is() -> None:
    w = bucket_weights(REAL_COUNTS, 0.0)
    assert w == pytest.approx(np.ones(8))


def test_power_one_makes_every_bucket_count_equally() -> None:
    w = bucket_weights(REAL_COUNTS, 1.0)
    assert effective_shares(REAL_COUNTS, w) == pytest.approx(np.full(8, 1 / 8))


@pytest.mark.parametrize("power", [0.0, 0.3, 0.5, 1.0])
def test_average_weight_per_position_is_one(power: float) -> None:
    counts = np.array(REAL_COUNTS)
    w = bucket_weights(counts, power)
    assert (counts * w).sum() == pytest.approx(counts.sum())


def test_half_power_is_in_between() -> None:
    w = bucket_weights(REAL_COUNTS, 0.5)
    assert w[0] > w[1] > 1 > w[4]
    share = effective_shares(REAL_COUNTS, w)
    data_share = np.array(REAL_COUNTS) / sum(REAL_COUNTS)
    assert data_share[0] < share[0] < 1 / 8
    assert share.max() / share.min() == pytest.approx(np.sqrt(max(REAL_COUNTS) / min(REAL_COUNTS)))


def test_empty_buckets_get_zero_weight() -> None:
    w = bucket_weights([100, 0, 300], 1.0)
    assert w[1] == 0
    assert effective_shares([100, 0, 300], w) == pytest.approx([0.5, 0, 0.5])


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="power"):
        bucket_weights([1, 2], 1.5)
    with pytest.raises(ValueError, match="empty"):
        bucket_weights([0, 0], 0.5)
    with pytest.raises(ValueError, match="non-negative"):
        bucket_weights([1, -1], 0.5)


def test_position_weights_and_table() -> None:
    w = bucket_weights([10, 30], 1.0)
    pw = position_weights(np.array([0, 1, 1, 0], dtype=np.int8), w)
    assert pw.dtype == np.float32 and pw.tolist() == pytest.approx([w[0], w[1], w[1], w[0]])
    table = format_balance(["low", "high"], [10, 30], w)
    assert "low" in table and "50.0%" in table and "25.0%" in table
