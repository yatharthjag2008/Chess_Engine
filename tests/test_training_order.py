"""Tests for the per-epoch position order (shuffled or resampled by bucket)."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.training.balancing import bucket_weights
from chess_engine.training.order import (
    batch_slices,
    batches_per_epoch,
    epoch_order,
    resample_counts,
    resampled_order,
    shuffled_order,
)


def test_shuffle_is_a_permutation_that_depends_on_seed_and_epoch() -> None:
    a = shuffled_order(1000, seed=1, epoch=0)
    assert sorted(a.tolist()) == list(range(1000))
    assert np.array_equal(a, shuffled_order(1000, seed=1, epoch=0))
    assert not np.array_equal(a, shuffled_order(1000, seed=1, epoch=1))
    assert not np.array_equal(a, shuffled_order(1000, seed=2, epoch=0))


def test_resample_counts() -> None:
    assert resample_counts(np.array([10, 0, 30]), np.array([2.0, 5.0, 0.01])).tolist() == [20, 0, 1]


def test_resampled_epoch_matches_the_weights() -> None:
    rng = np.random.default_rng(0)
    buckets = rng.choice(4, size=20_000, p=[0.05, 0.15, 0.5, 0.3])
    counts = np.bincount(buckets)
    weights = bucket_weights(counts, 1.0)
    order = resampled_order(buckets, weights, seed=3, epoch=0)
    drawn = np.bincount(buckets[order], minlength=4)
    assert drawn.tolist() == resample_counts(counts, weights).tolist()
    assert abs(len(order) - len(buckets)) <= 4  # epoch size stays the same (up to rounding)
    assert np.all(np.abs(drawn - len(buckets) / 4) <= 2)


def test_oversampled_bucket_uses_every_position_before_repeating() -> None:
    buckets = np.array([0] * 10 + [1] * 90)
    order = resampled_order(buckets, np.array([3.0, 0.5]), seed=0, epoch=0)
    rare = order[buckets[order] == 0]
    assert len(rare) == 30
    assert np.bincount(rare, minlength=10).tolist() == [3] * 10  # exactly 3 times each
    common = order[buckets[order] == 1]
    assert len(common) == 45 and len(set(common.tolist())) == 45  # a subset, no repeats


def test_undersampled_bucket_uses_different_positions_each_epoch() -> None:
    buckets = np.array([0] * 10 + [1] * 1000)
    weights = np.array([1.0, 0.1])
    first = set(resampled_order(buckets, weights, 0, 0).tolist())
    second = set(resampled_order(buckets, weights, 0, 1).tolist())
    assert first != second
    assert np.array_equal(resampled_order(buckets, weights, 0, 1), resampled_order(buckets, weights, 0, 1))


def test_epoch_order_dispatch_and_errors() -> None:
    assert np.array_equal(epoch_order(50, 1, 2), shuffled_order(50, 1, 2))
    buckets = np.zeros(50, dtype=np.int64)
    assert sorted(epoch_order(50, 1, 2, buckets, np.array([1.0])).tolist()) == list(range(50))
    with pytest.raises(ValueError, match="both"):
        epoch_order(50, 1, 2, buckets=buckets)
    with pytest.raises(ValueError, match="positions"):
        epoch_order(40, 1, 2, buckets, np.array([1.0]))
    with pytest.raises(ValueError, match="weight"):
        resampled_order(np.array([0, 3]), np.array([1.0]), 0, 0)


def test_batch_slices_and_resume_offset() -> None:
    order = np.arange(10)
    assert [b.tolist() for b in batch_slices(order, 4)] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [b.tolist() for b in batch_slices(order, 4, drop_last=False)][-1] == [8, 9]
    assert [b.tolist() for b in batch_slices(order, 4, start_batch=1)] == [[4, 5, 6, 7]]
    assert batches_per_epoch(10, 4) == 2 and batches_per_epoch(10, 4, drop_last=False) == 3
    with pytest.raises(ValueError):
        batch_slices(order, 0)
