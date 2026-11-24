"""Tests for the game-clustered bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.eval.bootstrap import cluster_bootstrap


def test_interval_contains_the_mean_and_shrinks_with_more_data() -> None:
    rng = np.random.default_rng(0)
    small = rng.random(200) < 0.4
    big = rng.random(20_000) < 0.4
    ci_small = cluster_bootstrap(small, np.arange(200), samples=500)
    ci_big = cluster_bootstrap(big, np.arange(20_000), samples=500)
    assert ci_small.overall[0] < small.mean() < ci_small.overall[1]
    assert ci_big.overall[0] < big.mean() < ci_big.overall[1]
    width = lambda ci: ci.overall[1] - ci.overall[0]  # noqa: E731
    assert width(ci_big) < width(ci_small) / 5
    # For independent positions it matches the textbook binomial interval.
    assert width(ci_big) == pytest.approx(2 * 1.96 * np.sqrt(0.24 / 20_000), rel=0.15)


def test_correlated_positions_widen_the_interval() -> None:
    """Whole games of identical outcomes: the effective sample size is the number of games."""
    rng = np.random.default_rng(1)
    games = np.repeat(np.arange(100), 50)
    values = np.repeat(rng.random(100) < 0.5, 50).astype(float)
    clustered = cluster_bootstrap(values, games, samples=500)
    naive = cluster_bootstrap(values, np.arange(len(values)), samples=500)
    assert (clustered.overall[1] - clustered.overall[0]) > 4 * (naive.overall[1] - naive.overall[0])


def test_groups_and_empty_groups() -> None:
    values = np.array([1, 1, 0, 0, 1, 0], dtype=float)
    groups = np.array([0, 0, 0, 2, 2, 2])
    games = np.array([0, 1, 2, 3, 4, 5])
    ci = cluster_bootstrap(values, games, groups, num_groups=3, samples=300, seed=2)
    assert len(ci.groups) == 3 and all(np.isnan(ci.groups[1]))
    assert 0 <= ci.groups[0][0] <= 2 / 3 <= ci.groups[0][1] <= 1
    again = cluster_bootstrap(values, games, groups, num_groups=3, samples=300, seed=2)
    np.testing.assert_array_equal(np.array(again.groups), np.array(ci.groups))  # seeded
    assert ci.confidence == 0.95 and ci.samples == 300


def test_paired_difference() -> None:
    rng = np.random.default_rng(3)
    a = rng.random(5000) < 0.5
    b = a & (rng.random(5000) < 0.9)  # b loses ~5 points, on the same positions
    ci = cluster_bootstrap(a.astype(float) - b, np.arange(5000), samples=400)
    assert 0 < ci.overall[0] < 0.05 < ci.overall[1] < 0.1


def test_edge_cases() -> None:
    ci = cluster_bootstrap(np.array([]), np.array([]), samples=10)
    assert all(np.isnan(ci.overall))
    assert all(np.isnan(cluster_bootstrap(np.ones(3), np.arange(3), samples=0).overall))
    with pytest.raises(ValueError):
        cluster_bootstrap(np.ones(3), np.arange(2))
    with pytest.raises(ValueError):
        cluster_bootstrap(np.ones(3), np.arange(3), confidence=1.0)
