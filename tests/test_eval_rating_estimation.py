"""Tests for rating estimation from per-move likelihoods."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.config import RatingEstimationConfig, RatingsConfig
from chess_engine.eval.rating_estimation import (
    bucket_centers,
    estimate_from_bucket_ll,
    grid_log_likelihood,
    posterior,
    rating_grid,
    summarize,
)

RATINGS = RatingsConfig(bucket_edges=[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200])
CFG = RatingEstimationConfig()


def test_bucket_centers_and_grid() -> None:
    assert bucket_centers(RATINGS.bucket_edges).tolist() == [900, 1100, 1300, 1500, 1700, 1900, 2100, 2300]
    assert bucket_centers([1000]).tolist() == [1100]
    grid = rating_grid(CFG)
    assert grid[0] == 600 and grid[-1] == 2800 and np.all(np.diff(grid) == 10)


def test_interpolation_is_linear_between_centres_and_flat_outside() -> None:
    centers = np.array([900.0, 1100.0, 1300.0])
    ll = np.array([-10.0, -6.0, -8.0])
    out = grid_log_likelihood(ll, centers, np.array([600, 900, 1000, 1100, 1250, 1600]))
    assert out.tolist() == pytest.approx([-10, -10, -8, -6, -7.5, -8])
    two = grid_log_likelihood(np.stack([ll, ll * 2]), centers, np.array([1000.0]))
    assert two.shape == (2, 1) and two[:, 0].tolist() == pytest.approx([-8, -16])


def test_posterior_normalizes_and_tempering_widens() -> None:
    grid = np.linspace(0, 1, 11)
    ll = -50 * (grid - 0.5) ** 2
    sharp = posterior(ll, np.zeros(11))
    wide = posterior(ll, np.zeros(11), temper=4)
    assert sharp.sum() == pytest.approx(1) and wide.sum() == pytest.approx(1)
    assert sharp.argmax() == 5 and wide.max() < sharp.max()
    with pytest.raises(ValueError):
        posterior(ll, np.zeros(11), temper=0)
    huge = posterior(ll * 1e6, np.zeros(11))  # no overflow
    assert np.isfinite(huge).all() and huge[5] == pytest.approx(1)


def test_summarize_a_symmetric_posterior() -> None:
    grid = np.arange(1000, 2001, 10, dtype=float)
    probs = np.exp(-0.5 * ((grid - 1500) / 100) ** 2)
    est = summarize(grid, probs / probs.sum(), interval=0.8, moves=40)
    assert est.estimate == pytest.approx(1500, abs=1) and est.median == pytest.approx(1500, abs=6)
    assert est.low == pytest.approx(1500 - 128, abs=12) and est.high == pytest.approx(1500 + 128, abs=12)
    assert "about 1500" in est.describe() and "from 40 moves" in est.describe()


def test_estimate_follows_the_best_explaining_bucket() -> None:
    # 30 moves that players around 1700 (bucket 4) play much more often than other ratings
    per_move = np.full((30, 8), -3.0)
    per_move[:, 4] = -2.0
    per_move[:, 3] = per_move[:, 5] = -2.4
    est = estimate_from_bucket_ll(per_move, RATINGS, CFG)
    assert 1600 < est.estimate < 1800 and est.low < est.estimate < est.high and est.moves == 30
    wider = estimate_from_bucket_ll(per_move, RATINGS, CFG, temper=10)
    assert (wider.high - wider.low) > (est.high - est.low)
    prior = np.where(rating_grid(CFG) < 1000, 0.0, -1e9)  # a prior that rules out everything above 1000
    assert estimate_from_bucket_ll(per_move, RATINGS, CFG, log_prior=prior).estimate < 1000
