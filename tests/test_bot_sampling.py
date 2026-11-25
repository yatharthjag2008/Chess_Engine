"""Tests for the bot's sampling rules (temperature, top-k, top-p, min probability)."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.bot.sampling import SamplingSettings, adjust_probabilities, sample_index

P = np.array([0.1, 0.5, 0.25, 0.1, 0.05])


def test_default_keeps_the_model_probabilities() -> None:
    np.testing.assert_allclose(adjust_probabilities(P), P)
    np.testing.assert_allclose(adjust_probabilities(P * 3), P)  # renormalized


def test_temperature() -> None:
    sharp = adjust_probabilities(P, SamplingSettings(temperature=0.5))
    np.testing.assert_allclose(sharp, P**2 / (P**2).sum())
    flat = adjust_probabilities(P, SamplingSettings(temperature=4.0))
    assert flat.max() < P.max() and flat.min() > P.min()
    greedy = adjust_probabilities(P, SamplingSettings(temperature=0))
    assert greedy.tolist() == [0, 1, 0, 0, 0]
    tie = adjust_probabilities(np.array([0.4, 0.4, 0.2]), SamplingSettings(temperature=0))
    assert tie.tolist() == [1, 0, 0]  # ties: the first one


def test_tiny_temperature_does_not_overflow() -> None:
    out = adjust_probabilities(P, SamplingSettings(temperature=1e-4))
    assert np.isfinite(out).all() and out[1] == pytest.approx(1.0)


def test_top_k_top_p_and_min_prob() -> None:
    assert np.flatnonzero(adjust_probabilities(P, SamplingSettings(top_k=2))).tolist() == [1, 2]
    # likeliest first: 0.5, 0.25 (cum 0.75), 0.1 -> crosses 0.8 at the third move, which is kept
    kept = np.flatnonzero(adjust_probabilities(P, SamplingSettings(top_p=0.8)))
    assert sorted(kept.tolist()) == [0, 1, 2]
    np.testing.assert_allclose(adjust_probabilities(P, SamplingSettings(top_p=0.75))[[1, 2]], [2 / 3, 1 / 3])
    out = adjust_probabilities(P, SamplingSettings(min_prob=0.1))
    assert out[4] == 0 and out[0] > 0 and out.sum() == pytest.approx(1)


def test_filters_never_remove_every_move() -> None:
    out = adjust_probabilities(P, SamplingSettings(min_prob=0.9, top_p=0.01, top_k=1))
    assert out.tolist() == [0, 1, 0, 0, 0]


def test_filters_apply_before_temperature() -> None:
    out = adjust_probabilities(P, SamplingSettings(top_k=2, temperature=0.5))
    np.testing.assert_allclose(out[[1, 2]], np.array([0.25, 0.0625]) / 0.3125)


@pytest.mark.parametrize("bad", [dict(temperature=-1), dict(top_k=0), dict(top_p=0), dict(top_p=1.5),
                                 dict(min_prob=1.0)])
def test_invalid_settings(bad: dict) -> None:
    with pytest.raises(ValueError):
        SamplingSettings(**bad)


def test_invalid_probabilities() -> None:
    for probs in (np.array([]), np.zeros(3), np.array([0.5, -0.1]), np.array([np.nan, 1.0])):
        with pytest.raises(ValueError):
            adjust_probabilities(probs)


def test_sampling_follows_the_probabilities() -> None:
    rng = np.random.default_rng(0)
    draws = np.bincount([sample_index(P, rng) for _ in range(20_000)], minlength=5) / 20_000
    np.testing.assert_allclose(draws, P, atol=0.01)
