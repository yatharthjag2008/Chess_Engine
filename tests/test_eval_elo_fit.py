"""Tests for the Elo fit (synthetic games from known ratings)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from chess_engine.eval.elo_fit import (
    Result,
    bootstrap_ratings,
    dial_summary,
    expected_score,
    fit_ratings,
    spearman,
)


def simulate(true: dict[str, float], pairs, games: int, seed: int = 0, draw_rate: float = 0.2) -> list[Result]:
    rng = np.random.default_rng(seed)
    results = []
    for a, b in pairs:
        for i in range(games):
            white, black = (a, b) if i % 2 == 0 else (b, a)
            e = expected_score(true[white], true[black])
            u = rng.random()
            # draws take draw_rate of the probability mass, split so that the expected score stays e
            draw = min(draw_rate, 2 * min(e, 1 - e))
            win = e - draw / 2
            score = 1.0 if u < win else 0.5 if u < win + draw else 0.0
            results.append(Result(white, black, score))
    return results


def test_expected_score() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)


def test_fit_recovers_known_ratings_on_the_anchor_scale() -> None:
    true = {"sf:1320": 1320, "sf:1900": 1900, "bot:a": 1100, "bot:b": 1500, "bot:c": 2000}
    pairs = [(bot, sf) for bot in ("bot:a", "bot:b", "bot:c") for sf in ("sf:1320", "sf:1900")]
    pairs += [("bot:a", "bot:b"), ("bot:b", "bot:c")]
    results = simulate(true, pairs, games=400)
    fit = fit_ratings(results, anchors={"sf:1320": 1320, "sf:1900": 1900})
    assert fit["sf:1320"] == 1320 and fit["sf:1900"] == 1900
    for bot in ("bot:a", "bot:b", "bot:c"):
        assert fit[bot] == pytest.approx(true[bot], abs=45)


def test_without_anchors_only_differences_matter() -> None:
    true = {"x": 1200, "y": 1600}
    fit = fit_ratings(simulate(true, [("x", "y")], 600), center=1500)
    assert (fit["x"] + fit["y"]) / 2 == pytest.approx(1500)
    assert fit["y"] - fit["x"] == pytest.approx(400, abs=60)


def test_a_clean_sweep_stays_finite() -> None:
    results = [Result("strong", "weak", 1.0) for _ in range(10)]
    fit = fit_ratings(results, anchors={"weak": 1500})
    assert 1900 < fit["strong"] < 4000


def test_bootstrap_intervals_cover_the_truth() -> None:
    true = {"sf": 1600, "bot": 1450}
    results = simulate(true, [("bot", "sf")], 200, seed=3)
    groups = ["bot vs sf"] * len(results)
    estimate, intervals, fits = bootstrap_ratings(results, groups, anchors={"sf": 1600}, samples=100)
    low, high = intervals["bot"]
    assert low < estimate["bot"] < high and low < 1450 < high
    assert intervals["sf"] == (1600, 1600) and len(fits) == 100


def test_dial_summary_and_spearman() -> None:
    dial = {"bot:800": 800, "bot:1400": 1400, "bot:2000": 2000}
    estimate = {"bot:800": 1200, "bot:1400": 1350, "bot:2000": 1700}
    fits = [estimate, {"bot:800": 1300, "bot:1400": 1250, "bot:2000": 1650}]
    summary = dial_summary(dial, estimate, fits)
    assert summary["spearman"] == pytest.approx(1.0)
    assert summary["slope"] == pytest.approx(np.polyfit([800, 1400, 2000], [1200, 1350, 1700], 1)[0])
    first = summary["steps"][0]
    assert first["dial_gap"] == 600 and first["measured_gap"] == 150 and first["prob_stronger"] == 0.5
    assert summary["steps"][1]["prob_stronger"] == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 2], [5, 5]))


def test_every_pair_pairs_helper_is_consistent() -> None:
    true = {f"p{i}": 1200 + 150 * i for i in range(4)}
    results = simulate(true, itertools.combinations(true, 2), 300, seed=5)
    fit = fit_ratings(results, anchors={"p0": 1200})
    order = sorted(fit, key=fit.get)
    assert order == ["p0", "p1", "p2", "p3"]
