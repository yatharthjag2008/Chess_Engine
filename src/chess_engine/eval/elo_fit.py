"""Estimate players' Elo ratings from game results, and check that strength rises with the dial.

The model is the Elo / Bradley-Terry one: player i scores against player j

    E = 1 / (1 + 10 ** (-(r_i - r_j) / 400))

per game (a draw counts as half a point). Ratings are fitted by maximum likelihood
(Newton's method). *Anchors* (Stockfish levels with a known UCI_Elo) keep their
ratings fixed, which puts everyone on Stockfish's scale; without anchors the
mean rating is fixed at ``center`` instead. A weak Gaussian prior (``prior_sd``
around ``center``) keeps a player who won or lost every game finite.

Confidence intervals come from a bootstrap that resamples games within each
pairing, refitting each time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

K = math.log(10) / 400


@dataclass(frozen=True)
class Result:
    """One game: ``score`` is ``white``'s points (1, 0.5 or 0)."""

    white: str
    black: str
    score: float


def fit_ratings(
    results: Sequence[Result],
    anchors: Mapping[str, float] | None = None,
    center: float = 1500.0,
    prior_sd: float = 800.0,
    iterations: int = 100,
) -> dict[str, float]:
    """Maximum-likelihood ratings of every player in ``results`` (anchors stay at their given values)."""
    anchors = dict(anchors or {})
    players = sorted({r.white for r in results} | {r.black for r in results} | set(anchors))
    index = {p: i for i, p in enumerate(players)}
    free = [p for p in players if p not in anchors]
    if not free:
        return dict(anchors)
    ratings = np.array([anchors.get(p, center) for p in players], dtype=np.float64)
    w = np.array([index[r.white] for r in results], dtype=np.int64)
    b = np.array([index[r.black] for r in results], dtype=np.int64)
    s = np.array([r.score for r in results], dtype=np.float64)
    free_idx = np.array([index[p] for p in free])
    prior_precision = 1 / prior_sd**2
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-K * (ratings[w] - ratings[b])))
        grad = np.zeros(len(players))
        np.add.at(grad, w, K * (s - p))
        np.add.at(grad, b, -K * (s - p))
        info = K * K * p * (1 - p)
        hess = np.zeros((len(players), len(players)))
        np.add.at(hess, (w, w), -info)
        np.add.at(hess, (b, b), -info)
        np.add.at(hess, (w, b), info)
        np.add.at(hess, (b, w), info)
        grad -= prior_precision * (ratings - center)
        hess -= prior_precision * np.eye(len(players))
        g = grad[free_idx]
        h = hess[np.ix_(free_idx, free_idx)]
        if not anchors:  # without anchors only differences are identified: also fix the mean
            g = np.append(g, 0.0)
            h = np.block([[h, np.ones((len(free), 1))], [np.ones((1, len(free))), np.zeros((1, 1))]])
        step = np.linalg.solve(h, -g)[: len(free)]
        step = np.clip(step, -400, 400)
        ratings[free_idx] += step
        if not anchors:
            ratings[free_idx] += center - ratings[free_idx].mean()
        if np.abs(step).max() < 1e-6:
            break
    return {p: float(ratings[index[p]]) for p in players}


def bootstrap_ratings(
    results: Sequence[Result],
    groups: Sequence[str],
    anchors: Mapping[str, float] | None = None,
    samples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
    **fit_kwargs: Any,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], list[dict[str, float]]]:
    """Point estimates, percentile intervals and every bootstrap fit.

    Args:
        groups: For each result, its pairing; games are resampled within each pairing,
            so every pairing keeps its number of games.
    """
    estimate = fit_ratings(results, anchors, **fit_kwargs)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(samples):
        picked = [i for members in by_group.values() for i in rng.choice(members, len(members))]
        fits.append(fit_ratings([results[i] for i in picked], anchors, **fit_kwargs))
    alpha = (1 - confidence) / 2
    intervals = {}
    for player in estimate:
        values = np.array([f[player] for f in fits if player in f]) if fits else np.array([estimate[player]])
        low, high = np.quantile(values, [alpha, 1 - alpha])
        intervals[player] = (float(low), float(high))
    return estimate, intervals, fits


def expected_score(rating_a: float, rating_b: float) -> float:
    """A's expected score per game against B."""
    return 1 / (1 + 10 ** (-(rating_a - rating_b) / 400))


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties sharing their average rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (NaN if either side is constant)."""
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def dial_summary(dial: Mapping[str, int], estimate: Mapping[str, float],
                 fits: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """How measured strength follows the dial.

    Args:
        dial: Player key -> the rating it was set to (the bots).

    Returns:
        ``spearman`` of dial vs. measured rating, ``slope`` (measured Elo per dial Elo, least squares),
        and for each step up the dial, the gap and the share of bootstrap fits in which the
        higher setting measured stronger.
    """
    keys = sorted(dial, key=lambda k: dial[k])
    x = np.array([dial[k] for k in keys], dtype=np.float64)
    y = np.array([estimate[k] for k in keys], dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0]) if len(keys) >= 2 else float("nan")
    steps = []
    for low, high in zip(keys, keys[1:]):
        wins = [f[high] > f[low] for f in fits if low in f and high in f]
        steps.append({"from": low, "to": high, "dial_gap": dial[high] - dial[low],
                      "measured_gap": estimate[high] - estimate[low],
                      "prob_stronger": float(np.mean(wins)) if wins else float("nan")})
    return {"spearman": spearman(x, y), "slope": slope, "steps": steps}
