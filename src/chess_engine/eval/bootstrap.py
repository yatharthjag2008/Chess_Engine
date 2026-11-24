"""Confidence intervals for per-position rates, resampling whole games (a cluster bootstrap).

Positions of the same game are not independent: one player's style and one
opening shape dozens of positions. Resampling positions as if they were
independent gives intervals that are too narrow, so each bootstrap replicate
draws *games* with replacement and keeps all of each drawn game's positions.

    ci = cluster_bootstrap(table.top1_hit, table["game"], table["bucket"], num_groups=8)
    ci.overall        # (low, high) for all positions
    ci.groups[3]      # (low, high) for bucket 3

The same function gives an interval for a *difference* between two predictors
on the same positions: pass ``hits_a - hits_b`` (the mean of differences is the
difference of means, and pairing keeps the interval tight).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile intervals: ``overall`` and one per group (NaN where a group has no data)."""

    overall: tuple[float, float]
    groups: list[tuple[float, float]]
    confidence: float
    samples: int


def cluster_bootstrap(
    values: np.ndarray,
    games: np.ndarray,
    groups: np.ndarray | None = None,
    num_groups: int = 1,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    chunk: int = 100,
) -> BootstrapCI:
    """Percentile interval of the mean of ``values`` overall and per group, resampling ``games``.

    Args:
        values: Per-position numbers (e.g. 0/1 hits, log-likelihoods).
        games: Per-position game id (the resampling unit).
        groups: Per-position group (e.g. rating bucket), or None for a single group.
        samples: Bootstrap replicates (0 gives NaN intervals).
        chunk: Replicates computed at once (memory: ``chunk x games`` integers).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    values = np.asarray(values, dtype=np.float64)
    groups = np.zeros(len(values), dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
    if not len(values) == len(games) == len(groups):
        raise ValueError("values, games and groups must have the same length")
    nan = (float("nan"), float("nan"))
    if samples == 0 or len(values) == 0:
        return BootstrapCI(nan, [nan] * num_groups, confidence, samples)

    _, game_index = np.unique(np.asarray(games), return_inverse=True)
    num_games = int(game_index.max()) + 1
    flat = game_index * num_groups + groups
    sums = np.bincount(flat, weights=values, minlength=num_games * num_groups).reshape(num_games, num_groups)
    counts = np.bincount(flat, minlength=num_games * num_groups).reshape(num_games, num_groups).astype(np.float64)

    rng = np.random.default_rng(seed)
    group_means, overall_means = [], []
    for start in range(0, samples, chunk):
        n = min(chunk, samples - start)
        weights = rng.multinomial(num_games, np.full(num_games, 1 / num_games), size=n).astype(np.float64)
        s, c = weights @ sums, weights @ counts
        with np.errstate(invalid="ignore", divide="ignore"):
            group_means.append(s / c)
            overall_means.append(s.sum(1) / c.sum(1))
    group_means_all = np.concatenate(group_means)
    overall_all = np.concatenate(overall_means)
    alpha = (1 - confidence) / 2

    def interval(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return nan
        low, high = np.quantile(x, [alpha, 1 - alpha])
        return float(low), float(high)

    per_group = [interval(group_means_all[:, g]) if counts[:, g].sum() else nan for g in range(num_groups)]
    return BootstrapCI(interval(overall_all), per_group, confidence, samples)
