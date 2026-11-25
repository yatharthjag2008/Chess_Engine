"""Estimate a player's rating (and a likely range) from the moves they played.

For every move, the model gives the probability that a player of each rating
bucket plays it. Summed over a game's moves, the log-probabilities say how well
each rating explains the game. Bucket ``b`` is placed at its centre rating and the
values are interpolated linearly to every rating on a grid; with a prior over
ratings (how ratings are spread among players) this gives a posterior:

    log p(rating | moves)  =  sum over moves of log p(move | rating) / temper  +  log prior(rating)  + const

The estimate is the posterior mean, the range its central ``interval`` (e.g. 80%).
``temper`` > 1 widens the posterior: the moves of one game are not independent
pieces of evidence (one opening, one style, one time scramble), so the raw sum
is overconfident; the right value is fitted on held-out games.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from chess_engine.config import RatingEstimationConfig, RatingsConfig


def bucket_centers(edges: Sequence[int]) -> np.ndarray:
    """Centre rating of each bucket; the open top bucket is as wide as the one below it.

    ``[800, 1000, ..., 2200]`` -> ``[900, 1100, ..., 2100, 2300]``.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) == 1:
        return edges + 100.0
    widths = np.diff(edges)
    return np.append(edges[:-1] + widths / 2, edges[-1] + widths[-1] / 2)


def rating_grid(cfg: RatingEstimationConfig) -> np.ndarray:
    """The ratings considered: ``rating_min`` to ``rating_max`` in steps of ``rating_step``."""
    return np.arange(cfg.rating_min, cfg.rating_max + 1, cfg.rating_step, dtype=np.float64)


def grid_log_likelihood(bucket_ll: np.ndarray, centers: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate per-bucket log-likelihoods ``(..., K)`` to the grid ``(..., G)``.

    Linear between bucket centres and flat beyond the outermost ones (the model knows
    nothing finer). Interpolating a sum equals summing the interpolations, so callers may
    sum over moves first.
    """
    bucket_ll = np.asarray(bucket_ll, dtype=np.float64)
    flat = bucket_ll.reshape(-1, bucket_ll.shape[-1])
    out = np.stack([np.interp(grid, centers, row) for row in flat])
    return out.reshape(*bucket_ll.shape[:-1], len(grid))


def posterior(log_likelihood: np.ndarray, log_prior: np.ndarray, temper: float = 1.0) -> np.ndarray:
    """Normalized posterior over the grid (rows are independent if 2-D)."""
    if temper <= 0:
        raise ValueError("temper must be positive")
    logp = np.asarray(log_likelihood, dtype=np.float64) / temper + log_prior
    logp = logp - logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class RatingEstimate:
    """A posterior over ratings, summarized."""

    estimate: float                 # posterior mean
    median: float
    low: float                      # central interval with probability `interval`
    high: float
    interval: float
    moves: int                      # moves the estimate is based on
    grid: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)

    def describe(self) -> str:
        return (f"about {round(self.estimate, -1):.0f} ({self.interval:.0%} range "
                f"{round(self.low, -1):.0f}-{round(self.high, -1):.0f}, from {self.moves} moves)")


def summarize(grid: np.ndarray, probabilities: np.ndarray, interval: float = 0.8, moves: int = 0) -> RatingEstimate:
    """Mean, median and central ``interval`` of a posterior on ``grid``."""
    cdf = np.cumsum(probabilities)
    tail = (1 - interval) / 2

    def quantile(q: float) -> float:
        return float(np.interp(q, cdf, grid))

    return RatingEstimate(float((grid * probabilities).sum()), quantile(0.5), quantile(tail), quantile(1 - tail),
                          interval, moves, grid, probabilities)


def estimate_from_bucket_ll(
    bucket_ll: np.ndarray,
    ratings: RatingsConfig,
    cfg: RatingEstimationConfig,
    log_prior: np.ndarray | None = None,
    temper: float | None = None,
) -> RatingEstimate:
    """Estimate from per-move bucket log-likelihoods ``(moves, K)`` (one player, any number of games)."""
    bucket_ll = np.atleast_2d(np.asarray(bucket_ll, dtype=np.float64))
    grid = rating_grid(cfg)
    total = grid_log_likelihood(bucket_ll.sum(axis=0), bucket_centers(ratings.bucket_edges), grid)
    prior = np.zeros_like(grid) if log_prior is None else log_prior
    temper = temper if temper is not None else (cfg.temper or 1.0)
    return summarize(grid, posterior(total, prior, temper), cfg.interval, moves=len(bucket_ll))
