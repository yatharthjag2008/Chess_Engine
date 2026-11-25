"""Turn the model's move probabilities into the probabilities the bot actually plays with.

The model says how often humans at a rating play each move. The bot can play
exactly that (the default) or be adjusted:

* ``temperature``: ``p ** (1 / T)``, renormalized. ``T = 1`` keeps the human
  frequencies, ``T < 1`` favours the most common moves, ``T = 0`` always plays the
  single likeliest move (ties: the first one), ``T > 1`` spreads probability out.
* ``top_k``: keep only the ``k`` likeliest moves.
* ``top_p``: keep the smallest set of likeliest moves whose probability adds up to ``top_p``.
* ``min_prob``: drop moves the model gives less than this (before temperature),
  e.g. to remove one-in-a-thousand blunders. The likeliest move is always kept.

Applied in that order: filters first, on the model's own probabilities, then temperature.

    probs = adjust_probabilities(model_probs, temperature=0.8, top_p=0.95)
    index = sample_index(probs, rng)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingSettings:
    """How the bot picks among the model's moves (see the module docstring)."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    min_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= self.min_prob < 1:
            raise ValueError("min_prob must be in [0, 1)")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def adjust_probabilities(probs: np.ndarray, settings: SamplingSettings | None = None) -> np.ndarray:
    """The bot's probabilities for each move (float64, sums to 1, same order as ``probs``).

    Args:
        probs: The model's probability for each candidate move (non-negative, not all zero).
    """
    settings = settings or SamplingSettings()
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probs must be a non-empty 1-D array")
    if (p < 0).any() or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("probs must be finite, non-negative and not all zero")
    p = p / p.sum()
    order = np.argsort(-p, kind="stable")  # likeliest first; ties keep the original order
    keep = np.zeros(len(p), dtype=bool)
    keep[order[0]] = True  # the likeliest move always survives the filters
    ranked = p[order]
    allowed = ranked >= settings.min_prob
    if settings.top_k is not None:
        allowed[settings.top_k:] = False
    if settings.top_p < 1:
        before = np.concatenate([[0.0], np.cumsum(ranked)[:-1]])
        allowed &= before < settings.top_p - 1e-12  # include the move that crosses top_p
    keep[order[allowed]] = True

    if settings.greedy:
        out = np.zeros_like(p)
        out[order[0]] = 1.0
        return out
    filtered = np.where(keep, p, 0.0)
    if settings.temperature != 1:
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            logp = np.where(filtered > 0, np.log(np.where(filtered > 0, filtered, 1.0)), -np.inf)
            scaled = logp / settings.temperature
            scaled -= scaled[np.isfinite(scaled)].max()
            filtered = np.where(np.isfinite(scaled), np.exp(scaled), 0.0)
    return filtered / filtered.sum()


def sample_index(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Draw one index with the given probabilities (which must sum to ~1)."""
    probs = np.asarray(probs, dtype=np.float64)
    return int(rng.choice(len(probs), p=probs / probs.sum()))
