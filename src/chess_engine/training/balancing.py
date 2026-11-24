"""Balance the rating buckets so low-data buckets are not under-learned.

The 50k-game sample has 6x fewer positions at 800-999 than at 1600-1799. With
plain training the rare buckets contribute little to the loss, so the model
spends its capacity on the common ones. Each bucket gets a weight

    w_b ∝ (n_b / mean n) ** -power,   normalized so that  sum_b n_b * w_b = sum_b n_b

``power = 0`` keeps the data as it is, ``power = 1`` makes every bucket count
equally, and ``0.5`` (the default) is in between: the 800 bucket counts ~2.4x
more per position and the 1600 bucket ~0.8x. Normalizing keeps the average
weight per position at 1, so the loss scale (and the learning rate) do not change.

The weights are used either as per-position loss weights (``bucket_balancing:
weight``) or as resampling rates (``sample``, see :mod:`chess_engine.training.order`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bucket_weights(counts: Sequence[int] | np.ndarray, power: float) -> np.ndarray:
    """Per-position weight of each bucket (float64); empty buckets get weight 0.

    Raises:
        ValueError: ``power`` is outside ``[0, 1]`` or every bucket is empty.
    """
    if not 0 <= power <= 1:
        raise ValueError(f"power must be in [0, 1], got {power}")
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or (counts < 0).any():
        raise ValueError("counts must be a 1-D array of non-negative numbers")
    present = counts > 0
    if not present.any():
        raise ValueError("every bucket is empty")
    weights = np.zeros_like(counts)
    weights[present] = (counts[present] / counts[present].mean()) ** -power
    weights *= counts.sum() / (counts * weights).sum()
    return weights


def effective_shares(counts: Sequence[int] | np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fraction of the total training signal each bucket gets with these weights (sums to 1)."""
    mass = np.asarray(counts, dtype=np.float64) * weights
    return mass / mass.sum()


def position_weights(buckets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """float32 loss weight of every position, looked up from its bucket."""
    return weights.astype(np.float32)[np.asarray(buckets, dtype=np.int64)]


def format_balance(labels: Sequence[str], counts: Sequence[int], weights: np.ndarray) -> str:
    """A small table: positions, weight and share of the training signal per bucket."""
    counts = np.asarray(counts)
    data_share = counts / counts.sum()
    shares = effective_shares(counts, weights)
    lines = [f"{'bucket':<10} {'positions':>10} {'weight':>7} {'data':>6} {'trained':>8}"]
    for label, n, w, d, s in zip(labels, counts, weights, data_share, shares):
        lines.append(f"{label:<10} {int(n):>10,} {w:>7.2f} {d:>6.1%} {s:>8.1%}")
    return "\n".join(lines)
