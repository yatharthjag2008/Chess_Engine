"""The order positions are visited in during one epoch: deterministic and resumable.

The order of epoch ``e`` depends only on ``(seed, e)``, so a resumed run
rebuilds exactly the same order and skips the batches it already trained on.

Two modes:

* **shuffle**: a permutation of every position (each one exactly once).
* **resample by bucket** (``bucket_balancing: sample``): bucket ``b`` contributes
  ``round(n_b * w_b)`` positions instead of ``n_b`` (see
  :mod:`chess_engine.training.balancing`), so rare buckets are seen more often and
  common ones less. Within a bucket, positions are taken from successive random
  permutations, so no position repeats before every position of its bucket has
  been used, and an under-sampled bucket uses a different subset each epoch.
  The epoch still has about the same number of positions as the data.
"""

from __future__ import annotations

import numpy as np


def epoch_rng(seed: int, epoch: int) -> np.random.Generator:
    """The random generator for epoch ``epoch`` of a run with ``seed``."""
    return np.random.default_rng([seed % 2**32, epoch])


def shuffled_order(num_positions: int, seed: int, epoch: int) -> np.ndarray:
    """A permutation of ``range(num_positions)`` for this epoch (int64)."""
    return epoch_rng(seed, epoch).permutation(num_positions).astype(np.int64)


def resample_counts(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Positions drawn from each bucket per epoch: ``round(n_b * w_b)``, at least 1 for non-empty buckets."""
    counts = np.asarray(counts, dtype=np.int64)
    drawn = np.rint(counts * np.asarray(weights, dtype=np.float64)).astype(np.int64)
    return np.where(counts > 0, np.maximum(drawn, 1), 0)


def _cycle(indices: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    """``total`` entries from ``indices``: whole random permutations, then part of one more."""
    reps, rest = divmod(total, len(indices))
    parts = [rng.permutation(indices) for _ in range(reps)]
    parts.append(rng.permutation(indices)[:rest])
    return np.concatenate(parts)


def resampled_order(buckets: np.ndarray, weights: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """Indices of one resampled epoch (see the module docstring), globally shuffled."""
    buckets = np.asarray(buckets, dtype=np.int64)
    rng = epoch_rng(seed, epoch)
    counts = np.bincount(buckets, minlength=len(weights))
    if len(counts) > len(weights):
        raise ValueError(f"bucket {len(counts) - 1} has no weight ({len(weights)} weights)")
    draws = resample_counts(counts, weights)
    by_bucket = np.argsort(buckets, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)])
    parts = [_cycle(by_bucket[starts[b]:starts[b + 1]], int(draws[b]), rng)
             for b in range(len(counts)) if draws[b] > 0]
    order = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return rng.permutation(order).astype(np.int64)


def epoch_order(
    num_positions: int,
    seed: int,
    epoch: int,
    buckets: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """The epoch's order: resampled by bucket when ``buckets`` and ``weights`` are given, else shuffled."""
    if (buckets is None) != (weights is None):
        raise ValueError("pass both buckets and weights (resampling) or neither (plain shuffle)")
    if buckets is None:
        return shuffled_order(num_positions, seed, epoch)
    if len(buckets) != num_positions:
        raise ValueError(f"{len(buckets)} buckets for {num_positions} positions")
    return resampled_order(buckets, weights, seed, epoch)


def batch_slices(order: np.ndarray, batch_size: int, start_batch: int = 0, drop_last: bool = True) -> list[np.ndarray]:
    """``order`` cut into batches, skipping the first ``start_batch`` (already trained on)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stop = len(order) // batch_size * batch_size if drop_last else len(order)
    return [order[i:i + batch_size] for i in range(start_batch * batch_size, stop, batch_size)]


def batches_per_epoch(epoch_size: int, batch_size: int, drop_last: bool = True) -> int:
    """Number of batches :func:`batch_slices` gives for an epoch of ``epoch_size`` positions."""
    return epoch_size // batch_size if drop_last else -(-epoch_size // batch_size)
