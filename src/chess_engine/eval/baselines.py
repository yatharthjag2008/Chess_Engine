"""Baselines the model has to beat.

**Most common move** (:class:`MoveFrequencyTable`): for a test position that also
occurs in the training games, predict the move played most often there in
training, either by everyone or only by players of the same rating bucket. This
is what an "opening book of human games" would do. It is strong in the opening,
where positions repeat, and has nothing to say once a game leaves known
positions, so its *coverage* is reported next to its accuracy.

**Random legal move**: matches the human move ``1 / legal moves`` of the time
(computed exactly, not sampled). It is also the fallback for positions a
frequency table has never seen.

Every baseline gives, per test position, a *hit* value (1 / 0, or the expected
hit rate for a random choice) and whether it had a real prediction (*covered*),
so it can be scored exactly like the model (:func:`summarize_baseline`).

Built with numpy only: positions are identified by their 64-bit key
(:mod:`chess_engine.eval.positions`), ``(key, move)`` pairs are counted by sorting,
and lookups are binary searches, so 3 million training positions take a second or two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from chess_engine.data.samples import Samples
from chess_engine.eval.move_matching import group_means
from chess_engine.eval.positions import combine_keys, sample_keys
from chess_engine.eval.predictions import PredictionTable


@dataclass(frozen=True)
class MoveFrequencyTable:
    """For every position key seen in training: its most common move and how often it was played."""

    keys: np.ndarray        # (K,) uint64, sorted, unique
    best_move: np.ndarray   # (K,) int16, most common move (ties: lowest move index)
    best_count: np.ndarray  # (K,) int32, how often that move was played
    total: np.ndarray       # (K,) int32, how often the position occurred

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def build(cls, keys: np.ndarray, moves: np.ndarray) -> MoveFrequencyTable:
        """Count the moves played in each position (``keys`` and ``moves`` are per sample)."""
        keys = np.asarray(keys, dtype=np.uint64)
        moves = np.asarray(moves, dtype=np.int64)
        if len(keys) != len(moves):
            raise ValueError("keys and moves must have the same length")
        if len(keys) == 0:
            empty = np.zeros(0, dtype=np.int32)
            return cls(np.zeros(0, dtype=np.uint64), empty.astype(np.int16), empty, empty)
        order = np.lexsort((moves, keys))
        k, m = keys[order], moves[order]
        pair_start = np.flatnonzero(np.r_[True, (k[1:] != k[:-1]) | (m[1:] != m[:-1])])
        pair_count = np.diff(np.r_[pair_start, len(k)])
        pair_key, pair_move = k[pair_start], m[pair_start]
        # Within each key: highest count first, then lowest move index.
        best_first = np.lexsort((pair_move, -pair_count, pair_key))
        pk, pm, pc = pair_key[best_first], pair_move[best_first], pair_count[best_first]
        key_start = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
        totals = np.add.reduceat(pc, key_start)
        return cls(pk[key_start], pm[key_start].astype(np.int16), pc[key_start].astype(np.int32),
                   totals.astype(np.int32))

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(found, best_move, total)`` for each key (``best_move`` is -1 and ``total`` 0 where not found)."""
        keys = np.asarray(keys, dtype=np.uint64)
        if len(self.keys) == 0:
            return np.zeros(len(keys), dtype=bool), np.full(len(keys), -1, np.int16), np.zeros(len(keys), np.int32)
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        found = self.keys[pos] == keys
        return (found, np.where(found, self.best_move[pos], -1).astype(np.int16),
                np.where(found, self.total[pos], 0).astype(np.int32))


# --------------------------------------------------------------------------- baseline predictions


@dataclass(frozen=True)
class BaselineResult:
    """A baseline's per-position hits on a prediction table's positions."""

    name: str
    hits: np.ndarray     # (N,) float64: 1 / 0, or the expected hit rate of a random fallback
    covered: np.ndarray  # (N,) bool: the baseline made its own prediction (no fallback)


def build_frequency_tables(train: Samples) -> tuple[MoveFrequencyTable, MoveFrequencyTable]:
    """Most-common-move tables from training samples: by position, and by (position, mover's bucket)."""
    keys = sample_keys(train)
    moves = train["move"]
    return MoveFrequencyTable.build(keys, moves), MoveFrequencyTable.build(combine_keys(keys, train["bucket"]), moves)


def random_baseline(table: PredictionTable) -> BaselineResult:
    """A uniformly random legal move: expected hit rate ``1 / legal moves``."""
    return BaselineResult("random legal move", 1.0 / table["legal_count"].astype(np.float64),
                          np.ones(len(table), dtype=bool))


def most_common_move_baseline(
    table: PredictionTable, overall: MoveFrequencyTable, by_bucket: MoveFrequencyTable | None = None
) -> BaselineResult:
    """Most common training move for each test position; a random legal move where the position is new.

    With ``by_bucket``, the move most often played *by the mover's rating bucket* is
    used where that (position, bucket) pair was seen, then the overall table, then random.
    """
    target = table["target"].astype(np.int64)
    hits = 1.0 / table["legal_count"].astype(np.float64)
    covered = np.zeros(len(table), dtype=bool)
    lookups = [overall.lookup(table["key"])]
    if by_bucket is not None:
        lookups.insert(0, by_bucket.lookup(combine_keys(table["key"], table["bucket"])))
    for found, move, _ in lookups:
        use = found & ~covered
        hits[use] = (move[use] == target[use]).astype(np.float64)
        covered |= found
    name = "most common move (same rating)" if by_bucket is not None else "most common move"
    return BaselineResult(name, hits, covered)


def summarize_baseline(result: BaselineResult, table: PredictionTable,
                       labels: Sequence[str] | None = None) -> dict[str, Any]:
    """Top-1 (with fallback), coverage and top-1 where covered, overall and per bucket.

    Also the model's own top-1 on the covered positions, for a like-for-like comparison.
    """
    labels = list(labels if labels is not None else table.meta.get("bucket_labels", []))
    model_hits = table.top1_hit.astype(np.float64)

    def stats(rows: np.ndarray) -> dict[str, float]:
        n = int(rows.sum())
        if n == 0:
            return {"positions": 0}
        cov = rows & result.covered
        out = {"positions": n, "top1": float(result.hits[rows].mean()), "coverage": float(cov.sum() / n)}
        if cov.any():
            out["top1_covered"] = float(result.hits[cov].mean())
            out["model_top1_covered"] = float(model_hits[cov].mean())
        return out

    buckets = {label: stats(table["bucket"] == b) for b, label in enumerate(labels)}
    top1, counts = group_means(result.hits, table["bucket"], len(labels))
    present = top1[counts > 0]
    return {"name": result.name, "overall": stats(np.ones(len(table), dtype=bool)), "buckets": buckets,
            "bucket_mean_top1": float(present.mean()) if len(present) else float("nan")}
