"""Baselines the model has to beat.

**Most common move** (:class:`MoveFrequencyTable`): for a test position that also
occurs in the training games, predict the move played most often there in
training, either by everyone or only by players of the same rating bucket. This
is what an "opening book of human games" would do. It is strong in the opening,
where positions repeat, and has nothing to say once a game leaves known
positions, so its *coverage* is reported next to its accuracy.

Built with numpy only: positions are identified by their 64-bit key
(:mod:`chess_engine.eval.positions`), ``(key, move)`` pairs are counted by sorting,
and lookups are binary searches, so 3 million training positions take a second or two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
