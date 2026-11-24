"""Identify and rebuild the positions of a dataset split.

* :func:`position_keys` gives every sample a 64-bit key that is equal exactly when
  two samples are the same position (same pieces, side to move, castling rights and
  legal en passant square; move counters are ignored, as in a FEN without clocks).
  The most-common-move baseline groups training positions by this key.
* :func:`board_of` rebuilds a ``chess.Board`` from a stored sample, for engines
  (the Stockfish baseline) and for showing positions in reports.
"""

from __future__ import annotations

import chess
import numpy as np

from chess_engine.data.samples import Samples
from chess_engine.encoding.compact import board_from_compact

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer, vectorized (uint64 arithmetic wraps around by design)."""
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def position_keys(pieces: np.ndarray, flags: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """``(N,)`` uint64 key per position from the compact board columns.

    Collisions between different positions are possible in principle but have a
    probability of about ``N**2 / 2**65`` (~1e-7 for 3.5 million positions).
    """
    pieces = np.asarray(pieces, dtype=np.uint64)
    if pieces.ndim != 2 or pieces.shape[1] != 12:
        raise ValueError(f"pieces must have shape (N, 12), got {pieces.shape}")
    with np.errstate(over="ignore"):
        key = np.zeros(len(pieces), dtype=np.uint64)
        for i in range(12):
            key = _mix(key ^ pieces[:, i] ^ (_GOLDEN * np.uint64(i + 1)))
        extra = (np.asarray(flags, dtype=np.uint64) << np.uint64(8)) | (np.asarray(ep, dtype=np.int64) + 1).astype(
            np.uint64)
        return _mix(key ^ extra)


def sample_keys(samples: Samples) -> np.ndarray:
    """:func:`position_keys` of every sample."""
    return position_keys(samples["pieces"], samples["flags"], samples["ep"])


def combine_keys(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A new key per (key, value) pair, e.g. (position, rating bucket)."""
    with np.errstate(over="ignore"):
        return _mix(np.asarray(keys, dtype=np.uint64) ^ (_GOLDEN * (np.asarray(values, dtype=np.uint64) + 1)))


def board_of(samples: Samples, index: int) -> chess.Board:
    """The position of sample ``index`` as a ``chess.Board`` (move counters from the halfmove clock only)."""
    return board_from_compact(samples["pieces"][index], int(samples["flags"][index]), int(samples["ep"][index]),
                              int(samples["halfmove"][index]))
