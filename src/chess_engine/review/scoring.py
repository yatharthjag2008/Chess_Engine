"""Turning engine evaluations into win percentages and accuracy, as Lichess does.

* :func:`win_percent`: Stockfish's centipawns as the side's chance of winning (0-100),
  ``50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)``; mates count as 100 / 0.
* :func:`move_accuracy`: how much of the win chance a move kept, 0-100:
  ``103.1668 * exp(-0.04354 * win% lost) - 3.1669``, plus 1 (Lichess's allowance for
  engine noise), clamped to [0, 100]. A move that loses nothing scores 100.
* :func:`game_accuracy`: one number per player for a game: the mean of a
  volatility-weighted mean and the harmonic mean of the move accuracies (the
  harmonic mean makes a single blunder count; the weighting makes moves in sharp
  positions count more than moves in dead-equal ones).

All evaluations here are from one side's point of view unless said otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import chess.engine
import numpy as np

#: Centipawn value used for "mate" when a single number is needed (ordering, graphs).
MATE_CP = 10_000
#: Evaluations are capped here before converting to win %, as Lichess does.
CP_CAP = 1_000


def score_to_cp(score: chess.engine.Score) -> int:
    """A python-chess score as centipawns; mate in n is ``+-(MATE_CP - n)``, so faster mates rank higher."""
    if score.is_mate():
        mate = score.mate()
        return MATE_CP - abs(mate) if mate > 0 else -(MATE_CP - abs(mate))
    return int(score.score())


def win_percent(cp: float) -> float:
    """The side's win chance in % for an evaluation of ``cp`` centipawns (mates give 100 / 0)."""
    if cp >= MATE_CP - 1000:
        return 100.0
    if cp <= -(MATE_CP - 1000):
        return 0.0
    cp = max(-CP_CAP, min(CP_CAP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Accuracy (0-100) of a move that took the mover's win chance from ``win_before`` to ``win_after``."""
    lost = max(0.0, win_before - win_after)
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * lost) - 3.166924740191411
    return max(0.0, min(100.0, raw + 1))


def _harmonic_mean(values: Sequence[float]) -> float:
    values = [max(v, 1e-6) for v in values]
    return len(values) / sum(1 / v for v in values)


def game_accuracy(move_accuracies: Sequence[float], white_wins: Sequence[float], color_moves: Sequence[int]) -> float:
    """A player's accuracy over a game (Lichess's method).

    Args:
        move_accuracies: Accuracy of each of the player's moves.
        white_wins: White's win % before every ply of the game and after the last one (length plies + 1);
            the volatility weights come from windows of this sequence.
        color_moves: The ply index of each of the player's moves (same order as ``move_accuracies``).
    """
    if not move_accuracies:
        return float("nan")
    wins = np.asarray(white_wins, dtype=np.float64)
    window = int(np.clip(len(wins) // 10, 2, 8))
    weights = []
    for ply in color_moves:
        start = max(0, min(ply + 1 - window, len(wins) - window))
        segment = wins[start:start + window]
        weights.append(float(np.clip(segment.std(), 0.5, 12)))
    weights = np.asarray(weights)
    acc = np.asarray(move_accuracies, dtype=np.float64)
    weighted = float((acc * weights).sum() / weights.sum())
    return (weighted + _harmonic_mean(acc)) / 2
