"""Tests for win %, move accuracy and game accuracy."""

from __future__ import annotations

import math

import chess.engine
import pytest

from chess_engine.review.scoring import MATE_CP, game_accuracy, move_accuracy, score_to_cp, win_percent


def test_score_to_cp() -> None:
    assert score_to_cp(chess.engine.Cp(35)) == 35
    assert score_to_cp(chess.engine.Mate(3)) == MATE_CP - 3
    assert score_to_cp(chess.engine.Mate(-2)) == -(MATE_CP - 2)
    assert score_to_cp(chess.engine.Mate(1)) > score_to_cp(chess.engine.Mate(5))  # faster mates rank higher


def test_win_percent() -> None:
    assert win_percent(0) == pytest.approx(50)
    assert win_percent(100) == pytest.approx(50 + 50 * (2 / (1 + math.exp(-0.368208)) - 1))
    assert win_percent(300) + win_percent(-300) == pytest.approx(100)
    assert win_percent(5000) == win_percent(1000)              # capped at +-1000 cp
    assert win_percent(MATE_CP - 3) == 100 and win_percent(-(MATE_CP - 3)) == 0
    values = [win_percent(cp) for cp in range(-1200, 1201, 100)]
    assert values == sorted(values)


def test_move_accuracy() -> None:
    assert move_accuracy(60, 60) == 100                        # nothing lost
    assert move_accuracy(40, 70) == 100                        # gains are not rewarded beyond 100
    assert move_accuracy(80, 70) == pytest.approx(103.1668100711649 * math.exp(-0.4354415386753951) - 2.166924740191411)
    assert move_accuracy(90, 10) < 5 and move_accuracy(100, 0) == 0
    assert move_accuracy(60, 50) > move_accuracy(60, 40) > move_accuracy(60, 20)


def test_game_accuracy() -> None:
    assert math.isnan(game_accuracy([], [50, 50], []))
    # one player, four moves, all perfect: 100
    wins = [50, 52, 51, 53, 52, 54, 53, 55, 54]
    assert game_accuracy([100, 100, 100, 100], wins, [0, 2, 4, 6]) == pytest.approx(100)
    # a single blunder drags the score down a lot (harmonic mean)
    blunder = game_accuracy([100, 100, 100, 5], wins, [0, 2, 4, 6])
    assert blunder < 70
    assert blunder < game_accuracy([90, 90, 90, 90], wins, [0, 2, 4, 6])
