"""Tests for "your level vs. next level"."""

from __future__ import annotations

import chess
import pytest

from chess_engine.review.level_compare import level_choices
from bot_helpers import small_predictor

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def test_level_choices_match_single_predictions() -> None:
    predictor = small_predictor()
    board = chess.Board(KIWIPETE)
    choices = level_choices(predictor, board, [1100, 1300, 1500], opp_rating=1250, top=3)
    assert [c.rating for c in choices] == [1100, 1300, 1500]
    assert all(len(c.moves) == 3 for c in choices)
    for choice in choices:
        direct = predictor.predict(board, choice.rating, opp_ratings=1250)
        assert [m.move for m in choice.moves] == [m.move for m in direct.moves[:3]]
        assert choice.top.prob == pytest.approx(direct.moves[0].prob, abs=1e-5)
    assert choices[0].moves[0].prob != pytest.approx(choices[2].moves[0].prob)   # the rating matters
    assert level_choices(predictor, board, []) == []


def test_unknown_opponent_means_same_rating() -> None:
    predictor = small_predictor()
    board = chess.Board()
    (choice,) = level_choices(predictor, board, [1700], top=5)
    direct = predictor.predict(board, 1700, opp_ratings=1700)
    assert [m.move for m in choice.moves] == [m.move for m in direct.moves[:5]]
