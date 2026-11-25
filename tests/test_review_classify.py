"""Tests for move classification."""

from __future__ import annotations

import pytest

from chess_engine.config import ReviewConfig
from chess_engine.review.classify import ORDER, MoveClass, classify

CFG = ReviewConfig()   # good 2, inaccuracy 10, mistake 20, blunder 30


@pytest.mark.parametrize("lost, engine, expected", [
    (0, True, MoveClass.BEST), (25, True, MoveClass.BEST),      # the engine's own move is always best
    (-5, False, MoveClass.BEST), (2, False, MoveClass.BEST),   # gaining, or an equally good move
    (5, False, MoveClass.GOOD), (9.9, False, MoveClass.GOOD),
    (10, False, MoveClass.INACCURACY), (19.9, False, MoveClass.INACCURACY),
    (20, False, MoveClass.MISTAKE), (30, False, MoveClass.BLUNDER), (95, False, MoveClass.BLUNDER),
])
def test_classify(lost: float, engine: bool, expected: MoveClass) -> None:
    assert classify(lost, engine, CFG) is expected


def test_symbols_nags_and_order() -> None:
    assert [c.symbol for c in ORDER] == ["!", "", "?!", "?", "??"]
    assert [c.nag for c in ORDER] == [None, None, 6, 2, 4]
    assert [c.is_error for c in ORDER] == [False, False, True, True, True]
    assert str(MoveClass.BLUNDER) == "blunder"


def test_custom_thresholds() -> None:
    strict = ReviewConfig(good_drop=0.5, inaccuracy_drop=5, mistake_drop=10, blunder_drop=15)
    assert classify(12, False, strict) is MoveClass.MISTAKE and classify(12, False, CFG) is MoveClass.INACCURACY
