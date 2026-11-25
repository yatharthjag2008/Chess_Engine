"""Tests for the chess clock (with a fake time source)."""

from __future__ import annotations

import chess
import pytest

from chess_engine.bot.clock import ChessClock, TimeControl, format_seconds


class FakeTime:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_parse_time_control() -> None:
    assert TimeControl.parse("3+2") == TimeControl(180, 2)
    assert TimeControl.parse("5") == TimeControl(300, 0)
    assert TimeControl.parse("0.5+0") == TimeControl(30, 0)
    assert TimeControl.parse("180+2") == TimeControl(180, 2)
    assert str(TimeControl(180, 2)) == "3+2"
    for bad in ("fast", "3+x", "0+2"):
        with pytest.raises(ValueError):
            TimeControl.parse(bad)


def test_clocks_start_after_both_first_moves_then_count_down() -> None:
    now = FakeTime()
    clock = ChessClock(TimeControl(180, 2), now)
    now.t += 30
    assert clock.press(chess.WHITE) == 0.0 and clock.running is None  # first moves are free
    now.t += 30
    clock.press(chess.BLACK)
    assert clock.running == chess.WHITE
    assert clock.remaining(chess.WHITE) == 180 and clock.remaining(chess.BLACK) == 180
    now.t += 10
    assert clock.remaining(chess.WHITE) == 170
    assert clock.press(chess.WHITE) == 10
    assert clock.remaining(chess.WHITE) == 172  # 180 - 10 + 2 increment
    assert clock.running == chess.BLACK
    now.t += 5
    assert clock.remaining(chess.BLACK) == 175


def test_flag_and_stop() -> None:
    now = FakeTime()
    clock = ChessClock(TimeControl(60, 0), now)
    clock.press(chess.WHITE)
    clock.press(chess.BLACK)
    now.t += 59
    assert clock.flagged() is None
    now.t += 2
    assert clock.flagged() == chess.WHITE
    clock.stop()
    now.t += 100
    assert clock.remaining(chess.WHITE) == pytest.approx(-1) and clock.running is None

    wrong_side = ChessClock(TimeControl(60, 0), now)
    wrong_side.press(chess.WHITE)
    wrong_side.press(chess.BLACK)  # White's clock is running now
    with pytest.raises(ValueError, match="not running"):
        wrong_side.press(chess.BLACK)


def test_format_seconds() -> None:
    assert format_seconds(185) == "3:05" and format_seconds(60) == "1:00"
    assert format_seconds(9.46) == "0:09.5" and format_seconds(-3) == "0:00.0"
