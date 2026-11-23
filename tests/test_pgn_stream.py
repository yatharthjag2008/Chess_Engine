"""Tests for the streaming PGN parser and game filters."""

from __future__ import annotations

import pytest

from chess_engine.data.pgn_stream import TimeControl, parse_time_control, speed_from_seconds, speed_of

# ----------------------------------------------------------------------------- time control


@pytest.mark.parametrize(
    "value, expected",
    [("180+2", TimeControl(180, 2)), ("300+0", TimeControl(300, 0)), (" 60+1 ", TimeControl(60, 1))],
)
def test_parse_time_control(value: str, expected: TimeControl) -> None:
    assert parse_time_control(value) == expected


@pytest.mark.parametrize("value", ["-", "", "300", "abc+1", "3+x", "+2", "-1+2"])
def test_parse_time_control_rejects_malformed(value: str) -> None:
    assert parse_time_control(value) is None


@pytest.mark.parametrize(
    "time_control, speed",
    [
        ("15+0", "ultrabullet"),
        ("0+1", "bullet"),       # 40 s estimated
        ("60+0", "bullet"),
        ("120+1", "bullet"),     # 160 s
        ("180+0", "blitz"),      # exactly 180 s -> blitz
        ("120+2", "blitz"),      # 200 s
        ("300+3", "blitz"),      # 420 s
        ("240+6", "rapid"),      # 480 s -> rapid boundary
        ("600+0", "rapid"),
        ("900+10", "rapid"),     # 1300 s
        ("1500+0", "classical"),
        ("1800+20", "classical"),
        ("-", "correspondence"),
        ("garbage", None),
    ],
)
def test_speed_of(time_control: str, speed: str | None) -> None:
    assert speed_of(time_control) == speed


def test_speed_boundaries() -> None:
    assert speed_from_seconds(29) == "ultrabullet"
    assert speed_from_seconds(30) == "bullet"
    assert speed_from_seconds(179) == "bullet"
    assert speed_from_seconds(479) == "blitz"
    assert speed_from_seconds(1499) == "rapid"


def test_time_control_properties() -> None:
    tc = TimeControl(180, 2)
    assert tc.estimated_seconds == 260
    assert tc.speed == "blitz"
