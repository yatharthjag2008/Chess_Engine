"""Tests for the streaming PGN parser and game filters."""

from __future__ import annotations

import pytest

from chess_engine.data.pgn_stream import (
    Movetext,
    RawGame,
    TimeControl,
    iter_raw_games,
    parse_clock,
    parse_header_line,
    parse_movetext,
    parse_time_control,
    speed_from_seconds,
    speed_of,
)

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


# ----------------------------------------------------------------------------- splitting games

TWO_GAMES = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc12345"]
[Result "1-0"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Qh5 { [%clk 0:02:59] } 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/def67890"]
[Result "0-1"]

1. d4 d5 0-1
"""


def _games(text: str) -> list[RawGame]:
    return list(iter_raw_games(text.splitlines(keepends=True)))


def test_parse_header_line() -> None:
    assert parse_header_line('[WhiteElo "1825"]') == ("WhiteElo", "1825")
    assert parse_header_line('[Opening "King\'s Pawn Game"]') == ("Opening", "King's Pawn Game")
    assert parse_header_line('[Event ""]') == ("Event", "")


def test_parse_header_line_unescapes() -> None:
    assert parse_header_line(r'[White "a \"quoted\" name \\ x"]') == ("White", 'a "quoted" name \\ x')


@pytest.mark.parametrize("line", ["1. e4 e5", "[%clk 0:03:00] }", "[Event]", '[Event "x"', "{ comment }"])
def test_parse_header_line_rejects_non_headers(line: str) -> None:
    assert parse_header_line(line) is None


def test_iter_raw_games_splits_games() -> None:
    games = _games(TWO_GAMES)
    assert len(games) == 2
    assert games[0].header("Site") == "https://lichess.org/abc12345"
    assert games[0].movetext.startswith("1. e4 { [%clk 0:03:00] }")
    assert games[0].movetext.endswith("1-0")
    assert games[1].headers == {"Event": "Rated Bullet game", "Site": "https://lichess.org/def67890", "Result": "0-1"}
    assert games[1].movetext == "1. d4 d5 0-1"
    assert games[1].header("Missing", "default") == "default"


def test_iter_raw_games_handles_crlf_bom_and_no_trailing_newline() -> None:
    text = chr(0xFEFF) + TWO_GAMES.replace("\n", "\r\n").rstrip()
    games = _games(text)
    assert [g.header("Result") for g in games] == ["1-0", "0-1"]
    assert games[0].header("Event") == "Rated Blitz game"


def test_iter_raw_games_joins_wrapped_movetext() -> None:
    text = '[Event "x"]\n\n1. e4 e5\n2. Nf3 { [%clk 0:01:00] }\n[%clk 0:01:00] Nc6 *\n'
    (game,) = _games(text)
    # A wrapped line starting with "[%clk" is movetext, not a header.
    assert game.movetext == "1. e4 e5 2. Nf3 { [%clk 0:01:00] } [%clk 0:01:00] Nc6 *"


def test_iter_raw_games_game_without_movetext() -> None:
    text = '[Event "a"]\n\n[Event "b"]\n\n1. e4 *\n'
    games = _games(text)
    assert [(g.header("Event"), g.movetext) for g in games] == [("a", ""), ("b", "1. e4 *")]


def test_iter_raw_games_ignores_leading_junk_and_empty_input() -> None:
    assert _games("") == []
    assert _games("\n\nsome junk\n") == []
    assert len(_games("junk line\n" + TWO_GAMES)) == 2


def test_iter_raw_games_is_lazy() -> None:
    def lines():
        # The first game is complete once the second game's first header line arrives.
        yield from TWO_GAMES.splitlines(keepends=True)[:7]
        raise AssertionError("read further than needed")  # pragma: no cover

    first = next(iter_raw_games(lines()))
    assert first.header("Result") == "1-0"


# ----------------------------------------------------------------------------- movetext


def test_parse_movetext_lichess_format() -> None:
    text = "1. d4 { [%clk 0:05:00] } 1... Nf6 { [%clk 0:04:58] } 2. Nf3 { [%clk 0:04:57.5] } 1/2-1/2"
    parsed = parse_movetext(text)
    assert parsed == Movetext(["d4", "Nf6", "Nf3"], [300.0, 298.0, 297.5], "1/2-1/2")
    assert parsed.num_plies == 3


def test_parse_movetext_with_evals_and_glyphs() -> None:
    # Analysed Lichess games carry [%eval] next to [%clk] and glyphs like ?! on moves.
    text = (
        "1. e4 { [%eval 0.18] [%clk 0:03:00] } 1... f6?! { [%eval 0.9] [%clk 0:03:00] } "
        "2. d4 { [%eval 0.8] [%clk 0:02:58] } 2... g5?? { [%eval #1] [%clk 0:02:57] } 3. Qh5# 1-0"
    )
    parsed = parse_movetext(text)
    assert parsed.moves == ["e4", "f6", "d4", "g5", "Qh5#"]
    assert parsed.clocks == []  # the last move has no clock, so clocks are dropped entirely
    assert parsed.result == "1-0"


def test_parse_movetext_without_clocks() -> None:
    parsed = parse_movetext("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *")
    assert parsed.moves == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert parsed.clocks == []
    assert parsed.result == "*"


def test_parse_movetext_skips_variations_nags_and_comments() -> None:
    text = "1. e4 $1 { best by test } (1. d4 d5 (1... Nf6) 2. c4) 1... c5 ; line comment" + chr(10) + "2. Nf3 0-1"
    parsed = parse_movetext(text)
    assert parsed.moves == ["e4", "c5", "Nf3"]
    assert parsed.result == "0-1"


def test_parse_movetext_castling_promotion_and_no_spaces() -> None:
    parsed = parse_movetext("1.e4 e5 2.O-O O-O-O 3.exd8=Q+ Kxd8 4.a8=N")
    assert parsed.moves == ["e4", "e5", "O-O", "O-O-O", "exd8=Q+", "Kxd8", "a8=N"]
    assert parsed.result is None


def test_parse_movetext_empty() -> None:
    assert parse_movetext("") == Movetext([], [], None)
    assert parse_movetext("0-1") == Movetext([], [], "0-1")


@pytest.mark.parametrize(
    "comment, seconds",
    [
        ("{ [%clk 0:03:00] }", 180.0),
        ("{ [%clk 1:00:05] }", 3605.0),
        ("{ [%clk 0:00:01.4] }", 1.4),
        ("{ [%eval 0.3] [%clk 0:0:9] }", 9.0),
        ("{ no clock }", None),
    ],
)
def test_parse_clock(comment: str, seconds: float | None) -> None:
    result = parse_clock(comment)
    if seconds is None:
        assert result is None
    else:
        assert result == pytest.approx(seconds)
