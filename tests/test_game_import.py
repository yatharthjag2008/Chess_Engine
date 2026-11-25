"""Tests for importing games (pasted PGN; the Lichess / Chess.com fetchers use a local fake server)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine.data.game_import import ImportedGame, parse_pgn_text, speed_of_time_control

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"

CHESSCOM_PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.11.20"]
[White "PlayerOne"]
[Black "PlayerTwo"]
[Result "1-0"]
[WhiteElo "1234"]
[BlackElo "1250"]
[TimeControl "600"]
[Link "https://www.chess.com/game/live/123456789"]

1. e4 {[%clk 0:09:58.5]} 1... e5 {[%clk 0:09:57]} 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

VARIANT = '[Event "Rated Chess960 game"]\n[Variant "Chess960"]\n[FEN "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1"]\n[SetUp "1"]\n[Result "*"]\n\n1. e4 *\n'


def test_speed_of_time_control() -> None:
    assert speed_of_time_control("180+2") == "blitz"
    assert speed_of_time_control("600") == "rapid"            # Chess.com writes no "+0"
    assert speed_of_time_control("60") == "bullet"
    assert speed_of_time_control("1/86400") == "correspondence"
    assert speed_of_time_control("-") is None and speed_of_time_control(None) is None
    assert speed_of_time_control("abc") is None


def test_real_lichess_games() -> None:
    games = parse_pgn_text(FIXTURE.read_text(encoding="utf-8"), source="lichess")
    assert len(games) == 16 and len({g.id for g in games}) == 16
    first = games[0]
    assert isinstance(first, ImportedGame) and first.source == "lichess"
    assert first.url.startswith("https://lichess.org/") and first.id == first.url.rsplit("/", 1)[-1]
    assert first.speed in {"ultrabullet", "bullet", "blitz", "rapid", "classical"} and first.plies > 0
    assert any(g.rated for g in games) and first.pgn.startswith("[Event")
    assert "pgn" not in first.summary() and first.summary()["white"] == first.white


def test_chesscom_pgn_and_duplicates_and_variants() -> None:
    games = parse_pgn_text(CHESSCOM_PGN + "\n" + CHESSCOM_PGN + "\n" + VARIANT)
    assert len(games) == 1                                   # the repeat and the Chess960 game are dropped
    g = games[0]
    assert g.id == "123456789" and g.url == "https://www.chess.com/game/live/123456789"
    assert (g.white, g.black, g.white_elo, g.black_elo, g.result) == ("PlayerOne", "PlayerTwo", 1234, 1250, "1-0")
    assert g.speed == "rapid" and g.date == "2026.11.20" and g.plies == 7 and g.rated is None
    assert "[%clk 0:09:58.5]" in g.pgn                       # clocks are kept for the rating estimate


def test_pasted_games_without_ids_and_junk() -> None:
    pasted = '[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. d4 d5 *\n\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. c4 *\n'
    games = parse_pgn_text(pasted)
    assert len(games) == 2 and games[0].id != games[1].id and len(games[0].id) == 12
    assert games[0].date is None and games[0].speed is None and games[0].white_elo is None
    assert parse_pgn_text("") == [] and parse_pgn_text("hello there") == []
    assert parse_pgn_text("1. e4 e5 2. Ke7 *") == []           # illegal moves


@pytest.mark.parametrize("event, rated", [("Rated Blitz game", True), ("Casual Rapid game", False),
                                          ("Live Chess", None), ("Unrated blitz", False)])
def test_rated_from_the_event(event: str, rated: bool | None) -> None:
    (g,) = parse_pgn_text(f'[Event "{event}"]\n[Result "*"]\n\n1. e4 *\n')
    assert g.rated is rated
