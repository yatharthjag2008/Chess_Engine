"""End-to-end parsing of real Lichess games (tests/fixtures/lichess_real.pgn).

The fixture holds the first games of every speed from the 2026-10 Lichess dump,
plus BOT games, with player names and game ids anonymized. It pins down the
real-world format: headers, [%clk] comments, tournament events and so on.
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from chess_engine.config import Config
from chess_engine.data.pgn_stream import FilterStats, GameFilter, RawGame, filter_games, iter_raw_games, parse_movetext

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"


@pytest.fixture(scope="module")
def raw_games() -> list[RawGame]:
    with FIXTURE.open(encoding="utf-8") as f:
        return list(iter_raw_games(f))


def test_fixture_game_count(raw_games: list[RawGame]) -> None:
    assert len(raw_games) == 16
    assert all(g.header("Site").startswith("https://lichess.org/game/fixt") for g in raw_games)


def test_every_game_has_one_clock_per_move(raw_games: list[RawGame]) -> None:
    for game in raw_games:
        parsed = parse_movetext(game.movetext)
        assert parsed.num_plies > 0
        assert len(parsed.clocks) == parsed.num_plies, game.header("Site")
        assert parsed.result == game.header("Result")


def test_parsed_moves_replay_legally(raw_games: list[RawGame]) -> None:
    """Our fast tokenizer must produce exactly the SAN that python-chess accepts."""
    for game in raw_games:
        board = chess.Board()
        for san in parse_movetext(game.movetext).moves:
            board.push_san(san)  # raises if a token is not a legal move


def test_clocks_never_exceed_base_plus_increments(raw_games: list[RawGame]) -> None:
    for game in raw_games:
        base, inc = (int(x) for x in game.header("TimeControl").split("+"))
        parsed = parse_movetext(game.movetext)
        for ply, clock in enumerate(parsed.clocks):
            moves_by_this_player = ply // 2 + 1
            assert 0 <= clock <= base + inc * moves_by_this_player


def test_default_filter_keeps_human_blitz_only(raw_games: list[RawGame], default_config: Config) -> None:
    stats = FilterStats()
    records = list(filter_games(raw_games, GameFilter.from_config(default_config.data), stats))

    assert [r.game_id for r in records] == [f"fixt{i:04d}" for i in range(1, 9)]
    assert all(r.speed == "blitz" for r in records)
    assert all(not r.white_title and not r.black_title for r in records)
    assert stats.as_dict()["rejected_by_reason"] == {"speed": 7, "title": 1}
    first = records[0]
    assert (first.white_elo, first.black_elo, first.result, first.num_plies) == (1825, 1790, "1/2-1/2", 120)
    assert first.clocks[0] == 300.0
