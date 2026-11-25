"""Tests for reading one player's moves from PGN."""

from __future__ import annotations

from pathlib import Path

import chess

from chess_engine.eval.player_games import player_games_from_pgn, read_games

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"

PGN = """[Event "Rated blitz game"]
[White "Alice"]
[Black "bob"]
[WhiteElo "1450"]
[BlackElo "1520"]
[TimeControl "180+2"]
[Result "0-1"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:58] } Nc6 { [%clk 0:02:55] }
3. Bc4 { [%clk 0:02:50] } Nf6 { [%clk 0:02:40] } 0-1

[Event "Casual"]
[White "bob"]
[Black "Carol"]
[Result "1-0"]

1. d4 d5 2. c4 1-0
"""


def test_pick_a_player_by_name_across_games() -> None:
    games = player_games_from_pgn(PGN, player="BOB")
    assert [g.color for g in games] == [chess.BLACK, chess.WHITE]
    first = games[0]
    assert first.name == "bob" and first.rating == 1520 and first.opp_rating == 1450 and first.result == "0-1"
    assert [m.uci() for m in first.moves] == ["e7e5", "b8c6", "g8f6"]
    assert all(b.turn == chess.BLACK for b in first.boards)
    after_nf3 = first.boards[1]  # the position before ...Nc6: 1. e4 e5 2. Nf3
    assert after_nf3.piece_at(chess.F3) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert after_nf3.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK)
    assert (first.base_seconds, first.increment) == (180, 2)
    # before each Black move: Black's own clock after its previous move, White's after White's last move
    assert first.clocks == [180, 180, 175] and first.opp_clocks == [180, 178, 170]
    second = games[1]
    assert second.rating is None and second.clocks is None and len(second) == 2 and second.base_seconds == 180


def test_pick_by_colour_or_both_sides() -> None:
    whites = player_games_from_pgn(PGN, color=chess.WHITE)
    assert [g.name for g in whites] == ["Alice", "bob"]
    both = player_games_from_pgn(PGN)
    assert len(both) == 4 and {len(g) for g in both} == {1, 2, 3}
    assert player_games_from_pgn(PGN, player="nobody") == []


def test_real_lichess_games() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    games = read_games(text)
    assert len(games) == 16
    sides = player_games_from_pgn(text)
    assert len(sides) >= 30
    timed = [g for g in sides if g.clocks is not None]
    assert timed and all(len(g.clocks) == len(g.moves) for g in timed)
    for g in sides:
        for board, move in zip(g.boards, g.moves):
            assert move in board.legal_moves and board.turn == g.color


def test_custom_start_position_and_broken_games() -> None:
    pgn = '[FEN "7k/8/6K1/8/8/8/R7/1R6 w - - 0 1"]\n[SetUp "1"]\n[Result "1-0"]\n\n1. Ra7 Kg8 2. Rb8# 1-0\n'
    (black,) = player_games_from_pgn(pgn, color=chess.BLACK)
    assert [m.uci() for m in black.moves] == ["h8g8"]
    assert player_games_from_pgn("1. e4 e5 2. Ke7 *\n") == []   # an illegal move: the game is skipped
    assert player_games_from_pgn("not a pgn at all") == []       # no moves: nothing to estimate from
