"""Tests for the Game wrapper: move input, results, takebacks and PGN."""

from __future__ import annotations

import io

import chess
import chess.pgn
import pytest

from chess_engine.bot.game import Game, IllegalMoveError, parse_move


def test_parse_san_and_uci() -> None:
    board = chess.Board()
    assert parse_move(board, "Nf3") == chess.Move.from_uci("g1f3")
    assert parse_move(board, " e4 ") == chess.Move.from_uci("e2e4")
    assert parse_move(board, "G1F3") == chess.Move.from_uci("g1f3")
    promo = chess.Board("8/4P1k1/8/8/8/8/8/K7 w - - 0 1")
    assert parse_move(promo, "e8=Q") == chess.Move.from_uci("e7e8q")
    assert parse_move(promo, "e7e8n") == chess.Move.from_uci("e7e8n")
    assert parse_move(promo, "e7e8") == chess.Move.from_uci("e7e8q")  # promotion defaults to a queen
    for bad, reason in (("", "empty"), ("hello", "not a move"), ("e2e5", "not legal")):
        with pytest.raises(IllegalMoveError, match=reason):
            parse_move(board, bad)


def test_fools_mate_ends_in_checkmate() -> None:
    game = Game("Alice", "Bot", black_elo=1200)
    for move in ("f3", "e5", "g4", "Qh4#"):
        game.push(move)
    assert game.outcome.result == "0-1" and game.outcome.termination == "checkmate"
    assert game.outcome.winner == chess.BLACK and game.is_over
    assert game.outcome.describe() == "Black wins by checkmate (0-1)"
    with pytest.raises(IllegalMoveError, match="over"):
        game.push("a3")
    assert game.move_list() == "1. f3 e5 2. g4 Qh4#"


def test_stalemate_repetition_and_insufficient_material() -> None:
    stalemate = Game(fen="k7/2K5/8/1Q6/8/8/8/8 w - - 0 1")
    stalemate.push("Qb6")
    assert stalemate.outcome.termination == "stalemate" and stalemate.outcome.result == "1/2-1/2"

    repeat = Game()
    for move in ("Nf3", "Nf6", "Ng1", "Ng8") * 2:
        repeat.push(move)
    assert repeat.outcome.termination == "threefold repetition"

    bare = Game(fen="8/8/4k3/8/8/2K5/8/8 w - - 0 1")
    assert bare.outcome.termination == "insufficient material"


def test_resign_time_forfeit_and_takeback() -> None:
    game = Game()
    game.push("e4")
    game.push(chess.Move.from_uci("e7e5"))
    assert game.takeback(2) == ["e4", "e5"] and game.board == chess.Board()
    with pytest.raises(ValueError):
        game.takeback(1)
    game.push("d4")
    assert game.resign(chess.WHITE).describe() == "Black wins by resignation (0-1)"
    with pytest.raises(ValueError):
        game.resign(chess.BLACK)
    with pytest.raises(ValueError, match="ended"):
        game.takeback(1)

    flag = Game()
    assert flag.forfeit_on_time(chess.WHITE).result == "0-1"
    # Black has only a king left: White's flag fall is a draw.
    lone = Game(fen="8/8/4k3/8/8/2K5/3Q4/8 w - - 0 1")
    assert lone.forfeit_on_time(chess.WHITE).result == "1/2-1/2"
    with pytest.raises(IllegalMoveError):
        Game().push(chess.Move.from_uci("e2e5"))


def test_pgn_round_trip() -> None:
    game = Game("You", "Bot 1500", black_elo=1500, event="Game vs. bot")
    for move in ("e4", "c5", "Nf3", "d6"):
        game.push(move)
    game.resign(chess.WHITE)
    parsed = chess.pgn.read_game(io.StringIO(game.pgn()))
    assert parsed.headers["White"] == "You" and parsed.headers["BlackElo"] == "1500"
    assert parsed.headers["Result"] == "0-1" and parsed.headers["Termination"] == "resignation"
    assert [m.uci() for m in parsed.mainline_moves()] == ["e2e4", "c7c5", "g1f3", "d7d6"]
    assert "WhiteElo" not in parsed.headers

    text = Game(fen="8/8/4k3/8/8/2K5/3Q4/8 b - - 3 40").pgn()
    assert '[FEN "8/8/4k3/8/8/2K5/3Q4/8 b - - 3 40"]' in text and '[Result "*"]' in text


def test_move_list_from_a_black_to_move_position() -> None:
    game = Game(fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    game.push("e5")
    game.push("Nf3")
    assert game.move_list() == "1... e5 2. Nf3"


def test_repetition_ends_the_game_only_when_it_happens() -> None:
    game = Game()
    for move in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"):
        game.push(move)
    assert not game.is_over  # Ng8 would repeat a third time, but it has not been played
    game.push("Ng8")
    assert game.outcome.termination == "threefold repetition"
    fifty = Game(fen="8/8/4k3/8/8/2K5/3R4/8 w - - 99 80")
    fifty.push("Rd1")
    assert fifty.outcome.termination == "fifty-move rule"
