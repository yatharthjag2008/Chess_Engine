"""Smoke tests: python-chess is installed and behaves as the rest of the project expects."""

from __future__ import annotations

import io

import chess
import chess.pgn


def test_play_a_legal_move() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    assert move in board.legal_moves
    board.push(move)
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_illegal_move_is_not_in_legal_moves() -> None:
    board = chess.Board()
    assert chess.Move.from_uci("e2e5") not in board.legal_moves


def test_starting_position_has_20_legal_moves() -> None:
    assert chess.Board().legal_moves.count() == 20


def test_san_round_trip() -> None:
    board = chess.Board()
    move = board.parse_san("Nf3")
    assert board.san(move) == "Nf3"
    assert move.uci() == "g1f3"


def test_detects_checkmate() -> None:
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:  # fool's mate
        board.push_san(san)
    assert board.is_checkmate()
    assert board.result() == "0-1"


def test_reads_pgn_with_lichess_style_headers() -> None:
    pgn = io.StringIO(
        '[Event "Rated Blitz game"]\n'
        '[WhiteElo "1500"]\n[BlackElo "1480"]\n'
        '[TimeControl "300+0"]\n[Termination "Normal"]\n\n'
        "1. e4 { [%clk 0:05:00] } e5 { [%clk 0:04:58] } 2. Nf3 Nc6 1-0\n"
    )
    game = chess.pgn.read_game(pgn)
    assert game is not None
    assert game.headers["TimeControl"] == "300+0"
    assert int(game.headers["WhiteElo"]) == 1500
    moves = list(game.mainline_moves())
    assert [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    first = game.next()
    assert first is not None and first.clock() == 300.0
