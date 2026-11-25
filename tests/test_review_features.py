"""Tests for the one-move facts used in explanations."""

from __future__ import annotations

import chess

from chess_engine.review.features import (
    captured_piece,
    gives_mate,
    hanging_pieces,
    newly_hanging,
    piece_name,
)


def test_captures_including_en_passant() -> None:
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert captured_piece(board, chess.Move.from_uci("e5f6")) == chess.PAWN      # en passant
    assert captured_piece(board, chess.Move.from_uci("e5e6")) is None
    rich = chess.Board("4k3/8/8/3q4/4N3/8/8/4K3 w - - 0 1")
    assert captured_piece(rich, chess.Move.from_uci("e4c3")) is None
    assert piece_name(chess.QUEEN) == "queen"


def test_gives_mate() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    assert gives_mate(board, chess.Move.from_uci("d1d8"))
    assert not gives_mate(board, chess.Move.from_uci("d1d7"))


def test_hanging_pieces() -> None:
    # The knight on e5 is attacked by a cheaper piece (the d6 pawn). The bishop on c4 is attacked by the
    # queen but defended by the knight, so it is safe; with the knight gone it hangs.
    board = chess.Board("4k3/5q2/3p4/4N3/2B5/8/8/R3K3 b - - 0 1")
    assert [h.describe() for h in hanging_pieces(board, chess.WHITE)] == ["knight on e5"]
    board.remove_piece_at(chess.E5)
    assert [h.describe() for h in hanging_pieces(board, chess.WHITE)] == ["bishop on c4"]
    assert hanging_pieces(chess.Board(), chess.WHITE) == []


def test_newly_hanging_after_a_move() -> None:
    board = chess.Board("4k3/8/8/3p4/8/8/8/2B1K3 w - - 0 1")
    moved = newly_hanging(board, chess.Move.from_uci("c1g5"))
    assert moved == []                                                   # g5 is not attacked
    board = chess.Board("4k3/8/4p3/8/8/8/8/2B1K3 w - - 0 1")
    moved = newly_hanging(board, chess.Move.from_uci("c1f4"))
    assert moved == []
    board = chess.Board("4k3/8/8/4p3/8/8/8/2B1K3 w - - 0 1")
    moved = newly_hanging(board, chess.Move.from_uci("c1d4"))
    assert [h.describe() for h in moved] == ["bishop on d4"]           # put en prise by the e5 pawn
