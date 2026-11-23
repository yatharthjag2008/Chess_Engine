"""Tests for the compact stored position format (must decode to exactly encode_board's planes)."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_engine.encoding import encode_board, encode_boards
from chess_engine.encoding.compact import (
    FLAG_MY_KINGSIDE,
    FLAG_MY_QUEENSIDE,
    FLAG_THEIR_KINGSIDE,
    FLAG_THEIR_QUEENSIDE,
    FLAG_WHITE_TO_MOVE,
    NO_EN_PASSANT,
    board_from_compact,
    compact_board,
    planes_from_compact,
)
from positions import all_boards, special_boards

BOARDS = all_boards()


def _stack(boards: list[chess.Board]) -> tuple[np.ndarray, ...]:
    compact = [compact_board(b) for b in boards]
    return (
        np.stack([c.pieces for c in compact]),
        np.array([c.flags for c in compact], dtype=np.uint8),
        np.array([c.ep for c in compact], dtype=np.int8),
        np.array([c.halfmove for c in compact], dtype=np.uint8),
    )


def test_batch_decode_equals_encode_board_for_every_position() -> None:
    planes = planes_from_compact(*_stack(BOARDS))
    np.testing.assert_array_equal(planes, encode_boards(BOARDS))


def test_decode_respects_dtype() -> None:
    planes = planes_from_compact(*_stack(BOARDS[:5]), dtype=np.float16)
    assert planes.dtype == np.float16
    np.testing.assert_array_equal(planes, encode_boards(BOARDS[:5], dtype=np.float16))


def test_start_position_flags() -> None:
    c = compact_board(chess.Board())
    assert c.flags == (FLAG_WHITE_TO_MOVE | FLAG_MY_KINGSIDE | FLAG_MY_QUEENSIDE
                       | FLAG_THEIR_KINGSIDE | FLAG_THEIR_QUEENSIDE)
    assert c.ep == NO_EN_PASSANT
    assert c.halfmove == 0
    assert c.pieces.dtype == np.uint64 and c.pieces.shape == (12,)


def test_partial_castling_for_black_is_seen_from_black() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1")
    c = compact_board(board)
    assert c.flags == FLAG_MY_QUEENSIDE | FLAG_THEIR_KINGSIDE


@pytest.mark.parametrize("name, expected", [("white_en_passant", chess.F6), ("black_en_passant", chess.A6)])
def test_en_passant_square_is_mover_oriented(name: str, expected: int) -> None:
    # Black's target a3 is mirrored to a6, like the board planes.
    assert compact_board(special_boards()[name]).ep == expected


def test_illegal_en_passant_is_not_stored() -> None:
    board = chess.Board()
    board.push_san("e4")  # python-chess sets ep_square = e3, but no black pawn can capture
    assert board.ep_square is not None
    assert compact_board(board).ep == NO_EN_PASSANT


def test_halfmove_clock_capped_at_255_and_planes_at_100() -> None:
    board = chess.Board("8/8/8/4k3/8/8/8/4K2R w - - 300 200")
    c = compact_board(board)
    assert c.halfmove == 255
    np.testing.assert_array_equal(planes_from_compact(*_stack([board]))[0], encode_board(board))


def test_board_from_compact_roundtrip() -> None:
    for board in BOARDS:
        rebuilt = board_from_compact(*compact_board(board))
        assert rebuilt.board_fen() == board.board_fen()
        assert rebuilt.turn == board.turn
        assert rebuilt.castling_rights == board.castling_rights
        assert set(rebuilt.legal_moves) == set(board.legal_moves)


def test_shape_mismatch_raises() -> None:
    pieces, flags, ep, halfmove = _stack(BOARDS[:3])
    with pytest.raises(ValueError, match="expected"):
        planes_from_compact(pieces, flags[:2], ep, halfmove)
