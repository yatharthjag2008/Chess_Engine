"""Tests for the board -> planes encoding."""

from __future__ import annotations

import chess
import numpy as np

from chess_engine.encoding.board import PIECE_TYPES, bitboards_to_planes, encode_pieces, piece_bitboards
from positions import all_boards

BOARDS = all_boards()

# ----------------------------------------------------------------------------- piece planes


def test_bitboards_to_planes_layout() -> None:
    planes = bitboards_to_planes(np.array([chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_E4], dtype=np.uint64))
    assert planes.shape == (4, 8, 8)
    assert planes.sum(axis=(1, 2)).tolist() == [1, 1, 1, 1]
    assert planes[0, 0, 0] == 1  # a1 -> rank 0, file 0
    assert planes[1, 0, 7] == 1  # h1 -> rank 0, file 7
    assert planes[2, 7, 0] == 1  # a8 -> rank 7, file 0
    assert planes[3, 3, 4] == 1  # e4 -> rank 3, file 4


def test_bitboards_to_planes_batched_shape() -> None:
    bbs = np.zeros((5, 12), dtype=np.uint64)
    assert bitboards_to_planes(bbs).shape == (5, 12, 8, 8)


def test_start_position_pieces_white_to_move() -> None:
    planes = encode_pieces(chess.Board())
    assert planes.shape == (12, 8, 8)
    assert planes.dtype == np.uint8
    assert planes.sum(axis=(1, 2)).tolist() == [8, 2, 2, 2, 1, 1] * 2
    assert planes[0, 1].tolist() == [1] * 8   # my pawns on my 2nd rank
    assert planes[6, 6].tolist() == [1] * 8   # their pawns on my 7th rank
    assert planes[5, 0, 4] == 1               # my king on e1
    assert planes[11, 7, 4] == 1              # their king on e8


def test_black_to_move_is_mirrored() -> None:
    board = chess.Board()
    board.push_san("e4")  # now Black to move
    planes = encode_pieces(board)
    # Black's own pawns are on the mover's 2nd rank, all eight of them.
    assert planes[0, 1].tolist() == [1] * 8
    # White's e-pawn (now "theirs") is on e4 -> from Black's view that is rank 5 (row 4), file e.
    assert planes[6, 4, 4] == 1
    assert planes[6, 1, 4] == 0  # e2 is empty
    # Black's king (mine) is on e8 -> mover's back rank, file e.
    assert planes[5, 0, 4] == 1


def test_piece_planes_match_board_everywhere() -> None:
    for board in BOARDS:
        planes = encode_pieces(board)
        for square, piece in board.piece_map().items():
            sq = square if board.turn == chess.WHITE else chess.square_mirror(square)
            offset = 0 if piece.color == board.turn else 6
            plane = offset + PIECE_TYPES.index(piece.piece_type)
            assert planes[plane, chess.square_rank(sq), chess.square_file(sq)] == 1
        assert planes.sum() == len(board.piece_map())
        assert planes.sum(axis=0).max() <= 1  # at most one piece per square


def test_piece_bitboards_mirror_symmetry() -> None:
    """A position and its colour-flipped mirror look identical to the side to move."""
    for board in BOARDS:
        assert np.array_equal(piece_bitboards(board), piece_bitboards(board.mirror()))


def test_empty_board() -> None:
    assert encode_pieces(chess.Board(None)).sum() == 0
