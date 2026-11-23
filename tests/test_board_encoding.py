"""Tests for the board -> planes encoding."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_engine.encoding.board import (
    NUM_PLANES,
    PIECE_TYPES,
    PLANE_INDEX,
    PLANE_NAMES,
    bitboards_to_planes,
    decode_board,
    encode_board,
    encode_boards,
    encode_pieces,
    piece_bitboards,
)
from positions import all_boards, special_boards

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


# ----------------------------------------------------------------------------- full encoding


def _plane(planes: np.ndarray, name: str) -> np.ndarray:
    return planes[PLANE_INDEX[name]]


def test_plane_names() -> None:
    assert NUM_PLANES == 20
    assert PLANE_NAMES[:2] == ("my_pawn", "my_knight")
    assert PLANE_NAMES[6] == "their_pawn"
    assert len(set(PLANE_NAMES)) == NUM_PLANES


def test_encode_board_shape_and_dtype() -> None:
    planes = encode_board(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes.dtype == np.float32
    assert encode_board(chess.Board(), dtype=np.uint8).dtype == np.uint8


def test_start_position_aux_planes() -> None:
    planes = encode_board(chess.Board())
    for name in ["white_to_move", "my_kingside_castle", "my_queenside_castle",
                 "their_kingside_castle", "their_queenside_castle", "ones"]:
        assert _plane(planes, name).min() == 1, name
    assert _plane(planes, "en_passant").sum() == 0
    assert _plane(planes, "fifty_move_clock").max() == 0


def test_colour_plane_for_black() -> None:
    board = chess.Board()
    board.push_san("Nf3")
    planes = encode_board(board)
    assert _plane(planes, "white_to_move").max() == 0


def test_castling_planes_are_from_movers_view() -> None:
    # White has only K, Black only q.
    white = encode_board(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"))
    assert _plane(white, "my_kingside_castle").min() == 1
    assert _plane(white, "my_queenside_castle").max() == 0
    assert _plane(white, "their_kingside_castle").max() == 0
    assert _plane(white, "their_queenside_castle").min() == 1
    black = encode_board(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1"))
    assert _plane(black, "my_kingside_castle").max() == 0
    assert _plane(black, "my_queenside_castle").min() == 1
    assert _plane(black, "their_kingside_castle").min() == 1


def test_en_passant_plane_white() -> None:
    planes = encode_board(special_boards()["white_en_passant"])  # e5xf6 possible
    ep = _plane(planes, "en_passant")
    assert ep.sum() == 1
    assert ep[5, 5] == 1  # f6 -> rank 6 (row 5), file f (col 5)


def test_en_passant_plane_black_is_mirrored() -> None:
    planes = encode_board(special_boards()["black_en_passant"])  # b4xa3 possible
    ep = _plane(planes, "en_passant")
    assert ep.sum() == 1
    assert ep[5, 0] == 1  # a3 mirrored -> a6 -> row 5, col 0


def test_en_passant_plane_empty_when_capture_impossible() -> None:
    board = chess.Board()
    board.push_san("e4")  # ep_square is e3, but no black pawn can capture
    assert board.ep_square == chess.E3
    assert _plane(encode_board(board), "en_passant").sum() == 0


def test_fifty_move_plane() -> None:
    assert _plane(encode_board(chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 37 80")), "fifty_move_clock").max() == \
        np.float32(0.37)
    assert _plane(encode_board(chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 150 80")), "fifty_move_clock").max() == 1


def test_values_are_binary_except_fifty_move_plane() -> None:
    fifty = PLANE_INDEX["fifty_move_clock"]
    for board in BOARDS:
        planes = encode_board(board)
        others = np.delete(planes, fifty, axis=0)
        assert set(np.unique(others)) <= {0.0, 1.0}
        assert 0 <= planes[fifty].min() <= planes[fifty].max() <= 1


# ----------------------------------------------------------------------------- symmetry, decoding, batches


def _comparable_fen(board: chess.Board) -> list[str]:
    """FEN fields the encoding keeps: pieces, turn, castling, legal ep square, halfmove (capped)."""
    fields = board.fen(en_passant="legal").split()
    fields[4] = str(min(int(fields[4]), 100))
    return fields[:5]


def test_mirror_symmetry_full_encoding() -> None:
    """board and board.mirror() differ only in the white_to_move plane."""
    color = PLANE_INDEX["white_to_move"]
    for board in BOARDS:
        ours, theirs = encode_board(board), encode_board(board.mirror())
        assert np.array_equal(np.delete(ours, color, 0), np.delete(theirs, color, 0)), board.fen()
        assert ours[color].max() != theirs[color].max()


def test_decode_inverts_encode() -> None:
    for board in BOARDS:
        decoded = decode_board(encode_board(board))
        assert _comparable_fen(decoded) == _comparable_fen(board), board.fen()
        assert decoded.is_valid()


def test_decode_keeps_legal_moves() -> None:
    for board in BOARDS:
        assert set(decode_board(encode_board(board)).legal_moves) == set(board.legal_moves)


def test_decode_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        decode_board(np.zeros((12, 8, 8)))


def test_encode_boards_batch() -> None:
    boards = BOARDS[:16]
    batch = encode_boards(boards)
    assert batch.shape == (16, NUM_PLANES, 8, 8)
    for i, board in enumerate(boards):
        assert np.array_equal(batch[i], encode_board(board))
    assert encode_boards([]).shape == (0, NUM_PLANES, 8, 8)


def test_transpositions_encode_identically() -> None:
    a, b = chess.Board(), chess.Board()
    for san in ["Nf3", "Nf6", "g3", "g6"]:
        a.push_san(san)
    for san in ["g3", "Nf6", "Nf3", "g6"]:  # same position, both end with a pawn move (clock 0)
        b.push_san(san)
    assert np.array_equal(encode_board(a), encode_board(b))
