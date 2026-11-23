"""Encode a chess.Board as a stack of 8x8 feature planes for the neural network.

Orientation
-----------
Positions are always seen **from the side to move**: when Black is to move the
board is mirrored vertically (rank 8 becomes rank 1) and the colours swap roles,
so "my pawns" always move up the board. This halves what the network has to
learn and matches the move encoding in :mod:`chess_engine.encoding.moves`.

In every plane, ``plane[row, col]`` is the square on rank ``row`` and file
``col`` *from the mover's point of view* (row 0 = the mover's back rank,
col 0 = the a-file).
"""

from __future__ import annotations

import chess
import numpy as np

PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING,
)
NUM_PIECE_PLANES = 2 * len(PIECE_TYPES)  # mine then theirs


def piece_bitboards(board: chess.Board) -> np.ndarray:
    """12 bitboards (uint64): the mover's P, N, B, R, Q, K, then the opponent's, mover-oriented.

    For Black to move, every bitboard is flipped vertically.
    """
    us, them = board.turn, not board.turn
    boards = [board.pieces_mask(pt, us) for pt in PIECE_TYPES] + [board.pieces_mask(pt, them) for pt in PIECE_TYPES]
    if us == chess.BLACK:
        boards = [chess.flip_vertical(bb) for bb in boards]
    return np.array(boards, dtype=np.uint64)


def bitboards_to_planes(bitboards: np.ndarray) -> np.ndarray:
    """Unpack uint64 bitboards of shape ``(..., N)`` into uint8 planes of shape ``(..., N, 8, 8)``.

    Bit ``i`` of a bitboard (python-chess square ``i`` = rank * 8 + file) lands at
    ``[rank, file]``.
    """
    bitboards = np.ascontiguousarray(bitboards, dtype="<u8")
    bits = np.unpackbits(bitboards.view(np.uint8), bitorder="little")
    return bits.reshape(*bitboards.shape, 8, 8)


def encode_pieces(board: chess.Board) -> np.ndarray:
    """The 12 piece planes of ``board`` as a ``(12, 8, 8)`` uint8 array (1 = piece present)."""
    return bitboards_to_planes(piece_bitboards(board))
