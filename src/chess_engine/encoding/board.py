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


# --------------------------------------------------------------------------- full encoding

#: Name of each input plane, in order. ``NUM_PLANES == len(PLANE_NAMES)``.
PLANE_NAMES: tuple[str, ...] = (
    *(f"my_{chess.piece_name(pt)}" for pt in PIECE_TYPES),
    *(f"their_{chess.piece_name(pt)}" for pt in PIECE_TYPES),
    "white_to_move",       # all 1 if White is to move (colours still play differently in practice)
    "my_kingside_castle",
    "my_queenside_castle",
    "their_kingside_castle",
    "their_queenside_castle",
    "en_passant",          # 1 on the en passant target square, only if the capture is actually legal
    "fifty_move_clock",    # halfmove clock / 100, capped at 1
    "ones",                # constant 1: lets convolutions see the board edge through zero padding
)
NUM_PLANES: int = len(PLANE_NAMES)
PLANE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PLANE_NAMES)}

_COLOR, _MY_K, _MY_Q, _THEIR_K, _THEIR_Q, _EP, _FIFTY, _ONES = range(NUM_PIECE_PLANES, NUM_PLANES)


def encode_board(board: chess.Board, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Encode ``board`` as a ``(NUM_PLANES, 8, 8)`` array, from the side to move's point of view.

    See :data:`PLANE_NAMES` for what each plane holds. Every plane is 0/1 except
    ``fifty_move_clock`` (a fraction), which an integer ``dtype`` truncates to 0/1.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=dtype)
    planes[:NUM_PIECE_PLANES] = encode_pieces(board)

    us, them = board.turn, not board.turn
    if us == chess.WHITE:
        planes[_COLOR] = 1
    planes[_MY_K] = board.has_kingside_castling_rights(us)
    planes[_MY_Q] = board.has_queenside_castling_rights(us)
    planes[_THEIR_K] = board.has_kingside_castling_rights(them)
    planes[_THEIR_Q] = board.has_queenside_castling_rights(them)

    # python-chess sets ep_square after every double push; only encode it when the capture
    # is legal, so identical positions (same legal moves) always encode identically.
    if board.ep_square is not None and board.has_legal_en_passant():
        ep = board.ep_square if us == chess.WHITE else chess.square_mirror(board.ep_square)
        planes[_EP, chess.square_rank(ep), chess.square_file(ep)] = 1

    planes[_FIFTY] = min(board.halfmove_clock, 100) / 100
    planes[_ONES] = 1
    return planes
