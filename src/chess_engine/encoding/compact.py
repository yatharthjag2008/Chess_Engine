"""A compact, fixed-size form of a position for storing millions of training samples.

The full input (:func:`chess_engine.encoding.board.encode_board`) is 20 x 8 x 8
float32 = 5 KB per position. Almost all of it is 0/1 bits, so on disk each
position is stored as

=========  ===========  =====================================================
field      dtype/shape  content (all from the side to move's point of view)
=========  ===========  =====================================================
pieces     uint64 (12)  the 12 piece bitboards: my P N B R Q K, their P N B R Q K
flags      uint8        bit 0 white to move; bits 1-4 castling rights:
                        my kingside, my queenside, their kingside, their queenside
ep         int8         en passant target square (0-63) if the capture is legal, else -1
halfmove   uint8        halfmove clock, capped at 255
=========  ===========  =====================================================

= 99 bytes, about 50x smaller. :func:`planes_from_compact` turns a whole batch
back into planes with a few numpy operations, giving exactly what
``encode_board`` would have produced (tested on every position in the test suite).
"""

from __future__ import annotations

from typing import NamedTuple

import chess
import numpy as np

from chess_engine.encoding.board import (
    NUM_PIECE_PLANES,
    NUM_PLANES,
    PLANE_INDEX,
    bitboards_to_planes,
    decode_board,
    piece_bitboards,
)

FLAG_WHITE_TO_MOVE = 1 << 0
FLAG_MY_KINGSIDE = 1 << 1
FLAG_MY_QUEENSIDE = 1 << 2
FLAG_THEIR_KINGSIDE = 1 << 3
FLAG_THEIR_QUEENSIDE = 1 << 4
NO_EN_PASSANT = -1

# (flag bit, plane index) for every plane that is all-0 or all-1.
_FLAG_PLANES = (
    (FLAG_WHITE_TO_MOVE, PLANE_INDEX["white_to_move"]),
    (FLAG_MY_KINGSIDE, PLANE_INDEX["my_kingside_castle"]),
    (FLAG_MY_QUEENSIDE, PLANE_INDEX["my_queenside_castle"]),
    (FLAG_THEIR_KINGSIDE, PLANE_INDEX["their_kingside_castle"]),
    (FLAG_THEIR_QUEENSIDE, PLANE_INDEX["their_queenside_castle"]),
)
_EP_PLANE = PLANE_INDEX["en_passant"]
_FIFTY_PLANE = PLANE_INDEX["fifty_move_clock"]
_ONES_PLANE = PLANE_INDEX["ones"]


class CompactBoard(NamedTuple):
    """One position in compact form (see the module docstring)."""

    pieces: np.ndarray  # (12,) uint64
    flags: int
    ep: int
    halfmove: int


def compact_board(board: chess.Board) -> CompactBoard:
    """The compact form of ``board`` (mover-oriented, like :func:`encode_board`)."""
    us, them = board.turn, not board.turn
    flags = FLAG_WHITE_TO_MOVE if us == chess.WHITE else 0
    if board.has_kingside_castling_rights(us):
        flags |= FLAG_MY_KINGSIDE
    if board.has_queenside_castling_rights(us):
        flags |= FLAG_MY_QUEENSIDE
    if board.has_kingside_castling_rights(them):
        flags |= FLAG_THEIR_KINGSIDE
    if board.has_queenside_castling_rights(them):
        flags |= FLAG_THEIR_QUEENSIDE

    ep = NO_EN_PASSANT
    if board.ep_square is not None and board.has_legal_en_passant():
        ep = board.ep_square if us == chess.WHITE else chess.square_mirror(board.ep_square)
    return CompactBoard(piece_bitboards(board), flags, ep, min(board.halfmove_clock, 255))


def planes_from_compact(
    pieces: np.ndarray,
    flags: np.ndarray,
    ep: np.ndarray,
    halfmove: np.ndarray,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Rebuild a batch of ``(N, NUM_PLANES, 8, 8)`` planes from compact arrays of length ``N``.

    The result equals ``encode_board`` of each position, plane for plane.
    """
    pieces = np.asarray(pieces, dtype=np.uint64)
    flags = np.asarray(flags, dtype=np.uint8)
    ep = np.asarray(ep, dtype=np.int16)
    halfmove = np.asarray(halfmove)
    n = len(pieces)
    if pieces.shape != (n, NUM_PIECE_PLANES) or not len(flags) == len(ep) == len(halfmove) == n:
        raise ValueError(
            f"expected pieces (N, {NUM_PIECE_PLANES}) and flags/ep/halfmove of length N, got "
            f"{pieces.shape}, {flags.shape}, {ep.shape}, {halfmove.shape}"
        )

    planes = np.zeros((n, NUM_PLANES, 8, 8), dtype=dtype)
    planes[:, :NUM_PIECE_PLANES] = bitboards_to_planes(pieces)
    for bit, plane in _FLAG_PLANES:
        planes[:, plane] = ((flags & bit) != 0)[:, None, None]
    has_ep = np.flatnonzero(ep >= 0)
    planes[has_ep, _EP_PLANE, ep[has_ep] // 8, ep[has_ep] % 8] = 1
    planes[:, _FIFTY_PLANE] = (np.minimum(halfmove, 100) / 100).astype(dtype)[:, None, None]
    planes[:, _ONES_PLANE] = 1
    return planes


def board_from_compact(pieces: np.ndarray, flags: int, ep: int, halfmove: int) -> chess.Board:
    """Rebuild one board from its compact form (debugging / inspecting stored samples).

    As with :func:`decode_board`, the halfmove clock is capped at 100 and the
    fullmove number is not stored (it is set to 1).
    """
    planes = planes_from_compact(
        np.asarray(pieces, dtype=np.uint64)[None], np.array([flags]), np.array([ep]), np.array([halfmove])
    )
    return decode_board(planes[0])
