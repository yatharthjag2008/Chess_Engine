"""Map the 1858-move vocabulary onto 73 "move type" planes per square, for a convolutional policy head.

Every move in the vocabulary (:data:`chess_engine.encoding.moves.MOVE_VOCABULARY`,
from the side to move's view) is a piece leaving one square in one way:

=========  =====  ==========================================================
planes     count  move type
=========  =====  ==========================================================
0 - 55     56     queen-line move: 8 directions x distance 1..7 (also covers
                  king moves, castling as a 2-square king move, pawn pushes and
                  captures, and queen promotions)
56 - 63    8      knight jump
64 - 72    9      underpromotion: (capture left, straight, capture right) x (N, B, R)
=========  =====  ==========================================================

So a head that outputs ``(73, 8, 8)`` scores "move type t from square s" at
``[t, rank(s), file(s)]``, and the 1858 policy logits are a fixed gather from
its flattened ``73 * 64`` outputs (:func:`policy_gather_index`). This is the
AlphaZero policy layout restricted to moves that can exist; a 1x1 convolution
produces it with a few thousand weights instead of a ~4M-weight dense layer, and
the same weights are shared by every square.
"""

from __future__ import annotations

import functools

import chess
import numpy as np

from chess_engine.encoding.moves import MOVE_VOCABULARY, UNDERPROMOTIONS

# (file step, rank step) per direction: N, NE, E, SE, S, SW, W, NW.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
KNIGHT_JUMPS: tuple[tuple[int, int], ...] = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
NUM_QUEEN_PLANES = len(QUEEN_DIRECTIONS) * 7
NUM_KNIGHT_PLANES = len(KNIGHT_JUMPS)
NUM_UNDERPROMOTION_PLANES = 3 * len(UNDERPROMOTIONS)
NUM_MOVE_TYPES = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + NUM_UNDERPROMOTION_PLANES  # 73


def move_type(move: chess.Move) -> int:
    """Move-type plane (0-72) of a vocabulary move (side to move's view).

    Raises:
        ValueError: if the move is not a queen-line move, a knight jump or a
            forward underpromotion.
    """
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    if move.promotion not in (None, chess.QUEEN, *UNDERPROMOTIONS):
        raise ValueError(f"{move.uci()} promotes to an impossible piece")
    if move.promotion in UNDERPROMOTIONS:
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError(f"{move.uci()} is not a forward underpromotion")
        return NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + (df + 1) * len(UNDERPROMOTIONS) + UNDERPROMOTIONS.index(move.promotion)
    if (df, dr) in KNIGHT_JUMPS:
        return NUM_QUEEN_PLANES + KNIGHT_JUMPS.index((df, dr))
    distance = max(abs(df), abs(dr))
    if distance == 0 or not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        raise ValueError(f"{move.uci()} is neither a queen-line move nor a knight jump")
    direction = QUEEN_DIRECTIONS.index((df // distance, dr // distance))
    return direction * 7 + (distance - 1)


@functools.cache
def policy_gather_index() -> np.ndarray:
    """``(1858,)`` int64: for each vocabulary move, its position in a flattened ``(73, 8, 8)`` output.

    Position = ``move_type * 64 + from_square`` (square = rank * 8 + file, matching
    the board planes' ``[rank, file]`` layout).
    """
    index = np.array([move_type(m) * 64 + m.from_square for m in MOVE_VOCABULARY], dtype=np.int64)
    index.flags.writeable = False
    return index
