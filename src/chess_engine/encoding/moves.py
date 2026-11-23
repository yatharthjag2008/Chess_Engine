"""Map chess moves to fixed policy indices (and back) for the model's output layer.

The move vocabulary
-------------------
Every move in chess goes from one square to another along a queen line (rank,
file or diagonal) or a knight jump. So the vocabulary is:

* **1792 from->to pairs**: every (from, to) where ``to`` is reachable from ``from``
  by a queen or knight move on an empty board. Castling is the king's two-square
  move (``e1g1``), as python-chess writes it. Promotion to a **queen** uses the
  plain from->to entry (``e7e8`` means ``e7e8q`` when a pawn makes it).
* **66 underpromotions**: a pawn on the 7th rank moving straight or capturing to
  the 8th rank and becoming a knight, bishop or rook (8 + 7 + 7 = 22 moves x 3 pieces).

Total: **1858** indices (the same size as Leela Chess Zero's policy).

Moves are always encoded from the **side to move's point of view**: when Black is
to move, the board is mirrored vertically (rank 1 <-> rank 8) so Black's pawns
also "move up". This matches the board encoding in :mod:`chess_engine.encoding.board`.
"""

from __future__ import annotations

import chess
import numpy as np

UNDERPROMOTIONS: tuple[chess.PieceType, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_KNIGHT_JUMPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


def _is_queen_or_knight_move(from_sq: chess.Square, to_sq: chess.Square) -> bool:
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    if (df, dr) == (0, 0):
        return False
    return df == 0 or dr == 0 or abs(df) == abs(dr) or (df, dr) in _KNIGHT_JUMPS


def _build_vocabulary() -> tuple[chess.Move, ...]:
    moves = [
        chess.Move(from_sq, to_sq)
        for from_sq in chess.SQUARES
        for to_sq in chess.SQUARES
        if _is_queen_or_knight_move(from_sq, to_sq)
    ]
    for from_sq in chess.SquareSet(chess.BB_RANK_7):
        file = chess.square_file(from_sq)
        for df in (-1, 0, 1):
            if 0 <= file + df < 8:
                to_sq = chess.square(file + df, 7)
                moves.extend(chess.Move(from_sq, to_sq, promotion=piece) for piece in UNDERPROMOTIONS)
    return tuple(moves)


#: Every move the model can output, from White's point of view, in index order.
MOVE_VOCABULARY: tuple[chess.Move, ...] = _build_vocabulary()
NUM_MOVES: int = len(MOVE_VOCABULARY)

# Fast lookup tables.
#   _FROM_TO[from * 64 + to]           -> index of the plain (or queen-promotion) move, -1 if none
#   _UNDERPROMO[from * 64 + to, piece] -> index of an underpromotion (piece: 0=N, 1=B, 2=R), -1 if none
_FROM_TO = np.full(64 * 64, -1, dtype=np.int32)
_UNDERPROMO = np.full((64 * 64, len(UNDERPROMOTIONS)), -1, dtype=np.int32)
for _index, _move in enumerate(MOVE_VOCABULARY):
    _key = _move.from_square * 64 + _move.to_square
    if _move.promotion is None:
        _FROM_TO[_key] = _index
    else:
        _UNDERPROMO[_key, UNDERPROMOTIONS.index(_move.promotion)] = _index
del _index, _move, _key
