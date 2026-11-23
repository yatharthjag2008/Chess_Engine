"""Board and move encoding: positions -> tensors, moves <-> indices, legal-move masks.

Both encodings are from the **side to move's point of view** (Black's positions
and moves are mirrored), so the network only ever learns "my move".

    board  encode_board(board) -> (20, 8, 8) planes; decode_board() inverts it
    moves  move_to_index / index_to_move over a 1858-move vocabulary; legal_move_mask(board)
"""

from chess_engine.encoding.board import (
    NUM_PLANES,
    PLANE_INDEX,
    PLANE_NAMES,
    decode_board,
    encode_board,
    encode_boards,
)
from chess_engine.encoding.moves import (
    MOVE_VOCABULARY,
    NUM_MOVES,
    index_to_move,
    legal_move_indices,
    legal_move_mask,
    move_to_index,
)

__all__ = [
    "MOVE_VOCABULARY",
    "NUM_MOVES",
    "NUM_PLANES",
    "PLANE_INDEX",
    "PLANE_NAMES",
    "decode_board",
    "encode_board",
    "encode_boards",
    "index_to_move",
    "legal_move_indices",
    "legal_move_mask",
    "move_to_index",
]
