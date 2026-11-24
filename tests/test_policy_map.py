"""Tests for the move-type layout used by the convolutional policy head."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_engine.encoding.moves import MOVE_VOCABULARY, NUM_MOVES, move_to_index
from chess_engine.model.policy_map import (
    NUM_KNIGHT_PLANES,
    NUM_MOVE_TYPES,
    NUM_QUEEN_PLANES,
    move_type,
    policy_gather_index,
)


def test_plane_counts() -> None:
    assert (NUM_QUEEN_PLANES, NUM_KNIGHT_PLANES, NUM_MOVE_TYPES) == (56, 8, 73)


def test_gather_index_is_injective_and_in_range() -> None:
    index = policy_gather_index()
    assert index.shape == (NUM_MOVES,) and index.dtype == np.int64
    assert len(np.unique(index)) == NUM_MOVES
    assert index.min() >= 0 and index.max() < NUM_MOVE_TYPES * 64
    assert not index.flags.writeable


@pytest.mark.parametrize(
    "uci, expected",
    [
        ("e2e3", 0),           # north, distance 1
        ("e2e4", 1),           # north, distance 2
        ("a1h8", 1 * 7 + 6),   # north-east, distance 7
        ("e1g1", 2 * 7 + 1),   # castling = king two squares east
        ("h8a8", 6 * 7 + 6),   # west, distance 7
        ("g1f3", 56 + 7),      # knight (-1, +2)
        ("b1c3", 56 + 0),      # knight (+1, +2)
        ("a7a8n", 64 + 3),     # straight underpromotion to knight
        ("b7a8r", 64 + 2),     # capture-left underpromotion to rook
        ("g7h8b", 64 + 7),     # capture-right underpromotion to bishop
        ("e7e8q", 0),          # queen promotion is a plain north move
    ],
)
def test_known_move_types(uci: str, expected: int) -> None:
    assert move_type(chess.Move.from_uci(uci)) == expected


def test_gather_index_matches_vocabulary_order() -> None:
    index = policy_gather_index()
    for uci in ("e2e4", "g1f3", "a7a8n", "e1c1"):
        move = chess.Move.from_uci(uci)
        assert index[move_to_index(move)] == move_type(move) * 64 + move.from_square


def test_every_type_and_square_decodes_back_to_one_move() -> None:
    """Decoding (type, from-square) back to a move gives the vocabulary move itself."""
    index = policy_gather_index()
    for vocab_index, move in enumerate(MOVE_VOCABULARY):
        plane, from_square = divmod(int(index[vocab_index]), 64)
        assert from_square == move.from_square
        assert plane == move_type(move)


@pytest.mark.parametrize("uci", ["a1c4", "e2e2", "e7e8k", "e6e8n"])
def test_impossible_moves_raise(uci: str) -> None:
    with pytest.raises(ValueError):
        move_type(chess.Move.from_uci(uci))
