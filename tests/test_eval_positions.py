"""Tests for position keys and rebuilding boards from dataset samples."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chess_engine.data.dataset import PositionDataset
from chess_engine.encoding.compact import compact_board
from chess_engine.encoding.moves import index_to_move
from chess_engine.eval.positions import board_of, combine_keys, position_keys, sample_keys


def _keys(*boards: chess.Board) -> np.ndarray:
    compact = [compact_board(b) for b in boards]
    return position_keys(np.stack([c.pieces for c in compact]), np.array([c.flags for c in compact]),
                         np.array([c.ep for c in compact]))


def test_same_position_same_key_whatever_the_move_counters() -> None:
    a = chess.Board()
    b = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):  # back to the start position, counters advanced
        b.push_san(san)
    keys = _keys(a, b)
    assert keys[0] == keys[1]


def test_different_positions_differ() -> None:
    start = chess.Board()
    e4 = chess.Board()
    e4.push_san("e4")
    no_castle = chess.Board(chess.STARTING_FEN.replace("KQkq", "Qkq"))
    black_to_move = chess.Board(chess.STARTING_FEN.replace(" w ", " b "))
    keys = _keys(start, e4, no_castle, black_to_move)
    assert len(set(keys.tolist())) == 4


def test_keys_on_a_dataset_are_mostly_unique_except_openings(tiny_dataset: Path) -> None:
    samples = PositionDataset(tiny_dataset, "train").samples
    keys = sample_keys(samples)
    assert keys.dtype == np.uint64 and len(keys) == len(samples)
    start = keys[samples["ply"] == 0]
    assert len(set(start.tolist())) == 1  # every game starts from the same position
    later = keys[samples["ply"] >= 12]
    assert len(set(later.tolist())) > 0.95 * len(later)  # random games rarely repeat a position


def test_combine_keys() -> None:
    keys = np.array([1, 1, 2], dtype=np.uint64)
    combined = combine_keys(keys, np.array([0, 1, 0]))
    assert len(set(combined.tolist())) == 3
    assert combine_keys(keys[:1], np.array([0]))[0] == combined[0]


def test_board_of_rebuilds_the_position_and_its_move(tiny_dataset: Path) -> None:
    samples = PositionDataset(tiny_dataset, "test").samples
    for i in range(0, len(samples), max(1, len(samples) // 25)):
        board = board_of(samples, i)
        assert board.turn == bool(samples["flags"][i] & 1)
        move = index_to_move(int(samples["move"][i]), board)
        assert move in board.legal_moves
        assert board.legal_moves.count() == int(samples["legal_count"][i])
