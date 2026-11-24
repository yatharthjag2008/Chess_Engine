"""Tests for the baselines (most common move, ...)."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.eval.baselines import MoveFrequencyTable


def test_most_common_move_per_position() -> None:
    keys = np.array([5, 5, 5, 9, 9, 7, 5], dtype=np.uint64)
    moves = np.array([3, 4, 4, 1, 2, 8, 3])  # key 5: move 3 twice, 4 twice -> tie, lowest index wins
    table = MoveFrequencyTable.build(keys, moves)
    assert table.keys.tolist() == [5, 7, 9]
    assert table.best_move.tolist() == [3, 8, 1]
    assert table.best_count.tolist() == [2, 1, 1] and table.total.tolist() == [4, 1, 2]


def test_clear_winner_beats_lower_index() -> None:
    table = MoveFrequencyTable.build(np.array([1, 1, 1], dtype=np.uint64), np.array([9, 9, 2]))
    assert table.best_move.tolist() == [9] and table.best_count.tolist() == [2]


def test_lookup() -> None:
    table = MoveFrequencyTable.build(np.array([10, 20, 20], dtype=np.uint64), np.array([1, 2, 2]))
    found, move, total = table.lookup(np.array([20, 15, 10, 99], dtype=np.uint64))
    assert found.tolist() == [True, False, True, False]
    assert move.tolist() == [2, -1, 1, -1] and total.tolist() == [2, 0, 1, 0]


def test_empty_table_and_errors() -> None:
    empty = MoveFrequencyTable.build(np.zeros(0, dtype=np.uint64), np.zeros(0))
    assert len(empty) == 0
    found, move, _ = empty.lookup(np.array([1], dtype=np.uint64))
    assert not found.any() and move.tolist() == [-1]
    with pytest.raises(ValueError):
        MoveFrequencyTable.build(np.zeros(2, dtype=np.uint64), np.zeros(3))


def test_large_random_table_matches_a_dictionary() -> None:
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 500, 20_000).astype(np.uint64)
    moves = rng.integers(0, 6, 20_000)
    table = MoveFrequencyTable.build(keys, moves)
    for key in rng.choice(500, 30, replace=False):
        counts = np.bincount(moves[keys == key], minlength=6)
        found, move, total = table.lookup(np.array([key], dtype=np.uint64))
        assert found[0] and move[0] == counts.argmax() and total[0] == counts.sum()
