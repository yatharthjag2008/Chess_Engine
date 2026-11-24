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


def _table(keys, targets, buckets, legal, ranks=None):
    from eval_helpers import make_table

    n = len(keys)
    t = make_table(rank=ranks or [1] * n, prob=[0.5] * n, bucket=buckets, legal=legal)
    t.columns["key"] = np.array(keys, dtype=np.uint64)
    t.columns["target"] = np.array(targets, dtype=np.int16)
    return t


def test_most_common_move_with_bucket_table_and_fallbacks() -> None:
    from chess_engine.eval.baselines import most_common_move_baseline, random_baseline, summarize_baseline
    from chess_engine.eval.positions import combine_keys

    train_keys = np.array([1, 1, 1, 1, 2], dtype=np.uint64)
    train_moves = np.array([5, 5, 5, 7, 9])
    train_buckets = np.array([0, 0, 0, 2, 0])
    overall = MoveFrequencyTable.build(train_keys, train_moves)
    by_bucket = MoveFrequencyTable.build(combine_keys(train_keys, train_buckets), train_moves)

    # pos 1 by bucket 2 plays 7 (only the bucket table knows), pos 2 by bucket 2 falls back to the
    # overall table, pos 3 is new (random, 1/4).
    test = _table(keys=[1, 1, 2, 3], targets=[7, 5, 9, 0], buckets=[2, 0, 2, 0], legal=[20, 20, 20, 4])
    plain = most_common_move_baseline(test, overall)
    assert plain.hits.tolist() == [0.0, 1.0, 1.0, 0.25]
    assert plain.covered.tolist() == [True, True, True, False]
    same_rating = most_common_move_baseline(test, overall, by_bucket)
    assert same_rating.hits.tolist() == [1.0, 1.0, 1.0, 0.25]
    assert "same rating" in same_rating.name

    rand = random_baseline(test)
    assert rand.hits.tolist() == [0.05, 0.05, 0.05, 0.25] and rand.covered.all()

    summary = summarize_baseline(plain, test)
    assert summary["overall"]["coverage"] == 0.75 and summary["overall"]["top1"] == pytest.approx(2.25 / 4)
    assert summary["overall"]["top1_covered"] == pytest.approx(2 / 3)
    assert summary["overall"]["model_top1_covered"] == 1.0
    assert summary["buckets"]["high"]["top1"] == 0.5 and summary["buckets"]["mid"] == {"positions": 0}


def test_frequency_tables_from_real_samples(tiny_dataset) -> None:
    from chess_engine.data.dataset import PositionDataset
    from chess_engine.eval.baselines import build_frequency_tables

    train = PositionDataset(tiny_dataset, "train").samples
    overall, by_bucket = build_frequency_tables(train)
    assert 0 < len(overall) <= len(by_bucket) <= len(train)
    assert overall.total.sum() == by_bucket.total.sum() == len(train)
