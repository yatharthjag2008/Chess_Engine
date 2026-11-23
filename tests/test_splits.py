"""Tests for the deterministic by-game train/val/test split."""

from __future__ import annotations

from collections import Counter

import pytest

from chess_engine.config import Config, SplitConfig
from chess_engine.data.splits import SPLITS, assign_split, assign_splits, split_value

IDS = [f"game{i:06d}" for i in range(20_000)]


def test_split_value_is_deterministic_and_in_range() -> None:
    values = [split_value(game_id, seed=42) for game_id in IDS[:1000]]
    assert values == [split_value(game_id, seed=42) for game_id in IDS[:1000]]
    assert all(0.0 <= v < 1.0 for v in values)
    assert len(set(values)) == len(values)


def test_split_value_depends_on_seed() -> None:
    assert [split_value(g, 1) for g in IDS[:50]] != [split_value(g, 2) for g in IDS[:50]]


def test_fractions_match_config(default_config: Config) -> None:
    split = default_config.data.split
    counts = Counter(assign_splits(IDS, split))
    assert set(counts) == set(SPLITS)
    for name in SPLITS:
        expected = getattr(split, name) * len(IDS)
        # Binomial standard deviation is at most ~sqrt(0.09 * 20000) ~ 42; allow 5 sigma.
        assert abs(counts[name] - expected) < 5 * (expected * (1 - expected / len(IDS))) ** 0.5


def test_each_game_gets_exactly_one_split(default_config: Config) -> None:
    splits = assign_splits(IDS[:500], default_config.data.split)
    assert len(splits) == 500
    assert all(s in SPLITS for s in splits)


def test_adding_games_does_not_move_existing_ones(default_config: Config) -> None:
    split = default_config.data.split
    before = assign_splits(IDS[:1000], split)
    after = assign_splits(IDS, split)[:1000]
    assert before == after


def test_empty_validation_split() -> None:
    split = SplitConfig(train=0.8, val=0.0, test=0.2, seed=7)
    counts = Counter(assign_splits(IDS[:5000], split))
    assert counts["val"] == 0
    assert counts["train"] + counts["test"] == 5000


@pytest.mark.parametrize("seed", [0, 42, 12345])
def test_boundaries_follow_split_value(seed: int) -> None:
    split = SplitConfig(train=0.5, val=0.25, test=0.25, seed=seed)
    for game_id in IDS[:300]:
        value = split_value(game_id, seed)
        expected = "train" if value < 0.5 else "val" if value < 0.75 else "test"
        assert assign_split(game_id, split) == expected
