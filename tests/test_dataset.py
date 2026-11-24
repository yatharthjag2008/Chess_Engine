"""Tests for reading a built dataset as training batches."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import pytest

from chess_engine.data.dataset import (
    FEATURE_COLUMNS,
    DatasetMismatchError,
    PositionDataset,
    check_compatible,
    make_batch,
)
from chess_engine.data.dataset_builder import GAMES_INDEX_NAME, MANIFEST_NAME, BuildResult, build_dataset
from chess_engine.encoding import encode_board, legal_move_mask, move_to_index
from dataset_helpers import random_game_rows, tmp_config, write_game_rows


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> BuildResult:
    tmp_path = tmp_path_factory.mktemp("dataset")
    games = write_game_rows(random_game_rows(40, seed=11), tmp_path / "games.parquet")
    cfg = tmp_config(tmp_path, shard_size=300, games_per_task=8)
    return build_dataset(cfg, [games], tmp_path / "ds", show_progress=False)


def test_batches_decode_to_the_original_positions(built: BuildResult) -> None:
    """End to end: a stored sample decodes to exactly what encoding the real game position gives."""
    index = pd.read_parquet(built.output_dir / GAMES_INDEX_NAME).set_index("game")
    rows = {row["game_id"]: row for row in random_game_rows(40, seed=11)}
    dataset = PositionDataset(built.output_dir, "train")
    batch = dataset.batch(np.arange(min(200, len(dataset))))
    for i in range(len(batch)):
        game = int(batch.features["game"][i])
        ply = int(batch.features["ply"][i])
        board = chess.Board()
        moves = rows[index.loc[game, "game_id"]]["moves"]
        for san in moves[:ply]:
            board.push_san(san)
        np.testing.assert_array_equal(batch.planes[i], encode_board(board))
        np.testing.assert_array_equal(batch.legal_mask[i], legal_move_mask(board))
        assert batch.move[i] == move_to_index(board.parse_san(moves[ply]), board.turn)


def test_batch_shapes_and_dtypes(built: BuildResult) -> None:
    batch = PositionDataset(built.output_dir, "train").batch(np.arange(10))
    assert batch.planes.shape == (10, 20, 8, 8) and batch.planes.dtype == np.float32
    assert batch.legal_mask.shape == (10, 1858) and batch.legal_mask.dtype == bool
    assert batch.move.dtype == np.int64 and len(batch) == 10
    assert set(batch.features) == set(FEATURE_COLUMNS)
    assert "bucket" in batch.features and "clock" in batch.features and "pieces" not in batch.features


@pytest.mark.parametrize("in_memory", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_one_epoch_covers_every_sample_once(built: BuildResult, in_memory: bool, shuffle: bool) -> None:
    dataset = PositionDataset(built.output_dir, "train", in_memory=in_memory)
    batches = list(dataset.iter_batches(batch_size=64, shuffle=shuffle, seed=3))
    assert all(len(b) == 64 for b in batches[:-1]) and 0 < len(batches[-1]) <= 64
    keys = np.concatenate([b.features["game"].astype(np.int64) * 1000 + b.features["ply"] for b in batches])
    assert len(keys) == len(dataset) == len(set(keys.tolist()))
    for b in batches:
        assert b.legal_mask[np.arange(len(b)), b.move].all()


def test_drop_last(built: BuildResult) -> None:
    dataset = PositionDataset(built.output_dir, "train")
    batches = list(dataset.iter_batches(batch_size=100, drop_last=True))
    assert len(batches) == len(dataset) // 100 and all(len(b) == 100 for b in batches)
    streamed = list(PositionDataset(built.output_dir, "train", in_memory=False).iter_batches(100, drop_last=True))
    assert len(streamed) == len(dataset) // 100


def test_shuffle_is_seeded(built: BuildResult) -> None:
    dataset = PositionDataset(built.output_dir, "train")
    first = [b.move for b in dataset.iter_batches(50, shuffle=True, seed=1)]
    again = [b.move for b in dataset.iter_batches(50, shuffle=True, seed=1)]
    other = [b.move for b in dataset.iter_batches(50, shuffle=True, seed=2)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_unshuffled_streaming_matches_in_memory(built: BuildResult) -> None:
    memory = list(PositionDataset(built.output_dir, "train").iter_batches(64))
    stream = list(PositionDataset(built.output_dir, "train", in_memory=False).iter_batches(64))
    assert len(memory) == len(stream)
    for a, b in zip(memory, stream):
        np.testing.assert_array_equal(a.planes, b.planes)
        np.testing.assert_array_equal(a.move, b.move)


def test_max_shards_and_val_split(built: BuildResult) -> None:
    first = built.manifest["splits"]["train"]["shards"][0]["positions"]
    assert len(PositionDataset(built.output_dir, "train", max_shards=1)) == first
    val = PositionDataset(built.output_dir, "val")
    assert len(val) == built.manifest["splits"]["val"]["positions"]


def test_streaming_mode_has_no_random_access(built: BuildResult) -> None:
    with pytest.raises(RuntimeError, match="in_memory"):
        PositionDataset(built.output_dir, "train", in_memory=False).batch(np.arange(3))


def test_bad_arguments(built: BuildResult) -> None:
    with pytest.raises(ValueError, match="split"):
        PositionDataset(built.output_dir, "training")
    with pytest.raises(ValueError, match="batch_size"):
        next(PositionDataset(built.output_dir, "val").iter_batches(0))


def test_incompatible_manifest(built: BuildResult) -> None:
    manifest = json.loads((built.output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    check_compatible(manifest)
    with pytest.raises(DatasetMismatchError, match="1859 moves"):
        check_compatible({**manifest, "encoding": {**manifest["encoding"], "num_moves": 1859}})
    with pytest.raises(DatasetMismatchError, match="planes"):
        check_compatible({**manifest, "encoding": {**manifest["encoding"], "plane_names": ["x"]}})
    with pytest.raises(DatasetMismatchError, match="format version"):
        check_compatible({**manifest, "format_version": 0})


def test_make_batch_float16(built: BuildResult) -> None:
    samples = PositionDataset(built.output_dir, "val").samples
    assert make_batch(samples, np.arange(3), dtype=np.float16).planes.dtype == np.float16
