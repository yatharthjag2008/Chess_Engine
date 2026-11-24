"""Tests for building the sharded, split-by-game training dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chess_engine.data import dataset_builder
from chess_engine.data.dataset_builder import (
    GAMES_INDEX_NAME,
    DatasetExistsError,
    build_dataset,
    dataset_output_dir,
    default_games_paths,
    iter_game_tasks,
    read_manifest,
)
from chess_engine.data.samples import Samples
from chess_engine.data.shards import read_shard
from chess_engine.data.splits import SPLITS, assign_split
from dataset_helpers import random_game_rows, tmp_config, write_game_rows


@pytest.fixture()
def games_file(tmp_path: Path) -> Path:
    rows = random_game_rows(60, seed=3)
    rows[5]["moves"] = [*rows[5]["moves"][:10], "Ke9", *rows[5]["moves"][11:]]  # one broken game
    return write_game_rows(rows, tmp_path / "games.parquet")


def _load_split(output: Path, manifest: dict, split: str) -> Samples:
    return Samples.concat([read_shard(output / split / s["file"]) for s in manifest["splits"][split]["shards"]])


def test_build_writes_splits_index_and_manifest(tmp_path: Path, games_file: Path) -> None:
    cfg = tmp_config(tmp_path, shard_size=500, games_per_task=16)
    result = build_dataset(cfg, [games_file], tmp_path / "ds", show_progress=False)
    manifest = result.manifest
    assert result.output_dir == tmp_path / "ds"
    assert read_manifest(result.output_dir) == manifest
    assert not (tmp_path / "ds.tmp").exists()

    stats = manifest["stats"]
    assert stats["games_in"] == 60 and stats["games_used"] == 59
    assert stats["games_rejected"] == {"illegal_move": 1}
    assert sum(result.positions.values()) == stats["positions"]
    assert sum(info["games"] for info in manifest["splits"].values()) == 59
    assert manifest["encoding"]["num_moves"] == 1858 and manifest["encoding"]["num_planes"] == 20

    index = pd.read_parquet(result.output_dir / GAMES_INDEX_NAME)
    assert len(index) == 60 and list(index["game"]) == list(range(60))
    assert index.loc[5, "positions"] == 0
    assert index["positions"].sum() == stats["positions"]

    for split in SPLITS:
        info = manifest["splits"][split]
        samples = _load_split(result.output_dir, manifest, split)
        assert len(samples) == info["positions"] == sum(s["positions"] for s in info["shards"])
        assert all(s["positions"] <= 500 for s in info["shards"])
        assert sum(info["positions_per_bucket"]) == info["positions"]
        np.testing.assert_array_equal(np.bincount(samples["bucket"], minlength=8), info["positions_per_bucket"])
        mask = samples.legal_mask()
        assert mask[np.arange(len(samples)), samples["move"]].all()


def test_no_game_leaks_across_splits(tmp_path: Path, games_file: Path) -> None:
    cfg = tmp_config(tmp_path, shard_size=300, games_per_task=7)
    result = build_dataset(cfg, [games_file], tmp_path / "ds", show_progress=False)
    index = pd.read_parquet(result.output_dir / GAMES_INDEX_NAME)
    seen: dict[str, set[int]] = {}
    for split in SPLITS:
        games = set(_load_split(result.output_dir, result.manifest, split)["game"].tolist())
        expected = set(index.loc[(index["split"] == split) & (index["positions"] > 0), "game"])
        assert games == expected
        seen[split] = games
        # the split stored for each game is the deterministic one
        for game_id, stored in zip(index["game_id"], index["split"]):
            assert stored == assign_split(game_id, cfg.data.split)
    assert not (seen["train"] & seen["val"]) and not (seen["train"] & seen["test"]) and not (seen["val"] & seen["test"])
    # every position of a game is kept: positions per game match its plies
    for split in SPLITS:
        samples = _load_split(result.output_dir, result.manifest, split)
        counts = np.bincount(samples["game"], minlength=60)
        for game in seen[split]:
            assert counts[game] == index.loc[game, "positions"]


def test_build_is_deterministic(tmp_path: Path, games_file: Path) -> None:
    cfg = tmp_config(tmp_path, shard_size=400)
    a = build_dataset(cfg, [games_file], tmp_path / "a", show_progress=False)
    b = build_dataset(cfg, [games_file], tmp_path / "b", show_progress=False)
    assert a.manifest["splits"] == b.manifest["splits"]
    for split in SPLITS:
        for shard in a.manifest["splits"][split]["shards"]:
            sa, sb = read_shard(a.output_dir / split / shard["file"]), read_shard(b.output_dir / split / shard["file"])
            np.testing.assert_array_equal(sa["pieces"], sb["pieces"])
            np.testing.assert_array_equal(sa.legal_moves, sb.legal_moves)


def test_multiple_files_and_max_games(tmp_path: Path) -> None:
    first = write_game_rows(random_game_rows(20, seed=1, prefix="a"), tmp_path / "a.parquet")
    second = write_game_rows(random_game_rows(20, seed=2, prefix="b"), tmp_path / "b.parquet")
    cfg = tmp_config(tmp_path, games_per_task=8)
    result = build_dataset(cfg, [first, second], tmp_path / "ds", max_games=30, show_progress=False)
    index = pd.read_parquet(result.output_dir / GAMES_INDEX_NAME)
    assert len(index) == 30
    assert index["game_id"].iloc[19] == "a00019" and index["game_id"].iloc[20] == "b00000"


def test_iter_game_tasks_numbering(tmp_path: Path) -> None:
    path = write_game_rows(random_game_rows(10, seed=0), tmp_path / "g.parquet")
    tasks = list(iter_game_tasks([path, path], games_per_task=4, max_games=15))
    assert [(t.first_index, len(t.games)) for t in tasks] == [(0, 4), (4, 4), (8, 2), (10, 4), (14, 1)]


def test_position_filters_apply(tmp_path: Path, games_file: Path) -> None:
    cfg = tmp_config(tmp_path, skip_first_plies=10)
    result = build_dataset(cfg, [games_file], tmp_path / "ds", show_progress=False)
    assert result.manifest["stats"]["skipped_opening"] == 59 * 10
    for split in SPLITS:
        assert (_load_split(result.output_dir, result.manifest, split)["ply"] >= 10).all()


def test_existing_output_needs_overwrite(tmp_path: Path, games_file: Path) -> None:
    cfg = tmp_config(tmp_path)
    build_dataset(cfg, [games_file], tmp_path / "ds", max_games=10, show_progress=False)
    with pytest.raises(DatasetExistsError, match="overwrite"):
        build_dataset(cfg, [games_file], tmp_path / "ds", show_progress=False)
    result = build_dataset(cfg, [games_file], tmp_path / "ds", max_games=20, overwrite=True, show_progress=False)
    assert result.manifest["stats"]["games_in"] == 20


def test_missing_games_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_games"):
        build_dataset(tmp_config(tmp_path), [tmp_path / "nope.parquet"], tmp_path / "ds", show_progress=False)


def test_failure_leaves_nothing_behind(tmp_path: Path, games_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(dataset_builder, "encode_task", boom)
    with pytest.raises(RuntimeError, match="crashed"):
        build_dataset(tmp_config(tmp_path), [games_file], tmp_path / "ds", show_progress=False)
    assert not (tmp_path / "ds").exists() and not (tmp_path / "ds.tmp").exists()


def test_default_paths(tmp_path: Path) -> None:
    cfg = tmp_config(tmp_path)
    assert default_games_paths(cfg) == [tmp_path / "processed" / "games" / "lichess_2026-10_sample.parquet"]
    assert dataset_output_dir(cfg) == tmp_path / "processed" / "datasets" / "lichess_2026-10_sample"


def test_read_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        read_manifest(tmp_path)
