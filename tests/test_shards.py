"""Tests for writing and reading sample shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_engine.config import Config, DatasetConfig
from chess_engine.data.samples import SAMPLE_COLUMNS, Samples, game_to_samples
from chess_engine.data.shards import (
    FORMAT_VERSION,
    ShardFormatError,
    ShardWriter,
    read_shard,
    write_shard,
)

GAMES = [
    ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
    ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"],
    ["f3", "e5", "g4", "Qh4#"],
]


@pytest.fixture()
def samples(default_config: Config) -> Samples:
    parts = []
    for index, moves in enumerate(GAMES):
        game = {"white_elo": 1200 + 300 * index, "black_elo": 1500, "base_seconds": 180, "increment_seconds": 2,
                "result": "1-0", "moves": moves, "clocks": [180.0] * len(moves)}
        parts.append(game_to_samples(game, index, default_config.ratings, DatasetConfig()))
    return Samples.concat(parts)


def _assert_same(a: Samples, b: Samples) -> None:
    assert len(a) == len(b)
    for name in SAMPLE_COLUMNS:
        np.testing.assert_array_equal(a[name], b[name])
        assert a[name].dtype == b[name].dtype
    np.testing.assert_array_equal(a.legal_moves, b.legal_moves)


@pytest.mark.parametrize("compress", [True, False])
def test_roundtrip(samples: Samples, tmp_path: Path, compress: bool) -> None:
    path = write_shard(tmp_path / "s.npz", samples, compress=compress)
    _assert_same(read_shard(path), samples)
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_roundtrip(tmp_path: Path) -> None:
    assert len(read_shard(write_shard(tmp_path / "e.npz", Samples.empty()))) == 0


def test_wrong_suffix(samples: Samples, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".npz"):
        write_shard(tmp_path / "s.bin", samples)


def test_version_mismatch(samples: Samples, tmp_path: Path) -> None:
    path = tmp_path / "old.npz"
    np.savez(path, **samples.columns, legal_moves=samples.legal_moves, format_version=np.array(FORMAT_VERSION + 1))
    with pytest.raises(ShardFormatError, match="rebuild"):
        read_shard(path)


def test_missing_column(samples: Samples, tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    columns = {k: v for k, v in samples.columns.items() if k != "clock"}
    np.savez(path, **columns, legal_moves=samples.legal_moves, format_version=np.array(FORMAT_VERSION))
    with pytest.raises(ShardFormatError, match="clock"):
        read_shard(path)


def test_writer_splits_into_full_shards_plus_remainder(samples: Samples, tmp_path: Path) -> None:
    assert len(samples) == 22
    with ShardWriter(tmp_path, shard_size=8, seed=0) as writer:
        writer.add(samples.take(np.arange(5)))
        writer.add(samples.take(np.arange(5, 22)))
        writer.add(Samples.empty())
    assert [(s.file, s.positions) for s in writer.shards] == [
        ("shard_00000.npz", 8), ("shard_00001.npz", 8), ("shard_00002.npz", 6),
    ]
    assert writer.positions_written == 22
    loaded = Samples.concat([read_shard(tmp_path / s.file) for s in writer.shards])
    # Same samples, shuffled within each shard: compare after sorting by a unique key.
    def key(s: Samples) -> np.ndarray:
        return np.lexsort((s["ply"], s["game"]))
    _assert_same(loaded.take(key(loaded)), samples.take(key(samples)))


def test_writer_shuffles_within_a_shard(samples: Samples, tmp_path: Path) -> None:
    with ShardWriter(tmp_path, shard_size=100, seed=1) as writer:
        writer.add(samples)
    shard = read_shard(tmp_path / "shard_00000.npz")
    assert not np.array_equal(shard["ply"], samples["ply"])
    mask = shard.legal_mask()
    assert mask[np.arange(len(shard)), shard["move"]].all()  # legal moves stayed with their position


def test_writer_is_deterministic(samples: Samples, tmp_path: Path) -> None:
    for name in ("a", "b"):
        with ShardWriter(tmp_path / name, shard_size=10, seed=5) as writer:
            writer.add(samples)
    for file in ("shard_00000.npz", "shard_00001.npz", "shard_00002.npz"):
        _assert_same(read_shard(tmp_path / "a" / file), read_shard(tmp_path / "b" / file))


def test_writer_with_nothing_writes_nothing(tmp_path: Path) -> None:
    with ShardWriter(tmp_path / "x", shard_size=10, seed=0) as writer:
        pass
    assert writer.shards == []
    assert not (tmp_path / "x").exists()


def test_writer_rejects_bad_shard_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ShardWriter(tmp_path, shard_size=0, seed=0)


def test_background_writes_match_foreground(samples: Samples, tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ShardWriter(tmp_path / "fg", shard_size=8, seed=3) as fg:
        fg.add(samples)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with ShardWriter(tmp_path / "bg", shard_size=8, seed=3, executor=pool) as bg:
            bg.add(samples)
    assert fg.shards == bg.shards
    for shard in fg.shards:
        _assert_same(read_shard(tmp_path / "fg" / shard.file), read_shard(tmp_path / "bg" / shard.file))


def test_background_write_errors_surface_on_close(samples: Samples, tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("a file where the shard directory should be", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = ShardWriter(blocker, shard_size=8, seed=0, executor=pool)
        writer.add(samples)  # submits two shard writes that will fail
        with pytest.raises(OSError):
            writer.close()
