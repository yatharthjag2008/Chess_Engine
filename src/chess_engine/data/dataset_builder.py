"""Turn filtered games (step 2 Parquet) into training-sample shards split by game (step 4).

Output layout::

    data/processed/datasets/lichess_2026-10_sample/
        manifest.json          what was built: sizes per split and bucket, config, encoding, timing
        games.parquet          one row per input game: its index, id, split, ratings, positions kept
        train/shard_00000.npz  shuffled samples (see chess_engine.data.samples for the columns)
        val/shard_00000.npz
        test/shard_00000.npz

Every game goes to exactly one split (:mod:`chess_engine.data.splits`), so no
position of a validation or test game is ever trained on. The whole directory is
built under ``<name>.tmp`` and renamed when complete.

Usage::

    result = build_dataset(cfg)                       # games of cfg.data.lichess.months
    result = build_dataset(cfg, games_paths=[...], output_dir=..., max_games=1000)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chess_engine.config import Config, DatasetConfig, RatingsConfig
from chess_engine.data.game_store import games_output_path
from chess_engine.data.samples import Samples, SampleStats, games_to_samples
from chess_engine.data.shards import FORMAT_VERSION, ShardWriter
from chess_engine.data.splits import SPLITS, assign_splits
from chess_engine.encoding.board import NUM_PLANES, PLANE_NAMES
from chess_engine.encoding.moves import NUM_MOVES

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
GAMES_INDEX_NAME = "games.parquet"
#: Columns of the step-2 games table that the builder needs.
GAME_COLUMNS = ["game_id", "white_elo", "black_elo", "base_seconds", "increment_seconds", "result", "moves", "clocks"]

GAMES_INDEX_SCHEMA = pa.schema(
    [
        ("game", pa.int32()),
        ("game_id", pa.string()),
        ("split", pa.string()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("base_seconds", pa.int32()),
        ("increment_seconds", pa.int16()),
        ("positions", pa.int32()),
    ]
)


class DatasetExistsError(FileExistsError):
    """The output directory already exists and ``overwrite`` was not set."""


def default_games_paths(cfg: Config) -> list[Path]:
    """The step-2 games file of every configured month."""
    return [games_output_path(cfg, month) for month in cfg.data.lichess.months]


def dataset_output_dir(cfg: Config) -> Path:
    """``processed_dir/datasets/lichess_<months>[_sample]``."""
    months = "+".join(cfg.data.lichess.months)
    suffix = "_sample" if cfg.data.sample.enabled else ""
    return cfg.paths.processed_dir / "datasets" / f"lichess_{months}{suffix}"


# --------------------------------------------------------------------------- reading games in tasks


@dataclass(frozen=True)
class _Task:
    first_index: int
    games: list[dict[str, Any]]


def count_games(paths: Sequence[Path]) -> int:
    """Total rows of the games files (from Parquet metadata, without reading them)."""
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def iter_game_tasks(paths: Sequence[Path], games_per_task: int, max_games: int | None = None) -> Iterator[_Task]:
    """The games of ``paths`` in order, ``games_per_task`` at a time, numbered from 0."""
    index = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=games_per_task, columns=GAME_COLUMNS):
            games = batch.to_pylist()
            if max_games is not None:
                games = games[: max_games - index]
            if not games:
                return
            yield _Task(index, games)
            index += len(games)
            if max_games is not None and index >= max_games:
                return


def encode_task(task: _Task, ratings: RatingsConfig, dataset: DatasetConfig) -> tuple[Samples, SampleStats, list[int]]:
    """Samples of one task's games (runs in a worker process when ``workers > 1``)."""
    return games_to_samples(task.games, task.first_index, ratings, dataset)


def _run_tasks(
    tasks: Iterable[_Task], ratings: RatingsConfig, dataset: DatasetConfig, workers: int
) -> Iterator[tuple[_Task, tuple[Samples, SampleStats, list[int]]]]:
    """Encode tasks and yield the results **in task order**, using ``workers`` processes.

    At most ``2 * workers`` tasks are in flight, so memory stays bounded on full
    months (millions of games) and the output is identical for any worker count.
    """
    if workers <= 1:
        for task in tasks:
            yield task, encode_task(task, ratings, dataset)
        return
    pool = ProcessPoolExecutor(max_workers=workers)
    pending: deque[tuple[_Task, Future[tuple[Samples, SampleStats, list[int]]]]] = deque()
    try:
        for task in tasks:
            pending.append((task, pool.submit(encode_task, task, ratings, dataset)))
            if len(pending) >= 2 * workers:
                done, future = pending.popleft()
                yield done, future.result()
        while pending:
            done, future = pending.popleft()
            yield done, future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------- the builder


@dataclass(frozen=True)
class BuildResult:
    """What :func:`build_dataset` produced."""

    output_dir: Path
    manifest: dict[str, Any]
    seconds: float

    @property
    def positions(self) -> dict[str, int]:
        return {name: info["positions"] for name, info in self.manifest["splits"].items()}


def build_dataset(
    cfg: Config,
    games_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    max_games: int | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    compress: bool = True,
    show_progress: bool = True,
) -> BuildResult:
    """Replay every game into samples and write shuffled shards per split.

    Args:
        cfg: Project config (``data.split``, ``data.dataset``, ``ratings``).
        games_paths: Step-2 games Parquet files; default: :func:`default_games_paths`.
        output_dir: Default: :func:`dataset_output_dir`.
        max_games: Use only the first ``max_games`` games (default: all).
        workers: Encoding processes; default ``cfg.data.dataset.resolved_workers()``.
        overwrite: Replace an existing dataset directory.
        compress: zlib-compress shards (about 3x smaller, slightly slower to load).
        show_progress: Show a tqdm progress bar.

    Raises:
        FileNotFoundError: a games file does not exist (run ``scripts/prepare_games.py`` first).
        DatasetExistsError: ``output_dir`` exists and ``overwrite`` is false.
    """
    start = time.perf_counter()
    paths = [Path(p) for p in (games_paths if games_paths is not None else default_games_paths(cfg))]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"games file not found: {path} (run scripts/prepare_games.py first)")
    output = Path(output_dir) if output_dir is not None else dataset_output_dir(cfg)
    if output.exists() and not overwrite:
        raise DatasetExistsError(f"{output} already exists (pass overwrite=True / --overwrite to rebuild)")
    workers = workers if workers is not None else cfg.data.dataset.resolved_workers()
    split_cfg, dataset_cfg, ratings = cfg.data.split, cfg.data.dataset, cfg.ratings

    tmp = output.with_name(output.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        total = count_games(paths)
        total = min(total, max_games) if max_games is not None else total
        writers = {
            name: ShardWriter(tmp / name, dataset_cfg.shard_size, seed=split_cfg.seed + i, compress=compress)
            for i, name in enumerate(SPLITS)
        }
        stats = SampleStats()
        games_per_split = dict.fromkeys(SPLITS, 0)
        bucket_counts = {name: np.zeros(ratings.num_buckets, dtype=np.int64) for name in SPLITS}
        index_columns: dict[str, list[Any]] = {name: [] for name in GAMES_INDEX_SCHEMA.names}

        tasks = iter_game_tasks(paths, dataset_cfg.games_per_task, max_games)
        progress = tqdm(total=total, unit=" games", desc="dataset", disable=not show_progress)
        for task, (samples, task_stats, per_game) in _run_tasks(tasks, ratings, dataset_cfg, workers):
            stats.merge(task_stats)
            game_splits = assign_splits((g["game_id"] for g in task.games), split_cfg)
            split_codes = np.array([SPLITS.index(s) for s in game_splits], dtype=np.int8)
            sample_split = split_codes[samples["game"] - task.first_index] if len(samples) else split_codes[:0]
            for code, name in enumerate(SPLITS):
                rows = np.flatnonzero(sample_split == code)
                if len(rows):
                    part = samples.take(rows)
                    writers[name].add(part)
                    bucket_counts[name] += np.bincount(part["bucket"], minlength=ratings.num_buckets)
            for offset, (game, split, positions) in enumerate(zip(task.games, game_splits, per_game)):
                if positions:
                    games_per_split[split] += 1
                index_columns["game"].append(task.first_index + offset)
                index_columns["game_id"].append(game["game_id"])
                index_columns["split"].append(split)
                index_columns["white_elo"].append(game["white_elo"])
                index_columns["black_elo"].append(game["black_elo"])
                index_columns["base_seconds"].append(game["base_seconds"])
                index_columns["increment_seconds"].append(game["increment_seconds"])
                index_columns["positions"].append(positions)
            progress.update(len(task.games))
            progress.set_postfix(positions=f"{stats.positions:,}")
        progress.close()

        shards = {name: writer.close() for name, writer in writers.items()}
        pq.write_table(pa.Table.from_pydict(index_columns, schema=GAMES_INDEX_SCHEMA), tmp / GAMES_INDEX_NAME,
                       compression="zstd")
        seconds = time.perf_counter() - start
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sources": [str(p) for p in paths],
            "encoding": {"num_planes": NUM_PLANES, "plane_names": list(PLANE_NAMES), "num_moves": NUM_MOVES},
            "ratings": {"bucket_edges": ratings.bucket_edges, "bucket_labels": ratings.bucket_labels()},
            "split_config": split_cfg.model_dump(),
            "dataset_config": dataset_cfg.model_dump(),
            "splits": {
                name: {
                    "games": games_per_split[name],
                    "positions": writers[name].positions_written,
                    "positions_per_bucket": bucket_counts[name].tolist(),
                    "shards": [{"file": s.file, "positions": s.positions} for s in shards[name]],
                }
                for name in SPLITS
            },
            "stats": stats.as_dict(),
            "workers": workers,
            "seconds": round(seconds, 1),
        }
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if output.exists():
        shutil.rmtree(output)
    tmp.rename(output)
    log.info("Built %s: %s positions in %.1fs", output, {k: v["positions"] for k, v in manifest["splits"].items()},
             seconds)
    return BuildResult(output, manifest, seconds)


def read_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """The ``manifest.json`` of a built dataset."""
    path = Path(dataset_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {dataset_dir} (build it with scripts/build_dataset.py)")
    return json.loads(path.read_text(encoding="utf-8"))
