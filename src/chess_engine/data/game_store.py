"""Save filtered games to Parquet (in batches) and run the whole stream -> filter -> save pipeline.

Output of step 2, input of step 4. One row per game, moves stored as SAN lists:

    data/processed/games/lichess_2026-10_sample.parquet
    data/processed/games/lichess_2026-10_sample.stats.json   (counts, rejection reasons, timing)

Usage::

    result = prepare_games(cfg, month="2026-10")       # streams from Lichess in sample mode
    games = read_games(result.output_path)              # pandas DataFrame
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chess_engine.config import Config
from chess_engine.data.pgn_stream import FilterStats, GameFilter, GameRecord, filter_games, iter_raw_games
from chess_engine.data.sources import is_url, lichess_filename, lichess_url, open_pgn

log = logging.getLogger(__name__)

GAME_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("event", pa.string()),
        ("utc_date", pa.string()),
        ("utc_time", pa.string()),
        ("white", pa.string()),
        ("black", pa.string()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("white_title", pa.string()),
        ("black_title", pa.string()),
        ("result", pa.string()),
        ("termination", pa.string()),
        ("base_seconds", pa.int32()),
        ("increment_seconds", pa.int16()),
        ("speed", pa.string()),
        ("eco", pa.string()),
        ("opening", pa.string()),
        ("num_plies", pa.int16()),
        ("moves", pa.list_(pa.string())),
        ("clocks", pa.list_(pa.float32())),
    ]
)
TMP_SUFFIX = ".tmp"


class GameWriter:
    """Append :class:`GameRecord` objects to a Parquet file in row-group batches.

    Writes to ``<path>.tmp`` and renames to ``path`` on a clean exit, so an
    interrupted run never leaves a file that looks finished. On error the
    temporary file is removed.
    """

    def __init__(self, path: str | Path, batch_size: int = 10_000) -> None:
        self.path = Path(path)
        self.batch_size = batch_size
        self.rows_written = 0
        self._tmp = self.path.with_name(self.path.name + TMP_SUFFIX)
        self._buffer: list[GameRecord] = []
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> GameWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._tmp, GAME_SCHEMA, compression="zstd")
        return self

    def write(self, record: GameRecord) -> None:
        """Buffer one record, flushing a row group when the buffer is full."""
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered records as one row group."""
        if not self._buffer or self._writer is None:
            return
        self._writer.write_table(records_to_table(self._buffer))
        self.rows_written += len(self._buffer)
        self._buffer.clear()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            if exc_type is None:
                self.flush()
        finally:
            if self._writer is not None:
                self._writer.close()
        if exc_type is None:
            self._tmp.replace(self.path)
        else:
            self._tmp.unlink(missing_ok=True)


def records_to_table(records: Iterable[GameRecord]) -> pa.Table:
    """Convert records to a pyarrow Table with :data:`GAME_SCHEMA`."""
    columns: dict[str, list[Any]] = {name: [] for name in GAME_SCHEMA.names}
    for record in records:
        for name, value in record.to_dict().items():
            columns[name].append(value)
        columns["num_plies"].append(record.num_plies)
    return pa.Table.from_pydict(columns, schema=GAME_SCHEMA)


def write_games(records: Iterable[GameRecord], path: str | Path, batch_size: int = 10_000) -> int:
    """Write all ``records`` to ``path``; returns the number of rows written."""
    with GameWriter(path, batch_size=batch_size) as writer:
        for record in records:
            writer.write(record)
    return writer.rows_written


def read_games(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a games Parquet file into a DataFrame (optionally only some columns)."""
    return pq.read_table(path, columns=columns).to_pandas()


# --------------------------------------------------------------------------- the step-2 pipeline


@dataclass(frozen=True)
class PrepareResult:
    """What :func:`prepare_games` produced."""

    output_path: Path
    stats_path: Path
    stats: FilterStats
    seconds: float
    bytes_read: int


def games_output_path(cfg: Config, month: str) -> Path:
    """Where the filtered games for ``month`` go (``_sample`` suffix in sample mode)."""
    suffix = "_sample" if cfg.data.sample.enabled else ""
    return cfg.paths.processed_dir / "games" / f"lichess_{month}{suffix}.parquet"


def default_source(cfg: Config, month: str) -> str | Path:
    """The downloaded dump in ``raw_dir`` if it exists, else the Lichess URL (streamed)."""
    local = cfg.paths.raw_dir / lichess_filename(month)
    return local if local.is_file() else lichess_url(cfg.data.lichess.base_url, month)


def prepare_games(
    cfg: Config,
    month: str,
    source: str | Path | None = None,
    output_path: str | Path | None = None,
    max_games: int | None = None,
    show_progress: bool = True,
    **reader_options: object,
) -> PrepareResult:
    """Stream one month of games, filter them, and save the survivors to Parquet.

    Args:
        cfg: Project config (filters, sample mode, paths).
        month: ``YYYY-MM`` of the Lichess dump.
        source: Local file or URL; default: the downloaded dump if present, else stream from Lichess.
        output_path: Default: :func:`games_output_path`.
        max_games: Overrides the sample size. ``None`` = ``cfg.data.sample.max_games`` in sample
            mode, unlimited otherwise.
        show_progress: Show a tqdm progress bar.
        reader_options: Passed to the HTTP reader (retries, backoff, timeout) for URL sources.
    """
    source = source if source is not None else default_source(cfg, month)
    output = Path(output_path) if output_path is not None else games_output_path(cfg, month)
    if max_games is None and cfg.data.sample.enabled:
        max_games = cfg.data.sample.max_games
    game_filter = GameFilter.from_config(cfg.data)
    stats = FilterStats()

    log.info("Reading %s -> %s (max_games=%s)", source, output, max_games)
    start = time.perf_counter()
    options = reader_options if is_url(source) else {}
    with open_pgn(source, **options) as stream:
        progress = tqdm(total=max_games, unit=" games", desc=f"lichess {month}", disable=not show_progress)
        with GameWriter(output) as writer:
            for record in filter_games(iter_raw_games(stream.text), game_filter, stats, max_games=max_games):
                writer.write(record)
                progress.update(1)
                if stats.games_accepted % 1000 == 0:
                    progress.set_postfix(read=stats.games_read, mb=f"{stream.bytes_read / 2**20:.0f}")
        progress.close()
        bytes_read = stream.bytes_read
    seconds = time.perf_counter() - start

    stats_path = output.with_name(output.name.removesuffix(".parquet") + ".stats.json")
    report = {
        "month": month,
        "source": str(source),
        "output": str(output),
        "sample_mode": cfg.data.sample.enabled,
        "max_games": max_games,
        **stats.as_dict(),
        "compressed_mb_read": round(bytes_read / 2**20, 2),
        "seconds": round(seconds, 2),
        "games_read_per_second": round(stats.games_read / seconds, 1) if seconds else None,
        "filters": {
            "speeds": sorted(game_filter.speeds),
            **cfg.data.filters.model_dump(),
        },
    }
    stats_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PrepareResult(output, stats_path, stats, seconds, bytes_read)
