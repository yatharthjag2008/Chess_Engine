"""Save samples as NumPy shard files and read them back.

A shard is one ``.npz`` file holding up to ``shard_size`` samples: one array per
column of :data:`chess_engine.data.samples.SAMPLE_COLUMNS`, the ragged
``legal_moves`` array and a ``format_version``. Positions inside a shard are
shuffled (positions from one game are strongly correlated, so they should not
sit next to each other in a training batch).

Files are written to ``<name>.tmp`` first and renamed once complete.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chess_engine.data.samples import SAMPLE_COLUMNS, Samples

FORMAT_VERSION = 1
SHARD_SUFFIX = ".npz"


class ShardFormatError(Exception):
    """A shard file is missing columns or was written by an incompatible version."""


def write_shard(path: str | Path, samples: Samples, compress: bool = True) -> Path:
    """Write ``samples`` to ``path`` (``.npz``); returns the path."""
    path = Path(path)
    if path.suffix != SHARD_SUFFIX:
        raise ValueError(f"shard files must end in {SHARD_SUFFIX}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arrays = {**samples.columns, "legal_moves": samples.legal_moves,
              "format_version": np.array(FORMAT_VERSION)}
    save = np.savez_compressed if compress else np.savez
    try:
        with tmp.open("wb") as f:  # a file object stops numpy from appending another ".npz"
            save(f, **arrays)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_shard(path: str | Path) -> Samples:
    """Load a shard written by :func:`write_shard`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"]) if "format_version" in data else None
        if version != FORMAT_VERSION:
            raise ShardFormatError(f"{path}: format version {version}, expected {FORMAT_VERSION}; rebuild the dataset")
        missing = [name for name in (*SAMPLE_COLUMNS, "legal_moves") if name not in data]
        if missing:
            raise ShardFormatError(f"{path}: missing arrays {missing}")
        return Samples({name: data[name] for name in SAMPLE_COLUMNS}, data["legal_moves"])


# --------------------------------------------------------------------------- writing many shards


@dataclass(frozen=True)
class ShardInfo:
    """One written shard: its file name (relative to the split directory) and sample count."""

    file: str
    positions: int


class ShardWriter:
    """Collect samples for one split and write them as shuffled shards of ``shard_size`` positions.

    Samples are buffered until a shard is full, shuffled (seeded) and written as
    ``shard_00000.npz``, ``shard_00001.npz``, ... in ``directory``. Call
    :meth:`close` (or use ``with``) to write the last, partial shard.

    With an ``executor`` (e.g. a ``ThreadPoolExecutor``), files are compressed and
    written in the background; zlib releases the GIL, so the caller keeps working
    meanwhile. :meth:`close` waits for every write and re-raises any error. The
    shuffle happens before handing off, so the output does not depend on timing.
    """

    def __init__(
        self,
        directory: str | Path,
        shard_size: int,
        seed: int,
        compress: bool = True,
        executor: Executor | None = None,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.directory = Path(directory)
        self.shard_size = shard_size
        self.compress = compress
        self.shards: list[ShardInfo] = []
        self._rng = np.random.default_rng(seed)
        self._buffer: list[Samples] = []
        self._buffered = 0
        self._executor = executor
        self._pending: list[Future[Path]] = []

    @property
    def positions_written(self) -> int:
        return sum(s.positions for s in self.shards)

    def add(self, samples: Samples) -> None:
        """Buffer ``samples``; writes every shard that becomes full."""
        if len(samples) == 0:
            return
        self._buffer.append(samples)
        self._buffered += len(samples)
        while self._buffered >= self.shard_size:
            pending = Samples.concat(self._buffer)
            self._write(pending.take(np.arange(self.shard_size)))
            rest = pending.take(np.arange(self.shard_size, len(pending)))
            self._buffer = [rest] if len(rest) else []
            self._buffered = len(rest)

    def close(self) -> list[ShardInfo]:
        """Write the remaining samples (if any) and return every shard written."""
        if self._buffered:
            self._write(Samples.concat(self._buffer))
            self._buffer, self._buffered = [], 0
        self.wait()
        return self.shards

    def wait(self) -> None:
        """Wait for background writes to finish; re-raises the first error."""
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def _write(self, samples: Samples) -> None:
        name = f"shard_{len(self.shards):05d}{SHARD_SUFFIX}"
        shuffled = samples.take(self._rng.permutation(len(samples)))
        if self._executor is None:
            write_shard(self.directory / name, shuffled, self.compress)
        else:
            self._pending.append(self._executor.submit(write_shard, self.directory / name, shuffled, self.compress))
        self.shards.append(ShardInfo(name, len(samples)))

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
