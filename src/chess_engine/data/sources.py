"""Where PGN text comes from: local ``.pgn`` / ``.pgn.zst`` files (and, later in this module, HTTP).

Everything is exposed as a *text stream of lines*, decompressed on the fly, so the
parser in :mod:`chess_engine.data.pgn_stream` never needs the whole file in memory
or on disk.

Usage::

    with open_pgn("data/raw/lichess_db_standard_rated_2026-10.pgn.zst") as stream:
        for raw_game in iter_raw_games(stream.text):
            ...
        print(stream.bytes_read)  # compressed bytes consumed so far (for progress bars)
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import zstandard

ZSTD_SUFFIX = ".zst"


def lichess_filename(month: str) -> str:
    """File name of a Lichess standard-rated monthly dump, e.g. ``lichess_db_standard_rated_2026-10.pgn.zst``."""
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def lichess_url(base_url: str, month: str) -> str:
    """Download URL of a Lichess monthly dump."""
    return f"{base_url.rstrip('/')}/{lichess_filename(month)}"


class CountingReader(io.RawIOBase):
    """Wrap a binary stream and count the bytes read through it (used for progress reporting)."""

    def __init__(self, raw: BinaryIO) -> None:
        super().__init__()
        self._raw = raw
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        data = self._raw.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        self.bytes_read += n
        return n

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


@dataclass
class PgnStream:
    """An open PGN source.

    Attributes:
        text: Line-iterable text stream of decompressed PGN.
        counter: Counts the *compressed* (on-disk / on-the-wire) bytes consumed.
        total_bytes: Size of the compressed source if known (for progress bars), else ``None``.
    """

    text: TextIO
    counter: CountingReader
    total_bytes: int | None = None

    @property
    def bytes_read(self) -> int:
        """Compressed bytes consumed so far."""
        return self.counter.bytes_read


def text_stream(raw: BinaryIO, compressed: bool) -> tuple[TextIO, CountingReader]:
    """Turn a binary stream into a UTF-8 text stream, decompressing zstd if ``compressed``.

    Invalid UTF-8 is replaced rather than raising, so one bad byte cannot stop a
    multi-hour run.
    """
    counter = CountingReader(raw)
    binary: BinaryIO = counter  # type: ignore[assignment]
    if compressed:
        # read_across_frames: a .zst file may contain several concatenated frames.
        binary = zstandard.ZstdDecompressor().stream_reader(counter, read_across_frames=True)  # type: ignore[assignment]
    buffered = io.BufferedReader(binary, buffer_size=1 << 20) if not compressed else binary
    text = io.TextIOWrapper(buffered, encoding="utf-8", errors="replace", newline=None)  # type: ignore[arg-type]
    return text, counter


@contextmanager
def open_local_pgn(path: str | Path) -> Iterator[PgnStream]:
    """Open a local ``.pgn`` or ``.pgn.zst`` file as a :class:`PgnStream`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PGN file not found: {path}")
    raw = path.open("rb")
    text, counter = text_stream(raw, compressed=path.suffix == ZSTD_SUFFIX)
    try:
        yield PgnStream(text=text, counter=counter, total_bytes=path.stat().st_size)
    finally:
        text.close()
