"""Where PGN text comes from: local ``.pgn`` / ``.pgn.zst`` files or an HTTP(S) URL.

Everything is exposed as a *text stream of lines*, decompressed on the fly, so the
parser in :mod:`chess_engine.data.pgn_stream` never needs the whole file in memory
or on disk. Streaming straight from the Lichess server is how sample mode works:
it reads only the first few tens of MB of a ~30 GB file and then stops.

Usage::

    with open_pgn("data/raw/lichess_db_standard_rated_2026-10.pgn.zst") as stream:  # or a URL
        for raw_game in iter_raw_games(stream.text):
            ...
        print(stream.bytes_read)  # compressed bytes consumed so far (for progress bars)
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import requests
import urllib3
import zstandard

log = logging.getLogger(__name__)

ZSTD_SUFFIX = ".zst"
# Generic client name only: no personal details are sent to outside servers.
USER_AGENT = "human-chess-engine-data-pipeline/0.1"


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


# --------------------------------------------------------------------------- HTTP


class HTTPStreamError(Exception):
    """Raised when an HTTP stream cannot be (re)established. Not retried."""


class _TransientHTTPError(ConnectionError):
    """A server error (5xx) or early end of stream that is worth retrying."""


# Errors that mean "the connection broke, try again from where we were".
_RETRYABLE = (requests.RequestException, urllib3.exceptions.HTTPError, OSError)


class HTTPRangeReader(io.RawIOBase):
    """A read-only binary stream over HTTP that resumes after dropped connections.

    If the connection breaks mid-file, it reconnects with a ``Range: bytes=<pos>-``
    request and carries on from the exact byte it stopped at, so a decompressor
    reading from it never notices. Gives up after ``max_retries`` failures in a row.
    """

    def __init__(
        self,
        url: str,
        *,
        start: int = 0,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__()
        self.url = url
        self.position = start
        self.total_bytes: int | None = None
        self.reconnects = 0
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._timeout = timeout_seconds
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._response: requests.Response | None = None
        self._eof = False
        self._read_with_retries(0)  # connect now so total_bytes is known (and errors surface early)

    def readable(self) -> bool:
        return True

    def _connect(self) -> None:
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if self.position:
            headers["Range"] = f"bytes={self.position}-"
        response = self._session.get(self.url, headers=headers, stream=True, timeout=self._timeout)

        if response.status_code == 416 and self.total_bytes is not None and self.position >= self.total_bytes:
            response.close()
            self._eof = True
            return
        if response.status_code >= 500:
            response.close()
            raise _TransientHTTPError(f"HTTP {response.status_code}")
        if response.status_code not in (200, 206):
            response.close()
            raise HTTPStreamError(f"GET {self.url} returned HTTP {response.status_code}")
        if self.position and response.status_code != 206:
            response.close()
            raise HTTPStreamError(f"{self.url} ignored the Range header, so the stream cannot be resumed")

        length = response.headers.get("Content-Length")
        if length is not None and length.isdigit():
            self.total_bytes = self.position + int(length)
        self._response = response

    def _drop_connection(self) -> None:
        if self._response is not None:
            self._response.close()
            self._response = None

    def _read_with_retries(self, size: int) -> bytes:
        """Read up to ``size`` bytes (``0`` = just make sure we're connected), reconnecting on failure."""
        attempt = 0
        while True:
            try:
                if self._response is None and not self._eof:
                    self._connect()
                if self._eof or size == 0:
                    return b""
                assert self._response is not None
                data = self._response.raw.read(size, decode_content=False)
                if not data and self.total_bytes is not None and self.position < self.total_bytes:
                    raise _TransientHTTPError(f"stream ended early ({self.position} of {self.total_bytes} bytes)")
                return data
            except _RETRYABLE as exc:
                self._drop_connection()
                attempt += 1
                if attempt > self._max_retries:
                    raise HTTPStreamError(
                        f"Giving up on {self.url} after {self._max_retries} retries at byte {self.position}: {exc}"
                    ) from exc
                delay = self._backoff * 2 ** (attempt - 1)
                log.warning("Connection to %s failed at byte %d (%s); retry %d in %.1fs",
                            self.url, self.position, exc, attempt, delay)
                time.sleep(delay)
                self.reconnects += 1

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        data = self._read_with_retries(len(buffer))
        if not data:
            self._eof = True
            return 0
        n = len(data)
        buffer[:n] = data
        self.position += n
        return n

    def close(self) -> None:
        self._drop_connection()
        if self._owns_session:
            self._session.close()
        super().close()


def is_url(source: str | Path) -> bool:
    """True if ``source`` is an http(s) URL rather than a local path."""
    return isinstance(source, str) and source.startswith(("http://", "https://"))


@contextmanager
def open_url_pgn(url: str, **reader_options: object) -> Iterator[PgnStream]:
    """Stream a remote ``.pgn`` / ``.pgn.zst`` file as a :class:`PgnStream`.

    ``reader_options`` are passed to :class:`HTTPRangeReader` (retries, backoff, timeout).
    """
    reader = HTTPRangeReader(url, **reader_options)  # type: ignore[arg-type]
    compressed = url.split("?", 1)[0].endswith(ZSTD_SUFFIX)
    text, counter = text_stream(reader, compressed=compressed)  # type: ignore[arg-type]
    try:
        yield PgnStream(text=text, counter=counter, total_bytes=reader.total_bytes)
    finally:
        text.close()


@contextmanager
def open_pgn(source: str | Path, **reader_options: object) -> Iterator[PgnStream]:
    """Open a local file or an http(s) URL as a :class:`PgnStream`."""
    if is_url(source):
        with open_url_pgn(str(source), **reader_options) as stream:
            yield stream
    else:
        with open_local_pgn(source) as stream:
            yield stream
