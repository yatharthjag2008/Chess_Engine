"""Tests for PGN sources: local files, compression, URLs."""

from __future__ import annotations

from pathlib import Path

import pytest
import zstandard

from chess_engine.data.pgn_stream import iter_raw_games
from chess_engine.data.sources import (
    HTTPRangeReader,
    HTTPStreamError,
    is_url,
    lichess_filename,
    lichess_url,
    open_local_pgn,
    open_pgn,
)
from range_server import RangeServer, ServedFile

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"


def _write_zst(path: Path, data: bytes, frames: int = 1) -> Path:
    """Compress ``data`` into ``frames`` concatenated zstd frames."""
    cctx = zstandard.ZstdCompressor(level=3)
    step = -(-len(data) // frames)
    with path.open("wb") as f:
        for i in range(0, len(data), step):
            f.write(cctx.compress(data[i : i + step]))
    return path


def test_lichess_names() -> None:
    assert lichess_filename("2026-10") == "lichess_db_standard_rated_2026-10.pgn.zst"
    assert (
        lichess_url("https://database.lichess.org/standard/", "2026-10")
        == "https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst"
    )


def test_open_plain_pgn() -> None:
    with open_local_pgn(FIXTURE) as stream:
        games = list(iter_raw_games(stream.text))
        assert stream.bytes_read == FIXTURE.stat().st_size
        assert stream.total_bytes == FIXTURE.stat().st_size
    assert len(games) == 16


@pytest.mark.parametrize("frames", [1, 3])
def test_open_zst_pgn_matches_plain(tmp_path: Path, frames: int) -> None:
    zst = _write_zst(tmp_path / "sample.pgn.zst", FIXTURE.read_bytes(), frames=frames)
    with open_local_pgn(zst) as stream:
        games = list(iter_raw_games(stream.text))
        assert stream.bytes_read == zst.stat().st_size  # counts compressed bytes
    with open_local_pgn(FIXTURE) as stream:
        expected = list(iter_raw_games(stream.text))
    assert games == expected


def test_crlf_and_unicode_names(tmp_path: Path) -> None:
    text = '[Event "Rated Blitz game"]\r\n[White "José_üß"]\r\n\r\n1. e4 *\r\n'
    zst = _write_zst(tmp_path / "u.pgn.zst", text.encode("utf-8"))
    with open_local_pgn(zst) as stream:
        (game,) = iter_raw_games(stream.text)
    assert game.header("White") == "José_üß"
    assert game.movetext == "1. e4 *"


def test_invalid_utf8_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "bad.pgn"
    path.write_bytes(b'[Event "Rated"]\n[White "a\xffb"]\n\n1. e4 *\n')
    with open_local_pgn(path) as stream:
        (game,) = iter_raw_games(stream.text)
    assert game.header("White") == "a�b"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with open_local_pgn(tmp_path / "nope.pgn.zst"):
            pass


# ----------------------------------------------------------------------------- HTTP streaming


@pytest.fixture()
def zst_bytes(tmp_path: Path) -> bytes:
    return _write_zst(tmp_path / "s.pgn.zst", FIXTURE.read_bytes(), frames=2).read_bytes()


def _fast(**extra: object) -> dict:
    return {"backoff_seconds": 0.0, "timeout_seconds": 5.0, **extra}


def _read_all(reader: HTTPRangeReader, chunk: int = 4096) -> bytes:
    out = bytearray()
    while data := reader.read(chunk):
        out += data
    return bytes(out)


def test_http_reader_reads_whole_file(range_server: RangeServer) -> None:
    data = bytes(range(256)) * 100
    range_server.files["f.bin"] = ServedFile(data)
    reader = HTTPRangeReader(range_server.url("f.bin"), **_fast())
    assert reader.total_bytes == len(data)
    assert _read_all(reader) == data
    assert reader.reconnects == 0
    reader.close()


def test_http_reader_resumes_after_dropped_connections(range_server: RangeServer) -> None:
    data = bytes(range(256)) * 400
    range_server.files["f.bin"] = ServedFile(data, drop_after=[10_000, 5_000])
    reader = HTTPRangeReader(range_server.url("f.bin"), **_fast())
    assert _read_all(reader) == data
    assert reader.reconnects == 2
    ranges = [r for _, r in range_server.requests]
    assert ranges == [None, "bytes=10000-", "bytes=15000-"]
    reader.close()


def test_http_reader_retries_server_errors(range_server: RangeServer) -> None:
    range_server.files["f.bin"] = ServedFile(b"hello", fail_first=2)
    reader = HTTPRangeReader(range_server.url("f.bin"), **_fast())
    assert _read_all(reader) == b"hello"
    assert reader.reconnects == 2


def test_http_reader_gives_up_after_max_retries(range_server: RangeServer) -> None:
    range_server.files["f.bin"] = ServedFile(b"hello", fail_first=10)
    with pytest.raises(HTTPStreamError, match="Giving up"):
        HTTPRangeReader(range_server.url("f.bin"), **_fast(max_retries=2))


def test_http_reader_404_is_not_retried(range_server: RangeServer) -> None:
    with pytest.raises(HTTPStreamError, match="404"):
        HTTPRangeReader(range_server.url("missing.bin"), **_fast())
    assert len(range_server.requests) == 1


def test_http_reader_fails_if_server_ignores_range(range_server: RangeServer) -> None:
    range_server.files["f.bin"] = ServedFile(b"x" * 10_000, drop_after=[1000], support_ranges=False)
    reader = HTTPRangeReader(range_server.url("f.bin"), **_fast())
    with pytest.raises(HTTPStreamError, match="Range"):
        _read_all(reader)


def test_http_reader_start_offset(range_server: RangeServer) -> None:
    range_server.files["f.bin"] = ServedFile(b"0123456789")
    reader = HTTPRangeReader(range_server.url("f.bin"), start=4, **_fast())
    assert reader.total_bytes == 10
    assert _read_all(reader) == b"456789"


def test_open_pgn_url_matches_local(range_server: RangeServer, zst_bytes: bytes) -> None:
    range_server.files["s.pgn.zst"] = ServedFile(zst_bytes, drop_after=[3000])
    with open_pgn(range_server.url("s.pgn.zst"), **_fast()) as stream:
        games = list(iter_raw_games(stream.text))
        assert stream.total_bytes == len(zst_bytes)
        assert stream.bytes_read == len(zst_bytes)
    with open_pgn(FIXTURE) as stream:
        assert games == list(iter_raw_games(stream.text))


def test_open_pgn_url_can_stop_early(range_server: RangeServer, zst_bytes: bytes) -> None:
    """Sample mode reads only the start of a huge file and then closes the stream."""
    range_server.files["s.pgn.zst"] = ServedFile(zst_bytes)
    with open_pgn(range_server.url("s.pgn.zst"), **_fast()) as stream:
        first = next(iter_raw_games(stream.text))
    assert first.header("Site").endswith("fixt0001")


def test_is_url() -> None:
    assert is_url("https://database.lichess.org/x.pgn.zst")
    assert is_url("http://localhost/x")
    assert not is_url("data/raw/x.pgn.zst")
    assert not is_url(Path("https:/x"))
