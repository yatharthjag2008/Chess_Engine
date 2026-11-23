"""Tests for PGN sources: local files, compression, URLs."""

from __future__ import annotations

from pathlib import Path

import pytest
import zstandard

from chess_engine.data.pgn_stream import iter_raw_games
from chess_engine.data.sources import lichess_filename, lichess_url, open_local_pgn

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
