"""Tests for resumable, checksum-verified downloads (against a local HTTP server)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chess_engine.data.sources import (
    ChecksumError,
    download_file,
    download_lichess_month,
    lichess_filename,
    parse_checksums,
    sha256_of_file,
)
from range_server import RangeServer, ServedFile

DATA = bytes(range(256)) * 4000  # ~1 MB
SHA = hashlib.sha256(DATA).hexdigest()
FAST = {"backoff_seconds": 0.0, "timeout_seconds": 5.0, "show_progress": False}


def test_parse_checksums() -> None:
    text = f"{'a' * 64}  lichess_db_standard_rated_2026-09.pgn.zst\n{'B' * 64} *other.zst\nnot a line\n\n"
    assert parse_checksums(text) == {
        "lichess_db_standard_rated_2026-09.pgn.zst": "a" * 64,
        "other.zst": "b" * 64,
    }


def test_download_file(range_server: RangeServer, tmp_path: Path) -> None:
    range_server.files["f.zst"] = ServedFile(DATA)
    dest = download_file(range_server.url("f.zst"), tmp_path / "sub" / "f.zst", expected_sha256=SHA,
                         chunk_size=65536, **FAST)
    assert dest.read_bytes() == DATA
    assert not (tmp_path / "sub" / "f.zst.part").exists()
    assert sha256_of_file(dest) == SHA


def test_download_survives_dropped_connections(range_server: RangeServer, tmp_path: Path) -> None:
    range_server.files["f.zst"] = ServedFile(DATA, drop_after=[100_000, 300_000])
    dest = download_file(range_server.url("f.zst"), tmp_path / "f.zst", expected_sha256=SHA, **FAST)
    assert dest.read_bytes() == DATA


def test_download_resumes_from_part_file(range_server: RangeServer, tmp_path: Path) -> None:
    range_server.files["f.zst"] = ServedFile(DATA)
    (tmp_path / "f.zst.part").write_bytes(DATA[:400_000])
    dest = download_file(range_server.url("f.zst"), tmp_path / "f.zst", expected_sha256=SHA, **FAST)
    assert dest.read_bytes() == DATA
    assert range_server.requests[0][1] == "bytes=400000-"


def test_download_part_file_already_complete(range_server: RangeServer, tmp_path: Path) -> None:
    range_server.files["f.zst"] = ServedFile(DATA)
    (tmp_path / "f.zst.part").write_bytes(DATA)
    dest = download_file(range_server.url("f.zst"), tmp_path / "f.zst", expected_sha256=SHA, **FAST)
    assert dest.read_bytes() == DATA


def test_download_checksum_mismatch_deletes_part(range_server: RangeServer, tmp_path: Path) -> None:
    range_server.files["f.zst"] = ServedFile(DATA)
    with pytest.raises(ChecksumError, match="mismatch"):
        download_file(range_server.url("f.zst"), tmp_path / "f.zst", expected_sha256="0" * 64, **FAST)
    assert not (tmp_path / "f.zst").exists()
    assert not (tmp_path / "f.zst.part").exists()


def test_download_skips_existing_file(range_server: RangeServer, tmp_path: Path) -> None:
    (tmp_path / "f.zst").write_bytes(b"already here")
    dest = download_file(range_server.url("f.zst"), tmp_path / "f.zst", **FAST)
    assert dest.read_bytes() == b"already here"
    assert range_server.requests == []


def test_download_lichess_month_verifies_checksum(range_server: RangeServer, tmp_path: Path) -> None:
    name = lichess_filename("2026-10")
    range_server.files[name] = ServedFile(DATA)
    range_server.files["sha256sums.txt"] = ServedFile(f"{SHA}  {name}\n".encode())
    base_url = range_server.url("").rstrip("/")
    dest = download_lichess_month(base_url, "2026-10", tmp_path, **FAST)
    assert dest == tmp_path / name
    assert dest.read_bytes() == DATA


def test_download_lichess_month_not_published(range_server: RangeServer, tmp_path: Path) -> None:
    range_server.files["sha256sums.txt"] = ServedFile(b"")
    with pytest.raises(ChecksumError, match="not listed"):
        download_lichess_month(range_server.url("").rstrip("/"), "2099-01", tmp_path, **FAST)


def test_download_lichess_month_without_verification(range_server: RangeServer, tmp_path: Path) -> None:
    name = lichess_filename("2026-10")
    range_server.files[name] = ServedFile(DATA)
    dest = download_lichess_month(range_server.url("").rstrip("/"), "2026-10", tmp_path, verify_checksum=False,
                                  **FAST)
    assert dest.read_bytes() == DATA
    assert all(path != "sha256sums.txt" for path, _ in range_server.requests)
