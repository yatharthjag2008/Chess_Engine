"""Tests for the command-line scripts in scripts/ (run in-process through their main())."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml
import zstandard

from chess_engine.config import PROJECT_ROOT
from chess_engine.data.game_store import read_games
from chess_engine.data.sources import lichess_filename
from range_server import RangeServer, ServedFile

SCRIPTS = PROJECT_ROOT / "scripts"
FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path, **data: object) -> Path:
    """A config override that points every data path into tmp_path."""
    override = {
        "paths": {
            "data_dir": str(tmp_path),
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "data": data,
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(override), encoding="utf-8")
    return path


def test_prepare_games_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load("prepare_games")
    code = script.main(["--config", str(_config(tmp_path)), "--month", "2026-10", "--source", str(FIXTURE),
                        "--no-progress"])
    assert code == 0
    out = tmp_path / "processed" / "games" / "lichess_2026-10_sample.parquet"
    assert len(read_games(out)) == 8
    printed = capsys.readouterr().out
    assert "kept 8 of 16 games" in printed
    assert "speed" in printed


def test_prepare_games_script_uses_downloaded_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    (raw / lichess_filename("2026-10")).write_bytes(zstandard.ZstdCompressor().compress(FIXTURE.read_bytes()))
    script = _load("prepare_games")
    assert script.main(["--config", str(_config(tmp_path)), "--max-games", "2", "--no-progress"]) == 0
    assert len(read_games(tmp_path / "processed" / "games" / "lichess_2026-10_sample.parquet")) == 2


def test_prepare_games_script_rejects_source_with_many_months(tmp_path: Path) -> None:
    script = _load("prepare_games")
    with pytest.raises(SystemExit):
        script.main(["--month", "2026-09", "--month", "2026-10", "--source", str(FIXTURE)])


def test_download_script(tmp_path: Path, range_server: RangeServer) -> None:
    data = b"pretend this is 30 GB" * 1000
    name = lichess_filename("2026-10")
    range_server.files[name] = ServedFile(data)
    range_server.files["sha256sums.txt"] = ServedFile(f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode())
    config = _config(tmp_path, lichess={"base_url": range_server.url("").rstrip("/"), "months": ["2026-10"]})

    script = _load("download_lichess")
    assert script.main(["--config", str(config), "--no-progress"]) == 0
    assert (tmp_path / "raw" / name).read_bytes() == data


def test_download_script_reports_checksum_failure(tmp_path: Path, range_server: RangeServer,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    name = lichess_filename("2026-10")
    range_server.files[name] = ServedFile(b"data")
    range_server.files["sha256sums.txt"] = ServedFile(f"{'0' * 64}  {name}\n".encode())
    config = _config(tmp_path, lichess={"base_url": range_server.url("").rstrip("/"), "months": ["2026-10"]})

    assert _load("download_lichess").main(["--config", str(config), "--no-progress"]) == 1
    assert "mismatch" in capsys.readouterr().err
