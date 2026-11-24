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


def test_benchmark_encoding_script_with_random_games(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load("benchmark_encoding")
    assert script.main(["--games-file", str(tmp_path / "missing.parquet"), "--games", "3"]) == 0
    out = capsys.readouterr().out
    assert "random games" in out
    assert "legal_move_mask" in out


def test_benchmark_encoding_script_with_games_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "g.parquet"
    assert _load("prepare_games").main(["--config", str(_config(tmp_path)), "--month", "2026-10", "--source",
                                        str(FIXTURE), "--output", str(out_path), "--no-progress"]) == 0
    capsys.readouterr()
    assert _load("benchmark_encoding").main(["--games-file", str(out_path), "--games", "8"]) == 0
    assert "positions from 8 games" in capsys.readouterr().out


def _games_for_dataset(tmp_path: Path) -> Path:
    from dataset_helpers import random_game_rows, write_game_rows

    return write_game_rows(random_game_rows(25, seed=4), tmp_path / "processed" / "games" / "lichess_2026-10_sample.parquet")


def test_build_dataset_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _games_for_dataset(tmp_path)
    config = _config(tmp_path, dataset={"workers": 1, "shard_size": 500})
    script = _load("build_dataset")
    assert script.main(["--config", str(config), "--no-progress"]) == 0
    out = capsys.readouterr().out
    assert "Used 25 of 25 games" in out and "train" in out and "2200+" in out
    dataset_dir = tmp_path / "processed" / "datasets" / "lichess_2026-10_sample"
    assert (dataset_dir / "manifest.json").is_file()

    # A second build without --overwrite fails cleanly; with it, succeeds.
    assert script.main(["--config", str(config), "--no-progress"]) == 1
    assert "overwrite" in capsys.readouterr().err
    assert script.main(["--config", str(config), "--no-progress", "--overwrite", "--max-games", "5",
                        "--no-compress"]) == 0
    assert "Used 5 of 5 games" in capsys.readouterr().out


def test_build_dataset_script_without_games(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _load("build_dataset").main(["--config", str(_config(tmp_path)), "--no-progress"]) == 1
    assert "prepare_games" in capsys.readouterr().err


def test_inspect_dataset_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _games_for_dataset(tmp_path)
    config = _config(tmp_path, dataset={"workers": 1})
    assert _load("build_dataset").main(["--config", str(config), "--no-progress"]) == 0
    capsys.readouterr()
    assert _load("inspect_dataset").main(["--config", str(config), "--samples", "2", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("--- train sample") == 2
    assert "to move, rated" in out and "played " in out and "legal moves" in out


def test_inspect_dataset_script_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _load("inspect_dataset").main(["--dataset", str(tmp_path / "none")]) == 1
    assert "build_dataset" in capsys.readouterr().err


def test_model_summary_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    override = tmp_path / "small.yaml"
    override.write_text(yaml.safe_dump({
        "paths": {"processed_dir": str(tmp_path / "processed")},
        "model": {"channels": 8, "residual_blocks": 1},
    }), encoding="utf-8")
    assert _load("model_summary").main(["--config", str(override), "--batch-size", "4", "--steps", "1"]) == 0
    out = capsys.readouterr().out
    assert "8 channels x 1 blocks" in out and "total" in out and "training step" in out
    assert "build the dataset" in out and "Untrained model on the start position" in out


def _train_override(tmp_path: Path) -> Path:
    override = tmp_path / "train.yaml"
    override.write_text(yaml.safe_dump({
        "paths": {"checkpoints_dir": str(tmp_path / "ckpt")},
        "model": {"channels": 8, "residual_blocks": 1, "rating_embedding_dim": 4},
        "training": {"device": "cpu", "batch_size": 32, "log_every_steps": 2, "warmup_steps": 1,
                     "prefetch_batches": 0},
    }), encoding="utf-8")
    return override


def test_train_script_new_run_then_resume(tmp_path: Path, tiny_dataset: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    script = _load("train")
    config = str(_train_override(tmp_path))
    args = ["--config", config, "--dataset", str(tiny_dataset), "--run-name", "r1", "--max-steps", "4"]
    assert script.main(args) == 0
    out = capsys.readouterr().out
    assert "step       4/4" in out and "validation at step 4" in out
    run = tmp_path / "ckpt" / "r1"
    assert (run / "final_model.pt").is_file() and (run / "best_model.pt").is_file()

    assert script.main(args) == 1  # the run exists already
    assert "--resume r1" in capsys.readouterr().err

    assert script.main(["--resume", str(run), "--max-steps", "6"]) == 0
    out = capsys.readouterr().out
    assert "resumed from step_00000004.pt" in out and "step       6/6" in out


def test_train_script_resume_latest_and_errors(tmp_path: Path, tiny_dataset: Path,
                                               capsys: pytest.CaptureFixture[str], monkeypatch) -> None:
    script = _load("train")
    config = str(_train_override(tmp_path))
    with pytest.raises(SystemExit):
        script.main(["--resume", "latest", "--config", config])
    assert script.main(["--config", config, "--dataset", str(tmp_path / "none")]) == 1
    assert "no dataset" in capsys.readouterr().err
    assert script.main(["--config", config, "--dataset", str(tiny_dataset), "--run-name", "a",
                        "--max-steps", "2"]) == 0
    assert script.find_run("latest", tmp_path / "ckpt") == tmp_path / "ckpt" / "a"
    assert script.find_run("a", tmp_path / "ckpt") == tmp_path / "ckpt" / "a"
    with pytest.raises(Exception, match="no run"):
        script.find_run("latest", tmp_path / "empty")
