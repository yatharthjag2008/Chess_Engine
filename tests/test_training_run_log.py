"""Tests for the run directory and the JSON-lines metrics log."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from chess_engine.config import Config
from chess_engine.training.run_log import MetricsLog, RunDirectory, default_run_name


def test_default_run_name(default_config: Config) -> None:
    name = default_run_name(default_config, datetime(2026, 11, 24, 15, 30, tzinfo=UTC))
    assert name == "20261124-1530_64x6"


def test_metrics_log_round_trip_and_cleaning(tmp_path: Path) -> None:
    log = MetricsLog(tmp_path / "m.jsonl")
    assert log.records() == []
    log.write("train", 10, loss=np.float32(2.5), lr=1e-3)
    log.write("val", 10, loss=float("nan"), buckets={"800-999": {"top1": 0.3}})
    log.write("train", 20, loss=2.0)
    assert [r["step"] for r in log.records("train")] == [10, 20]
    val = log.records("val")[0]
    assert val["loss"] is None and val["buckets"]["800-999"]["top1"] == 0.3
    assert log.records("train")[0]["loss"] == 2.5 and "time" in val


def test_partial_last_line_is_skipped(tmp_path: Path) -> None:
    log = MetricsLog(tmp_path / "m.jsonl")
    log.write("train", 1, loss=1.0)
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"kind": "train", "st')  # crash mid-write
    assert len(log.records()) == 1


def test_truncate_after_drops_repeated_steps(tmp_path: Path) -> None:
    log = MetricsLog(tmp_path / "m.jsonl")
    for step in (10, 20, 30, 40):
        log.write("train", step, loss=1.0)
    assert log.truncate_after(25) == 2
    assert [r["step"] for r in log.records()] == [10, 20]
    assert log.truncate_after(25) == 0


def test_run_directory(tmp_path: Path, default_config: Config) -> None:
    run = RunDirectory(tmp_path / "run1")
    assert not run.exists() and run.read_info() == {}
    run.create(default_config)
    assert run.exists() and run.name == "run1"
    assert run.read_config()["training"]["batch_size"] == default_config.training.batch_size
    with pytest.raises(FileExistsError):
        run.create(default_config)
    run.write_info(device="cpu", parameters=100)
    run.write_info(status="finished")
    assert run.read_info() == {"device": "cpu", "parameters": 100, "status": "finished"}
    assert run.best_model_path.name == "best_model.pt" and run.checkpoints_dir.name == "checkpoints"
