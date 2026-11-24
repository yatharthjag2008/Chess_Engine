"""One directory per training run, with its config, a JSON-lines metrics log and its checkpoints.

::

    checkpoints/<run name>/
        config.yaml        the full merged config the run was started with
        run.json           device, parameters, dataset, start time (updated when the run ends)
        metrics.jsonl      one JSON object per line: {"kind": "train" | "val" | "event", "step": ..., ...}
        checkpoints/       step_00001000.pt, ... (newest few, for resuming)
        best_model.pt      the model with the lowest validation loss (load with chess_engine.model.load_model)
        final_model.pt     the model at the end of the run

JSON lines are appended and flushed one at a time, so the log survives a crash
and can be read (or plotted) while the run is going.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from chess_engine.config import Config

CONFIG_NAME = "config.yaml"
RUN_INFO_NAME = "run.json"
METRICS_NAME = "metrics.jsonl"
CHECKPOINTS_DIR = "checkpoints"
BEST_MODEL_NAME = "best_model.pt"
FINAL_MODEL_NAME = "final_model.pt"


def default_run_name(cfg: Config, now: datetime | None = None) -> str:
    """``<UTC date-time>_<channels>x<blocks>``, e.g. ``20261124-1530_64x6``."""
    now = now or datetime.now(UTC)
    return f"{now:%Y%m%d-%H%M}_{cfg.model.channels}x{cfg.model.residual_blocks}"


def json_safe(value: Any) -> Any:
    """JSON-safe copy: NaN/inf become None, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MetricsLog:
    """Append-only JSON-lines log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, kind: str, step: int, **values: Any) -> dict[str, Any]:
        record = json_safe({"kind": kind, "step": step, "time": datetime.now(UTC).isoformat(timespec="seconds"),
                         **values})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def records(self, kind: str | None = None) -> list[dict[str, Any]]:
        return list(self.iter_records(kind))

    def iter_records(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Records in order (a partly written last line from a crash is skipped)."""
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind is None or record.get("kind") == kind:
                    yield record

    def truncate_after(self, step: int) -> int:
        """Drop records logged after ``step`` (a resumed run repeats those steps); returns how many."""
        records = self.records()
        keep = [r for r in records if r.get("step", 0) <= step]
        if len(keep) != len(records):
            with self.path.open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in keep)
        return len(records) - len(keep)


class RunDirectory:
    """Paths and files of one run (see the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metrics = MetricsLog(self.path / METRICS_NAME)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_NAME

    @property
    def info_path(self) -> Path:
        return self.path / RUN_INFO_NAME

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / CHECKPOINTS_DIR

    @property
    def best_model_path(self) -> Path:
        return self.path / BEST_MODEL_NAME

    @property
    def final_model_path(self) -> Path:
        return self.path / FINAL_MODEL_NAME

    def exists(self) -> bool:
        return self.config_path.is_file()

    def create(self, cfg: Config) -> None:
        """Start a new run here: write the config snapshot. Refuses to overwrite an existing run."""
        if self.exists():
            raise FileExistsError(f"run {self.path} already exists (resume it or pick another run name)")
        self.path.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    def read_config(self) -> dict[str, Any]:
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def write_info(self, **info: Any) -> dict[str, Any]:
        """Merge ``info`` into ``run.json``."""
        current = self.read_info()
        current.update(json_safe(info))
        self.path.mkdir(parents=True, exist_ok=True)
        self.info_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        return current

    def read_info(self) -> dict[str, Any]:
        if not self.info_path.is_file():
            return {}
        return json.loads(self.info_path.read_text(encoding="utf-8"))
