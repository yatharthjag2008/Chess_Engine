"""Helpers for training tests: tiny-model configs that write into a temp directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from chess_engine.config import Config, load_config

TINY_MODEL = {"channels": 8, "residual_blocks": 1, "rating_embedding_dim": 4}


def train_config(tmp_path: Path, model: dict[str, Any] | None = None, **training: Any) -> Config:
    """Default config with paths under ``tmp_path``, a tiny model and fast training settings."""
    override = {
        "paths": {
            "data_dir": str(tmp_path),
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "model": {**TINY_MODEL, **(model or {})},
        "training": {
            "device": "cpu", "batch_size": 32, "epochs": 1, "warmup_steps": 2, "learning_rate": 0.003,
            "log_every_steps": 5, "eval_every_steps": 1000, "checkpoint_every_steps": 1000,
            "prefetch_batches": 2, "val_max_positions": None, **training,
        },
    }
    path = tmp_path / "train_override.yaml"
    path.write_text(yaml.safe_dump(override), encoding="utf-8")
    return load_config(path)
