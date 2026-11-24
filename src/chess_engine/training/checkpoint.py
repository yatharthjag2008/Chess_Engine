"""Training checkpoints: everything needed to continue a run exactly where it stopped.

A checkpoint holds the model weights, the optimizer, learning-rate scheduler and
gradient-scaler states, the position in the data (epoch + batch within the
epoch) and the run's config. Because the data order of an epoch is a function
of ``(seed, epoch)`` (:mod:`chess_engine.training.order`), the step counter is
enough to rebuild it; no data-loader state needs saving.

Checkpoints are written atomically (``.tmp`` then rename) and read with
``torch.load(weights_only=True)``, so an untrusted file cannot run code. A
:class:`CheckpointManager` keeps the newest ``keep`` step checkpoints of a run.

(The *model files* for using a trained model, ``best_model.pt`` / ``final_model.pt``,
are written with :func:`chess_engine.model.inference.save_model` and are much smaller.)
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chess_engine.encoding.board import NUM_PLANES, PLANE_NAMES
from chess_engine.encoding.moves import NUM_MOVES
from chess_engine.model.network import RatingConditionedNet

CHECKPOINT_FORMAT_VERSION = 1
_STEP_FILE = re.compile(r"^step_(\d+)\.pt$")


class CheckpointError(Exception):
    """A checkpoint is unreadable or does not fit the model being trained."""


@dataclass
class TrainingState:
    """Where a run is: counters, the best validation result so far, time spent."""

    step: int = 0                  # optimizer steps done
    epoch: int = 0                 # current epoch (0-based)
    batch_in_epoch: int = 0        # batches of the current epoch already trained on
    positions_seen: int = 0
    best_metric: float | None = None   # best validation loss so far (lower is better)
    best_step: int | None = None
    elapsed_seconds: float = 0.0   # training time over all sessions of the run

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _encoding() -> dict[str, Any]:
    return {"num_planes": NUM_PLANES, "plane_names": list(PLANE_NAMES), "num_moves": NUM_MOVES}


def save_checkpoint(
    path: str | Path,
    model: RatingConditionedNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    config: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
) -> Path:
    """Write a checkpoint to ``path`` atomically.

    Args:
        config: The run's full config as plain data (``cfg.model_dump(mode="json")``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "encoding": _encoding(),
        "state": state.as_dict(),
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Read and check a checkpoint (tensors are mapped to ``device``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - torch raises many different errors for bad files
        raise CheckpointError(f"{path} is not a readable checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(f"{path} is not a training checkpoint of format {CHECKPOINT_FORMAT_VERSION}")
    if payload.get("encoding") != _encoding():
        raise CheckpointError(f"{path} was trained with a different board/move encoding than this code")
    return payload


def restore_checkpoint(
    payload: dict[str, Any],
    model: RatingConditionedNet,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainingState:
    """Load a checkpoint's states into the given objects and return its :class:`TrainingState`."""
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise CheckpointError(f"checkpoint does not fit this model (different model config?): {exc}") from exc
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if payload.get("torch_rng") is not None:
        torch.set_rng_state(payload["torch_rng"].cpu())
    return TrainingState(**payload["state"])


class CheckpointManager:
    """Step checkpoints of one run in ``directory`` (``step_00001000.pt``, ...), newest ``keep`` kept."""

    def __init__(self, directory: str | Path, keep: int = 3) -> None:
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self.directory = Path(directory)
        self.keep = keep

    def path_for(self, step: int) -> Path:
        return self.directory / f"step_{step:08d}.pt"

    def checkpoints(self) -> list[tuple[int, Path]]:
        """``(step, path)`` of every step checkpoint, oldest first."""
        if not self.directory.is_dir():
            return []
        found = []
        for path in self.directory.iterdir():
            match = _STEP_FILE.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return sorted(found)

    def latest(self) -> Path | None:
        """The newest step checkpoint, or None."""
        found = self.checkpoints()
        return found[-1][1] if found else None

    def save(self, step: int, **kwargs: Any) -> Path:
        """:func:`save_checkpoint` to :meth:`path_for` ``(step)``, then delete all but the newest ``keep``."""
        path = save_checkpoint(self.path_for(step), **kwargs)
        for _, old in self.checkpoints()[:-self.keep]:
            old.unlink(missing_ok=True)
        return path
