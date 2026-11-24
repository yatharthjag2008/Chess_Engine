"""Tests for training checkpoints (save, load, restore, rotation)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_engine.config import ModelConfig
from chess_engine.model.features import NUM_CONTEXT_FEATURES, ModelInputs
from chess_engine.model.network import RatingConditionedNet
from chess_engine.training.checkpoint import (
    CheckpointError,
    CheckpointManager,
    TrainingState,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from chess_engine.training.schedule import build_scheduler

TINY = ModelConfig(architecture="resnet", channels=8, residual_blocks=1, rating_embedding_dim=4)


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    model = RatingConditionedNet(TINY, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=100, warmup_steps=10)
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler) -> None:
    inputs = ModelInputs(torch.rand(4, 20, 8, 8), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
                         torch.rand(4, NUM_CONTEXT_FEATURES))
    loss = model(inputs).policy.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_round_trip_restores_every_state(tmp_path: Path) -> None:
    model, optimizer, scheduler = _setup()
    for _ in range(3):
        _step(model, optimizer, scheduler)
    state = TrainingState(step=3, epoch=1, batch_in_epoch=2, positions_seen=12, best_metric=2.5, best_step=3,
                          elapsed_seconds=1.5)
    path = save_checkpoint(tmp_path / "c.pt", model, optimizer, scheduler, state, {"training": {"seed": 1}})
    assert not (tmp_path / "c.pt.tmp").exists()

    model2, optimizer2, scheduler2 = _setup(seed=99)
    payload = load_checkpoint(path)
    assert payload["config"] == {"training": {"seed": 1}}
    restored = restore_checkpoint(payload, model2, optimizer2, scheduler2)
    assert restored == state
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)
    assert scheduler2.last_epoch == 3
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])

    # Both copies continue identically.
    torch.manual_seed(5)
    _step(model, optimizer, scheduler)
    torch.manual_seed(5)
    _step(model2, optimizer2, scheduler2)
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(a, b)


def test_bad_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="readable"):
        load_checkpoint(junk)
    other = tmp_path / "other.pt"
    torch.save({"format_version": 999}, other)
    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(other)


def test_checkpoint_of_a_different_model_is_rejected(tmp_path: Path) -> None:
    model, optimizer, scheduler = _setup()
    path = save_checkpoint(tmp_path / "c.pt", model, optimizer, scheduler, TrainingState(), {})
    bigger = RatingConditionedNet(TINY.model_copy(update={"channels": 16}), 8)
    with pytest.raises(CheckpointError, match="does not fit"):
        restore_checkpoint(load_checkpoint(path), bigger)


def test_manager_keeps_the_newest_checkpoints(tmp_path: Path) -> None:
    model, optimizer, scheduler = _setup()
    manager = CheckpointManager(tmp_path / "ckpt", keep=2)
    assert manager.latest() is None and manager.checkpoints() == []
    for step in (10, 20, 30, 40):
        manager.save(step, model=model, optimizer=optimizer, scheduler=scheduler,
                     state=TrainingState(step=step), config={})
    assert [s for s, _ in manager.checkpoints()] == [30, 40]
    assert manager.latest() == tmp_path / "ckpt" / "step_00000040.pt"
    assert TrainingState(**load_checkpoint(manager.latest())["state"]).step == 40
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, keep=0)
