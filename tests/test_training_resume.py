"""Resuming a run gives exactly the same model as training it without interruption."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_engine.model.inference import load_model
from chess_engine.training.train import Trainer, TrainingError
from training_helpers import train_config


def _quiet(_: str) -> None:
    pass


def _weights(path: Path) -> list[torch.Tensor]:
    return list(load_model(path).model.state_dict().values())


@pytest.mark.parametrize("pause_at", [13, None])  # mid-epoch, and exactly at the end of epoch 1
def test_paused_and_resumed_run_matches_an_uninterrupted_one(tmp_path: Path, tiny_dataset: Path,
                                                             pause_at: int | None) -> None:
    settings = dict(epochs=2, batch_size=64, log_every_steps=4, eval_every_steps=10, prefetch_batches=0)
    whole = Trainer(train_config(tmp_path / "a", **settings), tmp_path / "a" / "run", tiny_dataset, log=_quiet)
    expected = whole.train()

    cfg = train_config(tmp_path / "b", **settings)
    first = Trainer(cfg, tmp_path / "b" / "run", tiny_dataset, log=_quiet)
    pause_at = pause_at or first.steps_per_epoch
    paused = first.train(stop_after_steps=pause_at)
    assert paused.stop_reason == "paused" and paused.state.step == pause_at

    second = Trainer.from_run(tmp_path / "b" / "run", log=_quiet)
    state = second.resume()
    assert state.step == pause_at
    result = second.train()
    assert result.stop_reason == "finished" and result.state.step == expected.state.step
    assert result.state.positions_seen == expected.state.positions_seen

    for a, b in zip(_weights(expected.final_model), _weights(result.final_model)):
        assert torch.equal(a, b)
    # The log reads like one run: the same train steps (plus the pause), identical validation results.
    steps_a = {r["step"] for r in whole.run.metrics.records("train")}
    steps_b = {r["step"] for r in second.run.metrics.records("train")}
    assert steps_a <= steps_b and steps_b - steps_a <= {pause_at}
    val_a = {r["step"]: r["loss"] for r in whole.run.metrics.records("val")}
    val_b = {r["step"]: r["loss"] for r in second.run.metrics.records("val")}
    assert all(val_b[step] == val_a[step] for step in val_a)
    assert result.state.best_step == expected.state.best_step


def test_resume_drops_log_lines_after_the_checkpoint(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path, max_steps=30, log_every_steps=5, checkpoint_every_steps=10, keep_checkpoints=5)
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet)
    trainer.train()
    again = Trainer.from_run(tmp_path / "run", log=_quiet)
    again.resume(again.checkpoints.path_for(20))
    steps = [r["step"] for r in again.run.metrics.records("train")]
    assert max(steps) == 20


def test_from_run_overrides_and_errors(tmp_path: Path, tiny_dataset: Path) -> None:
    with pytest.raises(TrainingError, match="no run"):
        Trainer.from_run(tmp_path / "missing", log=_quiet)
    cfg = train_config(tmp_path, max_steps=3)
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet)
    with pytest.raises(TrainingError, match="no checkpoint"):
        trainer.resume()
    trainer.train()
    longer = Trainer.from_run(tmp_path / "run", training_overrides={"max_steps": 6}, log=_quiet)
    assert longer.cfg.training.max_steps == 6 and longer.dataset_dir == Path(tiny_dataset)
    longer.resume()
    assert longer.train().state.step == 6

    other = train_config(tmp_path / "other", model={"channels": 16})
    different = Trainer(other, tmp_path / "run", tiny_dataset, log=_quiet)
    with pytest.raises(TrainingError, match="different model config"):
        different.resume()
