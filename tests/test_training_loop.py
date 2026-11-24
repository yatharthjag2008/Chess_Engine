"""Tests for the training loop (tiny models on the tiny dataset, a few seconds each)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from chess_engine.model.inference import load_model
from chess_engine.training.train import Trainer, TrainingError, build_optimizer
from training_helpers import train_config


def _quiet(_: str) -> None:
    pass


def test_optimizer_decays_only_conv_and_linear_weights(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path)
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet)
    decay, no_decay = trainer.optimizer.param_groups
    assert decay["weight_decay"] == cfg.training.weight_decay and no_decay["weight_decay"] == 0
    decayed = {id(p) for p in decay["params"]}
    model = trainer.model
    assert id(model.condition.rating.weight) not in decayed      # embeddings
    assert id(model.stem[1].weight) not in decayed                # batch norm
    assert id(model.stem[0].weight) in decayed                    # conv
    total = sum(1 for _ in model.parameters())
    assert len(decay["params"]) + len(no_decay["params"]) == total
    sgd = build_optimizer(nn.Linear(2, 2), cfg.training.model_copy(update={"optimizer": "sgd"}))
    assert isinstance(sgd, torch.optim.SGD)


def test_training_reduces_the_loss_and_saves_a_usable_model(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path, epochs=4)
    lines: list[str] = []
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=lines.append)
    result = trainer.train()
    assert result.stop_reason == "finished"
    assert result.state.step == 4 * trainer.steps_per_epoch and result.state.epoch == 4
    records = trainer.run.metrics.records("train")
    assert records and records[-1]["step"] == result.state.step
    assert records[-1]["loss"] < records[0]["loss"]
    assert any("pos/s" in line for line in lines) and any("parameters" in line for line in lines)

    loaded = load_model(result.final_model)
    assert loaded.extra["step"] == result.state.step
    for a, b in zip(loaded.model.state_dict().values(), trainer.model.state_dict().values()):
        assert torch.equal(a, b.cpu())
    info = trainer.run.read_info()
    assert info["status"] == "finished" and info["steps_done"] == result.state.step


def test_max_steps_stops_early_and_shapes_the_schedule(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path, epochs=50, max_steps=7, lr_schedule="linear", min_lr_fraction=0.0)
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet)
    assert trainer.total_steps == 7
    result = trainer.train()
    assert result.stop_reason == "max_steps" and result.state.step == 7
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("balancing", ["none", "weight", "sample"])
def test_every_balancing_mode_trains(tmp_path: Path, tiny_dataset: Path, balancing: str) -> None:
    cfg = train_config(tmp_path, max_steps=3, bucket_balancing=balancing, balance_power=1.0)
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet)
    assert trainer.train().state.step == 3
    if balancing == "none":
        assert (trainer.bucket_weights == 1).all()
    else:
        assert "trained" in trainer.describe()


def test_streaming_dataset_and_label_smoothing(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path, max_steps=4, in_memory=False, label_smoothing=0.1, bucket_balancing="weight")
    assert Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet).train().state.step == 4


def test_setup_errors(tmp_path: Path, tiny_dataset: Path) -> None:
    with pytest.raises(TrainingError, match="no dataset"):
        Trainer(train_config(tmp_path), tmp_path / "run", tmp_path / "nothing", log=_quiet)
    with pytest.raises(TrainingError, match="fewer than one batch"):
        Trainer(train_config(tmp_path, batch_size=100_000), tmp_path / "run", tiny_dataset, log=_quiet)


def test_diverging_loss_is_reported(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path, max_steps=5, grad_clip_norm=None)
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=_quiet)
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.fill_(float("nan"))
    with pytest.raises(TrainingError, match="diverged"):
        trainer.train()


def test_validation_best_model_and_checkpoints(tmp_path: Path, tiny_dataset: Path) -> None:
    cfg = train_config(tmp_path, epochs=2, eval_every_steps=20, checkpoint_every_steps=25, keep_checkpoints=2)
    lines: list[str] = []
    trainer = Trainer(cfg, tmp_path / "run", tiny_dataset, log=lines.append)
    result = trainer.train()
    steps = result.state.step
    val_steps = [r["step"] for r in trainer.run.metrics.records("val")]
    assert val_steps == sorted(set(val_steps))  # never twice at the same step
    assert set(range(20, steps + 1, 20)) <= set(val_steps)
    assert trainer.steps_per_epoch in val_steps and steps in val_steps  # end of epoch 1 and end of run

    best = min(trainer.run.metrics.records("val"), key=lambda r: r["loss"])
    assert result.state.best_step == best["step"]
    assert result.best_model is not None and load_model(result.best_model).extra["step"] == best["step"]
    assert result.last_val["positions"] == len(trainer.val_data)

    saved = [s for s, _ in trainer.checkpoints.checkpoints()]
    assert len(saved) == 2 and saved[-1] == steps  # newest two, including the final one
    assert any("* best" in line for line in lines) and any("top5" in line for line in lines)
    assert trainer.run.read_info()["best_step"] == best["step"]
