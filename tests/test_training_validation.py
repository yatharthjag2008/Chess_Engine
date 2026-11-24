"""Tests for the validation pass used during training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.model.features import inputs_from_batch, targets_from_batch
from chess_engine.model.masking import legal_log_probs
from chess_engine.model.network import RatingConditionedNet
from chess_engine.training.validation import validate

TINY = ModelConfig(architecture="resnet", channels=8, residual_blocks=1, rating_embedding_dim=4)


@pytest.fixture()
def model() -> RatingConditionedNet:
    torch.manual_seed(0)
    return RatingConditionedNet(TINY, 8)


def test_validation_matches_a_direct_computation(tiny_dataset: Path, default_config: Config, model) -> None:
    dataset = PositionDataset(tiny_dataset, "val")
    assert len(dataset) > 50
    result = validate(model, dataset, default_config.ratings, batch_size=32)
    batch = dataset.batch(np.arange(len(dataset)))
    model.eval()  # validate() evaluates with BatchNorm's running statistics
    with torch.no_grad():
        targets = targets_from_batch(batch)
        log_probs = legal_log_probs(model(inputs_from_batch(batch, default_config.ratings)).policy,
                                    targets.legal_mask)
    nll = -log_probs.gather(1, targets.move[:, None]).squeeze(1)
    top1 = (log_probs.argmax(1) == targets.move).float().mean().item()
    assert result["positions"] == len(dataset)
    assert result["policy_loss"] == pytest.approx(nll.mean().item(), rel=1e-4)
    assert result["overall"]["nll"] == pytest.approx(nll.mean().item(), rel=1e-4)
    assert result["overall"]["top1"] == pytest.approx(top1)
    assert result["loss"] == pytest.approx(result["policy_loss"] + 0.25 * result["value_loss"], rel=1e-5)
    assert set(result["buckets"]) == set(default_config.ratings.bucket_labels())


def test_max_positions_limits_the_pass(tiny_dataset: Path, default_config: Config, model) -> None:
    dataset = PositionDataset(tiny_dataset, "val")
    result = validate(model, dataset, default_config.ratings, batch_size=16, max_positions=37)
    assert result["positions"] == 37
    again = validate(model, dataset, default_config.ratings, batch_size=8, max_positions=37, prefetch_batches=0)
    assert again["loss"] == pytest.approx(result["loss"], rel=1e-5)  # same positions whatever the batch size


def test_validation_restores_training_mode(tiny_dataset: Path, default_config: Config, model) -> None:
    model.train()
    validate(model, PositionDataset(tiny_dataset, "val"), default_config.ratings, max_positions=10)
    assert model.training
    model.eval()
    validate(model, PositionDataset(tiny_dataset, "val"), default_config.ratings, max_positions=10)
    assert not model.training


def test_without_value_head(tiny_dataset: Path, default_config: Config) -> None:
    model = RatingConditionedNet(TINY.model_copy(update={"value_head": False}), 8)
    result = validate(model, PositionDataset(tiny_dataset, "val"), default_config.ratings, max_positions=20)
    assert "value_loss" not in result and "value_accuracy" not in result["overall"]
    assert result["loss"] == pytest.approx(result["policy_loss"])
