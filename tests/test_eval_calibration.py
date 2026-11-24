"""Tests for reliability tables and calibration error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.calibration import binned_reliability, move_calibration, reliability
from chess_engine.eval.predictions import predict_split
from chess_engine.model.network import RatingConditionedNet


def test_perfectly_calibrated_predictions_have_small_ece() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    outcome = rng.random(200_000) < p
    result = binned_reliability(p, outcome, bins=10)
    assert result["ece"] < 0.01 and result["predictions"] == 200_000
    for b in result["bins"]:
        assert b["observed"] == pytest.approx(b["mean_predicted"], abs=0.02)


def test_overconfident_predictions_have_large_ece() -> None:
    p = np.full(1000, 0.9)
    outcome = np.arange(1000) < 500  # only half come true
    result = binned_reliability(p, outcome, bins=10)
    assert result["ece"] == pytest.approx(0.4)
    last = result["bins"][-1]
    assert last["count"] == 1000 and last["mean_predicted"] == pytest.approx(0.9) and last["observed"] == 0.5
    assert result["bins"][0]["observed"] is None


def test_reliability_from_counts_and_edges() -> None:
    result = reliability(np.array([2, 0]), np.array([0.6, 0.0]), np.array([1.0, 0.0]))
    assert [b["low"] for b in result["bins"]] == [0.0, 0.5] and result["bins"][1]["count"] == 0
    assert result["ece"] == pytest.approx(0.2)
    assert np.isnan(reliability(np.zeros(3), np.zeros(3), np.zeros(3))["ece"])
    assert binned_reliability(np.array([1.0]), np.array([1.0]), bins=4)["bins"][3]["count"] == 1


def test_move_calibration_of_a_model(tiny_dataset: Path, default_config: Config) -> None:
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(architecture="resnet", channels=8, residual_blocks=1,
                                             rating_embedding_dim=4), 8)
    table = predict_split(model, PositionDataset(tiny_dataset, "test"), default_config.ratings, calibration_bins=5)
    result = move_calibration(table)
    assert len(result["all_moves"]["bins"]) == len(result["top1"]["bins"]) == 5
    assert result["all_moves"]["predictions"] == table["legal_count"].sum()
    assert result["top1"]["predictions"] == len(table)
    assert 0 <= result["all_moves"]["ece"] <= 1 and 0 <= result["top1"]["ece"] <= 1
