"""Tests for turning dataset batches into model inputs and targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_engine.config import Config
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import build_dataset
from chess_engine.model.features import (
    NUM_CONTEXT_FEATURES,
    buckets_from_ratings,
    context_features,
    inputs_from_batch,
    make_inputs,
    targets_from_batch,
)
from dataset_helpers import random_game_rows, tmp_config, write_game_rows


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("features")
    games = write_game_rows(random_game_rows(10, seed=21), tmp_path / "games.parquet")
    result = build_dataset(tmp_config(tmp_path), [games], tmp_path / "ds", show_progress=False)
    return PositionDataset(result.output_dir, "train").batch(np.arange(64))


def test_context_features_values() -> None:
    ctx = context_features(np.array([90.0, 0.0, 600.0]), np.array([180.0, 10.0, 30.0]),
                           np.array([180, 60, 300]), np.array([2, 0, 40]))
    assert ctx.shape == (3, NUM_CONTEXT_FEATURES) and ctx.dtype == np.float32
    np.testing.assert_allclose(ctx[0, :2], [0.5, 1.0])
    np.testing.assert_allclose(ctx[2, 0], 2.0)          # fraction capped at 2
    np.testing.assert_allclose(ctx[1, 2], 0.0)          # no time left
    np.testing.assert_allclose(ctx[2, 2], 1.0)          # 10 minutes = 1
    np.testing.assert_allclose(ctx[:, 4], [0.2, 0.0, 3.0])  # increment / 10, capped at 3
    assert np.isfinite(ctx).all()


def test_context_features_handle_bad_clocks() -> None:
    ctx = context_features(np.array([-5.0]), np.array([0.0]), np.array([0]), np.array([0]))
    assert np.isfinite(ctx).all() and (ctx >= 0).all()


def test_buckets_from_ratings(default_config: Config) -> None:
    buckets = buckets_from_ratings([500, 800, 999, 1000, 1530, 2199, 2200, 3200], default_config.ratings)
    np.testing.assert_array_equal(buckets, [0, 0, 0, 1, 3, 6, 7, 7])
    assert buckets.dtype == np.int64


def test_inputs_from_batch(batch, default_config: Config) -> None:
    inputs = inputs_from_batch(batch, default_config.ratings)
    assert len(inputs) == 64
    assert inputs.planes.shape == (64, 20, 8, 8) and inputs.planes.dtype == torch.float32
    assert inputs.bucket.dtype == torch.int64 and inputs.opp_bucket.dtype == torch.int64
    assert inputs.context.shape == (64, NUM_CONTEXT_FEATURES)
    np.testing.assert_array_equal(inputs.bucket.numpy(), batch.features["bucket"])
    np.testing.assert_array_equal(inputs.opp_bucket.numpy(),
                                  buckets_from_ratings(batch.features["opp_rating"], default_config.ratings))
    np.testing.assert_array_equal(inputs.planes.numpy(), batch.planes)


def test_targets_from_batch(batch) -> None:
    targets = targets_from_batch(batch)
    assert targets.move.dtype == torch.int64 and targets.legal_mask.dtype == torch.bool
    assert targets.legal_mask[torch.arange(64), targets.move].all()
    np.testing.assert_array_equal(targets.result.numpy(), batch.features["result"] + 1)
    assert set(targets.result.tolist()) <= {0, 1, 2}


def test_to_device(batch, default_config: Config) -> None:
    inputs = inputs_from_batch(batch, default_config.ratings, device="cpu")
    assert inputs.to("cpu").planes.device.type == "cpu"
    assert targets_from_batch(batch, device="cpu").to("cpu").move.device.type == "cpu"


def test_make_inputs_converts_dtypes() -> None:
    inputs = make_inputs(np.zeros((2, 20, 8, 8), dtype=np.uint8), np.array([1, 2], dtype=np.int8),
                         np.array([0, 0]), np.zeros((2, NUM_CONTEXT_FEATURES)))
    assert inputs.planes.dtype == torch.float32 and inputs.bucket.dtype == torch.int64
    assert inputs.context.dtype == torch.float32
