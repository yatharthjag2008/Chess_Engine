"""Tests for the per-position prediction pass and its saved form."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.predictions import PredictionTable, predict_split
from chess_engine.model.features import inputs_from_batch
from chess_engine.model.masking import legal_probs
from chess_engine.model.network import RatingConditionedNet

TINY = ModelConfig(architecture="resnet", channels=8, residual_blocks=1, rating_embedding_dim=4)


def _model() -> RatingConditionedNet:
    torch.manual_seed(0)
    model = RatingConditionedNet(TINY, 8)
    with torch.no_grad():  # make the rating matter, as after training
        for block in model.blocks:
            block.modulation.weight.normal_(0, 0.5)
    return model.eval()


@pytest.fixture(scope="module")
def table(tiny_dataset: Path, default_config: Config) -> PredictionTable:
    return predict_split(_model(), PositionDataset(tiny_dataset, "test"), default_config.ratings, batch_size=64)


def test_columns_match_a_direct_computation(table: PredictionTable, tiny_dataset: Path,
                                            default_config: Config) -> None:
    dataset = PositionDataset(tiny_dataset, "test")
    assert len(table) == len(dataset) == table.meta["positions"]
    batch = dataset.batch(np.arange(len(dataset)))
    with torch.no_grad():
        probs = legal_probs(_model()(inputs_from_batch(batch, default_config.ratings)).policy,
                            torch.as_tensor(batch.legal_mask)).numpy()
    rows = np.arange(len(dataset))
    np.testing.assert_allclose(table["target_prob"], probs[rows, batch.move], rtol=1e-5, atol=1e-7)
    assert np.array_equal(table.top1_hit, probs.argmax(1) == batch.move)
    assert np.array_equal(table["top_moves"][:, 0], probs.argmax(1))
    assert np.array_equal(table["legal_count"], batch.legal_mask.sum(1))
    assert np.array_equal(table["bucket"], batch.features["bucket"])
    assert (table["target_rank"] >= 1).all() and (table["target_rank"] <= table["legal_count"]).all()
    assert np.all(np.diff(table["top_probs"], axis=1) <= 1e-7)  # sorted, likeliest first
    assert table["value"].shape == (len(dataset), 3)
    np.testing.assert_allclose(table["value"].sum(1), 1, rtol=1e-5)


def test_hits_and_calibration_histogram(table: PredictionTable) -> None:
    assert np.array_equal(table.hits(1), table.top1_hit)
    assert (table.hits(5) >= table.hits(3)).all() and table.hits(5).mean() >= table.hits(1).mean()
    hist = table.legal_calibration
    assert hist[:, 0].sum() == table["legal_count"].sum()  # every legal move of every position
    assert hist[:, 2].sum() == len(table)                  # exactly one played move per position
    assert hist[:, 1].sum() == pytest.approx(len(table), rel=1e-4)  # probabilities sum to 1 per position


def test_max_positions_and_conditioned_bucket(tiny_dataset: Path, default_config: Config,
                                              table: PredictionTable) -> None:
    dataset = PositionDataset(tiny_dataset, "test")
    first = predict_split(_model(), dataset, default_config.ratings, batch_size=16, max_positions=40)
    assert len(first) == 40
    np.testing.assert_allclose(first["target_prob"], table["target_prob"][:40], rtol=1e-5, atol=1e-7)
    low = predict_split(_model(), dataset, default_config.ratings, max_positions=40, conditioned_bucket=0)
    high = predict_split(_model(), dataset, default_config.ratings, max_positions=40, conditioned_bucket=7)
    assert np.array_equal(low["bucket"], first["bucket"])  # the true bucket is kept
    assert not np.allclose(low["target_prob"], high["target_prob"])
    assert low.meta["conditioned_bucket"] == 0
    with pytest.raises(ValueError, match="conditioned_bucket"):
        predict_split(_model(), dataset, default_config.ratings, conditioned_bucket=8)
    with pytest.raises(ValueError, match="in-memory"):
        predict_split(_model(), PositionDataset(tiny_dataset, "test", in_memory=False), default_config.ratings)


def test_progress_callback(tiny_dataset: Path, default_config: Config) -> None:
    seen = []
    predict_split(_model(), PositionDataset(tiny_dataset, "test"), default_config.ratings, batch_size=50,
                  max_positions=120, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(50, 120), (100, 120), (120, 120)]


def test_save_and_load_round_trip(table: PredictionTable, tmp_path: Path) -> None:
    path = table.save(tmp_path / "sub" / "predictions.npz")
    loaded = PredictionTable.load(path)
    assert set(loaded.columns) == set(table.columns)
    for name, values in table.columns.items():
        assert np.array_equal(loaded[name], values) and loaded[name].dtype == values.dtype
    assert np.array_equal(loaded.legal_calibration, table.legal_calibration)
    assert loaded.meta["split"] == "test" and loaded.meta["positions"] == len(table)
    part = loaded.take(np.arange(10))
    assert len(part) == 10 and "key" in part


def test_loading_another_format_fails(tmp_path: Path) -> None:
    import json

    path = tmp_path / "old.npz"
    np.savez(path, legal_calibration=np.zeros((2, 3)),
             meta=np.frombuffer(json.dumps({"format_version": 0}).encode(), dtype=np.uint8))
    with pytest.raises(ValueError, match="format"):
        PredictionTable.load(path)
