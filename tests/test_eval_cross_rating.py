"""Tests for the cross-rating matrix and the fixed-rating baseline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.cross_rating import cross_rating_from_tables, run_cross_rating
from chess_engine.model.network import RatingConditionedNet
from eval_helpers import LABELS, make_table


def _tables(ranks_per_model: list[list[int]], buckets: list[int]):
    return [make_table(rank=ranks, prob=[0.5] * len(ranks), bucket=buckets) for ranks in ranks_per_model]


def test_matrix_diagonal_and_fixed_rating() -> None:
    buckets = [0, 0, 1, 1, 2, 2]
    # Each model bucket predicts its own players perfectly and the others half the time / never.
    tables = _tables([
        [1, 1, 1, 2, 2, 2],   # model "low": low 100%, mid 50%, high 0%
        [1, 2, 1, 1, 1, 2],   # model "mid": low 50%, mid 100%, high 50%
        [2, 2, 2, 2, 1, 1],   # model "high": low 0%, mid 0%, high 100%
    ], buckets)
    result = cross_rating_from_tables(tables, LABELS)
    assert result["top1"][0] == [1.0, 0.5, 0.0]
    assert result["top1"][1] == [0.5, 1.0, 0.5]
    assert result["diagonal_top1"] == [1.0, 1.0, 1.0] and result["matched_top1"] == 1.0
    assert result["best_model_bucket"] == ["low", "mid", "high"]
    assert result["fixed_rating"]["bucket"] == "mid"
    assert result["fixed_rating"]["top1"] == pytest.approx(4 / 6)
    assert result["positions"] == [2, 2, 2]


def test_missing_player_bucket_and_errors() -> None:
    tables = _tables([[1, 2], [2, 1], [1, 1]], [0, 0])
    result = cross_rating_from_tables(tables, LABELS)
    assert result["top1"][0][1] is None and result["diagonal_top1"][1] is None
    assert result["best_model_bucket"] == ["high", None, None]
    with pytest.raises(ValueError, match="one per bucket"):
        cross_rating_from_tables(tables[:2], LABELS)
    shorter = make_table(rank=[1], prob=[0.5], bucket=[0])
    with pytest.raises(ValueError, match="same positions"):
        cross_rating_from_tables([tables[0], shorter, tables[2]], LABELS)


def test_run_on_a_model(tiny_dataset: Path, default_config: Config) -> None:
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(architecture="resnet", channels=8, residual_blocks=1,
                                             rating_embedding_dim=4), 8)
    seen = []
    result = run_cross_rating(model, PositionDataset(tiny_dataset, "test"), default_config.ratings,
                              max_positions=60, batch_size=32, progress=lambda d, t: seen.append(d))
    assert seen == list(range(1, 9))
    assert len(result["top1"]) == 8 and sum(result["positions"]) == 60
    # Untrained, the modulation layers are zero, so the rating changes nothing: all rows are equal.
    rows = result["fixed_rating"]["top1_per_model_bucket"]
    assert max(rows) - min(rows) < 1e-9
