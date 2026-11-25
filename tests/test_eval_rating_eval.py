"""Tests for scoring rating estimation on held-out games."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.rating_eval import EvalData, collect_eval_data
from chess_engine.model.network import RatingConditionedNet


@pytest.fixture(scope="module")
def eval_data(tiny_dataset: Path, default_config: Config) -> EvalData:
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(architecture="resnet", channels=8, residual_blocks=1,
                                             rating_embedding_dim=4), 8)
    with torch.no_grad():
        for block in model.blocks:
            block.modulation.weight.normal_(0, 0.5)
    return collect_eval_data(model.eval(), PositionDataset(tiny_dataset, "test"), default_config.ratings)


def test_player_games_and_true_ratings(eval_data: EvalData, tiny_dataset: Path) -> None:
    samples = PositionDataset(tiny_dataset, "test").samples
    assert eval_data.ll.shape == (len(samples), 8) and (eval_data.ll <= 0).all()
    games = set(samples["game"].tolist())
    assert eval_data.num_players == 2 * len(games)          # both sides of every test game
    assert set(eval_data.game.tolist()) == games
    for p in range(eval_data.num_players):
        moves = eval_data.player == p
        assert (eval_data.ply[moves] % 2 == eval_data.color[p]).all()
    # the true rating is the mover's rating of that side's moves
    rows = np.flatnonzero(eval_data.player == 0)
    assert eval_data.rating[0] == samples["rating"][rows[0]] and eval_data.opp_rating[0] == samples["opp_rating"][rows[0]]
    assert eval_data.moves_per_player().sum() == len(samples)


def test_summed_and_truncated(eval_data: EvalData) -> None:
    total = eval_data.summed()
    assert total.shape == (eval_data.num_players, 8)
    np.testing.assert_allclose(total.sum(), eval_data.ll.astype(np.float64).sum(), rtol=1e-6)
    first5 = eval_data.summed(max_moves=5)
    p = 0
    rows = np.flatnonzero(eval_data.player == p)
    rows = rows[np.argsort(eval_data.ply[rows])][:5]
    np.testing.assert_allclose(first5[p], eval_data.ll[rows].astype(np.float64).sum(0), rtol=1e-6)
    assert np.all(np.abs(first5) <= np.abs(total) + 1e-6)


def test_save_and_load(eval_data: EvalData, tmp_path: Path) -> None:
    loaded = EvalData.load(eval_data.save(tmp_path / "ll.npz"))
    for name in ("player", "ply", "ll", "rating", "opp_rating", "game", "color"):
        assert np.array_equal(getattr(loaded, name), getattr(eval_data, name))
