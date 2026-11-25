"""Helpers for bot tests: a small random model whose moves depend on the rating."""

from __future__ import annotations

import torch

from chess_engine.config import ModelConfig, RatingsConfig
from chess_engine.model.inference import MovePredictor
from chess_engine.model.network import RatingConditionedNet

SMALL = ModelConfig(architecture="resnet", channels=16, residual_blocks=2, rating_embedding_dim=8)
RATINGS = RatingsConfig(bucket_edges=[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200])


def small_predictor(seed: int = 0) -> MovePredictor:
    """An untrained small model whose rating modulation is switched on (as after training)."""
    torch.manual_seed(seed)
    net = RatingConditionedNet(SMALL, num_buckets=RATINGS.num_buckets)
    with torch.no_grad():
        for block in net.blocks:
            block.modulation.weight.normal_(0, 0.5)
    return MovePredictor(net, RATINGS)
