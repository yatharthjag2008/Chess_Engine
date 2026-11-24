"""Sanity checks that the network can actually learn what it is for (tiny models, a few seconds each).

These are not the training pipeline (step 6); they only show that gradients
reach every part that matters: the board trunk, the rating embedding, the
opponent embedding, the clock features and the value head.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from chess_engine.config import ModelConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import build_dataset
from chess_engine.encoding import encode_board, legal_move_mask, move_to_index
from chess_engine.model.features import (
    NUM_CONTEXT_FEATURES,
    ModelInputs,
    Targets,
    inputs_from_batch,
    targets_from_batch,
)
from chess_engine.model.masking import legal_probs, policy_loss
from chess_engine.model.network import RatingConditionedNet
from dataset_helpers import random_game_rows, tmp_config, write_game_rows

TINY = dict(architecture="resnet", channels=16, residual_blocks=2, rating_embedding_dim=8)


def _train(model: RatingConditionedNet, inputs: ModelInputs, targets: Targets, steps: int,
           value_weight: float = 0.0, lr: float = 3e-3) -> list[float]:
    torch.manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        out = model(inputs)
        loss = policy_loss(out.policy, targets.legal_mask, targets.move)
        if value_weight:
            loss = loss + value_weight * F.cross_entropy(out.value, targets.result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return losses


def _start_position_task(n: int, bucket: torch.Tensor, opp_bucket: torch.Tensor | None = None,
                         clock: torch.Tensor | None = None, targets_uci: list[str] | None = None) -> tuple:
    """n copies of the starting position with the given conditioning and target moves."""
    board = chess.Board()
    planes = torch.as_tensor(np.repeat(encode_board(board)[None], n, axis=0))
    mask = torch.as_tensor(np.repeat(legal_move_mask(board)[None], n, axis=0))
    context = torch.zeros(n, NUM_CONTEXT_FEATURES) if clock is None else clock[:, None].repeat(1, NUM_CONTEXT_FEATURES)
    inputs = ModelInputs(planes, bucket, opp_bucket if opp_bucket is not None else torch.zeros(n, dtype=torch.long),
                         context)
    move = torch.tensor([move_to_index(chess.Move.from_uci(u)) for u in targets_uci])
    return inputs, Targets(move, mask, torch.ones(n, dtype=torch.long))


def _probs(model: RatingConditionedNet, inputs: ModelInputs, targets: Targets) -> torch.Tensor:
    with torch.no_grad():
        return legal_probs(model(inputs).policy, targets.legal_mask)


@pytest.fixture(scope="module")
def real_batch(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("learning")
    games = write_game_rows(random_game_rows(12, seed=5), tmp_path / "games.parquet")
    result = build_dataset(tmp_config(tmp_path), [games], tmp_path / "ds", show_progress=False)
    return PositionDataset(result.output_dir, "train").batch(np.arange(64))


def test_memorizes_a_batch_of_real_positions(real_batch, default_config) -> None:
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(**TINY), 8)
    inputs, targets = inputs_from_batch(real_batch, default_config.ratings), targets_from_batch(real_batch)
    losses = _train(model, inputs, targets, steps=150)
    assert losses[-1] < 0.25 * losses[0]
    with torch.no_grad():
        predicted = model(inputs).policy.masked_fill(~targets.legal_mask, -1e9).argmax(dim=1)
    assert (predicted == targets.move).float().mean() > 0.9


@pytest.mark.parametrize("conditioning", ["add", "film"])
def test_learns_a_different_move_for_each_rating(conditioning: str) -> None:
    """Same position, three rating groups, three different 'favourite' moves."""
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(**TINY, conditioning=conditioning), 8)
    plan = {0: "e2e4", 4: "g1f3", 7: "d2d4"}
    buckets = torch.tensor([b for b in plan for _ in range(16)])
    inputs, targets = _start_position_task(len(buckets), buckets, targets_uci=[plan[int(b)] for b in buckets])
    _train(model, inputs, targets, steps=150)
    probs = _probs(model, inputs, targets)
    for bucket, uci in plan.items():
        row = int((buckets == bucket).nonzero()[0])
        assert probs[row, move_to_index(chess.Move.from_uci(uci))] > 0.8, (bucket, uci)


def test_learns_from_the_opponent_rating_and_the_clock() -> None:
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(**TINY), 8)
    n = 64
    opp = torch.tensor([0, 7] * (n // 2))
    clock = torch.tensor([0.05] * (n // 2) + [1.0] * (n // 2))
    # target depends on both: (weak opponent, low clock) -> a3, (weak, high) -> e4, (strong, low) -> h3, (strong, high) -> d4
    table = {(0, True): "a2a3", (0, False): "e2e4", (7, True): "h2h3", (7, False): "d2d4"}
    targets_uci = [table[(int(o), bool(c < 0.5))] for o, c in zip(opp, clock)]
    inputs, targets = _start_position_task(n, torch.full((n,), 3), opp, clock, targets_uci)
    _train(model, inputs, targets, steps=200)
    probs = _probs(model, inputs, targets)
    for i in range(4):
        assert probs[i, targets.move[i]] > 0.7, targets_uci[i]
    for i in (n // 2, n // 2 + 1):
        assert probs[i, targets.move[i]] > 0.7, targets_uci[i]


def test_ignores_the_opponent_when_configured_to() -> None:
    model = RatingConditionedNet(ModelConfig(**TINY, condition_on_opponent=False, use_clock=False), 8)
    n = 32
    opp = torch.tensor([0, 7] * (n // 2))
    inputs, targets = _start_position_task(n, torch.full((n,), 3), opp, None,
                                           ["e2e4" if o == 0 else "d2d4" for o in opp])
    _train(model, inputs, targets, steps=100)
    probs = _probs(model, inputs, targets)
    assert torch.allclose(probs[0], probs[1])  # cannot tell the two apart, so it cannot learn the split


def test_value_head_learns_results_from_ratings() -> None:
    torch.manual_seed(0)
    model = RatingConditionedNet(ModelConfig(**TINY), 8)
    n = 48
    bucket = torch.tensor([7, 0] * (n // 2))
    opp = torch.tensor([0, 7] * (n // 2))
    inputs, targets = _start_position_task(n, bucket, opp, None, ["e2e4"] * n)
    targets = Targets(targets.move, targets.legal_mask, torch.where(bucket > opp, 2, 0))  # stronger player wins
    _train(model, inputs, targets, steps=150, value_weight=1.0)
    with torch.no_grad():
        wdl = torch.softmax(model(inputs).value, dim=1)
    assert wdl[0, 2] > 0.8 and wdl[1, 0] > 0.8
