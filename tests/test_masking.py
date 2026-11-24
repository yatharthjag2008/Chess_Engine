"""Tests for restricting move logits to legal moves."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_engine.encoding import legal_move_mask, move_to_index
from chess_engine.encoding.moves import NUM_MOVES
from chess_engine.model.masking import legal_log_probs, legal_probs, masked_logits, policy_loss

BOARDS = [chess.Board(), chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1")]


@pytest.fixture()
def mask() -> torch.Tensor:
    return torch.as_tensor(np.stack([legal_move_mask(b) for b in BOARDS]))


@pytest.fixture()
def logits() -> torch.Tensor:
    return torch.randn(2, NUM_MOVES, generator=torch.Generator().manual_seed(0))


def test_probs_are_zero_on_illegal_moves_and_sum_to_one(logits: torch.Tensor, mask: torch.Tensor) -> None:
    probs = legal_probs(logits, mask)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    assert (probs[~mask] == 0).all() and (probs[mask] > 0).all()


def test_probs_equal_softmax_over_the_legal_subset(logits: torch.Tensor, mask: torch.Tensor) -> None:
    probs = legal_probs(logits, mask)
    for row in range(2):
        expected = torch.softmax(logits[row][mask[row]], dim=0)
        assert torch.allclose(probs[row][mask[row]], expected, atol=1e-6)


def test_temperature(logits: torch.Tensor, mask: torch.Tensor) -> None:
    sharp = legal_probs(logits, mask, temperature=0.01)
    best = masked_logits(logits, mask).argmax(dim=1)
    assert torch.allclose(sharp[torch.arange(2), best], torch.ones(2), atol=1e-3)
    flat = legal_probs(logits, mask, temperature=100.0)
    counts = mask.sum(dim=1, keepdim=True)
    assert torch.allclose(flat[mask], (1 / counts).expand_as(flat)[mask], atol=1e-2)
    with pytest.raises(ValueError, match="temperature"):
        legal_probs(logits, mask, temperature=0)


def test_loss_is_negative_log_prob_of_the_played_move(logits: torch.Tensor, mask: torch.Tensor) -> None:
    target = torch.tensor([move_to_index(chess.Move.from_uci("e2e4")),
                           move_to_index(chess.Move.from_uci("e8g8"), chess.BLACK)])
    loss = policy_loss(logits, mask, target, reduction="none")
    log_probs = legal_log_probs(logits, mask)
    assert torch.allclose(loss, -log_probs[torch.arange(2), target])
    assert torch.isclose(policy_loss(logits, mask, target), loss.mean())


def test_illegal_logits_get_no_gradient(logits: torch.Tensor, mask: torch.Tensor) -> None:
    logits = logits.clone().requires_grad_(True)
    target = mask.float().argmax(dim=1)
    policy_loss(logits, mask, target).backward()
    assert (logits.grad[~mask] == 0).all() and (logits.grad[mask] != 0).any()


def test_half_precision_is_finite(logits: torch.Tensor, mask: torch.Tensor) -> None:
    probs = legal_probs(logits.half(), mask)
    assert torch.isfinite(probs).all() and torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-3)
    assert torch.isfinite(policy_loss(logits.half(), mask, mask.float().argmax(dim=1)))


def test_game_over_position_raises(logits: torch.Tensor, mask: torch.Tensor) -> None:
    empty = mask.clone()
    empty[1] = False
    with pytest.raises(ValueError, match="no legal moves"):
        legal_probs(logits, empty)
    assert torch.isfinite(masked_logits(logits, empty, strict=False)).all()


def test_illegal_target_raises(logits: torch.Tensor, mask: torch.Tensor) -> None:
    illegal = (~mask).float().argmax(dim=1)
    with pytest.raises(ValueError, match="not legal"):
        policy_loss(logits, mask, illegal)


def test_shape_and_dtype_checks(logits: torch.Tensor, mask: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape"):
        masked_logits(logits[:, :10], mask)
    with pytest.raises(TypeError, match="bool"):
        masked_logits(logits, mask.float())
