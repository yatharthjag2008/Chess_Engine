"""Tests for the training loss (masked policy cross-entropy, label smoothing, value loss, weights)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from chess_engine.model.features import ModelInputs, Targets
from chess_engine.model.masking import policy_loss
from chess_engine.model.network import ModelOutput
from chess_engine.training.loader import TrainBatch
from chess_engine.training.losses import compute_loss, policy_losses


def _batch(weights: torch.Tensor | None = None, n: int = 6, moves: int = 1858) -> tuple[ModelOutput, TrainBatch]:
    g = torch.Generator().manual_seed(0)
    mask = torch.zeros(n, moves, dtype=torch.bool)
    for i in range(n):
        mask[i, torch.randperm(moves, generator=g)[: 5 + i]] = True
    target = torch.stack([torch.nonzero(mask[i])[0, 0] for i in range(n)])
    inputs = ModelInputs(torch.zeros(n, 20, 8, 8), torch.zeros(n, dtype=torch.long),
                         torch.zeros(n, dtype=torch.long), torch.zeros(n, 5))
    batch = TrainBatch(inputs, Targets(target, mask, torch.randint(0, 3, (n,), generator=g)), weights)
    out = ModelOutput(torch.randn(n, moves, generator=g, requires_grad=True),
                      torch.randn(n, 3, generator=g, requires_grad=True))
    return out, batch


def test_plain_loss_matches_masked_cross_entropy() -> None:
    out, batch = _batch()
    loss = compute_loss(out, batch, value_weight=0.0, strict=True)
    expected = policy_loss(out.policy, batch.targets.legal_mask, batch.targets.move)
    assert loss.total.item() == pytest.approx(expected.item(), rel=1e-6)
    assert loss.value is None
    assert loss.nll.mean().item() == pytest.approx(expected.item(), rel=1e-6)


def test_value_loss_is_added_with_its_weight() -> None:
    out, batch = _batch()
    loss = compute_loss(out, batch, value_weight=0.5)
    value = F.cross_entropy(out.value, batch.targets.result)
    assert loss.value.item() == pytest.approx(value.item(), rel=1e-6)
    assert loss.total.item() == pytest.approx(loss.policy.item() + 0.5 * value.item(), rel=1e-6)


def test_weights_scale_each_position() -> None:
    weights = torch.tensor([2.0, 0.0, 1.0, 1.0, 0.5, 1.5])
    out, batch = _batch(weights)
    loss = compute_loss(out, batch, value_weight=0.0)
    assert loss.total.item() == pytest.approx((loss.nll * weights).mean().item(), rel=1e-6)
    assert loss.nll.requires_grad is False


def test_label_smoothing_uses_only_legal_moves() -> None:
    log_probs = torch.log(torch.tensor([[0.5, 0.25, 0.25, 0.0]]).clamp_min(1e-30))
    mask = torch.tensor([[True, True, True, False]])
    smoothed, nll = policy_losses(log_probs, mask, torch.tensor([0]), label_smoothing=0.3)
    uniform = -(torch.log(torch.tensor(0.5)) + 2 * torch.log(torch.tensor(0.25))) / 3
    assert nll.item() == pytest.approx(-torch.log(torch.tensor(0.5)).item())
    assert smoothed.item() == pytest.approx(0.7 * nll.item() + 0.3 * uniform.item(), rel=1e-5)


def test_gradient_reaches_only_legal_logits() -> None:
    out, batch = _batch()
    compute_loss(out, batch, value_weight=0.25, label_smoothing=0.1).total.backward()
    grad = out.policy.grad
    assert torch.all(grad[~batch.targets.legal_mask] == 0)
    assert torch.all(grad[batch.targets.legal_mask].abs().sum(-1) > 0)
    assert out.value.grad is not None


def test_strict_mode_catches_an_illegal_target() -> None:
    out, batch = _batch()
    mask = batch.targets.legal_mask.clone()
    mask[:, 1857] = False
    bad = TrainBatch(batch.inputs, Targets(torch.full_like(batch.targets.move, 1857), mask, batch.targets.result))
    with pytest.raises(ValueError, match="not legal"):
        compute_loss(out, bad, strict=True)
