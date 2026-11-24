"""The training objective: cross-entropy of the human move over the legal moves, plus the value loss.

::

    total = mean_i w_i * [ policy_i + value_weight * value_i ]

    policy_i = (1 - eps) * -log p(played move)  +  eps * mean over legal moves of -log p(move)
    value_i  = cross-entropy of the game result (loss / draw / win for the mover)

``w_i`` are the per-position rating-bucket weights (1 when not balancing). They
average 1 over the dataset, so ``mean(w * loss)`` is an unbiased estimate of the
weighted objective and the loss scale does not depend on the balancing setting.
Label smoothing ``eps`` spreads probability only over the *legal* moves; spreading
it over all 1858 moves would teach the network nothing useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chess_engine.model.masking import legal_log_probs
from chess_engine.model.network import ModelOutput
from chess_engine.training.loader import TrainBatch


@dataclass(frozen=True)
class LossBreakdown:
    """The loss to back-propagate and its parts (the per-position tensors are detached)."""

    total: torch.Tensor                 # scalar, with gradient
    policy: torch.Tensor                # scalar, weighted mean policy loss (detached)
    value: torch.Tensor | None          # scalar, weighted mean value loss (detached), if there is a value head
    nll: torch.Tensor                   # (B,) -log p(played move), unweighted, unsmoothed (detached)
    log_probs: torch.Tensor             # (B, NUM_MOVES) legal log-probabilities (detached)


def policy_losses(
    log_probs: torch.Tensor, legal_mask: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position (smoothed loss, plain negative log-likelihood of the target)."""
    nll = -log_probs.gather(1, target[:, None]).squeeze(1)
    if label_smoothing == 0:
        return nll, nll
    legal = legal_mask.float()
    uniform = -(log_probs * legal).sum(1) / legal.sum(1).clamp_min(1)
    return (1 - label_smoothing) * nll + label_smoothing * uniform, nll


def compute_loss(
    out: ModelOutput,
    batch: TrainBatch,
    value_weight: float = 0.25,
    label_smoothing: float = 0.0,
    strict: bool = False,
) -> LossBreakdown:
    """The training loss for one batch (see the module docstring).

    Args:
        strict: Check that every position has a legal move and every target is legal
            (a device sync per batch; tests turn it on, training leaves it off).
    """
    targets = batch.targets
    if strict and not bool(targets.legal_mask.gather(1, targets.move[:, None]).all()):
        raise ValueError("a target move is not legal in its position")
    log_probs = legal_log_probs(out.policy, targets.legal_mask, strict=strict)
    per_position, nll = policy_losses(log_probs, targets.legal_mask, targets.move, label_smoothing)

    weights = batch.weights

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * weights).mean() if weights is not None else values.mean()

    policy = weighted_mean(per_position)
    total = policy
    value = None
    if out.value is not None and value_weight > 0:
        value = weighted_mean(F.cross_entropy(out.value.float(), targets.result, reduction="none"))
        total = policy + value_weight * value
    return LossBreakdown(total, policy.detach(), value.detach() if value is not None else None,
                         nll.detach(), log_probs.detach())
