"""Restrict the network's move logits to legal moves.

The network scores all 1858 vocabulary moves; for a given position only ~30 are
legal. Illegal moves get the most negative representable logit (not ``-inf``, so
nothing becomes NaN, also in float16), which gives them probability exactly 0
after softmax and no gradient.

    probs = legal_probs(out.policy, legal_mask)            # (B, 1858), 0 on illegal moves
    loss = policy_loss(out.policy, legal_mask, target)      # cross-entropy over legal moves only
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(logits: torch.Tensor, legal_mask: torch.Tensor, strict: bool) -> None:
    if logits.shape != legal_mask.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and legal_mask {tuple(legal_mask.shape)} differ in shape")
    if legal_mask.dtype != torch.bool:
        raise TypeError(f"legal_mask must be bool, got {legal_mask.dtype}")
    if strict and not bool(legal_mask.any(dim=-1).all()):
        raise ValueError("a position has no legal moves (game over); there is no move to predict")


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor, strict: bool = True) -> torch.Tensor:
    """``logits`` with every illegal move set to the lowest finite value of the dtype.

    Args:
        strict: Raise if some position has no legal move at all (costs a device sync;
            the training loop can turn it off because the dataset never contains such positions).
    """
    _check(logits, legal_mask, strict)
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def legal_log_probs(logits: torch.Tensor, legal_mask: torch.Tensor, strict: bool = True) -> torch.Tensor:
    """Log-probabilities over legal moves (illegal entries are very negative, not ``-inf``)."""
    return F.log_softmax(masked_logits(logits, legal_mask, strict).float(), dim=-1)


def legal_probs(
    logits: torch.Tensor, legal_mask: torch.Tensor, temperature: float = 1.0, strict: bool = True
) -> torch.Tensor:
    """Probabilities over legal moves; ``temperature`` < 1 sharpens, > 1 flattens.

    Computed in float32 for stability; exactly 0 for illegal moves.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive (use argmax of the masked logits for greedy play)")
    probs = F.softmax(masked_logits(logits, legal_mask, strict).float() / temperature, dim=-1)
    return probs.masked_fill(~legal_mask, 0.0)


def policy_loss(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    strict: bool = True,
) -> torch.Tensor:
    """Cross-entropy of the played move ``target`` under the legal-move distribution.

    With ``strict``, also checks that every target is a legal move (a wrong target
    would otherwise give a huge, silent loss).
    """
    if strict and not bool(legal_mask.gather(-1, target[:, None]).all()):
        raise ValueError("a target move is not legal in its position")
    return F.cross_entropy(masked_logits(logits, legal_mask, strict).float(), target, reduction=reduction)
