"""Train a style adapter on one player's moves (the base model stays frozen).

Each epoch shuffles the training positions; after it, the held-out games are scored. Training stops when the
held-out loss has not improved for ``patience`` epochs, and the best epoch's adapter is kept, so a style never
ends up worse on unseen games than where it started (the untrained adapter equals the base model).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from chess_engine.config import StyleConfig
from chess_engine.model.masking import masked_logits, policy_loss
from chess_engine.model.network import RatingConditionedNet
from chess_engine.style.adapter import StyleAdapter, StyledNet
from chess_engine.style.data import StyleExamples


@dataclass(frozen=True)
class Score:
    """Average loss (cross-entropy of the moves played) and how often the top move was the move played."""

    loss: float
    top1: float


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    train_loss: float
    val: Score | None


@dataclass
class StyleTrainResult:
    adapter: StyleAdapter
    base: Score | None                # the base model on the held-out games
    styled: Score | None              # the chosen adapter on the held-out games
    best_epoch: int
    history: list[EpochLog] = field(default_factory=list)
    train_positions: int = 0
    val_positions: int = 0


@torch.no_grad()
def score(net: torch.nn.Module, examples: StyleExamples, batch_size: int = 512) -> Score:
    """Loss and top-1 move match of ``net`` on ``examples``."""
    net.eval()
    total_loss, hits = 0.0, 0
    n = len(examples)
    for start in range(0, n, batch_size):
        part = examples.subset(np.arange(start, min(n, start + batch_size)))
        logits = net(part.inputs).policy
        legal = torch.as_tensor(part.legal)
        target = torch.as_tensor(part.targets)
        total_loss += float(policy_loss(logits, legal, target, reduction="sum"))
        hits += int((masked_logits(logits, legal).argmax(dim=-1) == target).sum())
    return Score(total_loss / n, hits / n)


def train_style(base: RatingConditionedNet, train: StyleExamples, val: StyleExamples | None, cfg: StyleConfig,
                log: Callable[[str], None] | None = None) -> StyleTrainResult:
    """Train an adapter for ``base`` on ``train``; ``val`` (held-out games) picks the best epoch and stops early."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = StyledNet(base)
    optimizer = torch.optim.AdamW(net.adapter.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    base_score = score(base, val) if val is not None and len(val) else None
    best = copy.deepcopy(net.adapter.state_dict())
    best_val, best_epoch, stale = (base_score.loss if base_score else float("inf")), 0, 0
    history: list[EpochLog] = []
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            part = train.subset(order[start:start + cfg.batch_size])
            loss = policy_loss(net(part.inputs).policy, torch.as_tensor(part.legal), torch.as_tensor(part.targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_score = score(net, val) if base_score is not None else None
        history.append(EpochLog(epoch, float(np.mean(losses)), val_score))
        if log:
            log(f"epoch {epoch}: train loss {history[-1].train_loss:.3f}"
                + (f", held-out loss {val_score.loss:.3f}, top-1 {val_score.top1:.1%}" if val_score else ""))
        if val_score is None:
            best, best_epoch = copy.deepcopy(net.adapter.state_dict()), epoch
            continue
        if val_score.loss < best_val - 1e-4:
            best, best_val, best_epoch, stale = copy.deepcopy(net.adapter.state_dict()), val_score.loss, epoch, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    net.adapter.load_state_dict(best)
    styled_score = score(net, val) if base_score is not None else None
    return StyleTrainResult(net.adapter, base_score, styled_score, best_epoch, history, len(train),
                            len(val) if val is not None else 0)
