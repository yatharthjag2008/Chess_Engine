"""Rating-conditioned policy network: board planes + who is moving -> move logits (and win/draw/loss).

::

    planes (B, 20, 8, 8) --stem conv--> (B, C, 8, 8) --N conditioned residual blocks--> features
                                                         ^ every block
    bucket, opp_bucket, clock features --ConditionEncoder--> cond (B, C)
                                                         |
    features --policy head--> move logits (B, 1858)      |
    features + cond --value head--> win/draw/loss logits (B, 3)   (optional)

The policy logits are **unmasked**: illegal moves are removed with
:func:`chess_engine.model.masking.masked_logits` (training loss, sampling and
inference all do this), so the network itself never needs to know the rules.

Usage::

    model = build_model(cfg)                         # from cfg.model and cfg.ratings
    out = model(inputs_from_batch(batch, cfg.ratings))
    out.policy, out.value
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine.config import Config, ModelConfig
from chess_engine.encoding.board import NUM_PLANES
from chess_engine.model.features import ModelInputs
from chess_engine.model.heads import ConvPolicyHead, FCPolicyHead, ValueHead
from chess_engine.model.layers import ConditionedResidualBlock, ConditionEncoder, ConvBlock


@dataclass(frozen=True)
class ModelOutput:
    """Raw network outputs for a batch."""

    policy: torch.Tensor               # (B, NUM_MOVES) unmasked move logits
    value: torch.Tensor | None = None  # (B, 3) logits for loss / draw / win, if the model has a value head


class RatingConditionedNet(nn.Module):
    """ResNet over the board whose every residual block is modulated by the rating/clock conditioning.

    Args:
        config: Model hyperparameters (``cfg.model``).
        num_buckets: Number of rating buckets (``cfg.ratings.num_buckets``).
    """

    def __init__(self, config: ModelConfig, num_buckets: int) -> None:
        super().__init__()
        if config.architecture != "resnet":
            raise ValueError(f"unknown architecture {config.architecture!r}")
        self.config = config
        self.num_buckets = num_buckets
        channels = config.channels
        self.condition = ConditionEncoder(
            num_buckets,
            config.rating_embedding_dim,
            out_dim=channels,
            condition_on_opponent=config.condition_on_opponent,
            use_clock=config.use_clock,
        )
        self.stem = ConvBlock(NUM_PLANES, channels, 3)
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(channels, channels, config.conditioning) for _ in range(config.residual_blocks)
        )
        self.policy = ConvPolicyHead(channels) if config.policy_head == "conv" else FCPolicyHead(channels)
        self.value = ValueHead(channels, channels) if config.value_head else None

    def forward(self, inputs: ModelInputs) -> ModelOutput:
        cond = self.condition(inputs)
        x = self.stem(inputs.planes)
        for block in self.blocks:
            x = block(x, cond)
        value = self.value(x, cond) if self.value is not None else None
        return ModelOutput(self.policy(x), value)


def build_model(cfg: Config) -> RatingConditionedNet:
    """The network described by ``cfg.model``, sized for ``cfg.ratings``' buckets (randomly initialized)."""
    return RatingConditionedNet(cfg.model, cfg.ratings.num_buckets)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable parameters per top-level part (``condition``, ``stem``, ``blocks``, ...) and ``total``."""
    counts = {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in model.named_children()
    }
    counts["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts
