"""A style adapter: a few thousand parameters that bend the frozen rating model towards one player.

The base network stays as it is (weights and batch-norm statistics). The adapter adds, all starting at zero:

* an **offset to the conditioning vector** (the "who is moving" input every residual block reads),
* a **shift to each block's modulation** (per-channel FiLM scale and shift),
* a **bias per move** in the policy (moves are encoded from the mover's side, so "likes 1. e4 / ...e5" is one
  preference for both colours).

At zero the styled network is exactly the base network, so training starts from the rating model and only learns
how the player differs from players of their rating.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_engine.encoding.moves import NUM_MOVES
from chess_engine.model.features import ModelInputs
from chess_engine.model.network import ModelOutput, RatingConditionedNet


class StyleAdapter(nn.Module):
    """The adapter's parameters, sized for a base network."""

    def __init__(self, cond_dim: int, channels: int, num_blocks: int, conditioning: str = "film") -> None:
        super().__init__()
        width = channels * (2 if conditioning == "film" else 1)
        self.offset = nn.Parameter(torch.zeros(cond_dim))
        self.block_shift = nn.Parameter(torch.zeros(num_blocks, width))
        self.policy_bias = nn.Parameter(torch.zeros(NUM_MOVES))

    @classmethod
    def for_model(cls, base: RatingConditionedNet) -> StyleAdapter:
        cfg = base.config
        return cls(cfg.channels, cfg.channels, cfg.residual_blocks, cfg.conditioning)


class StyledNet(nn.Module):
    """The frozen base network with a style adapter; same inputs and outputs as the base."""

    def __init__(self, base: RatingConditionedNet, adapter: StyleAdapter | None = None) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.adapter = adapter or StyleAdapter.for_model(base)
        self.config = base.config
        self.num_buckets = base.num_buckets
        self.base.eval()

    def train(self, mode: bool = True) -> StyledNet:
        """Only the adapter trains; the base keeps its batch-norm statistics (always in eval mode)."""
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, inputs: ModelInputs) -> ModelOutput:
        base, adapter = self.base, self.adapter
        cond = base.condition(inputs) + adapter.offset
        x = base.stem(inputs.planes)
        for index, block in enumerate(base.blocks):
            x = block(x, cond, adapter.block_shift[index])
        value = base.value(x, cond) if base.value is not None else None
        return ModelOutput(base.policy(x) + adapter.policy_bias, value)
