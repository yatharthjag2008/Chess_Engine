"""Output heads: move logits (policy) and an optional win/draw/loss prediction (value)."""

from __future__ import annotations

import torch
from torch import nn

from chess_engine.encoding.moves import NUM_MOVES
from chess_engine.model.layers import ConvBlock
from chess_engine.model.policy_map import NUM_MOVE_TYPES, policy_gather_index


class ConvPolicyHead(nn.Module):
    """Score each of 73 move types from each square with convolutions, then pick the 1858 real moves.

    ``conv3x3-BN-ReLU -> conv1x1 (73 planes)`` gives a ``(B, 73, 8, 8)`` map, flattened
    and gathered with :func:`~chess_engine.model.policy_map.policy_gather_index`.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(channels, channels, 3)
        self.moves = nn.Conv2d(channels, NUM_MOVE_TYPES, 1)
        self.register_buffer("gather_index", torch.as_tensor(policy_gather_index().copy()), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        planes = self.moves(self.block(features))      # (B, 73, 8, 8)
        return planes.flatten(1).index_select(1, self.gather_index)  # (B, 1858)


class FCPolicyHead(nn.Module):
    """``conv1x1 (32 planes)-BN-ReLU -> flatten -> dense layer`` to 1858 logits (~3.8M weights)."""

    def __init__(self, channels: int, planes: int = 32) -> None:
        super().__init__()
        self.block = ConvBlock(channels, planes, 1)
        self.fc = nn.Linear(planes * 64, NUM_MOVES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.block(features).flatten(1))


class ValueHead(nn.Module):
    """Win/draw/loss logits for the side to move, from the board features and the conditioning vector.

    The conditioning vector is an input because the expected result depends on
    both players' ratings and clocks, not only on the position.
    """

    def __init__(self, channels: int, cond_dim: int, planes: int = 8, hidden: int = 128) -> None:
        super().__init__()
        self.block = ConvBlock(channels, planes, 1)
        self.mlp = nn.Sequential(
            nn.Linear(planes * 64 + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.block(features).flatten(1), cond], dim=1))  # (B, 3): loss, draw, win
