"""Building blocks of the rating-conditioned network.

* :class:`ConditionEncoder` turns "who is moving and under what conditions" (the
  mover's rating bucket, optionally the opponent's bucket and the clocks) into one
  conditioning vector per position.
* :class:`ConditionedResidualBlock` is a standard two-convolution residual block
  whose features are modulated by that vector after the first convolution:

  - ``add``:  ``h + beta(c)``                 (a per-channel shift, "rating embedding added")
  - ``film``: ``h * (1 + gamma(c)) + beta(c)`` (feature-wise linear modulation)

  The modulation layers start at zero, so an untrained network is a plain ResNet
  and the rating's influence is learned from there (stable at the start of training).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from chess_engine.model.features import NUM_CONTEXT_FEATURES, ModelInputs

Conditioning = Literal["add", "film"]


class ConvBlock(nn.Sequential):
    """Convolution (no bias) + batch norm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConditionEncoder(nn.Module):
    """Rating bucket(s) + clock features -> a ``(B, out_dim)`` conditioning vector.

    Args:
        num_buckets: Number of rating buckets (embedding table size).
        embedding_dim: Size of each bucket's learned vector.
        out_dim: Size of the conditioning vector.
        condition_on_opponent: Also embed the opponent's bucket (separate table).
        use_clock: Also read the clock features (``ModelInputs.context``).
    """

    def __init__(
        self,
        num_buckets: int,
        embedding_dim: int,
        out_dim: int,
        condition_on_opponent: bool = True,
        use_clock: bool = True,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.rating = nn.Embedding(num_buckets, embedding_dim)
        self.opponent = nn.Embedding(num_buckets, embedding_dim) if condition_on_opponent else None
        self.clock = nn.Linear(NUM_CONTEXT_FEATURES, embedding_dim) if use_clock else None
        parts = 1 + (self.opponent is not None) + (self.clock is not None)
        self.mlp = nn.Sequential(
            nn.Linear(parts * embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, inputs: ModelInputs) -> torch.Tensor:
        for name, buckets in (("bucket", inputs.bucket), ("opp_bucket", inputs.opp_bucket)):
            if len(buckets) and (int(buckets.min()) < 0 or int(buckets.max()) >= self.num_buckets):
                raise IndexError(f"{name} values must be in [0, {self.num_buckets}), got "
                                 f"{int(buckets.min())}..{int(buckets.max())}")
        parts = [self.rating(inputs.bucket)]
        if self.opponent is not None:
            parts.append(self.opponent(inputs.opp_bucket))
        if self.clock is not None:
            parts.append(self.clock(inputs.context))
        return self.mlp(torch.cat(parts, dim=-1))


class ConditionedResidualBlock(nn.Module):
    """conv-BN-ReLU -> condition -> conv-BN -> + skip -> ReLU, conditioned per channel."""

    def __init__(self, channels: int, cond_dim: int, conditioning: Conditioning = "film") -> None:
        super().__init__()
        if conditioning not in ("add", "film"):
            raise ValueError(f"conditioning must be 'add' or 'film', got {conditioning!r}")
        self.conditioning = conditioning
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.modulation = nn.Linear(cond_dim, channels * (2 if conditioning == "film" else 1))
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """``extra``: an optional ``(channels * k,)`` shift added to the modulation (style adapters, step 19)."""
        h = torch.relu(self.bn1(self.conv1(x)))
        mod = self.modulation(cond)
        if extra is not None:
            mod = mod + extra
        mod = mod[:, :, None, None]
        if self.conditioning == "film":
            gamma, beta = mod.chunk(2, dim=1)
            h = h * (1 + gamma) + beta
        else:
            h = h + mod
        h = self.bn2(self.conv2(h))
        return torch.relu(h + x)
