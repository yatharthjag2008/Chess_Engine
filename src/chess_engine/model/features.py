"""Turn dataset batches (numpy) into the tensors the network takes, and the training targets.

Model inputs (:class:`ModelInputs`):

* ``planes``     ``(B, 20, 8, 8)`` float32, the board (see ``chess_engine.encoding.board``)
* ``bucket``     ``(B,)`` int64, the mover's rating bucket (what the model imitates)
* ``opp_bucket`` ``(B,)`` int64, the opponent's rating bucket
* ``context``    ``(B, 5)`` float32 clock features, all roughly in ``[0, 1]``:

  ====================  ===========================================================
  clock_fraction        mover's time left / starting time (capped at 2)
  opp_clock_fraction    the same for the opponent
  clock_log             log(1 + mover's seconds left) / log(1 + 600): absolute time
                        pressure, so 10 s left means the same in 3+0 and 5+3
  opp_clock_log         the same for the opponent
  increment             increment seconds / 10 (capped at 3)
  ====================  ===========================================================

Targets (:class:`Targets`): the played move's policy index, the legal-move mask
and the game result for the mover as a class (0 = loss, 1 = draw, 2 = win).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch

from chess_engine.config import RatingsConfig
from chess_engine.data.dataset import Batch

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "clock_fraction", "opp_clock_fraction", "clock_log", "opp_clock_log", "increment",
)
NUM_CONTEXT_FEATURES = len(CONTEXT_FEATURE_NAMES)
_LOG_CLOCK_SCALE = float(np.log1p(600.0))


def context_features(
    clock: np.ndarray, opp_clock: np.ndarray, base_seconds: np.ndarray, increment: np.ndarray
) -> np.ndarray:
    """``(B, NUM_CONTEXT_FEATURES)`` float32 clock features (see the module docstring)."""
    clock = np.maximum(np.asarray(clock, dtype=np.float32), 0)
    opp_clock = np.maximum(np.asarray(opp_clock, dtype=np.float32), 0)
    base = np.maximum(np.asarray(base_seconds, dtype=np.float32), 1)
    increment = np.asarray(increment, dtype=np.float32)
    return np.stack(
        [
            np.minimum(clock / base, 2),
            np.minimum(opp_clock / base, 2),
            np.log1p(clock) / _LOG_CLOCK_SCALE,
            np.log1p(opp_clock) / _LOG_CLOCK_SCALE,
            np.clip(increment / 10, 0, 3),
        ],
        axis=-1,
    ).astype(np.float32)


def buckets_from_ratings(ratings: np.ndarray | Sequence[int], cfg: RatingsConfig) -> np.ndarray:
    """Bucket index of each rating (int64); ratings below the lowest edge go to bucket 0."""
    edges = np.asarray(cfg.bucket_edges)
    buckets = np.searchsorted(edges, np.asarray(ratings), side="right") - 1
    return np.maximum(buckets, 0).astype(np.int64)


@dataclass(frozen=True)
class ModelInputs:
    """Everything the network reads for a batch of positions (torch tensors)."""

    planes: torch.Tensor
    bucket: torch.Tensor
    opp_bucket: torch.Tensor
    context: torch.Tensor

    def __len__(self) -> int:
        return len(self.bucket)

    def to(self, device: torch.device | str) -> ModelInputs:
        return ModelInputs(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class Targets:
    """What the network is trained to predict, plus the legal-move mask."""

    move: torch.Tensor        # (B,) int64 policy index of the human move
    legal_mask: torch.Tensor  # (B, NUM_MOVES) bool
    result: torch.Tensor      # (B,) int64: 0 loss, 1 draw, 2 win (for the mover)

    def to(self, device: torch.device | str) -> Targets:
        return Targets(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def make_inputs(
    planes: np.ndarray,
    bucket: np.ndarray,
    opp_bucket: np.ndarray,
    context: np.ndarray,
    device: torch.device | str | None = None,
) -> ModelInputs:
    """Wrap numpy arrays as :class:`ModelInputs` (zero-copy on CPU where possible)."""
    inputs = ModelInputs(
        planes=torch.as_tensor(np.ascontiguousarray(planes, dtype=np.float32)),
        bucket=torch.as_tensor(np.asarray(bucket, dtype=np.int64)),
        opp_bucket=torch.as_tensor(np.asarray(opp_bucket, dtype=np.int64)),
        context=torch.as_tensor(np.ascontiguousarray(context, dtype=np.float32)),
    )
    return inputs.to(device) if device is not None else inputs


def inputs_from_batch(batch: Batch, ratings: RatingsConfig, device: torch.device | str | None = None) -> ModelInputs:
    """Model inputs for a dataset batch."""
    f = batch.features
    return make_inputs(
        batch.planes,
        f["bucket"],
        buckets_from_ratings(f["opp_rating"], ratings),
        context_features(f["clock"], f["opp_clock"], f["base_seconds"], f["increment"]),
        device,
    )


def targets_from_batch(batch: Batch, device: torch.device | str | None = None) -> Targets:
    """Training targets for a dataset batch."""
    targets = Targets(
        move=torch.as_tensor(np.asarray(batch.move, dtype=np.int64)),
        legal_mask=torch.as_tensor(np.ascontiguousarray(batch.legal_mask, dtype=bool)),
        result=torch.as_tensor(np.asarray(batch.features["result"], dtype=np.int64) + 1),
    )
    return targets.to(device) if device is not None else targets
