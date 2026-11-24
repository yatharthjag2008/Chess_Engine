"""Validation pass during training: loss, top-1/3/5 move matching and value accuracy per rating bucket.

    result = validate(model, val_dataset, cfg.ratings, batch_size=1024, max_positions=50_000)
    result["loss"], result["overall"]["top1"], result["buckets"]["1400-1599"]["top1"]

Always the same positions in the same order (the first ``max_positions`` of the
split, which is already shuffled on disk), so numbers from different steps and
runs are directly comparable. The fuller evaluation with baselines and
calibration is step 7 (:mod:`chess_engine.eval`).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import torch

from chess_engine.config import RatingsConfig
from chess_engine.data.dataset import Batch, PositionDataset
from chess_engine.model.network import RatingConditionedNet
from chess_engine.training.device import Precision
from chess_engine.training.loader import prefetch, to_train_batch
from chess_engine.training.losses import compute_loss
from chess_engine.training.metrics import MoveMetrics, RunningMeans


def _first_batches(dataset: PositionDataset, batch_size: int, max_positions: int | None) -> Iterator[Batch]:
    remaining = len(dataset) if max_positions is None else min(max_positions, len(dataset))
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        if remaining <= 0:
            return
        if len(batch) > remaining:
            batch = Batch(batch.planes[:remaining], batch.legal_mask[:remaining], batch.move[:remaining],
                          {name: values[:remaining] for name, values in batch.features.items()})
        remaining -= len(batch)
        yield batch


@torch.no_grad()
def validate(
    model: RatingConditionedNet,
    dataset: PositionDataset,
    ratings: RatingsConfig,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
    precision: Precision | None = None,
    max_positions: int | None = None,
    value_weight: float = 0.25,
    prefetch_batches: int = 2,
) -> dict[str, Any]:
    """Evaluate ``model`` on (the first ``max_positions`` of) ``dataset``.

    Returns:
        ``{"loss", "policy_loss", "value_loss", "overall", "buckets", "bucket_mean_top1",
        "positions", "seconds"}``; ``loss`` is unweighted (every position counts once),
        so it does not depend on the balancing setting.
    """
    start = time.perf_counter()
    device = torch.device(device)
    precision = precision or Precision(device.type)
    was_training = model.training
    model.eval()
    metrics = MoveMetrics(ratings.num_buckets, ratings.bucket_labels())
    losses = RunningMeans()
    try:
        raw = _first_batches(dataset, batch_size, max_positions)
        for batch in prefetch((to_train_batch(b, ratings) for b in raw), prefetch_batches):
            batch = batch.to(device)
            with precision.autocast():
                out = model(batch.inputs)
            loss = compute_loss(out, batch, value_weight=value_weight)
            n = len(batch)
            losses.add("loss", loss.total, n)
            losses.add("policy_loss", loss.policy, n)
            if loss.value is not None:
                losses.add("value_loss", loss.value, n)
            metrics.update(batch.inputs.bucket, loss.log_probs, batch.targets.move,
                           out.value, batch.targets.result)
    finally:
        model.train(was_training)
    summary = metrics.summary()
    return {**losses.means(), **summary, "positions": metrics.positions,
            "seconds": round(time.perf_counter() - start, 2)}
