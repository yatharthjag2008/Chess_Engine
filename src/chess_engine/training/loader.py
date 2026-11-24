"""Turn a dataset split into training batches of tensors, decoded ahead on a background thread.

    loader = BatchLoader(dataset, cfg.ratings, batch_size=256, device=device, prefetch=4)
    for batch in loader.epoch(epoch=0, seed=42):            # TrainBatch
        out = model(batch.inputs)
        loss = policy_loss(out.policy, batch.targets.legal_mask, batch.targets.move)

Decoding a batch (compact boards -> planes, legal-move lists -> masks) takes
~2 ms for 256 positions; numpy releases the GIL for most of it, so a producer
thread keeps the next batches ready while the model trains on the current one.

In-memory datasets use :mod:`chess_engine.training.order` (shuffle or resample by
bucket, resumable at any batch). Streaming datasets (``in_memory=False``) read one
shard at a time in a seeded order; resuming skips the batches already trained on.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

from chess_engine.config import RatingsConfig
from chess_engine.data.dataset import Batch, PositionDataset
from chess_engine.model.features import ModelInputs, Targets, inputs_from_batch, targets_from_batch
from chess_engine.training.order import batch_slices, batches_per_epoch, epoch_order

T = TypeVar("T")
_DONE = object()


@dataclass(frozen=True)
class TrainBatch:
    """Model inputs, targets and optional per-position loss weights for one batch."""

    inputs: ModelInputs
    targets: Targets
    weights: torch.Tensor | None = None  # (B,) float32 loss weight per position

    def __len__(self) -> int:
        return len(self.inputs)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TrainBatch:
        def move(t: torch.Tensor) -> torch.Tensor:
            return t.to(device, non_blocking=non_blocking)

        inputs = ModelInputs(*(move(t) for t in (self.inputs.planes, self.inputs.bucket, self.inputs.opp_bucket,
                                                  self.inputs.context)))
        targets = Targets(*(move(t) for t in (self.targets.move, self.targets.legal_mask, self.targets.result)))
        return TrainBatch(inputs, targets, move(self.weights) if self.weights is not None else None)

    def pin_memory(self) -> TrainBatch:
        def pin(t: torch.Tensor) -> torch.Tensor:
            return t.pin_memory()

        inputs = ModelInputs(*(pin(t) for t in (self.inputs.planes, self.inputs.bucket, self.inputs.opp_bucket,
                                                 self.inputs.context)))
        targets = Targets(*(pin(t) for t in (self.targets.move, self.targets.legal_mask, self.targets.result)))
        return TrainBatch(inputs, targets, pin(self.weights) if self.weights is not None else None)


def to_train_batch(batch: Batch, ratings: RatingsConfig, bucket_weights: np.ndarray | None = None) -> TrainBatch:
    """CPU tensors for a dataset batch; ``bucket_weights`` (per bucket) gives each position a loss weight."""
    weights = None
    if bucket_weights is not None:
        table = torch.as_tensor(np.asarray(bucket_weights, dtype=np.float32))
        weights = table[torch.as_tensor(batch.features["bucket"].astype(np.int64))]
    return TrainBatch(inputs_from_batch(batch, ratings), targets_from_batch(batch), weights)


def prefetch(items: Iterable[T], size: int) -> Iterator[T]:
    """Iterate ``items`` on a background thread, keeping up to ``size`` results ready.

    Exceptions in the producer are re-raised in the consumer. If the consumer stops
    early (``break``, an exception, garbage collection), the producer is told to stop
    and the thread is joined, so no thread outlives the loop. ``size == 0`` iterates
    in the calling thread.
    """
    if size <= 0:
        yield from items
        return
    buffer: queue.Queue = queue.Queue(maxsize=size)
    stop = threading.Event()

    def put(item: object) -> bool:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in items:
                if not put(item):
                    return
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            put(exc)
            return
        put(_DONE)

    thread = threading.Thread(target=produce, name="batch-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


class BatchLoader:
    """Epochs of :class:`TrainBatch` from one dataset split.

    Args:
        dataset: The split to read.
        ratings: Bucket edges (the opponent's rating is bucketed with them).
        batch_size: Positions per batch; the last incomplete batch of an epoch is dropped.
        device: Batches are moved here (pinned + non-blocking on CUDA).
        prefetch: Batches decoded ahead on a background thread (0 = decode inline).
        loss_weights: Per-bucket loss weights (``bucket_balancing: weight``), or None.
        resample_weights: Per-bucket resampling rates (``bucket_balancing: sample``, in-memory only), or None.
    """

    def __init__(
        self,
        dataset: PositionDataset,
        ratings: RatingsConfig,
        batch_size: int,
        device: torch.device | str = "cpu",
        prefetch: int = 4,
        loss_weights: np.ndarray | None = None,
        resample_weights: np.ndarray | None = None,
    ) -> None:
        if resample_weights is not None and not dataset.in_memory:
            raise ValueError("resampling by bucket needs an in-memory dataset (training.in_memory: true)")
        self.dataset = dataset
        self.ratings = ratings
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.prefetch = prefetch
        self.loss_weights = loss_weights
        self.resample_weights = resample_weights
        self._buckets = (dataset.samples["bucket"].astype(np.int64)
                         if resample_weights is not None else None)

    def epoch_size(self, epoch: int = 0, seed: int = 0) -> int:
        """Positions in one epoch (differs slightly from the split size when resampling)."""
        if self.resample_weights is None:
            return len(self.dataset)
        return len(self._order(epoch, seed))

    def batches_per_epoch(self, epoch: int = 0, seed: int = 0) -> int:
        return batches_per_epoch(self.epoch_size(epoch, seed), self.batch_size)

    def _order(self, epoch: int, seed: int) -> np.ndarray:
        return epoch_order(len(self.dataset), seed, epoch, self._buckets, self.resample_weights)

    def _raw_batches(self, epoch: int, seed: int, start_batch: int) -> Iterator[Batch]:
        if self.dataset.in_memory:
            for indices in batch_slices(self._order(epoch, seed), self.batch_size, start_batch):
                yield self.dataset.batch(indices)
            return
        stream_seed = int(np.random.default_rng([seed % 2**32, epoch]).integers(2**31))
        batches = self.dataset.iter_batches(self.batch_size, shuffle=True, seed=stream_seed, drop_last=True)
        for i, batch in enumerate(batches):
            if i >= start_batch:
                yield batch

    def epoch(self, epoch: int, seed: int, start_batch: int = 0) -> Iterator[TrainBatch]:
        """The batches of ``epoch`` (seeded by ``seed``), starting at batch ``start_batch``."""
        pin = self.device.type == "cuda"

        def produce() -> Iterator[TrainBatch]:
            for raw in self._raw_batches(epoch, seed, start_batch):
                batch = to_train_batch(raw, self.ratings, self.loss_weights)
                yield batch.pin_memory() if pin else batch

        for batch in prefetch(produce(), self.prefetch):
            yield batch.to(self.device, non_blocking=pin) if self.device.type != "cpu" else batch
