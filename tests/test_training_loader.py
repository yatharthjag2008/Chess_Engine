"""Tests for the training batch loader and background prefetching."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_engine.config import Config
from chess_engine.data.dataset import PositionDataset
from chess_engine.training.balancing import bucket_weights
from chess_engine.training.loader import BatchLoader, prefetch, to_train_batch


def test_prefetch_yields_everything_in_order() -> None:
    assert list(prefetch(range(100), 3)) == list(range(100))
    assert list(prefetch(range(5), 0)) == list(range(5))


def test_prefetch_reraises_producer_errors() -> None:
    def broken():
        yield 1
        raise RuntimeError("bad shard")

    with pytest.raises(RuntimeError, match="bad shard"):
        list(prefetch(broken(), 2))


def test_prefetch_stops_its_thread_when_the_consumer_stops_early() -> None:
    produced = []

    def endless():
        i = 0
        while True:
            produced.append(i)
            yield i
            i += 1

    before = threading.active_count()
    gen = prefetch(endless(), 2)
    assert next(gen) == 0
    gen.close()
    time.sleep(0.05)
    assert threading.active_count() == before
    assert len(produced) <= 5  # it stopped instead of running forever


def test_prefetch_runs_ahead_of_the_consumer() -> None:
    produced = []

    def items():
        for i in range(4):
            produced.append(i)
            yield i

    gen = prefetch(items(), 4)
    assert next(gen) == 0
    deadline = time.time() + 2
    while len(produced) < 4 and time.time() < deadline:
        time.sleep(0.01)
    assert produced == [0, 1, 2, 3]
    gen.close()


def test_train_batch_contents(tiny_dataset: Path, default_config: Config) -> None:
    dataset = PositionDataset(tiny_dataset, "train")
    raw = dataset.batch(np.arange(8))
    weights = np.arange(1, 9, dtype=np.float64)
    batch = to_train_batch(raw, default_config.ratings, weights)
    assert len(batch) == 8
    assert batch.inputs.planes.shape == (8, 20, 8, 8)
    assert batch.targets.legal_mask.dtype == torch.bool
    expected = weights[raw.features["bucket"].astype(int)]
    assert batch.weights.tolist() == pytest.approx(expected.tolist())
    assert to_train_batch(raw, default_config.ratings).weights is None


@pytest.mark.parametrize("prefetch_size", [0, 3])
def test_epoch_covers_the_split_once_in_a_seeded_order(tiny_dataset: Path, default_config: Config,
                                                        prefetch_size: int) -> None:
    dataset = PositionDataset(tiny_dataset, "train")
    loader = BatchLoader(dataset, default_config.ratings, batch_size=64, prefetch=prefetch_size)
    batches = list(loader.epoch(epoch=0, seed=1))
    assert len(batches) == loader.batches_per_epoch() == len(dataset) // 64
    moves = torch.cat([b.targets.move for b in batches])
    again = torch.cat([b.targets.move for b in loader.epoch(epoch=0, seed=1)])
    assert torch.equal(moves, again)
    other = torch.cat([b.targets.move for b in loader.epoch(epoch=1, seed=1)])
    assert not torch.equal(moves, other)


def test_resuming_at_a_batch_gives_the_remaining_batches(tiny_dataset: Path, default_config: Config) -> None:
    dataset = PositionDataset(tiny_dataset, "train")
    loader = BatchLoader(dataset, default_config.ratings, batch_size=50, prefetch=2)
    full = [b.targets.move for b in loader.epoch(epoch=2, seed=5)]
    rest = [b.targets.move for b in loader.epoch(epoch=2, seed=5, start_batch=3)]
    assert len(rest) == len(full) - 3
    assert all(torch.equal(a, b) for a, b in zip(full[3:], rest))


def test_streaming_dataset_epochs_and_resume(tiny_dataset: Path, default_config: Config) -> None:
    dataset = PositionDataset(tiny_dataset, "train", in_memory=False)
    loader = BatchLoader(dataset, default_config.ratings, batch_size=50, prefetch=2)
    full = [b.targets.move for b in loader.epoch(epoch=0, seed=5)]
    assert len(full) == len(dataset) // 50
    rest = [b.targets.move for b in loader.epoch(epoch=0, seed=5, start_batch=2)]
    assert all(torch.equal(a, b) for a, b in zip(full[2:], rest))
    with pytest.raises(ValueError, match="in-memory"):
        BatchLoader(dataset, default_config.ratings, 50, resample_weights=np.ones(8))


def test_resampling_changes_the_bucket_mix(tiny_dataset: Path, default_config: Config) -> None:
    dataset = PositionDataset(tiny_dataset, "train")
    counts = np.bincount(dataset.samples["bucket"].astype(int), minlength=8)
    weights = bucket_weights(counts, 1.0)
    loader = BatchLoader(dataset, default_config.ratings, batch_size=32, prefetch=0, resample_weights=weights)
    seen = torch.cat([b.inputs.bucket for b in loader.epoch(epoch=0, seed=0)]).numpy()
    shares = np.bincount(seen, minlength=8)[counts > 0] / len(seen)
    assert shares.max() - shares.min() < 0.05  # roughly equal buckets after resampling
    assert abs(loader.epoch_size() - len(dataset)) <= 8
