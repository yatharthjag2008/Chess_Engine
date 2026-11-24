"""Read a built dataset (step 4 output) as batches of model inputs and targets.

    dataset = PositionDataset("data/processed/datasets/lichess_2026-10_sample", "train")
    for batch in dataset.iter_batches(batch_size=256, shuffle=True, seed=0):
        batch.planes      # (B, 20, 8, 8) float32 board planes (exactly encode_board's output)
        batch.legal_mask  # (B, 1858) bool, True for legal moves
        batch.move        # (B,) int64 policy index of the human move (the target)
        batch.features["bucket"], batch.features["clock"], ...   # (B,) per-position extras

Batches are plain numpy so this module does not depend on PyTorch; the training
loop (step 6) wraps them in tensors.

Two loading modes:

* ``in_memory=True`` (default): every shard of the split is loaded once, and
  shuffling is a single global permutation. The 50k-game sample's training split
  is roughly 0.7 GB in memory.
* ``in_memory=False``: shards are loaded one at a time (in a shuffled order, each
  shard shuffled again), so memory use is one shard however big the dataset is.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_engine.data.dataset_builder import read_manifest
from chess_engine.data.samples import SAMPLE_COLUMNS, Samples
from chess_engine.data.shards import FORMAT_VERSION, read_shard
from chess_engine.data.splits import SPLITS
from chess_engine.encoding.board import NUM_PLANES, PLANE_NAMES
from chess_engine.encoding.compact import planes_from_compact
from chess_engine.encoding.moves import NUM_MOVES

#: Columns used to rebuild the planes; every other column (except ``move``) becomes a feature.
_BOARD_COLUMNS = ("pieces", "flags", "ep", "halfmove")
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    name for name in SAMPLE_COLUMNS if name not in (*_BOARD_COLUMNS, "move", "legal_count")
)


class DatasetMismatchError(Exception):
    """The dataset on disk was built with a different format or encoding than this code uses."""


@dataclass(frozen=True)
class Batch:
    """One batch of model inputs and targets (numpy arrays, first axis = batch)."""

    planes: np.ndarray
    legal_mask: np.ndarray
    move: np.ndarray
    features: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.move)


def make_batch(samples: Samples, indices: np.ndarray | None = None, dtype: np.dtype | type = np.float32) -> Batch:
    """Decode ``samples`` (or the rows ``indices`` of them) into a :class:`Batch`."""
    if indices is not None:
        samples = samples.take(indices)
    planes = planes_from_compact(*(samples[name] for name in _BOARD_COLUMNS), dtype=dtype)
    features = {name: samples[name] for name in FEATURE_COLUMNS}
    return Batch(planes, samples.legal_mask(), samples["move"].astype(np.int64), features)


def check_compatible(manifest: dict[str, Any]) -> None:
    """Raise :class:`DatasetMismatchError` if the dataset does not match this code's encoding."""
    problems = []
    if manifest.get("format_version") != FORMAT_VERSION:
        problems.append(f"format version {manifest.get('format_version')} (expected {FORMAT_VERSION})")
    encoding = manifest.get("encoding", {})
    if encoding.get("num_planes") != NUM_PLANES or encoding.get("plane_names") != list(PLANE_NAMES):
        problems.append("different board planes")
    if encoding.get("num_moves") != NUM_MOVES:
        problems.append(f"{encoding.get('num_moves')} moves (expected {NUM_MOVES})")
    if problems:
        raise DatasetMismatchError("dataset was built with " + ", ".join(problems) + "; rebuild it")


class PositionDataset:
    """The samples of one split of a built dataset.

    Args:
        dataset_dir: Directory written by :func:`chess_engine.data.dataset_builder.build_dataset`.
        split: ``"train"``, ``"val"`` or ``"test"``.
        in_memory: Load all shards now (fast global shuffling) or one at a time while iterating.
        max_shards: Use only the first ``max_shards`` shards (quick experiments).
    """

    def __init__(
        self, dataset_dir: str | Path, split: str, in_memory: bool = True, max_shards: int | None = None
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.directory = Path(dataset_dir)
        self.split = split
        self.manifest = read_manifest(self.directory)
        check_compatible(self.manifest)
        shards = self.manifest["splits"][split]["shards"]
        shards = shards[:max_shards] if max_shards is not None else shards
        self.shard_paths = [self.directory / split / s["file"] for s in shards]
        self._length = sum(s["positions"] for s in shards)
        self.in_memory = in_memory
        self._samples = Samples.concat([read_shard(p) for p in self.shard_paths]) if in_memory else None

    def __len__(self) -> int:
        return self._length

    @property
    def bucket_labels(self) -> list[str]:
        return list(self.manifest["ratings"]["bucket_labels"])

    @property
    def samples(self) -> Samples:
        """All samples of the split (``in_memory`` mode only)."""
        if self._samples is None:
            raise RuntimeError("samples is only available with in_memory=True")
        return self._samples

    def batch(self, indices: np.ndarray, dtype: np.dtype | type = np.float32) -> Batch:
        """The samples at ``indices`` as a batch (``in_memory`` mode only)."""
        return make_batch(self.samples, np.asarray(indices, dtype=np.int64), dtype)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
        dtype: np.dtype | type = np.float32,
    ) -> Iterator[Batch]:
        """Yield batches covering the split once (one epoch).

        With ``shuffle``, the order is a seeded permutation (global in memory;
        shard order + within-shard order when streaming). ``drop_last`` skips a
        final batch smaller than ``batch_size``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        if self._samples is not None:
            yield from _batches(self._samples, batch_size, shuffle, rng, drop_last, dtype)
            return
        order = rng.permutation(len(self.shard_paths)) if shuffle else range(len(self.shard_paths))
        carry: Samples | None = None
        for shard_index in order:
            shard = read_shard(self.shard_paths[shard_index])
            if shuffle:
                shard = shard.take(rng.permutation(len(shard)))
            if carry is not None and len(carry):
                shard = Samples.concat([carry, shard])
            full = len(shard) // batch_size * batch_size
            yield from _batches(shard.take(np.arange(full)), batch_size, False, rng, False, dtype)
            carry = shard.take(np.arange(full, len(shard)))
        if carry is not None and len(carry) and not drop_last:
            yield make_batch(carry, dtype=dtype)


def _batches(
    samples: Samples, batch_size: int, shuffle: bool, rng: np.random.Generator, drop_last: bool, dtype: Any
) -> Iterator[Batch]:
    order = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        if drop_last and len(indices) < batch_size:
            return
        yield make_batch(samples, indices, dtype)
