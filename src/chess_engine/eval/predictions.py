"""Run a model once over a dataset split and keep, per position, what every metric needs.

    table = predict_split(model, PositionDataset(dataset_dir, "test"), cfg.ratings)
    table["target_rank"]   # 1 if the model's favourite move is the one the human played
    table["target_prob"]   # probability the model gave the human move
    table.save("reports/eval/run/predictions.npz")

Columns (one row per position, in the split's stored order):

=============  ==========  ====================================================
column         dtype       content
=============  ==========  ====================================================
game           int32       game row in the dataset's ``games.parquet``
key            uint64      position key (:func:`chess_engine.eval.positions.position_keys`)
bucket         int8        the mover's true rating bucket
rating         int16       the mover's rating; ``opp_rating`` the opponent's
ply            int16       0 = White's first move
clock          float32     the mover's seconds left; ``base_seconds``, ``increment``: time control
legal_count    int16       number of legal moves
result         int8        game result for the mover (+1 / 0 / -1)
target         int16       policy index of the move the human played
target_prob    float32     model probability of that move
target_rank    int16       1 + number of legal moves the model rated strictly higher
entropy        float32     entropy of the model's move distribution (nats)
top_moves      int16 x5    the model's five likeliest moves (policy indices) ...
top_probs      float32 x5  ... and their probabilities
value          float32 x3  value head: probabilities of loss / draw / win for the mover (if any)
=============  ==========  ====================================================

``conditioned_bucket`` lets the model be told a different rating than the
mover's real one (all positions at one bucket), which is how the cross-rating
evaluation asks "how well does the 1500 model predict 1100 players?".

The pass also accumulates a reliability histogram over *every* legal move
(predicted probability vs. whether it was the move played), which would be too
big to keep per move; see :mod:`chess_engine.eval.calibration`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_engine.config import RatingsConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.positions import sample_keys
from chess_engine.model.features import inputs_from_batch
from chess_engine.model.masking import legal_probs
from chess_engine.model.network import RatingConditionedNet
from chess_engine.training.loader import prefetch
from chess_engine.training.validation import first_batches

TOP_STORED = 5
PREDICTIONS_FORMAT_VERSION = 1
_COPIED = {  # dataset feature -> (column, dtype)
    "game": np.int32, "bucket": np.int8, "rating": np.int16, "opp_rating": np.int16, "ply": np.int16,
    "clock": np.float32, "base_seconds": np.int16, "increment": np.int16, "result": np.int8,
}


@dataclass
class PredictionTable:
    """Per-position predictions (``columns``) plus the all-moves reliability histogram and metadata."""

    columns: dict[str, np.ndarray]
    legal_calibration: np.ndarray  # (bins, 3): count, sum of predicted probability, number played
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.columns["target"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    @property
    def top1_hit(self) -> np.ndarray:
        """bool per position: the model's favourite move is the human move."""
        return self.columns["target_rank"] == 1

    def hits(self, k: int) -> np.ndarray:
        """bool per position: the human move is among the model's ``k`` likeliest."""
        return self.columns["target_rank"] <= k

    def take(self, rows: np.ndarray) -> PredictionTable:
        """A table of some rows (the calibration histogram is kept as is, so it covers all rows)."""
        return PredictionTable({k: v[rows] for k, v in self.columns.items()}, self.legal_calibration, dict(self.meta))

    # ----------------------------------------------------------------- files

    def save(self, path: str | Path) -> Path:
        """Write to an ``.npz`` file (columns + histogram + JSON metadata), atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format_version": PREDICTIONS_FORMAT_VERSION, **self.meta}
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, legal_calibration=self.legal_calibration,
                            meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
                            **{f"col_{k}": v for k, v in self.columns.items()})
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> PredictionTable:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            if meta.get("format_version") != PREDICTIONS_FORMAT_VERSION:
                raise ValueError(f"{path}: predictions format {meta.get('format_version')}, "
                                 f"expected {PREDICTIONS_FORMAT_VERSION}")
            columns = {name[4:]: data[name] for name in data.files if name.startswith("col_")}
            return cls(columns, data["legal_calibration"], meta)


def _accumulate_calibration(hist: np.ndarray, probs: np.ndarray, mask: np.ndarray, target: np.ndarray) -> None:
    bins = len(hist)
    rows, cols = np.nonzero(mask)
    p = probs[rows, cols]
    played = (cols == target[rows]).astype(np.float64)
    which = np.minimum((p * bins).astype(np.int64), bins - 1)
    hist[:, 0] += np.bincount(which, minlength=bins)
    hist[:, 1] += np.bincount(which, weights=p, minlength=bins)
    hist[:, 2] += np.bincount(which, weights=played, minlength=bins)


@torch.no_grad()
def predict_split(
    model: RatingConditionedNet,
    dataset: PositionDataset,
    ratings: RatingsConfig,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
    max_positions: int | None = None,
    conditioned_bucket: int | None = None,
    calibration_bins: int = 10,
    progress: Callable[[int, int], None] | None = None,
) -> PredictionTable:
    """Score the first ``max_positions`` positions of ``dataset`` (all if None); see the module docstring.

    Args:
        conditioned_bucket: Tell the model every mover has this rating bucket (cross-rating evaluation).
        progress: Called with ``(positions done, total)`` after every batch.
    """
    if not dataset.in_memory:
        raise ValueError("evaluation needs an in-memory dataset")
    if conditioned_bucket is not None and not 0 <= conditioned_bucket < ratings.num_buckets:
        raise ValueError(f"conditioned_bucket must be in [0, {ratings.num_buckets})")
    start = time.perf_counter()
    device = torch.device(device)
    total = len(dataset) if max_positions is None else min(max_positions, len(dataset))
    was_training = model.training
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    hist = np.zeros((calibration_bins, 3), dtype=np.float64)
    done = 0

    def add(name: str, values: np.ndarray) -> None:
        parts.setdefault(name, []).append(values)

    try:
        for batch in prefetch(first_batches(dataset, batch_size, max_positions), 2):
            inputs = inputs_from_batch(batch, ratings)
            if conditioned_bucket is not None:
                inputs = type(inputs)(inputs.planes, torch.full_like(inputs.bucket, conditioned_bucket),
                                      inputs.opp_bucket, inputs.context)
            out = model(inputs.to(device))
            mask = torch.as_tensor(batch.legal_mask, device=device)
            probs = legal_probs(out.policy, mask, strict=False)
            target = torch.as_tensor(batch.move, device=device)
            target_prob = probs.gather(1, target[:, None]).squeeze(1)
            rank = 1 + (probs > target_prob[:, None]).sum(1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(1)
            top = probs.topk(TOP_STORED, dim=1)

            probs_np = probs.cpu().numpy()
            _accumulate_calibration(hist, probs_np, batch.legal_mask, batch.move)
            for name, dtype in _COPIED.items():
                add(name, batch.features[name].astype(dtype))
            add("legal_count", batch.legal_mask.sum(1).astype(np.int16))
            add("target", batch.move.astype(np.int16))
            add("target_prob", target_prob.float().cpu().numpy())
            add("target_rank", rank.cpu().numpy().astype(np.int16))
            add("entropy", entropy.float().cpu().numpy())
            add("top_moves", top.indices.cpu().numpy().astype(np.int16))
            add("top_probs", top.values.float().cpu().numpy())
            if out.value is not None:
                add("value", torch.softmax(out.value.float(), dim=-1).cpu().numpy())
            done += len(batch)
            if progress is not None:
                progress(done, total)
    finally:
        model.train(was_training)

    columns = {name: np.concatenate(values) for name, values in parts.items()}
    columns["key"] = sample_keys(dataset.samples.take(np.arange(done)))
    meta = {
        "split": dataset.split, "dataset": str(dataset.directory), "positions": done,
        "conditioned_bucket": conditioned_bucket, "bucket_labels": ratings.bucket_labels(),
        "seconds": round(time.perf_counter() - start, 1),
    }
    return PredictionTable(columns, hist, meta)
