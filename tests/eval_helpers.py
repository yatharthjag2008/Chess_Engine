"""Hand-made prediction tables for evaluation tests."""

from __future__ import annotations

import numpy as np

from chess_engine.eval.predictions import TOP_STORED, PredictionTable

LABELS = ["low", "mid", "high"]


def make_table(rank: list[int], prob: list[float], bucket: list[int], game: list[int] | None = None,
               legal: list[int] | None = None, **extra: list) -> PredictionTable:
    """A :class:`PredictionTable` with the given per-position values (other columns filled plausibly)."""
    n = len(rank)
    columns = {
        "target_rank": np.array(rank, dtype=np.int16),
        "target_prob": np.array(prob, dtype=np.float32),
        "bucket": np.array(bucket, dtype=np.int8),
        "game": np.array(game if game is not None else range(n), dtype=np.int32),
        "legal_count": np.array(legal if legal is not None else [20] * n, dtype=np.int16),
        "target": np.zeros(n, dtype=np.int16),
        "ply": np.array(extra.pop("ply", [10] * n), dtype=np.int16),
        "clock": np.array(extra.pop("clock", [120.0] * n), dtype=np.float32),
        "base_seconds": np.full(n, 180, dtype=np.int16),
        "increment": np.zeros(n, dtype=np.int16),
        "result": np.array(extra.pop("result", [0] * n), dtype=np.int8),
        "top_moves": np.zeros((n, TOP_STORED), dtype=np.int16),
        "top_probs": np.zeros((n, TOP_STORED), dtype=np.float32),
        "key": np.arange(n, dtype=np.uint64),
    }
    columns.update({name: np.asarray(values) for name, values in extra.items()})
    return PredictionTable(columns, np.zeros((10, 3)), {"bucket_labels": LABELS})
