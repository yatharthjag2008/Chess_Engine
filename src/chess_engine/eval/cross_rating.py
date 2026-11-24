"""Cross-rating evaluation: does the model at rating R predict players rated R best?

The same test positions are scored once per rating bucket, each time telling the
model "the mover is in bucket m" whatever the real rating. Entry ``[m, p]`` of the
result is the model-at-``m``'s top-1 accuracy on players whose real bucket is ``p``.

If rating conditioning works, every column peaks at (or next to) the diagonal:
1100-rated players are predicted best by the model playing at 1100, not by the
model playing at 1900. The best single row is the **fixed-rating baseline**,
"one model rating for every player", which shows how much knowing the rating
adds (it is the same network, so the comparison isolates the conditioning).

    tables = [predict_split(model, test, ratings, conditioned_bucket=m, max_positions=40_000) for m in ...]
    result = cross_rating_from_tables(tables, ratings.bucket_labels())
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from chess_engine.config import RatingsConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.move_matching import group_means
from chess_engine.eval.predictions import PredictionTable, predict_split
from chess_engine.model.network import RatingConditionedNet


def cross_rating_from_tables(tables: Sequence[PredictionTable], labels: Sequence[str]) -> dict[str, Any]:
    """Combine one prediction table per conditioned bucket (same positions, in bucket order) into the matrix.

    Returns:
        ``top1`` and ``log_likelihood``: ``[model bucket][player bucket]`` lists (None where a
        player bucket has no positions); ``positions`` per player bucket; ``best_model_bucket``
        per player bucket; ``diagonal_top1`` (each player bucket at its own rating);
        ``fixed_rating``: the best single model bucket for everyone and its accuracy;
        ``matched_top1``: accuracy when every player is predicted at their own rating.
    """
    k = len(labels)
    if len(tables) != k:
        raise ValueError(f"expected {k} tables (one per bucket), got {len(tables)}")
    player = tables[0]["bucket"].astype(np.int64)
    for table in tables[1:]:
        if len(table) != len(player) or not np.array_equal(table["bucket"], tables[0]["bucket"]):
            raise ValueError("tables must cover the same positions")
    top1 = np.full((k, k), np.nan)
    loglik = np.full((k, k), np.nan)
    counts = np.bincount(player, minlength=k)[:k]
    overall = np.zeros(k)
    for m, table in enumerate(tables):
        hits = table.top1_hit.astype(np.float64)
        top1[m], _ = group_means(hits, player, k)
        loglik[m], _ = group_means(np.log(np.maximum(table["target_prob"].astype(np.float64), 1e-12)), player, k)
        overall[m] = hits.mean() if len(hits) else np.nan
    present = counts > 0
    diagonal = np.diag(top1)
    matched = float((diagonal[present] * counts[present]).sum() / counts[present].sum()) if present.any() else np.nan
    best_fixed = int(np.nanargmax(overall)) if np.isfinite(overall).any() else 0

    def listed(matrix: np.ndarray) -> list[list[float | None]]:
        return [[None if np.isnan(v) else float(v) for v in row] for row in matrix]

    return {
        "labels": list(labels),
        "top1": listed(top1),
        "log_likelihood": listed(loglik),
        "positions": counts.tolist(),
        "best_model_bucket": [labels[int(np.nanargmax(top1[:, p]))] if present[p] else None for p in range(k)],
        "diagonal_top1": [None if not present[p] else float(diagonal[p]) for p in range(k)],
        "matched_top1": matched,
        "fixed_rating": {"bucket": labels[best_fixed], "top1": float(overall[best_fixed]),
                         "top1_per_model_bucket": [float(v) for v in overall]},
    }


def run_cross_rating(
    model: RatingConditionedNet,
    dataset: PositionDataset,
    ratings: RatingsConfig,
    max_positions: int | None = 40_000,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Score the first ``max_positions`` positions at every bucket and build the matrix.

    ``progress`` is called with ``(buckets done, number of buckets)``.
    """
    tables = []
    for bucket in range(ratings.num_buckets):
        tables.append(predict_split(model, dataset, ratings, batch_size, device, max_positions,
                                    conditioned_bucket=bucket))
        if progress is not None:
            progress(bucket + 1, ratings.num_buckets)
    return cross_rating_from_tables(tables, ratings.bucket_labels())
