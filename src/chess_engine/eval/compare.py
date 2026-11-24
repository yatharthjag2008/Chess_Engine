"""Compare models on exactly the same positions, with paired confidence intervals.

    tables = {"best": predict_split(best, test, ratings), "final": predict_split(final, test, ratings)}
    result = compare_tables(tables, ratings.bucket_labels())
    print(format_comparison(result))

The first model is the reference; every other model gets its top-1 difference to it,
overall and per rating bucket. Because both models see the same positions, the
interval comes from bootstrapping the per-position *difference* (resampling whole
games), which is much tighter than comparing two separate intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chess_engine.eval.bootstrap import cluster_bootstrap
from chess_engine.eval.move_matching import move_matching
from chess_engine.eval.predictions import PredictionTable


def compare_tables(
    tables: Mapping[str, PredictionTable],
    labels: Sequence[str],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-model move matching plus each model's top-1 difference to the first one."""
    names = list(tables)
    if len(names) < 2:
        raise ValueError("need at least two models to compare")
    reference = tables[names[0]]
    for name in names[1:]:
        other = tables[name]
        if len(other) != len(reference) or not np.array_equal(other["key"], reference["key"]):
            raise ValueError(f"{name!r} was scored on different positions than {names[0]!r}")
    models = {name: move_matching(table, labels) for name, table in tables.items()}
    differences = {}
    for name in names[1:]:
        diff = tables[name].top1_hit.astype(np.float64) - reference.top1_hit
        ci = cluster_bootstrap(diff, reference["game"], reference["bucket"], len(labels), samples, confidence, seed)
        per_bucket = {}
        for b, label in enumerate(labels):
            rows = reference["bucket"] == b
            if rows.any():
                per_bucket[label] = {"difference": float(diff[rows].mean()), "ci": list(ci.groups[b])}
        differences[name] = {"overall": {"difference": float(diff.mean()), "ci": list(ci.overall)},
                             "buckets": per_bucket}
    return {"reference": names[0], "labels": list(labels), "models": models, "differences": differences,
            "positions": len(reference), "confidence": confidence}


def format_comparison(result: dict[str, Any]) -> str:
    """Top-1 per bucket for every model, and the difference to the reference with its interval."""
    names = list(result["models"])
    reference = result["reference"]
    width = max(10, *(len(n) for n in names))
    header = f"{'bucket':<10} " + " ".join(f"{n:>{width}}" for n in names)
    for name in names[1:]:
        header += f"  {name + ' - ' + reference:>{width + 22}}"
    lines = [header]

    def row(label: str, stats_key: str | None) -> str:
        cells = []
        for name in names:
            s = result["models"][name]["overall" if stats_key is None else "buckets"]
            s = s if stats_key is None else s[stats_key]
            cells.append(f"{s['top1']:>{width}.1%}" if s.get("positions") else f"{'-':>{width}}")
        text = f"{label:<10} " + " ".join(cells)
        for name in names[1:]:
            d = result["differences"][name]
            d = d["overall"] if stats_key is None else d["buckets"].get(stats_key)
            if d is None:
                text += f"  {'-':>{width + 22}}"
                continue
            low, high = d["ci"]
            interval = f"[{low * 100:+.1f}, {high * 100:+.1f}]" if low == low else ""
            text += f"  {d['difference'] * 100:>+{width + 6}.1f} pts {interval:>14}"
        return text

    lines += [row(label, label) for label in result["labels"]]
    lines.append(row("all", None))
    lines.append(f"({result['positions']:,} positions; {int(result['confidence'] * 100)}% intervals of the paired "
                 "difference, resampling whole games)")
    return "\n".join(lines)
