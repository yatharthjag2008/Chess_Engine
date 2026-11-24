"""Metrics for training logs and validation: running means, top-k move matching, per-bucket results.

* :class:`RunningMeans` averages scalar losses between log lines. Values stay on the
  device until :meth:`RunningMeans.means` is called, so a GPU is not forced to
  synchronize on every step.
* :class:`MoveMetrics` accumulates, per rating bucket, the number of positions,
  the negative log-likelihood of the human move, top-1/3/5 matches and the value
  head's accuracy, then reports them overall and per bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

TOP_K: tuple[int, ...] = (1, 3, 5)


class RunningMeans:
    """Sums and counts of named scalars (tensors or floats) since the last :meth:`reset`."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor | float] = {}
        self._counts: dict[str, int] = {}

    def add(self, name: str, value: torch.Tensor | float, count: int = 1) -> None:
        value = value.detach() * count if isinstance(value, torch.Tensor) else float(value) * count
        self._sums[name] = self._sums.get(name, 0.0) + value
        self._counts[name] = self._counts.get(name, 0) + count

    def means(self) -> dict[str, float]:
        return {name: float(total) / self._counts[name] for name, total in self._sums.items()}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def __bool__(self) -> bool:
        return bool(self._counts)


def topk_hits(log_probs: torch.Tensor, target: torch.Tensor, ks: Sequence[int] = TOP_K) -> dict[int, torch.Tensor]:
    """For each k, a bool tensor: is the played move among the model's k likeliest moves?

    Ties are broken by ``topk`` (in practice exact ties only occur between illegal moves,
    which are never in the top k while the position has at least k legal moves; with
    fewer than k legal moves the played move always counts as a top-k hit).
    """
    top = log_probs.topk(min(max(ks), log_probs.shape[-1]), dim=-1).indices
    hits = top == target[:, None]
    return {k: hits[:, :k].any(dim=-1) for k in ks}


class MoveMetrics:
    """Per-bucket accumulation of move-prediction quality.

    Args:
        num_buckets: Number of rating buckets.
        labels: Bucket labels for the report (default ``"0"``, ``"1"``, ...).
    """

    _FIELDS = ("positions", "nll", *(f"top{k}" for k in TOP_K), "value_positions", "value_correct", "value_nll")

    def __init__(self, num_buckets: int, labels: Sequence[str] | None = None) -> None:
        self.num_buckets = num_buckets
        self.labels = list(labels) if labels is not None else [str(b) for b in range(num_buckets)]
        self._sums = {name: np.zeros(num_buckets, dtype=np.float64) for name in self._FIELDS}

    def _add(self, name: str, buckets: np.ndarray, values: np.ndarray | None = None) -> None:
        self._sums[name] += np.bincount(buckets, weights=values, minlength=self.num_buckets)

    def update(
        self,
        buckets: torch.Tensor,
        log_probs: torch.Tensor,
        target: torch.Tensor,
        value_logits: torch.Tensor | None = None,
        result: torch.Tensor | None = None,
    ) -> None:
        """Add a batch: mover's buckets, legal log-probabilities, played moves, optional value logits + results."""
        b = buckets.detach().cpu().numpy().astype(np.int64)
        nll = -log_probs.detach().gather(1, target[:, None]).squeeze(1)
        self._add("positions", b)
        self._add("nll", b, nll.float().cpu().numpy())
        for k, hit in topk_hits(log_probs.detach(), target).items():
            self._add(f"top{k}", b, hit.float().cpu().numpy())
        if value_logits is not None and result is not None:
            value = value_logits.detach().float()
            correct = (value.argmax(-1) == result).float().cpu().numpy()
            value_nll = torch.nn.functional.cross_entropy(value, result, reduction="none").cpu().numpy()
            self._add("value_positions", b)
            self._add("value_correct", b, correct)
            self._add("value_nll", b, value_nll)

    def merge(self, other: MoveMetrics) -> None:
        for name in self._FIELDS:
            self._sums[name] += other._sums[name]

    @property
    def positions(self) -> int:
        return int(self._sums["positions"].sum())

    def _rates(self, mask: np.ndarray) -> dict[str, float]:
        s = {name: values[mask].sum() for name, values in self._sums.items()}
        n = s["positions"]
        out: dict[str, float] = {"positions": int(n)}
        if n:
            out["nll"] = s["nll"] / n
            out.update({f"top{k}": s[f"top{k}"] / n for k in TOP_K})
        if s["value_positions"]:
            out["value_accuracy"] = s["value_correct"] / s["value_positions"]
            out["value_nll"] = s["value_nll"] / s["value_positions"]
        return out

    def summary(self) -> dict[str, Any]:
        """``{"overall": {...}, "buckets": {label: {...}}, "bucket_mean_top1": ...}``.

        ``overall`` averages over positions; ``bucket_mean_top1`` averages the
        per-bucket top-1 rates, so every rating level counts equally.
        """
        buckets = {label: self._rates(np.arange(self.num_buckets) == b) for b, label in enumerate(self.labels)}
        top1 = [r["top1"] for r in buckets.values() if r["positions"]]
        return {
            "overall": self._rates(np.ones(self.num_buckets, dtype=bool)),
            "buckets": buckets,
            "bucket_mean_top1": float(np.mean(top1)) if top1 else float("nan"),
        }


def format_metrics(summary: dict[str, Any]) -> str:
    """A per-bucket table of a :meth:`MoveMetrics.summary`."""
    lines = [f"{'bucket':<10} {'positions':>9} {'top1':>6} {'top3':>6} {'top5':>6} {'nll':>6} {'value':>6}"]

    def row(label: str, r: dict[str, float]) -> str:
        if not r["positions"]:
            return f"{label:<10} {0:>9}"
        value = f"{r['value_accuracy']:>6.1%}" if "value_accuracy" in r else f"{'-':>6}"
        return (f"{label:<10} {r['positions']:>9,} {r['top1']:>6.1%} {r['top3']:>6.1%} {r['top5']:>6.1%} "
                f"{r['nll']:>6.3f} {value}")

    for label, r in summary["buckets"].items():
        lines.append(row(label, r))
    lines.append(row("all", summary["overall"]))
    return "\n".join(lines)
