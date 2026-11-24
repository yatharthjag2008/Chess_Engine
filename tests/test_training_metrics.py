"""Tests for running means, top-k matching and per-bucket move metrics."""

from __future__ import annotations

import math

import pytest
import torch

from chess_engine.training.metrics import MoveMetrics, RunningMeans, format_metrics, topk_hits


def test_running_means_with_tensors_and_counts() -> None:
    means = RunningMeans()
    assert not means
    means.add("loss", torch.tensor(2.0))
    means.add("loss", torch.tensor(4.0), count=3)
    means.add("lr", 0.1)
    assert means.means() == pytest.approx({"loss": 3.5, "lr": 0.1})
    means.reset()
    assert means.means() == {} and not means


def test_topk_hits() -> None:
    log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.06, 0.04, 0.0],
                                        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0]]).clamp_min(1e-30))
    hits = topk_hits(log_probs, torch.tensor([1, 0]), ks=(1, 3, 4))
    assert hits[1].tolist() == [False, False]
    assert hits[3].tolist() == [True, False]
    assert hits[4].tolist() == [True, True]


def _log_probs(rows: list[list[float]]) -> torch.Tensor:
    return torch.log(torch.tensor(rows).clamp_min(1e-30))


def test_move_metrics_per_bucket_and_overall() -> None:
    metrics = MoveMetrics(3, ["low", "mid", "high"])
    probs = [[0.7, 0.2, 0.1, 0.0, 0.0, 0.0]] * 4
    target = torch.tensor([0, 0, 1, 5])          # top-1 hit, hit, top-3 hit, miss
    buckets = torch.tensor([0, 0, 2, 2])
    value = torch.tensor([[0.0, 0.0, 5.0]] * 4)  # always predicts "win"
    result = torch.tensor([2, 0, 2, 2])
    metrics.update(buckets, _log_probs(probs), target, value, result)
    s = metrics.summary()
    assert metrics.positions == 4
    assert s["buckets"]["low"]["top1"] == 1.0 and s["buckets"]["high"]["top1"] == 0.0
    assert s["buckets"]["high"]["top3"] == 0.5
    assert s["buckets"]["mid"] == {"positions": 0}
    assert s["overall"]["top1"] == 0.5 and s["overall"]["value_accuracy"] == 0.75
    assert s["bucket_mean_top1"] == 0.5  # mean of 1.0 (low) and 0.0 (high); mid has no data
    assert s["buckets"]["low"]["nll"] == pytest.approx(-math.log(0.7))


def test_merge_equals_one_big_update() -> None:
    probs = _log_probs([[0.6, 0.4, 0.0]] * 6)
    target = torch.tensor([0, 1, 0, 1, 0, 0])
    buckets = torch.tensor([0, 1, 0, 1, 1, 0])
    whole = MoveMetrics(2)
    whole.update(buckets, probs, target)
    a, b = MoveMetrics(2), MoveMetrics(2)
    a.update(buckets[:2], probs[:2], target[:2])
    b.update(buckets[2:], probs[2:], target[2:])
    a.merge(b)
    assert a.summary() == whole.summary()
    assert "value_accuracy" not in whole.summary()["overall"]


def test_format_metrics_table() -> None:
    metrics = MoveMetrics(2, ["800-999", "1000+"])
    metrics.update(torch.tensor([0]), _log_probs([[1.0, 0.0]]), torch.tensor([0]))
    table = format_metrics(metrics.summary())
    assert "800-999" in table and "100.0%" in table and "all" in table
