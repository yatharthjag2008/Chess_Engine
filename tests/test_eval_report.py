"""Tests for the Markdown evaluation report."""

from __future__ import annotations

import copy

from chess_engine.eval.report import CHARTS, render_report

LABELS = ["800-999", "1000-1199"]


def _stats(top1: float, positions: int = 100) -> dict:
    return {"positions": positions, "top1": top1, "top3": top1 + 0.2, "top5": top1 + 0.3, "log_likelihood": -2.1,
            "perplexity": 8.2, "mean_legal_moves": 30.0, "random_top1": 0.05}


def _baseline(name: str, top1: float, **overall: float) -> dict:
    return {"name": name, "overall": {"positions": 200, "top1": top1, **overall},
            "buckets": {label: {"positions": 100, "top1": top1} for label in LABELS}}


RESULTS = {
    "labels": LABELS,
    "setup": {"model_name": "laptop_64x6 best", "model_path": "checkpoints/x/best_model.pt", "model_step": 9000,
              "dataset": "data/ds", "split": "test", "positions": 200, "games": 7, "confidence": 0.95,
              "bootstrap_samples": 1000, "stockfish": "not installed", "seconds": 12.3},
    "move_matching": {"overall": _stats(0.41, 200), "buckets": {"800-999": _stats(0.40), "1000-1199": _stats(0.42)},
                      "bucket_mean_top1": 0.41, "ci": {"overall": [0.39, 0.43], "buckets": [[0.37, 0.43], None]}},
    "baselines": {
        "most_common_move": _baseline("most common move", 0.2, coverage=0.3, top1_covered=0.5, model_top1_covered=0.6),
        "random": _baseline("random legal move", 0.05),
        "stockfish": {"name": "Stockfish", "skipped": "Stockfish was not found"},
    },
    "model_vs_most_common": {"difference": 0.21, "ci": [0.19, 0.23]},
    "cross_rating": {"labels": LABELS, "top1": [[0.40, 0.39], [0.38, 0.42]], "positions": [100, 100],
                     "diagonal_top1": [0.40, 0.42], "best_model_bucket": ["800-999", "1000-1199"],
                     "matched_top1": 0.41, "fixed_rating": {"bucket": "1000-1199", "top1": 0.40}},
    "breakdowns": {key: {"band A": _stats(0.5), "band B": {"positions": 0}}
                   for key in ("phase", "clock", "legal_moves")},
    "calibration": {"all_moves": {"ece": 0.012, "predictions": 6000}, "top1": {"ece": 0.03, "predictions": 200}},
    "value_head": {"accuracy": 0.55, "log_loss": 0.9, "brier": 0.5, "score": {"ece": 0.02},
                   "result_shares": {"loss": 0.45, "draw": 0.1, "win": 0.45}},
}


def test_report_contains_every_section_and_number() -> None:
    text = render_report(RESULTS)
    assert text.startswith("# Evaluation: laptop_64x6 best")
    for heading in ("## Summary", "## Accuracy by rating", "## Does the model play like each rating?",
                    "## When is the model right?", "## Calibration", "## Setup"):
        assert heading in text
    assert "**Top-1 move match: 41.0%** (39.0-43.0)" in text
    assert "Stockfish: skipped (Stockfish was not found)" in text
    assert "it knows 30.0% of the positions" in text and "+21.0 points (19.0-23.0)" in text
    assert "40.0% (37.0-43.0)" in text  # bucket CI in the table
    assert "band A" in text and "band B" not in text  # empty bands are left out
    for chart in CHARTS.values():
        assert f"]({chart})" in text


def test_report_without_optional_sections() -> None:
    results = copy.deepcopy(RESULTS)
    for key in ("cross_rating", "breakdowns", "calibration", "value_head", "model_vs_most_common"):
        results.pop(key)
    results["baselines"] = {}
    text = render_report(results)
    assert "## Calibration" not in text and "cross-rating" not in text.lower()
    assert "| **all** |" in text
