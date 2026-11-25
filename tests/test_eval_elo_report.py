"""Tests for the Elo-validation charts and report."""

from __future__ import annotations

from pathlib import Path

from chess_engine.eval.elo_report import CHARTS, plot_dial, plot_scores, render_report
from chess_engine.eval.elo_validation import analyze
from test_eval_elo_validation import _synthetic_log

PNG = b"\x89PNG\r\n\x1a\n"
SETUP = {"model_name": "run_best_model", "model_path": "checkpoints/run/best_model.pt", "temperature": 1.0,
         "stockfish_move_seconds": 0.05, "judge_nodes": 20000, "adjudicate_cp": 800, "adjudicate_plies": 8,
         "max_plies": 300, "bootstrap": 60, "seconds": 600, "self_play": "adjacent", "stockfish_version": "19"}


def test_charts(tmp_path: Path) -> None:
    analysis = analyze(_synthetic_log(), bootstrap=40)
    assert plot_dial(analysis, tmp_path / "d.png").read_bytes()[:8] == PNG
    assert plot_scores(analysis, tmp_path / "s.png").read_bytes()[:8] == PNG
    relative = analyze([r for r in _synthetic_log() if "sf:" not in r["pairing"]], bootstrap=10)
    assert plot_dial(relative, tmp_path / "r.png").is_file()


def test_report() -> None:
    analysis = analyze(_synthetic_log(), bootstrap=40)
    text = render_report(analysis, SETUP)
    assert text.startswith("# Elo-dial validation: run_best_model")
    for part in ("## Summary", "Rank correlation", "Elo of measured strength per 100 points", "## Every pairing",
                 "vs Stockfish 1320", "bot 1000 -> bot 1600", "not** the Lichess blitz scale", "Run time: 10 minutes"):
        assert part in text, part
    for chart in CHARTS.values():
        assert f"]({chart})" in text
    assert "Run time" not in render_report(analysis, {**SETUP, "seconds": None})
