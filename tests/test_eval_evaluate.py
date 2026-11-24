"""End-to-end test of the evaluation pipeline (tiny model, tiny dataset, fake UCI engine)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from chess_engine.config import ModelConfig
from chess_engine.eval.evaluate import EvaluationError, evaluate_model, model_name
from chess_engine.eval.predictions import PredictionTable
from chess_engine.eval.report import CHARTS
from chess_engine.model.inference import save_model
from chess_engine.model.network import RatingConditionedNet
from training_helpers import train_config

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci_engine.py"), "--first-legal"]


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    cfg = train_config(tmp_path)
    model = RatingConditionedNet(ModelConfig(architecture="resnet", channels=8, residual_blocks=1,
                                             rating_embedding_dim=4), cfg.ratings.num_buckets)
    return save_model(model, tmp_path / "run" / "best_model.pt", cfg.ratings, extra={"step": 12})


def _cfg(tmp_path: Path, **evaluation):
    cfg = train_config(tmp_path)
    settings = {"bootstrap_samples": 50, "cross_rating_positions": 80, "stockfish_positions": 15, **evaluation}
    return cfg.model_copy(update={"evaluation": cfg.evaluation.model_copy(update=settings)})


def test_full_evaluation_writes_results_report_and_charts(tmp_path: Path, tiny_dataset: Path, model_file: Path) -> None:
    lines: list[str] = []
    out = evaluate_model(_cfg(tmp_path), model_file, tiny_dataset, tmp_path / "eval", stockfish_command=FAKE,
                         log=lines.append)
    results = json.loads((out.output_dir / "results.json").read_text(encoding="utf-8"))
    assert results == out.results
    assert results["setup"]["model_step"] == 12 and results["setup"]["model_name"] == "run_best_model"
    matching = results["move_matching"]
    assert matching["overall"]["positions"] == results["setup"]["positions"] > 0
    low, high = matching["ci"]["overall"]
    assert low <= matching["overall"]["top1"] <= high
    assert set(results["baselines"]) == {"most_common_move", "most_common_move_same_rating", "random", "stockfish"}
    assert results["baselines"]["stockfish"]["overall"]["positions"] == 15
    assert sum(results["cross_rating"]["positions"]) == 80
    assert results["value_head"]["positions"] == matching["overall"]["positions"]
    for chart in CHARTS.values():
        assert (out.output_dir / chart).is_file()
    report = out.report_path.read_text(encoding="utf-8")
    assert "# Evaluation: run_best_model" in report and "Stockfish depth" in report
    assert len(PredictionTable.load(out.output_dir / "predictions.npz")) == matching["overall"]["positions"]
    assert any("Report:" in line for line in lines)
    assert (tmp_path / "stockfish_moves.json").exists() is False  # the cache lives under data_dir/cache
    assert (tmp_path / "cache" / "stockfish_moves.json").is_file()


def test_optional_parts_can_be_skipped(tmp_path: Path, tiny_dataset: Path, model_file: Path) -> None:
    cfg = _cfg(tmp_path, cross_rating_positions=0, max_positions=50)
    out = evaluate_model(cfg, model_file, tiny_dataset, tmp_path / "eval", use_stockfish=False, log=lambda _: None)
    assert out.results["setup"]["positions"] == 50
    assert out.results["baselines"]["stockfish"] == {"name": "Stockfish", "skipped": "turned off"}
    assert "cross_rating" not in out.results
    assert not (out.output_dir / CHARTS["cross_rating"]).exists()
    assert "skipped (turned off)" in out.report_path.read_text(encoding="utf-8")


def test_errors_and_names(tmp_path: Path, model_file: Path) -> None:
    with pytest.raises(EvaluationError, match="no dataset"):
        evaluate_model(_cfg(tmp_path), model_file, tmp_path / "none", log=lambda _: None)
    assert model_name("checkpoints/laptop_64x6/best_model.pt") == "laptop_64x6_best_model"
    assert model_name("model.pt") == "model"
