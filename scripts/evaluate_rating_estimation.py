"""How well does the model estimate a player's rating from one game? (step 12)

Scores every held-out game at every rating bucket (cached as <out>/<split>_ll.npz,
~15 minutes per split on a laptop the first time), fits the range tempering on the
calibration games, then measures the error, range coverage and convergence on the
evaluation games against two baselines. Writes report.md, results.json and charts.

Usage:
    python scripts/evaluate_rating_estimation.py
    python scripts/evaluate_rating_estimation.py --model checkpoints/my_run/best_model.pt --out reports/rating/x
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import dataset_output_dir
from chess_engine.eval.evaluate import model_name, shown_path
from chess_engine.eval.rating_estimation import make_log_prior, training_player_ratings
from chess_engine.eval.rating_eval import EvalData, collect_eval_data, convergence, estimate_all, fit_temper, score
from chess_engine.eval.rating_report import CHARTS, plot_convergence, plot_error_by_rating, plot_scatter, render_report
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.training.run_log import json_safe


def load_or_collect(path: Path, loaded, dataset_dir: Path, split: str) -> EvalData:
    if path.is_file():
        print(f"{split}: using cached {path}", flush=True)
        return EvalData.load(path)
    start = time.perf_counter()
    data = collect_eval_data(loaded.model, PositionDataset(dataset_dir, split), loaded.ratings,
                             progress=lambda d, t: print(f"{split}: bucket {d}/{t} ({time.perf_counter() - start:.0f}s)",
                                                         flush=True))
    data.save(path)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="model file (default: bot.model)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None, help="output directory (default: reports/rating/<run>_<model>)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        r = cfg.rating_estimation
        model_path = Path(args.model).resolve() if args.model else cfg.bot.model
        dataset_dir = Path(args.dataset) if args.dataset else dataset_output_dir(cfg)
        out = Path(args.out) if args.out else cfg.paths.reports_dir / "rating" / model_name(model_path)
        loaded = load_model(model_path)
        ratings = loaded.ratings
        log_prior = make_log_prior(r, dataset_dir)
        mean_rating = float(training_player_ratings(dataset_dir).mean())
        calibration = load_or_collect(out / f"{r.calibration_split}_ll.npz", loaded, dataset_dir, r.calibration_split)
        evaluation = load_or_collect(out / f"{r.evaluation_split}_ll.npz", loaded, dataset_dir, r.evaluation_split)
    except (ConfigError, ModelFileError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if r.temper is not None:
        temper_fit = {"temper": r.temper, "fitted": False}
    else:
        temper_fit = {**fit_temper(calibration, ratings, r, log_prior), "fitted": True}
    temper = temper_fit["temper"]
    est = estimate_all(evaluation, ratings, r, log_prior, temper)
    scores = score(est, evaluation, ratings, mean_rating)
    rows = convergence(evaluation, ratings, r, log_prior, temper)
    setup = {"model_name": model_name(model_path), "model_path": shown_path(model_path), "split": r.evaluation_split,
             "calibration_split": r.calibration_split, "prior": r.prior, "rating_min": r.rating_min,
             "rating_max": r.rating_max}
    plot_scatter(est, evaluation, out / CHARTS["scatter"])
    plot_error_by_rating(scores, out / CHARTS["error"])
    plot_convergence(rows, out / CHARTS["convergence"])
    (out / "results.json").write_text(json.dumps(json_safe({"setup": setup, "temper": temper_fit, "scores": scores,
                                                            "convergence": rows}), indent=1) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(scores, rows, temper_fit, setup), encoding="utf-8")
    o = scores["overall"]
    print(f"one game: error {o['mae']:.0f} Elo (opponent's rating: {o['baseline_opponent']['mae']:.0f}, "
          f"average: {o['baseline_average']['mae']:.0f}); {r.interval:.0%} ranges cover {o['coverage']:.0%}, "
          f"width {o['mean_width']:.0f}; temper {temper:.2f}")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
