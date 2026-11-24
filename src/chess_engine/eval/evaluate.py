"""The whole step-7 evaluation of one trained model, driven by ``cfg.evaluation``.

    output = evaluate_model(cfg, "checkpoints/laptop_64x6/best_model.pt")
    output.report_path      # reports/eval/laptop_64x6_best_model/report.md

Steps (each logged with its time):

1. predictions on the held-out split (:mod:`.predictions`), saved as ``predictions.npz``;
2. move matching per rating bucket with game-clustered confidence intervals, and
   breakdowns by game phase, clock and number of legal moves (:mod:`.move_matching`);
3. baselines: most common move (overall and same rating, from the training split),
   random legal move, and Stockfish on a sample if it is installed (:mod:`.baselines`,
   :mod:`.stockfish_baseline`);
4. the cross-rating matrix and the fixed-rating baseline (:mod:`.cross_rating`);
5. calibration of the move probabilities and of the value head (:mod:`.calibration`);
6. ``results.json``, ``report.md`` and the charts in the output directory.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_engine.config import Config
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import dataset_output_dir
from chess_engine.eval import plots
from chess_engine.eval.baselines import (
    build_frequency_tables,
    most_common_move_baseline,
    random_baseline,
    summarize_baseline,
)
from chess_engine.eval.bootstrap import cluster_bootstrap
from chess_engine.eval.calibration import move_calibration, value_calibration
from chess_engine.eval.cross_rating import run_cross_rating
from chess_engine.eval.move_matching import breakdowns, format_move_matching, move_matching
from chess_engine.eval.predictions import predict_split
from chess_engine.eval.report import CHARTS, render_report
from chess_engine.eval.stockfish_baseline import sample_rows, stockfish_baseline
from chess_engine.model.inference import load_model
from chess_engine.stockfish import StockfishError, find_stockfish
from chess_engine.training.run_log import json_safe

Logger = Callable[[str], None]


class EvaluationError(RuntimeError):
    """The evaluation cannot run (missing dataset, mismatched rating buckets, ...)."""


@dataclass(frozen=True)
class EvaluationOutput:
    """What :func:`evaluate_model` wrote."""

    results: dict[str, Any]
    output_dir: Path
    report_path: Path


def model_name(model_path: str | Path) -> str:
    """``<run directory>_<file stem>``, e.g. ``laptop_64x6_best_model``."""
    path = Path(model_path)
    return f"{path.parent.name}_{path.stem}" if path.parent.name else path.stem


def _ci_lists(ci: Any) -> dict[str, Any]:
    return {"overall": list(ci.overall), "buckets": [list(g) for g in ci.groups]}


def evaluate_model(
    cfg: Config,
    model_path: str | Path,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_stockfish: bool = True,
    stockfish_command: str | list[str] | None = None,
    log: Logger = print,
) -> EvaluationOutput:
    """Run every evaluation step for the model at ``model_path`` (see the module docstring).

    Args:
        dataset_dir: Built dataset; default: the config's dataset directory.
        output_dir: Default: ``<reports_dir>/eval/<model name>``.
        use_stockfish: Run the Stockfish baseline if Stockfish can be found.
        stockfish_command: Engine command instead of the configured Stockfish (tests use a fake engine).
    """
    start = time.perf_counter()
    e = cfg.evaluation
    model_path = Path(model_path)
    loaded = load_model(model_path)
    model, ratings = loaded.model, loaded.ratings
    labels = ratings.bucket_labels()
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else dataset_output_dir(cfg)
    if not (dataset_dir / "manifest.json").is_file():
        raise EvaluationError(f"no dataset at {dataset_dir} (build it with scripts/build_dataset.py)")
    output = Path(output_dir) if output_dir is not None else cfg.paths.reports_dir / "eval" / model_name(model_path)
    output.mkdir(parents=True, exist_ok=True)

    def step(message: str) -> Callable[[], None]:
        t0 = time.perf_counter()
        log(f"{message} ...")
        return lambda: log(f"  done in {time.perf_counter() - t0:.0f} s")

    data = PositionDataset(dataset_dir, e.split)
    if data.manifest["ratings"]["bucket_edges"] != ratings.bucket_edges:
        raise EvaluationError("the model and the dataset use different rating buckets")
    if len(data) == 0:
        raise EvaluationError(f"the {e.split} split is empty")

    done = step(f"Predicting {min(len(data), e.max_positions or len(data)):,} {e.split} positions")
    table = predict_split(model, data, ratings, e.batch_size, max_positions=e.max_positions,
                          calibration_bins=e.calibration_bins)
    table.save(output / "predictions.npz")
    done()

    games = table["game"]
    matching = move_matching(table, labels)
    matching["ci"] = _ci_lists(cluster_bootstrap(table.top1_hit, games, table["bucket"], len(labels),
                                                 e.bootstrap_samples, e.confidence, e.seed))
    log(format_move_matching(matching))
    results: dict[str, Any] = {"labels": labels, "move_matching": matching,
                               "breakdowns": breakdowns(table, labels), "baselines": {}}

    done = step("Baselines: most common move (from the training split), random legal move")
    train = PositionDataset(dataset_dir, "train").samples
    overall_table, bucket_table = build_frequency_tables(train)
    del train
    common = most_common_move_baseline(table, overall_table)
    results["baselines"]["most_common_move"] = summarize_baseline(common, table, labels)
    results["baselines"]["most_common_move_same_rating"] = summarize_baseline(
        most_common_move_baseline(table, overall_table, bucket_table), table, labels)
    results["baselines"]["random"] = summarize_baseline(random_baseline(table), table, labels)
    diff = cluster_bootstrap(table.top1_hit - common.hits, games, samples=e.bootstrap_samples,
                             confidence=e.confidence, seed=e.seed)
    results["model_vs_most_common"] = {"difference": float((table.top1_hit - common.hits).mean()),
                                       "ci": list(diff.overall)}
    done()

    stockfish_note = "turned off"
    if use_stockfish and e.stockfish_positions > 0:
        command = stockfish_command
        if command is None:
            try:
                command = str(find_stockfish(cfg.stockfish.path))
            except StockfishError:
                stockfish_note = "not installed"
        if command is not None:
            rows = sample_rows(len(table), e.stockfish_positions, e.seed)
            done = step(f"Stockfish baseline on {len(rows):,} positions at depth {e.stockfish_depth}")
            summary = stockfish_baseline(table, data.samples, rows, command, e.stockfish_depth, labels,
                                         cfg.paths.data_dir / "cache" / "stockfish_moves.json",
                                         cfg.stockfish.threads, cfg.stockfish.hash_mb)
            summary.pop("rows")
            summary.pop("hits")
            results["baselines"]["stockfish"] = summary
            stockfish_note = f"depth {e.stockfish_depth}, {len(rows):,} sampled positions"
            done()
    if "stockfish" not in results["baselines"]:
        results["baselines"]["stockfish"] = {"name": "Stockfish", "skipped": stockfish_note}

    if e.cross_rating_positions > 0:
        done = step(f"Cross-rating: {min(len(data), e.cross_rating_positions):,} positions x {len(labels)} ratings")
        results["cross_rating"] = run_cross_rating(model, data, ratings, e.cross_rating_positions, e.batch_size)
        done()

    results["calibration"] = move_calibration(table)
    results["value_head"] = value_calibration(table, e.calibration_bins)
    results["setup"] = {
        "model_name": model_name(model_path), "model_path": str(model_path), "model_step": loaded.extra.get("step"),
        "dataset": str(dataset_dir), "split": e.split, "positions": len(table),
        "games": int(len(np.unique(games))), "confidence": e.confidence, "bootstrap_samples": e.bootstrap_samples,
        "stockfish": stockfish_note, "seconds": round(time.perf_counter() - start, 1),
    }

    _write_charts(results, labels, output)
    results = json_safe(results)
    (output / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    report = output / "report.md"
    report.write_text(render_report(results), encoding="utf-8")
    log(f"Report: {report}")
    return EvaluationOutput(results, output, report)


def _write_charts(results: dict[str, Any], labels: list[str], output: Path) -> None:
    matching = results["move_matching"]
    present = [label for label in labels if matching["buckets"][label].get("positions")]
    cis = dict(zip(labels, matching["ci"]["buckets"]))
    series = [plots.Series("model", [matching["buckets"][b]["top1"] for b in present],
                           [cis[b][0] for b in present], [cis[b][1] for b in present])]
    cross = results.get("cross_rating")
    if cross:
        fixed = cross["fixed_rating"]["bucket"]
        row = cross["top1"][cross["labels"].index(fixed)]
        series.append(plots.Series("model, one rating for all", [row[labels.index(b)] for b in present]))
    for key in ("most_common_move", "stockfish"):
        baseline = results["baselines"].get(key)
        if baseline and not baseline.get("skipped"):
            name = "Stockfish" if key == "stockfish" else baseline["name"]
            series.append(plots.Series(name, [baseline["buckets"][b].get("top1") for b in present]))
    random_top1 = [results["baselines"]["random"]["buckets"][b]["top1"] for b in present]
    plots.plot_accuracy_by_rating(present, series, output / CHARTS["accuracy"], random=random_top1)
    plots.plot_calibration(results["calibration"], output / CHARTS["calibration"])
    plots.plot_breakdowns(results["breakdowns"], labels, output / CHARTS["breakdowns"])
    if cross:
        plots.plot_cross_rating(cross, output / CHARTS["cross_rating"])
