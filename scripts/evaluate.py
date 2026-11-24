"""Evaluate a trained model on held-out games: accuracy per rating vs. baselines, calibration (step 7).

Writes results.json, report.md, the charts and predictions.npz to reports/eval/<run>_<model>/.
Stockfish is used for its baseline when it is installed (see README), and skipped otherwise.

Usage:
    python scripts/evaluate.py                                 # best_model.pt of the most recent run
    python scripts/evaluate.py laptop_64x6                     # a run name: its best_model.pt
    python scripts/evaluate.py checkpoints/laptop_64x6/final_model.pt
    python scripts/evaluate.py laptop_64x6 --max-positions 20000 --no-stockfish   # quick look
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_engine.config import ConfigError, EvaluationConfig, load_config
from chess_engine.eval.evaluate import EvaluationError, evaluate_model
from chess_engine.model.inference import ModelFileError
from chess_engine.training.run_log import BEST_MODEL_NAME, FINAL_MODEL_NAME


def find_model(target: str | None, checkpoints_dir: Path) -> Path:
    """A model file given as a path, a run (directory or name: its best, else final model), or None (newest run)."""
    if target is None:
        candidates = list(checkpoints_dir.glob(f"*/{BEST_MODEL_NAME}"))
        if not candidates:
            raise FileNotFoundError(f"no trained model in {checkpoints_dir} (train one with scripts/train.py)")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    path = Path(target)
    if path.is_file():
        return path
    run = path if path.is_dir() else checkpoints_dir / target
    for name in (BEST_MODEL_NAME, FINAL_MODEL_NAME):
        if (run / name).is_file():
            return run / name
    raise FileNotFoundError(f"no model file or run named {target!r} (looked in {run})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default=None, help="model file, run directory or run name")
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="dataset directory (default: from the config)")
    parser.add_argument("--out", default=None, help="output directory (default: reports/eval/<run>_<model>)")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--stockfish-positions", type=int, default=None)
    parser.add_argument("--no-stockfish", action="store_true", help="skip the Stockfish baseline")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        overrides = {k: v for k, v in (("split", args.split), ("max_positions", args.max_positions),
                                       ("stockfish_positions", args.stockfish_positions)) if v is not None}
        if overrides:
            evaluation = EvaluationConfig.model_validate({**cfg.evaluation.model_dump(), **overrides})
            cfg = cfg.model_copy(update={"evaluation": evaluation})
        model = find_model(args.model, cfg.paths.checkpoints_dir)
        print(f"Evaluating {model}", flush=True)
        evaluate_model(cfg, model, args.dataset, args.out, use_stockfish=not args.no_stockfish,
                       log=lambda line: print(line, flush=True))
    except (ConfigError, EvaluationError, ModelFileError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
