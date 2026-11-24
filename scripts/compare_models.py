"""Compare two or more trained models on the same held-out positions (step 7).

Prints top-1 per rating bucket for every model and each model's difference to the
first one, with paired confidence intervals (bootstrap over whole games).

Usage:
    python scripts/compare_models.py checkpoints/laptop_64x6/best_model.pt checkpoints/laptop_64x6/final_model.pt
    python scripts/compare_models.py laptop_64x6 other_run --max-positions 50000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import dataset_output_dir
from chess_engine.eval.compare import compare_tables, format_comparison
from chess_engine.eval.evaluate import model_name
from chess_engine.eval.predictions import predict_split
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.training.run_log import json_safe


def _find_model():  # reuse evaluate.py's lookup (scripts are not a package)
    spec = importlib.util.spec_from_file_location("evaluate_script", Path(__file__).with_name("evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="model files, run directories or run names (first = reference)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--json", default=None, help="also write the comparison to this JSON file")
    args = parser.parse_args(argv)
    if len(args.models) < 2:
        parser.error("give at least two models")
    try:
        cfg = load_config(args.config)
        find_model = _find_model()
        paths = [find_model(m, cfg.paths.checkpoints_dir) for m in args.models]
        dataset = PositionDataset(args.dataset or dataset_output_dir(cfg), cfg.evaluation.split)
        tables, labels = {}, None
        for path in paths:
            loaded = load_model(path)
            labels = loaded.ratings.bucket_labels()
            name = model_name(path)
            print(f"scoring {name} ...", flush=True)
            tables[name] = predict_split(loaded.model, dataset, loaded.ratings, cfg.evaluation.batch_size,
                                         max_positions=args.max_positions or cfg.evaluation.max_positions)
        e = cfg.evaluation
        result = compare_tables(tables, labels, e.bootstrap_samples, e.confidence, e.seed)
    except (ConfigError, ModelFileError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_comparison(result))
    if args.json:
        Path(args.json).write_text(json.dumps(json_safe(result), indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
