"""Write what the website needs to estimate a rating in the browser: the prior over ratings and the fitted temper.

Usage:
    python scripts/build_rating_prior.py        # -> site/data/rating.json

The browser computes each move's probability at every rating bucket itself (site/js/rating.js); what it cannot
compute is how ratings are spread among the training players (the prior, from the dataset's games) and the temper
fitted on held-out games (scripts/evaluate_rating_estimation.py). Without the dataset the prior is flat. The file also
holds a test vector: the Python estimate for made-up per-move likelihoods, which the browser must reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_engine.config import load_config  # noqa: E402
from chess_engine.data.dataset_builder import dataset_output_dir  # noqa: E402
from chess_engine.eval.rating_estimation import (  # noqa: E402
    bucket_centers, estimate_from_bucket_ll, fitted_temper, rating_grid, setup_from_config,
)

TARGET = ROOT / "site" / "data" / "rating.json"
RESULTS = ROOT / "reports" / "rating" / "laptop_64x6_best_model" / "results.json"   # the model the site runs


def test_vector(cfg, ratings, log_prior: np.ndarray, temper: float) -> dict:
    """Made-up per-move likelihoods (moves x buckets) and the Python estimate from them."""
    rng = np.random.default_rng(7)
    k = len(ratings.bucket_edges)
    bucket_ll = np.round(np.log(rng.uniform(0.05, 0.6, size=(12, k))), 6)   # as written to the file
    est = estimate_from_bucket_ll(bucket_ll, ratings, cfg, log_prior, temper)
    return {"bucket_ll": bucket_ll.tolist(),
            "estimate": est.estimate, "median": est.median, "low": est.low, "high": est.high, "moves": est.moves}


def build(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    cfg, log_prior, temper = setup_from_config(config.rating_estimation, dataset_output_dir(config), RESULTS)
    temper = temper or fitted_temper(RESULTS) or 1.0
    log_prior = np.round(log_prior, 5)       # what the browser will use, so the test vector matches it exactly
    grid = rating_grid(cfg)
    return {
        "grid": {"min": cfg.rating_min, "max": cfg.rating_max, "step": cfg.rating_step},
        "prior": cfg.prior,
        "log_prior": log_prior.tolist(),
        "temper": round(float(temper), 6),
        "interval": cfg.interval,
        "bucket_edges": list(config.ratings.bucket_edges),
        "centers": bucket_centers(config.ratings.bucket_edges).tolist(),
        "points": len(grid),
        "test": test_vector(cfg, config.ratings, log_prior, round(float(temper), 6)),
    }


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="config file (default: configs/default.yaml)")
    parser.add_argument("--out", default=str(TARGET), help="where to write the JSON (default: site/data/rating.json)")
    args = parser.parse_args(argv)
    data = build(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    write(f"{data['prior']} prior over {data['points']} ratings, temper {data['temper']:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
