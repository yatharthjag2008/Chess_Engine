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
    bucket_centers, estimate_from_bucket_ll, fitted_temper, grid_log_likelihood, opponent_log_prior, posterior,
    rating_grid, setup_from_config, summarize,
)
from chess_engine.eval.rating_eval import EvalData  # noqa: E402

TARGET = ROOT / "site" / "data" / "rating.json"
RESULTS = ROOT / "reports" / "rating" / "laptop_64x6_best_model" / "results.json"   # the model the site runs
HELD_OUT = RESULTS.parent                    # val_ll.npz / test_ll.npz: every held-out move scored at every bucket
SIGMAS = (40, 50, 60, 70, 80, 100, 120, 150)
TEMPERS = (1.0, 1.5, 2.0, 3.0, 5.0)


def score_opponent_prior(data: EvalData, grid: np.ndarray, centers: np.ndarray, sigma: float, temper: float,
                         interval: float) -> dict[str, float]:
    """Average error and coverage of the central interval, with the opponent prior, on held-out player-games."""
    ll = grid_log_likelihood(data.summed(), centers, grid)
    prior = -0.5 * ((grid[None, :] - data.opp_rating[:, None]) / sigma) ** 2
    post = posterior(ll, prior, temper)
    estimate = post @ grid
    cdf = np.cumsum(post, axis=1)
    tail = (1 - interval) / 2
    low = np.array([np.interp(tail, row, grid) for row in cdf])
    high = np.array([np.interp(1 - tail, row, grid) for row in cdf])
    return {"mae": float(np.abs(estimate - data.rating).mean()),
            "coverage": float(np.mean((data.rating >= low) & (data.rating <= high))), "players": int(len(estimate))}


def fit_opponent_prior(cfg, ratings, held_out: Path = HELD_OUT) -> dict | None:
    """Sigma and temper of the opponent prior: on the validation games, the smallest error among the settings
    whose 80% range holds the true rating 78-82% of the time; then scored once on the test games."""
    if not ((held_out / "val_ll.npz").is_file() and (held_out / "test_ll.npz").is_file()):
        return None
    val, test = EvalData.load(held_out / "val_ll.npz"), EvalData.load(held_out / "test_ll.npz")
    grid, centers = rating_grid(cfg), bucket_centers(ratings.bucket_edges)
    trials = [(s, t, score_opponent_prior(val, grid, centers, s, t, cfg.interval)) for s in SIGMAS for t in TEMPERS]
    calibrated = [x for x in trials if abs(x[2]["coverage"] - cfg.interval) <= 0.02] or trials
    sigma, temper, on_val = min(calibrated, key=lambda x: x[2]["mae"])
    on_test = score_opponent_prior(test, grid, centers, sigma, temper, cfg.interval)
    return {"sigma": float(sigma), "temper": float(temper), "val": on_val, "test": on_test,
            "opponent_guess_mae": float(np.abs(test.opp_rating - test.rating).mean())}


def opponent_test_vector(cfg, ratings, opponent: dict) -> dict:
    """Made-up likelihoods with two opponents (1500 and 1620), and the Python estimate from them."""
    rng = np.random.default_rng(11)
    bucket_ll = np.round(np.log(rng.uniform(0.05, 0.6, size=(9, len(ratings.bucket_edges)))), 6)
    grid = rating_grid(cfg)
    prior = opponent_log_prior(grid, [1500, 1620], opponent["sigma"])
    ll = grid_log_likelihood(bucket_ll.sum(axis=0), bucket_centers(ratings.bucket_edges), grid)
    est = summarize(grid, posterior(ll, prior, opponent["temper"]), cfg.interval, moves=len(bucket_ll))
    return {"bucket_ll": bucket_ll.tolist(), "opp_ratings": [1500, 1620], "estimate": est.estimate, "low": est.low,
            "high": est.high}


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
        **opponent_part(cfg, config.ratings),
    }


def opponent_part(cfg, ratings) -> dict:
    opponent = fit_opponent_prior(cfg, ratings)
    if opponent is None:
        return {}
    opponent["test_vector"] = opponent_test_vector(cfg, ratings, opponent)
    return {"opponent": opponent}


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
    if "opponent" in data:
        o = data["opponent"]
        write(f"opponent prior: sigma {o['sigma']:.0f}, temper {o['temper']:.1f}; on the test games average error "
              f"{o['test']['mae']:.1f} Elo (guessing the opponent's rating: {o['opponent_guess_mae']:.1f}), "
              f"80% range coverage {o['test']['coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
