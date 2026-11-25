"""Score rating estimation on held-out games: data collection, metrics, tempering, convergence.

Every position of a split is scored once per rating bucket (the model told "the mover
is in bucket b", with the real opponent rating and clocks), giving a per-move
log-likelihood matrix. Positions are grouped into *player-games*: one side of one
game, whose true rating the dataset knows.

    data = collect_eval_data(model, PositionDataset(ds, "test"), ratings)
    data.save("reports/rating/test_ll.npz")
    est = estimate_all(data, ratings, cfg.rating_estimation, log_prior, temper=2.0)
    score(est, data, ratings)          # errors, interval coverage, baselines, by true rating

The baselines are what the estimate must beat to be worth anything: "everyone is
average" (the mean training rating) and "the opponent's rating" (Lichess pairs
players of similar strength, so this is a strong guess when the opponent is known).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_engine.config import RatingEstimationConfig, RatingsConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.predictions import predict_split
from chess_engine.eval.rating_estimation import bucket_centers, grid_log_likelihood, posterior, rating_grid
from chess_engine.model.network import RatingConditionedNet


@dataclass
class EvalData:
    """Per-move bucket log-likelihoods of a split, grouped into player-games."""

    player: np.ndarray        # (N,) player-game index of each move (0 .. P-1)
    ply: np.ndarray           # (N,) ply of the move in its game
    ll: np.ndarray            # (N, K) log-probability of the move at each bucket
    rating: np.ndarray        # (P,) the player's true rating
    opp_rating: np.ndarray    # (P,) the opponent's rating
    game: np.ndarray          # (P,) game row in the dataset
    color: np.ndarray         # (P,) 0 White, 1 Black

    @property
    def num_players(self) -> int:
        return len(self.rating)

    def moves_per_player(self) -> np.ndarray:
        return np.bincount(self.player, minlength=self.num_players)

    def summed(self, max_moves: int | None = None) -> np.ndarray:
        """``(P, K)`` log-likelihood per player-game (only its first ``max_moves`` moves if given)."""
        keep = np.ones(len(self.player), dtype=bool)
        if max_moves is not None:
            order = np.lexsort((self.ply, self.player))
            rank = np.empty(len(order), dtype=np.int64)
            starts = np.searchsorted(self.player[order], np.arange(self.num_players))
            rank[order] = np.arange(len(order)) - starts[self.player[order]]
            keep = rank < max_moves
        out = np.zeros((self.num_players, self.ll.shape[1]))
        np.add.at(out, self.player[keep], self.ll[keep])
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, player=self.player, ply=self.ply, ll=self.ll, rating=self.rating,
                            opp_rating=self.opp_rating, game=self.game, color=self.color)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EvalData:
        with np.load(Path(path)) as f:
            return cls(**{name: f[name] for name in f.files})


def collect_eval_data(
    model: RatingConditionedNet,
    dataset: PositionDataset,
    ratings: RatingsConfig,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
    max_positions: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvalData:
    """Score the split at every bucket and group the moves into player-games (see the module docstring).

    ``progress`` is called with (buckets done, number of buckets).
    """
    k = ratings.num_buckets
    columns = []
    base = None
    for bucket in range(k):
        table = predict_split(model, dataset, ratings, batch_size, device, max_positions, conditioned_bucket=bucket)
        if base is None:
            base = table
        columns.append(np.log(np.maximum(table["target_prob"].astype(np.float64), 1e-12)))
        if progress is not None:
            progress(bucket + 1, k)
    ll = np.stack(columns, axis=1).astype(np.float32)
    game = base["game"].astype(np.int64)
    color = (base["ply"].astype(np.int64) % 2)          # games start from the initial position: ply 0 is White
    key = game * 2 + color
    unique, player = np.unique(key, return_inverse=True)
    first = np.zeros(len(unique), dtype=np.int64)
    first[player[::-1]] = np.arange(len(player))[::-1]  # one representative move per player-game
    return EvalData(player=player.astype(np.int64), ply=base["ply"].astype(np.int64), ll=ll,
                    rating=base["rating"][first].astype(np.float64),
                    opp_rating=base["opp_rating"][first].astype(np.float64),
                    game=unique // 2, color=(unique % 2).astype(np.int8))



# --------------------------------------------------------------------------- estimates and scores


@dataclass
class Estimates:
    """Estimates for every player-game: posterior mean and central interval."""

    estimate: np.ndarray
    low: np.ndarray
    high: np.ndarray
    interval: float


def estimate_all(data: EvalData, ratings: RatingsConfig, cfg: RatingEstimationConfig, log_prior: np.ndarray,
                 temper: float = 1.0, max_moves: int | None = None) -> Estimates:
    """The estimate of every player-game (optionally from its first ``max_moves`` moves only)."""
    grid = rating_grid(cfg)
    ll = grid_log_likelihood(data.summed(max_moves), bucket_centers(ratings.bucket_edges), grid)
    post = posterior(ll, log_prior, temper)
    cdf = np.cumsum(post, axis=1)
    tail = (1 - cfg.interval) / 2
    low = np.array([np.interp(tail, row, grid) for row in cdf])
    high = np.array([np.interp(1 - tail, row, grid) for row in cdf])
    return Estimates(post @ grid, low, high, cfg.interval)


def _errors(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = predicted - truth
    return {"players": int(len(err)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err**2).mean())),
            "bias": float(err.mean()), "median_abs_error": float(np.median(np.abs(err)))}


def score(est: Estimates, data: EvalData, ratings: RatingsConfig, mean_rating: float | None = None) -> dict[str, Any]:
    """Errors of the estimate and of the baselines, overall and per true-rating bucket.

    Args:
        mean_rating: The "everyone is average" guess (default: the mean true rating of ``data``).
    """
    truth = data.rating
    mean_rating = float(truth.mean()) if mean_rating is None else mean_rating
    inside = (est.low <= truth) & (truth <= est.high)
    opp = np.where(np.isfinite(data.opp_rating), data.opp_rating, mean_rating)

    def block(rows: np.ndarray) -> dict[str, Any]:
        if not rows.any():
            return {"players": 0}
        out = _errors(est.estimate[rows], truth[rows])
        out["coverage"] = float(inside[rows].mean())
        out["mean_width"] = float((est.high - est.low)[rows].mean())
        out["baseline_average"] = _errors(np.full(rows.sum(), mean_rating), truth[rows])
        out["baseline_opponent"] = _errors(opp[rows], truth[rows])
        out["mean_moves"] = float(data.moves_per_player()[rows].mean())
        return out

    labels = ratings.bucket_labels()
    edges = np.asarray(ratings.bucket_edges)
    bucket = np.maximum(np.searchsorted(edges, truth, side="right") - 1, 0)
    return {
        "interval": est.interval,
        "mean_rating": mean_rating,
        "overall": block(np.ones(len(truth), dtype=bool)),
        "buckets": {label: block(bucket == b) for b, label in enumerate(labels)},
    }


# --------------------------------------------------------------------------- tempering


def fit_temper(data: EvalData, ratings: RatingsConfig, cfg: RatingEstimationConfig, log_prior: np.ndarray,
               candidates: np.ndarray | None = None) -> dict[str, Any]:
    """The ``temper`` whose ranges contain the true rating ``cfg.interval`` of the time on ``data``.

    Coverage grows with ``temper`` (wider posteriors), so the first candidate that reaches the
    target coverage is taken. Returns the chosen value and the coverage / error at every candidate.
    """
    candidates = np.geomspace(1, 64, 49) if candidates is None else np.asarray(candidates, dtype=np.float64)
    target = cfg.interval
    rows = []
    for temper in candidates:
        est = estimate_all(data, ratings, cfg, log_prior, float(temper))
        inside = (est.low <= data.rating) & (data.rating <= est.high)
        rows.append({"temper": float(temper), "coverage": float(inside.mean()),
                     "mae": float(np.abs(est.estimate - data.rating).mean())})
    reached = [r for r in rows if r["coverage"] >= target]
    chosen = reached[0] if reached else rows[-1]
    return {"temper": chosen["temper"], "target_coverage": target, "coverage": chosen["coverage"],
            "mae": chosen["mae"], "candidates": rows}


# --------------------------------------------------------------------------- convergence


def convergence(data: EvalData, ratings: RatingsConfig, cfg: RatingEstimationConfig, log_prior: np.ndarray,
                temper: float = 1.0, steps: tuple[int, ...] = (5, 10, 20, 30, 40)) -> list[dict[str, float]]:
    """How the estimate improves with more moves: error, coverage and range width after each player's
    first ``n`` moves (players with fewer moves use all of theirs), and with every move (``n`` = None)."""
    rows = []
    moves = data.moves_per_player()
    for n in (*steps, None):
        est = estimate_all(data, ratings, cfg, log_prior, temper, max_moves=n)
        err = est.estimate - data.rating
        inside = (est.low <= data.rating) & (data.rating <= est.high)
        used = moves if n is None else np.minimum(moves, n)
        rows.append({"max_moves": n, "mean_moves": float(used.mean()), "mae": float(np.abs(err).mean()),
                     "rmse": float(np.sqrt((err**2).mean())), "coverage": float(inside.mean()),
                     "mean_width": float((est.high - est.low).mean())})
    return rows
