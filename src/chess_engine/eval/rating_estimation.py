"""Estimate a player's rating (and a likely range) from the moves they played.

For every move, the model gives the probability that a player of each rating
bucket plays it. Summed over a game's moves, the log-probabilities say how well
each rating explains the game. Bucket ``b`` is placed at its centre rating and the
values are interpolated linearly to every rating on a grid; with a prior over
ratings (how ratings are spread among players) this gives a posterior:

    log p(rating | moves)  =  sum over moves of log p(move | rating) / temper  +  log prior(rating)  + const

The estimate is the posterior mean, the range its central ``interval`` (e.g. 80%).
Several games of one player multiply their likelihoods (their log-likelihoods add up)::

    games = player_games_from_pgn(pgn_text, player="someone")
    result = estimate_rating(predictor, games, cfg.rating_estimation, log_prior)
    result.overall.describe()          # "about 1430 (80% range 1250-1620, from 112 moves)"
    result.per_game[0].estimate
``temper`` > 1 widens the posterior: the moves of one game are not independent
pieces of evidence (one opening, one style, one time scramble), so the raw sum
is overconfident; the right value is fitted on held-out games.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch

from chess_engine.config import RatingEstimationConfig, RatingsConfig
from chess_engine.encoding.board import encode_boards
from chess_engine.encoding.moves import legal_move_mask, move_to_index
from chess_engine.eval.player_games import PlayerGame
from chess_engine.model.features import buckets_from_ratings, context_features, make_inputs
from chess_engine.model.inference import MovePredictor
from chess_engine.model.masking import legal_log_probs


def bucket_centers(edges: Sequence[int]) -> np.ndarray:
    """Centre rating of each bucket; the open top bucket is as wide as the one below it.

    ``[800, 1000, ..., 2200]`` -> ``[900, 1100, ..., 2100, 2300]``.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) == 1:
        return edges + 100.0
    widths = np.diff(edges)
    return np.append(edges[:-1] + widths / 2, edges[-1] + widths[-1] / 2)


def rating_grid(cfg: RatingEstimationConfig) -> np.ndarray:
    """The ratings considered: ``rating_min`` to ``rating_max`` in steps of ``rating_step``."""
    return np.arange(cfg.rating_min, cfg.rating_max + 1, cfg.rating_step, dtype=np.float64)


def grid_log_likelihood(bucket_ll: np.ndarray, centers: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate per-bucket log-likelihoods ``(..., K)`` to the grid ``(..., G)``.

    Linear between bucket centres and flat beyond the outermost ones (the model knows
    nothing finer). Interpolating a sum equals summing the interpolations, so callers may
    sum over moves first.
    """
    bucket_ll = np.asarray(bucket_ll, dtype=np.float64)
    flat = bucket_ll.reshape(-1, bucket_ll.shape[-1])
    out = np.stack([np.interp(grid, centers, row) for row in flat])
    return out.reshape(*bucket_ll.shape[:-1], len(grid))


def posterior(log_likelihood: np.ndarray, log_prior: np.ndarray, temper: float = 1.0) -> np.ndarray:
    """Normalized posterior over the grid (rows are independent if 2-D)."""
    if temper <= 0:
        raise ValueError("temper must be positive")
    logp = np.asarray(log_likelihood, dtype=np.float64) / temper + log_prior
    logp = logp - logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- per-move likelihoods


@torch.no_grad()
def move_bucket_log_likelihoods(
    predictor: MovePredictor,
    boards: Sequence[chess.Board],
    moves: Sequence[chess.Move],
    opp_ratings: Sequence[int | None] | int | None = None,
    clocks: Sequence[float] | None = None,
    opp_clocks: Sequence[float] | None = None,
    base_seconds: int = 180,
    increment: int = 0,
    chunk: int = 2048,
) -> np.ndarray:
    """``(moves, K)``: log-probability that a player of each rating bucket plays ``moves[i]`` in ``boards[i]``.

    Args:
        opp_ratings: The opponent's rating at each move (or one for all). Where it is unknown
            (None), the opponent is assumed to be as strong as the candidate rating: Lichess
            pairs players of similar strength.
        clocks / opp_clocks: Seconds left before each move (default: the full base time).
    """
    n, k = len(boards), predictor.ratings.num_buckets
    if n != len(moves):
        raise ValueError("boards and moves must have the same length")
    if n == 0:
        return np.zeros((0, k))
    for board, move in zip(boards, moves):
        if move not in board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {board.fen()}")
    opp = [opp_ratings] * n if opp_ratings is None or isinstance(opp_ratings, int) else list(opp_ratings)
    clock = np.full(n, float(base_seconds)) if clocks is None else np.asarray(clocks, dtype=np.float64)
    opp_clock = np.full(n, float(base_seconds)) if opp_clocks is None else np.asarray(opp_clocks, dtype=np.float64)
    planes = encode_boards(list(boards))
    masks = np.stack([legal_move_mask(b) for b in boards])
    targets = np.array([move_to_index(m, b.turn) for b, m in zip(boards, moves)], dtype=np.int64)
    known = np.array([o is not None for o in opp])
    opp_bucket = buckets_from_ratings(np.array([o if o is not None else 0 for o in opp]), predictor.ratings)
    context = context_features(clock, opp_clock, np.full(n, base_seconds), np.full(n, increment))

    # Every (move, bucket) pair as one row: move i at bucket b is row b * n + i.
    rows_bucket = np.repeat(np.arange(k), n)
    rows_move = np.tile(np.arange(n), k)
    rows_opp = np.where(known[rows_move], opp_bucket[rows_move], rows_bucket)
    out = np.empty(n * k)
    model = predictor.model
    for start in range(0, n * k, chunk):
        sl = slice(start, start + chunk)
        idx = rows_move[sl]
        inputs = make_inputs(planes[idx], rows_bucket[sl], rows_opp[sl], context[idx], predictor.device)
        mask = torch.as_tensor(masks[idx], device=predictor.device)
        log_probs = legal_log_probs(model(inputs).policy, mask)
        target = torch.as_tensor(targets[idx], device=predictor.device)
        out[sl] = log_probs.gather(1, target[:, None]).squeeze(1).cpu().numpy()
    return out.reshape(k, n).T


# --------------------------------------------------------------------------- prior

#: Share of the prior spread evenly over the whole grid, so no rating is ever ruled out completely.
PRIOR_FLOOR = 0.01


def data_log_prior(player_ratings: np.ndarray, grid: np.ndarray, smoothing: float = 75.0) -> np.ndarray:
    """Log of the smoothed rating distribution of ``player_ratings`` on ``grid`` (plus a uniform floor)."""
    player_ratings = np.asarray(player_ratings, dtype=np.float64)
    if len(player_ratings) == 0:
        raise ValueError("no player ratings for the prior")
    step = grid[1] - grid[0] if len(grid) > 1 else 1.0
    edges = np.append(grid - step / 2, grid[-1] + step / 2)
    counts = np.histogram(np.clip(player_ratings, grid[0], grid[-1]), bins=edges)[0].astype(np.float64)
    if smoothing > 0:
        half = int(np.ceil(4 * smoothing / step))
        offsets = np.arange(-half, half + 1) * step
        kernel = np.exp(-0.5 * (offsets / smoothing) ** 2)
        counts = np.convolve(counts, kernel / kernel.sum(), mode="same")
    density = counts / counts.sum()
    density = (1 - PRIOR_FLOOR) * density + PRIOR_FLOOR / len(grid)
    return np.log(density)


def training_player_ratings(dataset_dir: str | Path, split: str = "train") -> np.ndarray:
    """Both players' ratings of every game of ``split`` in a built dataset (from its ``games.parquet``)."""
    games = pd.read_parquet(Path(dataset_dir) / "games.parquet", columns=["split", "white_elo", "black_elo",
                                                                          "positions"])
    games = games[(games["split"] == split) & (games["positions"] > 0)]
    return np.concatenate([games["white_elo"].to_numpy(), games["black_elo"].to_numpy()]).astype(np.float64)


def make_log_prior(cfg: RatingEstimationConfig, dataset_dir: str | Path | None = None) -> np.ndarray:
    """The configured prior on the grid: ``flat``, or ``data`` (needs the dataset's training games)."""
    grid = rating_grid(cfg)
    if cfg.prior == "flat":
        return np.zeros_like(grid)
    if dataset_dir is None:
        raise ValueError("the data prior needs the dataset directory (or set rating_estimation.prior: flat)")
    return data_log_prior(training_player_ratings(dataset_dir), grid, cfg.prior_smoothing)


def opponent_log_prior(grid: np.ndarray, opp_ratings: Sequence[float], sigma: float) -> np.ndarray:
    """Log of a prior centred on the opponents' ratings: Lichess pairs players of similar strength, so a player's
    rating is near their opponent's. Several games (several opponents) multiply: ``sum_i -((r - opp_i) / sigma)^2 / 2``.

    This prior carries most of the information when the opponent's rating is known (see
    ``scripts/build_rating_prior.py`` for how much); it is wrong for games against a bot whose rating was chosen.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    opp = np.asarray(list(opp_ratings), dtype=np.float64)
    if len(opp) == 0:
        raise ValueError("no opponent ratings")
    grid = np.asarray(grid, dtype=np.float64)
    return -0.5 * (((grid[None, :] - opp[:, None]) / sigma) ** 2).sum(axis=0)


@dataclass(frozen=True)
class RatingEstimate:
    """A posterior over ratings, summarized."""

    estimate: float                 # posterior mean
    median: float
    low: float                      # central interval with probability `interval`
    high: float
    interval: float
    moves: int                      # moves the estimate is based on
    grid: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)

    def describe(self) -> str:
        return (f"about {round(self.estimate, -1):.0f} ({self.interval:.0%} range "
                f"{round(self.low, -1):.0f}-{round(self.high, -1):.0f}, from {self.moves} moves)")


def summarize(grid: np.ndarray, probabilities: np.ndarray, interval: float = 0.8, moves: int = 0) -> RatingEstimate:
    """Mean, median and central ``interval`` of a posterior on ``grid``."""
    cdf = np.cumsum(probabilities)
    tail = (1 - interval) / 2

    def quantile(q: float) -> float:
        return float(np.interp(q, cdf, grid))

    return RatingEstimate(float((grid * probabilities).sum()), quantile(0.5), quantile(tail), quantile(1 - tail),
                          interval, moves, grid, probabilities)


def estimate_from_bucket_ll(
    bucket_ll: np.ndarray,
    ratings: RatingsConfig,
    cfg: RatingEstimationConfig,
    log_prior: np.ndarray | None = None,
    temper: float | None = None,
) -> RatingEstimate:
    """Estimate from per-move bucket log-likelihoods ``(moves, K)`` (one player, any number of games)."""
    bucket_ll = np.atleast_2d(np.asarray(bucket_ll, dtype=np.float64))
    grid = rating_grid(cfg)
    total = grid_log_likelihood(bucket_ll.sum(axis=0), bucket_centers(ratings.bucket_edges), grid)
    prior = np.zeros_like(grid) if log_prior is None else log_prior
    temper = temper if temper is not None else (cfg.temper or 1.0)
    return summarize(grid, posterior(total, prior, temper), cfg.interval, moves=len(bucket_ll))


# --------------------------------------------------------------------------- games -> estimate


def game_bucket_ll(predictor: MovePredictor, game: PlayerGame, use_opponent_rating: bool = True) -> np.ndarray:
    """``(moves, K)`` bucket log-likelihoods of one player's moves in one game.

    With ``use_opponent_rating`` the opponent's rating from the PGN is given to the model
    (when present); otherwise the opponent is assumed to be as strong as each candidate rating.
    """
    opp = game.opp_rating if use_opponent_rating else None
    return move_bucket_log_likelihoods(predictor, game.boards, game.moves, opp, game.clocks, game.opp_clocks,
                                       game.base_seconds, game.increment)


@dataclass(frozen=True)
class PlayerEstimate:
    """The combined estimate over all games, and each game's own."""

    overall: RatingEstimate
    per_game: list[RatingEstimate]


def estimate_rating(
    predictor: MovePredictor,
    games: Sequence[PlayerGame],
    cfg: RatingEstimationConfig,
    log_prior: np.ndarray | None = None,
    temper: float | None = None,
    use_opponent_rating: bool = True,
) -> PlayerEstimate:
    """Estimate a player's rating from their moves in ``games`` (see the module docstring).

    Raises:
        ValueError: no games, or the games contain no moves of the player.
    """
    if not games:
        raise ValueError("no games to estimate from")
    per_game_ll = [game_bucket_ll(predictor, g, use_opponent_rating) for g in games]
    if sum(len(ll) for ll in per_game_ll) == 0:
        raise ValueError("the games contain no moves of the player")
    ratings = predictor.ratings
    per_game = [estimate_from_bucket_ll(ll, ratings, cfg, log_prior, temper) for ll in per_game_ll if len(ll)]
    overall = estimate_from_bucket_ll(np.concatenate(per_game_ll), ratings, cfg, log_prior, temper)
    return PlayerEstimate(overall, per_game)


# --------------------------------------------------------------------------- setup from a config


def fitted_temper(results_path: str | Path) -> float | None:
    """The temper fitted by ``scripts/evaluate_rating_estimation.py`` (its ``results.json``), if any."""
    try:
        return float(json.loads(Path(results_path).read_text(encoding="utf-8"))["temper"]["temper"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def setup_from_config(cfg: RatingEstimationConfig, dataset_dir: str | Path,
                      results_path: str | Path) -> tuple[RatingEstimationConfig, np.ndarray, float | None]:
    """The config, prior and temper to estimate with: the data prior if the dataset is there (else flat),
    and ``cfg.temper`` or else the fitted one (None if neither exists)."""
    if cfg.prior == "data" and not (Path(dataset_dir) / "games.parquet").is_file():
        cfg = cfg.model_copy(update={"prior": "flat"})
    return cfg, make_log_prior(cfg, dataset_dir), cfg.temper or fitted_temper(results_path)
