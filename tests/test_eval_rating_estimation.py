"""Tests for rating estimation from per-move likelihoods."""

from __future__ import annotations

import numpy as np
import pytest

from chess_engine.config import RatingEstimationConfig, RatingsConfig
from chess_engine.eval.rating_estimation import (
    bucket_centers,
    estimate_from_bucket_ll,
    grid_log_likelihood,
    posterior,
    rating_grid,
    summarize,
)

RATINGS = RatingsConfig(bucket_edges=[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200])
CFG = RatingEstimationConfig()


def test_bucket_centers_and_grid() -> None:
    assert bucket_centers(RATINGS.bucket_edges).tolist() == [900, 1100, 1300, 1500, 1700, 1900, 2100, 2300]
    assert bucket_centers([1000]).tolist() == [1100]
    grid = rating_grid(CFG)
    assert grid[0] == 600 and grid[-1] == 2800 and np.all(np.diff(grid) == 10)


def test_interpolation_is_linear_between_centres_and_flat_outside() -> None:
    centers = np.array([900.0, 1100.0, 1300.0])
    ll = np.array([-10.0, -6.0, -8.0])
    out = grid_log_likelihood(ll, centers, np.array([600, 900, 1000, 1100, 1250, 1600]))
    assert out.tolist() == pytest.approx([-10, -10, -8, -6, -7.5, -8])
    two = grid_log_likelihood(np.stack([ll, ll * 2]), centers, np.array([1000.0]))
    assert two.shape == (2, 1) and two[:, 0].tolist() == pytest.approx([-8, -16])


def test_posterior_normalizes_and_tempering_widens() -> None:
    grid = np.linspace(0, 1, 11)
    ll = -50 * (grid - 0.5) ** 2
    sharp = posterior(ll, np.zeros(11))
    wide = posterior(ll, np.zeros(11), temper=4)
    assert sharp.sum() == pytest.approx(1) and wide.sum() == pytest.approx(1)
    assert sharp.argmax() == 5 and wide.max() < sharp.max()
    with pytest.raises(ValueError):
        posterior(ll, np.zeros(11), temper=0)
    huge = posterior(ll * 1e6, np.zeros(11))  # no overflow
    assert np.isfinite(huge).all() and huge[5] == pytest.approx(1)


def test_summarize_a_symmetric_posterior() -> None:
    grid = np.arange(1000, 2001, 10, dtype=float)
    probs = np.exp(-0.5 * ((grid - 1500) / 100) ** 2)
    est = summarize(grid, probs / probs.sum(), interval=0.8, moves=40)
    assert est.estimate == pytest.approx(1500, abs=1) and est.median == pytest.approx(1500, abs=6)
    assert est.low == pytest.approx(1500 - 128, abs=12) and est.high == pytest.approx(1500 + 128, abs=12)
    assert "about 1500" in est.describe() and "from 40 moves" in est.describe()


def test_estimate_follows_the_best_explaining_bucket() -> None:
    # 30 moves that players around 1700 (bucket 4) play much more often than other ratings
    per_move = np.full((30, 8), -3.0)
    per_move[:, 4] = -2.0
    per_move[:, 3] = per_move[:, 5] = -2.4
    est = estimate_from_bucket_ll(per_move, RATINGS, CFG)
    assert 1600 < est.estimate < 1800 and est.low < est.estimate < est.high and est.moves == 30
    wider = estimate_from_bucket_ll(per_move, RATINGS, CFG, temper=10)
    assert (wider.high - wider.low) > (est.high - est.low)
    prior = np.where(rating_grid(CFG) < 1000, 0.0, -1e9)  # a prior that rules out everything above 1000
    assert estimate_from_bucket_ll(per_move, RATINGS, CFG, log_prior=prior).estimate < 1000


def test_data_prior_follows_the_rating_distribution() -> None:
    from chess_engine.eval.rating_estimation import PRIOR_FLOOR, data_log_prior

    grid = rating_grid(CFG)
    rng = np.random.default_rng(0)
    players = rng.normal(1600, 250, 20_000)
    prior = np.exp(data_log_prior(players, grid, smoothing=50))
    assert prior.sum() == pytest.approx(1)
    assert abs(grid[prior.argmax()] - 1600) <= 60
    assert prior.min() >= PRIOR_FLOOR / len(grid) * 0.999  # nothing is ruled out
    assert prior[grid == 1600][0] > 20 * prior[grid == 2700][0]
    rough = np.exp(data_log_prior(players, grid, smoothing=0))
    assert np.abs(np.diff(rough)).sum() > np.abs(np.diff(prior)).sum()  # smoothing smooths
    with pytest.raises(ValueError):
        data_log_prior(np.array([]), grid)


def test_prior_from_a_dataset(tiny_dataset) -> None:
    from chess_engine.eval.rating_estimation import make_log_prior, training_player_ratings

    ratings = training_player_ratings(tiny_dataset)
    assert len(ratings) > 100 and ratings.min() >= 800
    prior = make_log_prior(CFG, tiny_dataset)
    assert prior.shape == rating_grid(CFG).shape and np.exp(prior).sum() == pytest.approx(1)
    flat = make_log_prior(CFG.model_copy(update={"prior": "flat"}))
    assert (flat == 0).all()
    with pytest.raises(ValueError, match="dataset"):
        make_log_prior(CFG)


def test_move_bucket_log_likelihoods_match_the_predictor() -> None:
    import chess

    from bot_helpers import small_predictor
    from chess_engine.eval.rating_estimation import move_bucket_log_likelihoods

    predictor = small_predictor()
    board = chess.Board()
    boards, moves = [], []
    for san in ("e4", "e5", "Nf3", "Nc6"):
        move = board.parse_san(san)
        boards.append(board.copy())
        moves.append(move)
        board.push(move)
    ll = move_bucket_log_likelihoods(predictor, boards, moves, opp_ratings=1500, chunk=5)
    assert ll.shape == (4, 8) and (ll < 0).all()
    # Same numbers as asking the predictor directly, bucket by bucket (bucket b at its lowest rating).
    for b, rating in enumerate(predictor.ratings.bucket_edges):
        direct = predictor.predict(boards[2], rating, opp_ratings=1500).as_dict()["g1f3"]
        assert ll[2, b] == pytest.approx(np.log(direct), abs=1e-4)
    assert not np.allclose(ll[:, 0], ll[:, 7])  # the rating matters (random modulation switched on)

    unknown = move_bucket_log_likelihoods(predictor, boards, moves)  # opponent = candidate rating
    direct = predictor.predict(boards[0], 2300, opp_ratings=2300).as_dict()["e2e4"]
    assert unknown[0, 7] == pytest.approx(np.log(direct), abs=1e-4)
    assert move_bucket_log_likelihoods(predictor, [], []).shape == (0, 8)
    with pytest.raises(ValueError, match="not legal"):
        move_bucket_log_likelihoods(predictor, [chess.Board()], [chess.Move.from_uci("e2e5")])


def test_estimate_rating_from_pgn_games() -> None:
    from bot_helpers import small_predictor
    from chess_engine.eval.player_games import player_games_from_pgn
    from chess_engine.eval.rating_estimation import estimate_rating, game_bucket_ll

    pgn = (FIXTURE_DIR / "lichess_real.pgn").read_text(encoding="utf-8")
    games = player_games_from_pgn(pgn)[:4]
    predictor = small_predictor()
    result = estimate_rating(predictor, games, CFG)
    assert len(result.per_game) == 4
    assert result.overall.moves == sum(len(g) for g in games)
    assert CFG.rating_min <= result.overall.low <= result.overall.estimate <= result.overall.high <= CFG.rating_max
    # more games -> at least as narrow a range as the first game alone (likelihoods add up)
    one = result.per_game[0]
    assert (result.overall.high - result.overall.low) <= (one.high - one.low) + 1e-6
    # the opponent's rating changes the evidence (it is a model input)
    assert not np.allclose(game_bucket_ll(predictor, games[0], True), game_bucket_ll(predictor, games[0], False))
    with pytest.raises(ValueError):
        estimate_rating(predictor, [], CFG)


FIXTURE_DIR = __import__("pathlib").Path(__file__).parent / "fixtures"
