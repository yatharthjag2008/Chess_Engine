"""Tests for the bot's move choice."""

from __future__ import annotations

from collections import Counter

import chess
import numpy as np
import pytest

from chess_engine.bot.player import BotError, BotPlayer, check_rating, load_bot, rating_label
from chess_engine.bot.sampling import SamplingSettings, adjust_probabilities
from bot_helpers import small_predictor

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def test_greedy_bot_plays_the_models_favourite() -> None:
    predictor = small_predictor()
    bot = BotPlayer(predictor, rating=1300, sampling=SamplingSettings(temperature=0))
    board = chess.Board(KIWIPETE)
    choice = bot.choose_move(board)
    favourite = predictor.predict(board, 1300).moves[0]
    assert choice.move == favourite.move and choice.san == favourite.san
    assert choice.play_prob == 1.0 and choice.prob == pytest.approx(favourite.prob)
    assert board.fen() == KIWIPETE  # the board is not changed
    assert len(choice.candidates) == board.legal_moves.count() and choice.wdl is not None


def test_sampled_moves_follow_the_adjusted_probabilities() -> None:
    predictor = small_predictor()
    board = chess.Board()
    settings = SamplingSettings(temperature=1.5, top_k=6)
    bot = BotPlayer(predictor, rating=1500, sampling=settings, seed=1)
    counts = Counter(bot.choose_move(board).move for _ in range(500))
    candidates = predictor.predict(board, 1500).moves
    expected = adjust_probabilities(np.array([c.prob for c in candidates]), settings)
    for candidate, p in zip(candidates, expected):
        assert counts[candidate.move] / 500 == pytest.approx(p, abs=0.07)
    assert len(counts) <= 6


def test_seeded_bots_repeat_and_ratings_differ() -> None:
    predictor = small_predictor()
    board = chess.Board(KIWIPETE)
    first = [BotPlayer(predictor, 1500, seed=7).choose_move(board).move for _ in range(3)]
    again = [BotPlayer(predictor, 1500, seed=7).choose_move(board).move for _ in range(3)]
    assert first == again
    low = BotPlayer(predictor, 900, SamplingSettings(temperature=0)).choose_move(board)
    high = BotPlayer(predictor, 2300, SamplingSettings(temperature=0)).choose_move(board)
    assert low.candidates[0].prob != pytest.approx(high.candidates[0].prob)  # the rating reaches the model


def test_context_is_passed_to_the_model() -> None:
    predictor = small_predictor()
    board = chess.Board()
    bot = BotPlayer(predictor, 1500, SamplingSettings(temperature=0))
    calm = bot.choose_move(board, opp_rating=1500, clock=180, opp_clock=180)
    rushed = bot.choose_move(board, opp_rating=2400, clock=3, opp_clock=170, base_seconds=180, increment=2)
    assert calm.candidates[0].prob != pytest.approx(rushed.candidates[0].prob)


def test_forced_move_and_game_over() -> None:
    bot = BotPlayer(small_predictor(), 1500)
    forced = chess.Board("k7/8/1K6/8/8/8/8/1R6 b - - 0 1")  # Kb8 is the only legal move
    choice = bot.choose_move(forced)
    assert choice.san == "Kb8" and choice.play_prob == 1.0 and choice.prob == pytest.approx(1.0)
    mated = chess.Board("7k/8/8/8/8/8/6R1/K6R b - - 0 1")  # checkmate
    with pytest.raises(ValueError, match="game is over"):
        bot.choose_move(mated)
    with pytest.raises(BotError, match="between"):
        BotPlayer(small_predictor(), 0)


def test_rating_checks_and_labels() -> None:
    from bot_helpers import RATINGS

    assert check_rating(400) == 400 and check_rating(3000) == 3000
    for bad in (399, 3001):
        with pytest.raises(BotError):
            check_rating(bad)
    assert rating_label(1450, RATINGS) == "1400-1599"
    assert rating_label(600, RATINGS) == "800-999" and rating_label(2600, RATINGS) == "2200+"


def test_describe() -> None:
    predictor = small_predictor()
    assert "plays like 1400-1599" in BotPlayer(predictor, 1450).describe()
    assert "as often as humans" in BotPlayer(predictor, 1450).describe()
    assert "likeliest move" in BotPlayer(predictor, 1450, SamplingSettings(temperature=0)).describe()
    text = BotPlayer(predictor, 1450, SamplingSettings(temperature=0.7, top_k=5)).describe()
    assert "temperature 0.7" in text and "top-5" in text


def test_load_bot_from_a_model_file(tmp_path, default_config) -> None:
    from chess_engine.model.inference import save_model

    predictor = small_predictor()
    path = save_model(predictor.model, tmp_path / "m.pt", predictor.ratings)
    bot = load_bot(default_config, rating=1800, model_path=path, seed=3)
    assert bot.rating == 1800 and bot.sampling == SamplingSettings()
    assert bot.choose_move(chess.Board()).move in chess.Board().legal_moves
    greedy = load_bot(default_config, model_path=path, sampling=SamplingSettings(temperature=0))
    assert greedy.rating == default_config.bot.rating and greedy.sampling.greedy
    with pytest.raises(BotError, match="not found"):
        load_bot(default_config, model_path=tmp_path / "missing.pt")
    with pytest.raises(BotError, match="between"):
        load_bot(default_config, rating=5000, model_path=path)
