"""Adaptive difficulty (step 18): Elo between games, the profile store, live adjustment."""

from __future__ import annotations

import pytest

from chess_engine.bot.adaptive import bot_rating_for, expected_score, k_for, update_rating
from chess_engine.config import AdaptiveConfig

CFG = AdaptiveConfig()


def test_expected_score() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)        # 400 points: 10 to 1
    assert expected_score(1500, 1900) == pytest.approx(1 / 11)


def test_k_is_larger_while_new() -> None:
    assert k_for(0, CFG) == k_for(CFG.new_games - 1, CFG) == CFG.k_new
    assert k_for(CFG.new_games, CFG) == CFG.k_factor


def test_update_rating() -> None:
    assert update_rating(1500, 1500, 1, 10, CFG) == 1500 + CFG.k_factor / 2
    assert update_rating(1500, 1500, 0.5, 10, CFG) == 1500
    assert update_rating(1500, 1500, 0, 0, CFG) == 1500 - CFG.k_new / 2
    assert update_rating(1500, 1900, 1, 10, CFG) == pytest.approx(1500 + 40 * 10 / 11)   # an upset counts more
    assert update_rating(CFG.max_rating, 800, 1, 0, CFG) == CFG.max_rating                # kept in range
    assert update_rating(CFG.min_rating, 2400, 0, 0, CFG) == CFG.min_rating
    with pytest.raises(ValueError, match="score"):
        update_rating(1500, 1500, 2, 0, CFG)


def test_bot_rating_for() -> None:
    assert bot_rating_for(1512, CFG) == 1500 and bot_rating_for(1526, CFG) == 1550
    assert bot_rating_for(100, CFG) == CFG.min_rating and bot_rating_for(5000, CFG) == CFG.max_rating


def test_repeated_results_move_the_rating_towards_the_players_level() -> None:
    # A player who beats 1200 bots and loses to 1600 bots: the bot's rating settles in between.
    player, games = float(CFG.start_rating), 0
    for _ in range(60):
        bot = bot_rating_for(player, CFG)
        score = 1.0 if bot < 1400 else 0.0 if bot > 1400 else 0.5
        player = update_rating(player, bot, score, games, CFG)
        games += 1
    assert 1300 <= bot_rating_for(player, CFG) <= 1500
