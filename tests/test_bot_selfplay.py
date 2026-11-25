"""Tests for bot-vs-bot games and match scoring."""

from __future__ import annotations

import math

import chess
import pytest

from chess_engine.bot.game import Outcome
from chess_engine.bot.player import BotPlayer
from chess_engine.bot.selfplay import MatchResult, play_game, play_match
from bot_helpers import small_predictor


@pytest.fixture(scope="module")
def bots():
    predictor = small_predictor()
    return BotPlayer(predictor, 1000, seed=1), BotPlayer(predictor, 2000, seed=2)


def test_game_runs_to_an_end(bots) -> None:
    a, b = bots
    game = play_game(a, b, max_plies=40)
    assert game.is_over and len(game.sans) <= 40
    assert game.white == "Bot 1000" and game.black_elo == 2000
    if len(game.sans) == 40:
        assert game.outcome.termination.startswith("adjudicated draw")


def test_match_alternates_colours_and_counts(bots) -> None:
    a, b = bots
    seen = []
    result = play_match(a, b, games=4, max_plies=30, progress=lambda i, r: seen.append(i))
    assert seen == [1, 2, 3, 4] and result.games == 4
    assert result.wins_a + result.wins_b + result.draws == 4 and len(result.plies) == 4
    assert sum(result.terminations.values()) == 4
    assert "2000 vs 1000" in result.summary()


def test_scoring() -> None:
    r = MatchResult(1000, 2000)
    assert math.isnan(r.score_b)
    r.add(Outcome("1-0", "checkmate", chess.WHITE), b_was_white=True, plies=50)   # B wins as White
    r.add(Outcome("1-0", "checkmate", chess.WHITE), b_was_white=False, plies=60)  # A wins as White
    r.add(Outcome("1/2-1/2", "stalemate", None), b_was_white=True, plies=70)
    r.add(Outcome("0-1", "resignation", chess.BLACK), b_was_white=False, plies=40)  # B wins as Black
    assert (r.wins_b, r.wins_a, r.draws) == (2, 1, 1)
    assert r.score_b == pytest.approx(2.5 / 4)
    assert r.elo_difference() == pytest.approx(-400 * math.log10(1 / 0.625 - 1))
    low, high = r.score_interval()
    assert low < 0.625 < high
    clean = MatchResult(1, 2, games=2, wins_b=2)
    assert clean.elo_difference() == math.inf
