"""Adaptive difficulty (step 18): Elo between games, the profile store, live adjustment."""

from __future__ import annotations

import pytest

from chess_engine.bot.adaptive import (
    HISTORY,
    LiveAdjuster,
    ProfileStore,
    bot_rating_for,
    expected_score,
    k_for,
    update_rating,
)
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


def test_profile_store_records_games(tmp_path) -> None:
    store = ProfileStore(tmp_path / "sub" / "adaptive.json", CFG)
    fresh = store.load()
    assert fresh.rating == CFG.start_rating and fresh.games == 0 and fresh.history == []
    assert fresh.bot_rating(CFG) == 1200
    after = store.record(1200, 1.0, "between")
    assert after.rating == 1200 + CFG.k_new / 2 and after.games == 1
    again = ProfileStore(tmp_path / "sub" / "adaptive.json", CFG).load()   # read back from the file
    assert again.rating == after.rating and again.history[0].score == 1.0 and again.history[0].mode == "between"
    assert again.history[0].rating_before == 1200 and again.history[0].bot_rating == 1200
    for _ in range(HISTORY + 5):
        store.record(1300, 0.5, "live")
    assert len(store.load().history) == HISTORY and store.load().games == HISTORY + 6


def test_profile_store_reset_and_bad_file(tmp_path) -> None:
    path = tmp_path / "adaptive.json"
    path.write_text("{not json", encoding="utf-8")
    store = ProfileStore(path, CFG)
    assert store.load().rating == CFG.start_rating, "an unreadable file starts fresh"
    store.record(1200, 0, "between")
    assert store.reset(1650).rating == 1650 and store.load().games == 0
    assert store.reset(9999).rating == CFG.max_rating and store.reset().rating == CFG.start_rating
    assert not (tmp_path / "adaptive.json.tmp").exists()


LOSING = (0.1, 0.2, 0.7)      # the bot expects 0.2
WINNING = (0.8, 0.1, 0.1)     # the bot expects 0.85
EVEN = (0.35, 0.3, 0.35)


def test_live_waits_for_the_opening_then_steps_up_when_losing() -> None:
    live = LiveAdjuster(1500, CFG)
    changes = [live.observe(LOSING) for _ in range(CFG.live_after)]
    assert changes == [None] * CFG.live_after and live.rating == 1500, "no change during the opening"
    assert live.observe(LOSING) == 1550
    assert [live.observe(LOSING) for _ in range(CFG.live_every - 1)] == [None] * (CFG.live_every - 1)
    assert live.observe(LOSING) == 1600, "the next step after live_every bot moves"
    assert [c.rating for c in live.changes] == [1550, 1600] and live.changes[0].expected == pytest.approx(0.2)
    assert live.played_at[:CFG.live_after + 1] == [1500] * (CFG.live_after + 1)


def test_live_is_capped_and_steps_down_when_winning() -> None:
    live = LiveAdjuster(1500, CFG)
    for _ in range(100):
        live.observe(LOSING)
    assert live.rating == 1500 + CFG.live_cap
    down = LiveAdjuster(900, CFG)
    for _ in range(100):
        down.observe(WINNING)
    assert down.rating == CFG.min_rating, "the configured range wins over the cap"


def test_live_even_positions_and_no_value_head() -> None:
    live = LiveAdjuster(1500, CFG)
    for _ in range(20):
        assert live.observe(EVEN) is None
    blind = LiveAdjuster(1500, CFG)
    for _ in range(20):
        assert blind.observe(None) is None
    assert blind.mean_rating == 1500 and len(blind.played_at) == 20


def test_live_averages_recent_moves_and_reports_the_mean_rating() -> None:
    live = LiveAdjuster(1500, CFG)
    for _ in range(CFG.live_after):
        live.observe(EVEN)
    assert live.observe(LOSING) is None, "one bad move is not enough: the window is averaged"
    assert LiveAdjuster(1500, CFG).mean_rating == 1500
    for _ in range(10):
        live.observe(LOSING)
    assert 1500 < live.mean_rating < live.rating
