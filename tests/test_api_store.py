"""Tests for the in-memory game store."""

from __future__ import annotations

import threading

import chess
import pytest

from chess_engine.api.store import GameNotFoundError, GameStore, Session
from chess_engine.bot.game import Game
from chess_engine.bot.player import BotPlayer
from bot_helpers import small_predictor

BOT = BotPlayer(small_predictor(), 1500)


def _session(game_id: str) -> Session:
    return Session(game_id, Game(), BOT, chess.WHITE, None)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_add_get_remove() -> None:
    store = GameStore()
    store.add(_session("a"))
    assert store.get("a").id == "a" and len(store) == 1
    store.remove("a")
    with pytest.raises(GameNotFoundError):
        store.get("a")
    with pytest.raises(GameNotFoundError):
        store.remove("a")


def test_ids_are_random_and_url_safe() -> None:
    ids = {GameStore.new_id() for _ in range(100)}
    assert len(ids) == 100 and all(len(i) >= 16 and i.replace("-", "").replace("_", "").isalnum() for i in ids)


def test_full_store_drops_the_least_recently_used() -> None:
    clock = Clock()
    store = GameStore(max_games=2, now=clock)
    store.add(_session("a"))
    clock.t = 1
    store.add(_session("b"))
    clock.t = 2
    store.get("a")  # a is now more recent than b
    store.add(_session("c"))
    assert len(store) == 2
    store.get("a")
    store.get("c")
    with pytest.raises(GameNotFoundError):
        store.get("b")


def test_idle_games_expire() -> None:
    clock = Clock()
    store = GameStore(idle_seconds=60, now=clock)
    store.add(_session("a"))
    clock.t = 30
    store.add(_session("b"))
    clock.t = 70
    with pytest.raises(GameNotFoundError):
        store.get("a")
    assert store.get("b").id == "b" and len(store) == 1
    with pytest.raises(ValueError):
        GameStore(max_games=0)


def test_concurrent_adds() -> None:
    store = GameStore(max_games=1000)
    threads = [threading.Thread(target=lambda i=i: store.add(_session(f"g{i}"))) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 50
