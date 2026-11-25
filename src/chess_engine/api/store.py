"""In-memory storage of the games being played through the API.

Games live in memory only (a restart forgets them; finished games can be saved
as PGN). Each game gets a random, unguessable id. To keep memory bounded,
games idle for ``idle_minutes`` are dropped, and when ``max_games`` are stored the
least recently used one makes room for a new one.

The store is thread-safe: FastAPI runs plain (non-async) endpoints on a thread pool.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from chess_engine.bot.clock import ChessClock
from chess_engine.bot.game import Game
from chess_engine.bot.player import BotMove, BotPlayer


class GameNotFoundError(KeyError):
    """No game with this id (never existed, or dropped after being idle)."""


@dataclass
class Session:
    """One game against the bot."""

    id: str
    game: Game
    bot: BotPlayer
    human_color: chess.Color
    your_rating: int | None
    clock: ChessClock | None = None
    last_bot_move: BotMove | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_used: float = 0.0


class GameStore:
    """Sessions by id, least recently used first.

    Args:
        max_games: Most sessions kept at once.
        idle_seconds: Sessions untouched for this long are dropped.
        now: Time source (injectable for tests).
    """

    def __init__(self, max_games: int = 200, idle_seconds: float = 7200,
                 now: Callable[[], float] = time.monotonic) -> None:
        if max_games < 1 or idle_seconds <= 0:
            raise ValueError("max_games must be >= 1 and idle_seconds > 0")
        self.max_games = max_games
        self.idle_seconds = idle_seconds
        self._now = now
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)

    @staticmethod
    def new_id() -> str:
        return secrets.token_urlsafe(12)

    def add(self, session: Session) -> Session:
        """Store a session (dropping idle ones, and the least recently used one if the store is full)."""
        with self._lock:
            self._expire()
            while len(self._sessions) >= self.max_games:
                self._sessions.popitem(last=False)
            session.last_used = self._now()
            self._sessions[session.id] = session
        return session

    def get(self, game_id: str) -> Session:
        """The session with this id (marks it as used).

        Raises:
            GameNotFoundError: unknown or expired id.
        """
        with self._lock:
            self._expire()
            session = self._sessions.get(game_id)
            if session is None:
                raise GameNotFoundError(game_id)
            session.last_used = self._now()
            self._sessions.move_to_end(game_id)
            return session

    def remove(self, game_id: str) -> None:
        with self._lock:
            if self._sessions.pop(game_id, None) is None:
                raise GameNotFoundError(game_id)

    def _expire(self) -> None:
        cutoff = self._now() - self.idle_seconds
        while self._sessions:
            oldest = next(iter(self._sessions.values()))
            if oldest.last_used > cutoff:
                break
            self._sessions.popitem(last=False)
