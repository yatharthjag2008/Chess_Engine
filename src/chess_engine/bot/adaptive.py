"""Adaptive difficulty: the bot follows your level, between games or live within a game.

* **off**: the bot plays at the rating you choose.
* **between**: your rating is kept like an Elo rating, updated after every rated game (one without hints) from
  the result against the bot's rating; the next bot plays at your rating, so you should win about half the time.
* **live**: during a game the bot watches its own winning chances (the model's value head, averaged over its last
  few moves) and becomes a step stronger when it is losing or a step weaker when it is winning, never further than
  a cap from where the game started.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from chess_engine.config import AdaptiveConfig


# --------------------------------------------------------------------------- between games


def expected_score(player: float, bot: float) -> float:
    """Your expected score (0..1) against a bot rated ``bot`` (Elo formula)."""
    return 1.0 / (1.0 + 10 ** ((bot - player) / 400.0))


def k_for(games_played: int, cfg: AdaptiveConfig) -> float:
    """The Elo K: larger while your rating is new, so the bot finds your level quickly."""
    return cfg.k_new if games_played < cfg.new_games else cfg.k_factor


def update_rating(player: float, bot: float, score: float, games_played: int, cfg: AdaptiveConfig) -> float:
    """Your rating after scoring ``score`` (1 win, 0.5 draw, 0 loss) against a bot rated ``bot``, kept in range."""
    if not 0 <= score <= 1:
        raise ValueError(f"score must be between 0 and 1, got {score}")
    new = player + k_for(games_played, cfg) * (score - expected_score(player, bot))
    return min(max(new, cfg.min_rating), cfg.max_rating)


def bot_rating_for(player: float, cfg: AdaptiveConfig) -> int:
    """The bot's rating for your next game: your rating, rounded and kept within the bot's range."""
    rounded = int(round(player / cfg.round_to) * cfg.round_to)
    return min(max(rounded, cfg.min_rating), cfg.max_rating)


# --------------------------------------------------------------------------- your profile

HISTORY = 50   # games kept in the profile


@dataclass
class GameRecord:
    """One rated game's effect on your rating."""

    date: str
    bot_rating: int
    score: float                  # 1 win, 0.5 draw, 0 loss (yours)
    rating_before: float
    rating_after: float
    mode: str


@dataclass
class Profile:
    """Your rating for adaptive difficulty, and your latest rated games."""

    rating: float
    games: int = 0
    history: list[GameRecord] = field(default_factory=list)

    def bot_rating(self, cfg: AdaptiveConfig) -> int:
        return bot_rating_for(self.rating, cfg)

    def to_dict(self) -> dict:
        return asdict(self)


class ProfileStore:
    """The profile in a JSON file (a missing or unreadable file starts a fresh profile); safe across threads."""

    def __init__(self, path: str | Path, cfg: AdaptiveConfig) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self._lock = threading.Lock()

    def fresh(self, rating: float | None = None) -> Profile:
        return Profile(float(rating if rating is not None else self.cfg.start_rating))

    def load(self) -> Profile:
        with self._lock:
            return self._load()

    def _load(self) -> Profile:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            history = [GameRecord(**g) for g in data.get("history", [])]
            return Profile(float(data["rating"]), int(data.get("games", 0)), history)
        except (OSError, ValueError, KeyError, TypeError):
            return self.fresh()

    def _save(self, profile: Profile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(profile.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def record(self, bot_rating: int, score: float, mode: str) -> Profile:
        """Update your rating with one rated game's result and save it."""
        with self._lock:
            profile = self._load()
            before = profile.rating
            profile.rating = update_rating(before, bot_rating, score, profile.games, self.cfg)
            profile.games += 1
            profile.history.append(GameRecord(datetime.date.today().isoformat(), bot_rating, score,
                                              round(before, 1), round(profile.rating, 1), mode))
            profile.history = profile.history[-HISTORY:]
            self._save(profile)
            return profile

    def reset(self, rating: float | None = None) -> Profile:
        """Start again (at ``rating``, else the configured start rating)."""
        with self._lock:
            if rating is not None:
                rating = min(max(rating, self.cfg.min_rating), self.cfg.max_rating)
            profile = self.fresh(rating)
            self._save(profile)
            return profile
