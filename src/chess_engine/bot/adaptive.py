"""Adaptive difficulty: the bot follows your level, between games or live within a game.

* **off**: the bot plays at the rating you choose.
* **between**: your rating is kept like an Elo rating, updated after every rated game (one without hints) from
  the result against the bot's rating; the next bot plays at your rating, so you should win about half the time.
* **live**: during a game the bot watches its own winning chances (the model's value head, averaged over its last
  few moves) and becomes a step stronger when it is losing or a step weaker when it is winning, never further than
  a cap from where the game started.
"""

from __future__ import annotations

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
