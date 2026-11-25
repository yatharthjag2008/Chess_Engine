"""Bot-vs-bot games: a quick check that the bots finish games and that a higher rating plays better.

    result = play_match(bot_1100, bot_1900, games=40)   # colours alternate
    result.score_b                                      # points of bot_1900 per game (0..1)

Games have no clock (both sides are told they have the full base time) and are
adjudicated as a draw after ``max_plies``. The full Elo-dial validation, including
games against Stockfish levels, is step 11.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from chess_engine.bot.game import Game, Outcome
from chess_engine.bot.player import BotPlayer


def play_game(white: BotPlayer, black: BotPlayer, max_plies: int = 300, fen: str | None = None) -> Game:
    """One game between two bots (each is told the other's rating)."""
    game = Game(f"Bot {white.rating}", f"Bot {black.rating}", white.rating, black.rating, fen=fen,
                event="Bot vs. bot")
    while not game.is_over:
        if len(game.sans) >= max_plies:
            game._end(None, f"adjudicated draw after {max_plies} plies")  # noqa: SLF001
            break
        mover, other = (white, black) if game.turn == chess.WHITE else (black, white)
        game.push(mover.choose_move(game.board, opp_rating=other.rating).move)
    return game


@dataclass
class MatchResult:
    """Results of a match between bot A and bot B."""

    rating_a: int
    rating_b: int
    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    terminations: Counter = field(default_factory=Counter)
    plies: list[int] = field(default_factory=list)

    @property
    def score_b(self) -> float:
        """Bot B's points per game (win 1, draw 0.5)."""
        return (self.wins_b + 0.5 * self.draws) / self.games if self.games else float("nan")

    def elo_difference(self) -> float:
        """B's performance minus A's in Elo, from the score (infinite for a whitewash)."""
        s = self.score_b
        if s <= 0:
            return -math.inf
        if s >= 1:
            return math.inf
        return -400 * math.log10(1 / s - 1)

    def score_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Approximate 95% interval of B's score per game (normal approximation, per-game variance)."""
        if not self.games:
            return (float("nan"), float("nan"))
        s = self.score_b
        mean_sq = (self.wins_b + 0.25 * self.draws) / self.games
        se = math.sqrt(max(mean_sq - s * s, 0.0) / self.games)
        return max(0.0, s - z * se), min(1.0, s + z * se)

    def add(self, outcome: Outcome, b_was_white: bool, plies: int) -> None:
        self.games += 1
        self.terminations[outcome.termination] += 1
        self.plies.append(plies)
        if outcome.winner is None:
            self.draws += 1
        elif (outcome.winner == chess.WHITE) == b_was_white:
            self.wins_b += 1
        else:
            self.wins_a += 1

    def summary(self) -> str:
        low, high = self.score_interval()
        return (f"{self.rating_b} vs {self.rating_a}: +{self.wins_b} ={self.draws} -{self.wins_a} in {self.games} games, "
                f"score {self.score_b:.1%} (95% {low:.0%}-{high:.0%}), Elo difference {self.elo_difference():+.0f}, "
                f"average {sum(self.plies) / max(len(self.plies), 1):.0f} plies")


def play_match(bot_a: BotPlayer, bot_b: BotPlayer, games: int = 20, max_plies: int = 300,
               progress: Callable[[int, MatchResult], None] | None = None) -> MatchResult:
    """``games`` games between two bots, alternating colours (B is White in odd-numbered games)."""
    result = MatchResult(bot_a.rating, bot_b.rating)
    for i in range(games):
        b_white = i % 2 == 0
        white, black = (bot_b, bot_a) if b_white else (bot_a, bot_b)
        game = play_game(white, black, max_plies)
        result.add(game.outcome, b_white, len(game.sans))
        if progress is not None:
            progress(i + 1, result)
    return result
