"""Choose bot moves by sampling from the model at a target rating.

    bot = BotPlayer(predictor, rating=1200)                 # plays like a 1200
    choice = bot.choose_move(board)
    board.push(choice.move)
    choice.san, choice.prob                                 # e.g. "Nf3", 0.31 (how often 1200s play it)
    choice.candidates[:3]                                   # the model's three likeliest moves

The opponent's rating and both clocks are passed to the model too (they change
what humans play), so a bot in time trouble plays like a human in time trouble.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from chess_engine.bot.sampling import SamplingSettings, adjust_probabilities, sample_index
from chess_engine.model.inference import MovePredictor, MoveProbability


@dataclass(frozen=True)
class BotMove:
    """The move the bot chose and why."""

    move: chess.Move
    san: str
    prob: float                          # the model's probability of this move (how often humans play it)
    play_prob: float                     # the chance the bot had of choosing it (after the sampling settings)
    candidates: list[MoveProbability]    # every legal move, likeliest first, with the model's probabilities
    wdl: tuple[float, float, float] | None = None  # model's (win, draw, loss) for the bot before the move


class BotPlayer:
    """A bot that plays like humans of one rating.

    Args:
        predictor: The model wrapped for ``chess.Board`` positions.
        rating: The rating the bot imitates.
        sampling: How to pick among the model's moves (default: exactly as often as humans do).
        seed: Seed for reproducible move choices (None = different every game).
    """

    def __init__(self, predictor: MovePredictor, rating: int, sampling: SamplingSettings | None = None,
                 seed: int | None = None) -> None:
        if rating <= 0:
            raise ValueError("rating must be positive")
        self.predictor = predictor
        self.rating = rating
        self.sampling = sampling or SamplingSettings()
        self.rng = np.random.default_rng(seed)

    def choose_move(
        self,
        board: chess.Board,
        opp_rating: int | None = None,
        clock: float | None = None,
        opp_clock: float | None = None,
        base_seconds: int = 180,
        increment: int = 0,
    ) -> BotMove:
        """Pick a move for the side to move in ``board`` (the board is not changed).

        Args:
            opp_rating: The opponent's rating (default: the bot's own).
            clock / opp_clock: Seconds left for the bot / its opponent (default: the full base time).
            base_seconds / increment: The game's time control.

        Raises:
            ValueError: the game is over (no legal moves).
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(f"no legal moves in {board.fen()} (the game is over)")
        prediction = self.predictor.predict(
            board, self.rating, opp_ratings=opp_rating if opp_rating is not None else self.rating,
            clocks=clock, opp_clocks=opp_clock, base_seconds=base_seconds, increment=increment,
        )
        candidates = prediction.moves
        if len(candidates) == 1:
            only = candidates[0]
            return BotMove(only.move, only.san, only.prob, 1.0, candidates, prediction.wdl)
        play = adjust_probabilities(np.array([c.prob for c in candidates]), self.sampling)
        index = sample_index(play, self.rng)
        chosen = candidates[index]
        return BotMove(chosen.move, chosen.san, chosen.prob, float(play[index]), candidates, prediction.wdl)
