"""Choose bot moves by sampling from the model at a target rating.

    bot = BotPlayer(predictor, rating=1200)                 # plays like a 1200
    choice = bot.choose_move(board)
    board.push(choice.move)
    choice.san, choice.prob                                 # e.g. "Nf3", 0.31 (how often 1200s play it)
    choice.candidates[:3]                                   # the model's three likeliest moves

The opponent's rating and both clocks are passed to the model too (they change
what humans play), so a bot in time trouble plays like a human in time trouble.

Building a bot from the config (``bot:`` section) or a model file::

    bot = load_bot(cfg)                              # cfg.bot.model at cfg.bot.rating
    bot = load_bot(cfg, rating=1800, model_path="checkpoints/my_run/best_model.pt")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from chess_engine.bot.sampling import SamplingSettings, adjust_probabilities, sample_index
from chess_engine.config import BotConfig, Config, RatingsConfig
from chess_engine.model.inference import MovePredictor, MoveProbability, load_model

#: Ratings a bot accepts. Below the lowest bucket edge there is no training data; far above the
#: open-ended top bucket the model only knows "2200+", so very high ratings are refused, not faked.
MIN_RATING = 400
MAX_RATING = 3000


class BotError(Exception):
    """A bot cannot be created (missing model file, rating out of range, ...)."""


def check_rating(rating: int) -> int:
    """``rating`` if it is in ``[MIN_RATING, MAX_RATING]``; raises :class:`BotError` otherwise."""
    if not MIN_RATING <= rating <= MAX_RATING:
        raise BotError(f"rating must be between {MIN_RATING} and {MAX_RATING}, got {rating}")
    return rating


def rating_label(rating: int, ratings: RatingsConfig) -> str:
    """The training bucket a rating falls in, e.g. ``1400-1599`` (ratings below the lowest edge use bucket 0)."""
    index = ratings.bucket_index(rating) if rating >= ratings.bucket_edges[0] else 0
    return ratings.bucket_labels()[index]


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
        self.predictor = predictor
        self.rating = check_rating(rating)
        self.sampling = sampling or SamplingSettings()
        self.rng = np.random.default_rng(seed)

    @property
    def bucket(self) -> str:
        """The rating bucket whose players the bot imitates, e.g. ``1400-1599``."""
        return rating_label(self.rating, self.predictor.ratings)

    def describe(self) -> str:
        s = self.sampling
        how = ("always the likeliest move" if s.greedy else
               "moves as often as humans play them" if s == SamplingSettings() else
               f"temperature {s.temperature:g}" + (f", top-{s.top_k}" if s.top_k else "")
               + (f", top-p {s.top_p:g}" if s.top_p < 1 else "") + (f", min {s.min_prob:g}" if s.min_prob else ""))
        return f"bot rated {self.rating} (plays like {self.bucket} players; {how})"

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


def sampling_from_config(bot: BotConfig) -> SamplingSettings:
    return SamplingSettings(bot.temperature, bot.top_k, bot.top_p, bot.min_prob)


def load_bot(
    cfg: Config,
    rating: int | None = None,
    model_path: str | Path | None = None,
    sampling: SamplingSettings | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> BotPlayer:
    """A bot from a saved model; anything not given comes from ``cfg.bot``.

    Raises:
        BotError: the model file does not exist or the rating is out of range.
    """
    path = Path(model_path) if model_path is not None else cfg.bot.model
    if not path.is_file():
        raise BotError(f"model file not found: {path} (train one with scripts/train.py, or pass --model)")
    rating = check_rating(rating if rating is not None else cfg.bot.rating)
    predictor = load_model(path, device).predictor(device)
    return BotPlayer(predictor, rating, sampling or sampling_from_config(cfg.bot),
                     seed if seed is not None else cfg.bot.seed)
