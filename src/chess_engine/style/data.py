"""Training examples for a style: one player's positions and moves, encoded like the training data.

    games = player_games_from_pgn(text, player="someone")
    games = select_games(games, until=datetime.date(2024, 6, 30))     # ghost mode: up to a cutoff date
    train, val = split_games(games, 0.15, seed=0)                     # held out by game, not by position
    examples = examples_from_games(train, ratings, default_rating=1500)
"""

from __future__ import annotations

import datetime
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chess_engine.config import RatingsConfig
from chess_engine.encoding.board import encode_boards
from chess_engine.encoding.moves import legal_move_mask, move_to_index
from chess_engine.eval.player_games import PlayerGame
from chess_engine.model.features import ModelInputs, buckets_from_ratings, context_features, make_inputs


def game_date(game: PlayerGame) -> datetime.date | None:
    """The game's date from ``UTCDate`` or ``Date`` (``2024.03.15``); None when missing or partial (``????``)."""
    for key in ("UTCDate", "Date"):
        text = game.headers.get(key, "")
        try:
            year, month, day = (int(part) for part in text.split("."))
            return datetime.date(year, month, day)
        except ValueError:
            continue
    return None


def select_games(games: Sequence[PlayerGame], until: datetime.date | None = None,
                 since: datetime.date | None = None) -> list[PlayerGame]:
    """Games within the dates (inclusive). With a date limit, games without a date are left out."""
    if until is None and since is None:
        return list(games)
    out = []
    for game in games:
        date = game_date(game)
        if date is None or (until is not None and date > until) or (since is not None and date < since):
            continue
        out.append(game)
    return out


def split_games(games: Sequence[PlayerGame], val_fraction: float, seed: int = 0) -> tuple[list[PlayerGame],
                                                                                         list[PlayerGame]]:
    """Shuffle and hold out ``val_fraction`` of the games (at least one when there are two or more)."""
    order = list(games)
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(order) * val_fraction)) if len(order) >= 2 else 0
    return order[n_val:], order[:n_val]


@dataclass
class StyleExamples:
    """Encoded positions (inputs), the moves played (targets) and the legal moves of each position."""

    inputs: ModelInputs
    targets: np.ndarray          # (N,) move indices
    legal: np.ndarray            # (N, NUM_MOVES) bool

    def __len__(self) -> int:
        return len(self.targets)

    def subset(self, index: np.ndarray) -> StyleExamples:
        i = np.asarray(index)
        x = self.inputs
        return StyleExamples(ModelInputs(x.planes[i], x.bucket[i], x.opp_bucket[i], x.context[i]),
                             self.targets[i], self.legal[i])


def examples_from_games(games: Sequence[PlayerGame], ratings: RatingsConfig, default_rating: int,
                        skip_opening_plies: int = 0, max_positions: int | None = None) -> StyleExamples:
    """Every position where the player moved (after the first ``skip_opening_plies`` plies of the game).

    The player's rating comes from each game (else ``default_rating``), the opponent's likewise; clocks from
    ``[%clk]`` when recorded, else the full base time.
    """
    boards, targets, mine, theirs, clock, opp_clock, base, inc = [], [], [], [], [], [], [], []
    for game in games:
        for i, (board, move) in enumerate(zip(game.boards, game.moves)):
            if board.ply() < skip_opening_plies:
                continue
            boards.append(board)
            targets.append(move_to_index(move, board.turn))
            mine.append(game.rating or default_rating)
            theirs.append(game.opp_rating or game.rating or default_rating)
            clock.append(game.clocks[i] if game.clocks else game.base_seconds)
            opp_clock.append(game.opp_clocks[i] if game.opp_clocks else game.base_seconds)
            base.append(game.base_seconds)
            inc.append(game.increment)
            if max_positions is not None and len(boards) >= max_positions:
                break
        if max_positions is not None and len(boards) >= max_positions:
            break
    if not boards:
        raise ValueError("no positions: the player has no moves in these games")
    inputs = make_inputs(
        encode_boards(boards),
        buckets_from_ratings(np.array(mine), ratings),
        buckets_from_ratings(np.array(theirs), ratings),
        context_features(np.array(clock, dtype=np.float64), np.array(opp_clock, dtype=np.float64),
                         np.array(base), np.array(inc)),
    )
    legal = np.stack([legal_move_mask(b) for b in boards]).astype(bool)
    return StyleExamples(inputs, np.array(targets, dtype=np.int64), legal)
