"""Your level vs. the next level: what players of your rating play here, and what stronger players play.

For a position, the model gives the likeliest moves at your rating and at higher
ratings (+200, +400 by default). Where the levels *disagree* and Stockfish says
the stronger level's choice is clearly better (by at least ``min_win_gap`` win-%
points), the position is a lesson: it shows a habit that separates you from the
next level. Everywhere else the levels agree, or the difference does not matter,
and nothing is flagged.

    choices = level_choices(predictor, board, [1400, 1600, 1800], opp_rating=1450)
    choices[0].moves[0].san, choices[0].moves[0].prob     # the 1400s' favourite move here
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess

from chess_engine.config import CoachConfig
from chess_engine.model.inference import MovePredictor, MoveProbability
from chess_engine.review.analyzer import GameReview, MoveReview
from chess_engine.review.engine import Analyzer, EngineLine, PositionAnalysis
from chess_engine.review.scoring import win_percent


@dataclass(frozen=True)
class LevelChoice:
    """The likeliest moves of players at one rating in one position."""

    rating: int
    moves: list[MoveProbability]          # likeliest first

    @property
    def top(self) -> MoveProbability:
        return self.moves[0]


def level_choices(
    predictor: MovePredictor,
    board: chess.Board,
    ratings: Sequence[int],
    opp_rating: int | None = None,
    top: int = 3,
    clock: float | None = None,
    opp_clock: float | None = None,
    base_seconds: int = 180,
    increment: int = 0,
) -> list[LevelChoice]:
    """The ``top`` likeliest moves at each rating, in one batched model call.

    The opponent's rating is the same for every level (default: each level's own rating):
    the question is what a stronger player would play against *this* opponent.
    """
    if not ratings:
        return []
    n = len(ratings)
    predictions = predictor.predict_many(
        [board] * n, list(ratings), opp_ratings=[opp_rating] * n if opp_rating is not None else list(ratings),
        clocks=clock, opp_clocks=opp_clock, base_seconds=base_seconds, increment=increment,
    )
    return [LevelChoice(rating, prediction.moves[:top]) for rating, prediction in zip(ratings, predictions)]


def move_values(board: chess.Board, moves: Sequence[chess.Move], analyzer: Analyzer,
                known: PositionAnalysis | None = None) -> dict[str, int]:
    """Stockfish's evaluation (centipawns, the mover's view) of each move in ``moves``.

    Moves already among the engine lines of ``known`` (the analysis of ``board``, e.g. from the
    game review) reuse that evaluation; others are scored by analysing the position after them.
    """
    values: dict[str, int] = {}
    for move in moves:
        if move.uci() in values:
            continue
        line = known.line_for(move) if known is not None else None
        if line is not None:
            values[move.uci()] = line.cp
            continue
        after = board.copy(stack=False)
        after.push(move)
        values[move.uci()] = -analyzer.analyse(after).best_cp
    return values


# --------------------------------------------------------------------------- one position


@dataclass(frozen=True)
class LevelOption:
    """One level's favourite move here, how often that level plays it, and how good it is."""

    rating: int
    san: str
    uci: str
    prob: float          # how often players of this rating play it
    cp: int              # Stockfish, the mover's view
    win: float           # the mover's win % after it
    alternatives: list[tuple[str, float]]   # the level's next likeliest moves: (san, prob)


@dataclass(frozen=True)
class PositionComparison:
    """Your level against the higher levels in one position."""

    levels: list[LevelOption]           # your level first, then the higher ones
    stronger: LevelOption | None        # the first higher level whose favourite move differs from yours
    win_gap: float                      # how many more win-% points the stronger level's move keeps (or 0)
    matters: bool                       # the levels disagree AND the gap is at least min_win_gap

    @property
    def yours(self) -> LevelOption:
        return self.levels[0]


def compare_position(
    predictor: MovePredictor,
    board: chess.Board,
    rating: int,
    analyzer: Analyzer,
    cfg: CoachConfig,
    known: PositionAnalysis | None = None,
    opp_rating: int | None = None,
    **clock: object,
) -> PositionComparison:
    """What your rating and the ratings ``cfg.level_offsets`` above it play here, valued by Stockfish."""
    ratings = [rating] + [rating + offset for offset in cfg.level_offsets]
    choices = level_choices(predictor, board, ratings, opp_rating, cfg.top_moves, **clock)
    values = move_values(board, [c.top.move for c in choices], analyzer, known)
    levels = [LevelOption(c.rating, c.top.san, c.top.move.uci(), c.top.prob, values[c.top.move.uci()],
                          win_percent(values[c.top.move.uci()]), [(m.san, m.prob) for m in c.moves[1:]])
              for c in choices]
    yours = levels[0]
    stronger = next((level for level in levels[1:] if level.uci != yours.uci), None)
    gap = (stronger.win - yours.win) if stronger is not None else 0.0
    return PositionComparison(levels, stronger, gap, stronger is not None and gap >= cfg.min_win_gap)


# --------------------------------------------------------------------------- a whole game


@dataclass(frozen=True)
class LevelMoment:
    """One of your moves, with the level comparison of the position before it."""

    ply: int
    move_number: str
    played_san: str
    played_uci: str
    played_win: float                     # your win % after the move you played
    comparison: PositionComparison


@dataclass(frozen=True)
class GameComparison:
    """Every move of one player compared with the levels; the moments that matter; match rates."""

    color: str
    rating: int
    moments: list[LevelMoment]
    flagged: list[LevelMoment]            # the biggest gaps that matter, in game order
    your_level_match: float               # share of your moves that were your level's favourite
    next_level_match: float               # share that were the favourite of the first higher level


def known_from_review(move: MoveReview) -> PositionAnalysis:
    """The engine lines stored in a review, as an analysis (so a cached review needs no new search)."""
    board = chess.Board(move.fen_before)
    lines = [EngineLine(chess.Move.from_uci(a.uci), a.san, a.cp, [a.uci]) for a in move.alternatives]
    return PositionAnalysis(move.fen_before, lines, move.best_cp)


def compare_game(
    review: GameReview,
    color: str,
    rating: int,
    predictor: MovePredictor,
    analyzer: Analyzer,
    cfg: CoachConfig,
    opp_rating: int | None = None,
) -> GameComparison:
    """Compare every move ``color`` played in ``review`` with the levels (see the module docstring)."""
    moments = []
    for move in review.for_color(color):
        board = chess.Board(move.fen_before)
        comparison = compare_position(predictor, board, rating, analyzer, cfg, known_from_review(move), opp_rating)
        moments.append(LevelMoment(move.ply, move.move_number, move.san, move.uci, move.win_after, comparison))
    mattering = [m for m in moments if m.comparison.matters]
    flagged = sorted(sorted(mattering, key=lambda m: -m.comparison.win_gap)[:cfg.max_moments], key=lambda m: m.ply)
    n = len(moments)
    yours = sum(m.played_uci == m.comparison.levels[0].uci for m in moments)
    nxt = sum(m.played_uci == m.comparison.levels[1].uci for m in moments)
    return GameComparison(color, rating, moments, flagged, yours / n if n else 0.0, nxt / n if n else 0.0)
