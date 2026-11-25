"""Request and response bodies of the API (pydantic models), and converters from chess objects.

Colours are ``"white"`` / ``"black"``, squares are ``"e2"``, moves are given both
in UCI (``"e2e4"``, what a board widget needs) and SAN (``"e4"``, what people read).
"""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chess_engine.bot.clock import TimeControl
from chess_engine.bot.game import Outcome
from chess_engine.bot.player import MAX_RATING, MIN_RATING, BotMove

Color = Literal["white", "black"]


def color_name(color: chess.Color | None) -> Color | None:
    return None if color is None else ("white" if color == chess.WHITE else "black")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- requests


class NewGameRequest(_Body):
    """Start a game against the bot. Everything is optional."""

    rating: int = Field(default=1500, ge=MIN_RATING, le=MAX_RATING, description="the rating the bot plays like")
    color: Literal["white", "black", "random"] = Field(default="white", description="your side")
    your_rating: int | None = Field(default=None, ge=MIN_RATING, le=MAX_RATING,
                                    description="your rating (the bot is told); default: the bot's")
    time_control: str | None = Field(default=None, description="e.g. 3+2 (minutes + seconds); null = untimed")
    temperature: float | None = Field(default=None, ge=0, le=10)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, description="reproducible bot moves")
    fen: str | None = Field(default=None, description="start from this position instead of the start position")

    @field_validator("time_control")
    @classmethod
    def _check_time_control(cls, value: str | None) -> str | None:
        if value is not None:
            TimeControl.parse(value)  # raises ValueError with a readable message
        return value

    @field_validator("fen")
    @classmethod
    def _check_fen(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            board = chess.Board(value)
        except ValueError as exc:
            raise ValueError(f"invalid FEN: {exc}") from None
        if not board.is_valid():
            raise ValueError("the FEN is not a legal chess position")
        if board.is_game_over():
            raise ValueError("the game is already over in this position")
        return value


class MoveRequest(_Body):
    """Your move, in SAN (``Nf3``) or UCI (``g1f3``)."""

    move: str = Field(min_length=1, max_length=10)
    reply: bool = Field(default=True, description="let the bot answer in the same request")


# --------------------------------------------------------------------------- responses


class LegalMove(BaseModel):
    uci: str
    san: str
    from_square: str
    to_square: str
    promotion: str | None = None


class Candidate(BaseModel):
    uci: str
    san: str
    prob: float = Field(description="how often players of the bot's rating play this move here")


class BotMoveInfo(BaseModel):
    uci: str
    san: str
    prob: float = Field(description="how often players of the bot's rating play this move here")
    play_prob: float = Field(description="the chance the bot had of choosing it")
    candidates: list[Candidate] = Field(description="the likeliest moves, at most five")
    wdl: dict[str, float] | None = Field(default=None, description="the bot's win/draw/loss estimate before moving")


class ClockState(BaseModel):
    white: float = Field(description="seconds left")
    black: float
    running: Color | None
    base_seconds: int
    increment: int


class OutcomeInfo(BaseModel):
    result: str
    termination: str
    winner: Color | None
    description: str


class GameState(BaseModel):
    id: str
    fen: str
    turn: Color
    human_color: Color
    bot_rating: int
    bot_description: str
    your_rating: int | None
    status: Literal["ongoing", "over"]
    outcome: OutcomeInfo | None
    check: bool
    moves_san: list[str]
    moves_uci: list[str]
    move_list: str
    last_move: str | None
    legal_moves: list[LegalMove] = Field(description="the side to move's legal moves (empty when the game is over)")
    clock: ClockState | None
    last_bot_move: BotMoveInfo | None


class MoveResponse(BaseModel):
    played: str = Field(description="your move in SAN")
    bot_move: BotMoveInfo | None = Field(description="the bot's reply, if it moved")
    state: GameState


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    games: int


# --------------------------------------------------------------------------- converters


def legal_moves(board: chess.Board) -> list[LegalMove]:
    """Every legal move, sorted by UCI."""
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    return [LegalMove(uci=m.uci(), san=board.san(m), from_square=chess.square_name(m.from_square),
                      to_square=chess.square_name(m.to_square),
                      promotion=chess.piece_symbol(m.promotion) if m.promotion else None) for m in moves]


def bot_move_info(choice: BotMove, top: int = 5) -> BotMoveInfo:
    wdl = None
    if choice.wdl is not None:
        wdl = {"win": choice.wdl[0], "draw": choice.wdl[1], "loss": choice.wdl[2]}
    return BotMoveInfo(uci=choice.move.uci(), san=choice.san, prob=choice.prob, play_prob=choice.play_prob,
                       candidates=[Candidate(uci=c.move.uci(), san=c.san, prob=c.prob) for c in choice.candidates[:top]],
                       wdl=wdl)


def outcome_info(outcome: Outcome | None) -> OutcomeInfo | None:
    if outcome is None:
        return None
    return OutcomeInfo(result=outcome.result, termination=outcome.termination, winner=color_name(outcome.winner),
                       description=outcome.describe())


# --------------------------------------------------------------------------- game import (step 13)


class ImportPgnRequest(_Body):
    """Games pasted as PGN text."""

    pgn: str = Field(min_length=1, max_length=5_000_000)
    max_games: int | None = Field(default=None, ge=1, le=500)


class ImportedGameInfo(BaseModel):
    source: str
    id: str
    white: str
    black: str
    white_elo: int | None
    black_elo: int | None
    result: str
    date: str | None
    time_control: str | None
    speed: str | None
    rated: bool | None
    url: str | None
    plies: int
    pgn: str


class ImportResponse(BaseModel):
    source: str
    username: str | None
    count: int
    games: list[ImportedGameInfo]


# --------------------------------------------------------------------------- rating estimate (step 12 over the API)


class RatingEstimateRequest(_Body):
    """Estimate the rating of one player from PGN games: name them, or pick a colour."""

    pgn: str = Field(min_length=1, max_length=5_000_000)
    player: str | None = Field(default=None, max_length=50)
    color: Literal["white", "black"] | None = None
    use_opponent_rating: bool = True


class RatingRange(BaseModel):
    estimate: float
    median: float
    low: float
    high: float
    interval: float
    moves: int


class GameRatingEstimate(RatingRange):
    white: str
    black: str
    color: Color
    rating_in_game: int | None


class RatingEstimateResponse(BaseModel):
    overall: RatingRange
    games: list[GameRatingEstimate]
    temper: float
    note: str | None = None


# --------------------------------------------------------------------------- review (step 14)


class ReviewRequest(_Body):
    """A game to review, as PGN (``game_index``: which game of the text, 0 = first)."""

    pgn: str = Field(min_length=1, max_length=5_000_000)
    game_index: int = Field(default=0, ge=0)
