"""Tests for API request validation and converters."""

from __future__ import annotations

import chess
import pytest
from pydantic import ValidationError

from chess_engine.api.schemas import (
    MoveRequest,
    NewGameRequest,
    bot_move_info,
    color_name,
    legal_moves,
    outcome_info,
)
from chess_engine.bot.game import Outcome
from chess_engine.bot.player import BotPlayer
from chess_engine.bot.sampling import SamplingSettings
from bot_helpers import small_predictor


def test_new_game_defaults_and_validation() -> None:
    request = NewGameRequest()
    assert request.rating == 1500 and request.color == "white" and request.time_control is None
    NewGameRequest(time_control="3+2", fen="8/8/4k3/8/8/2K5/3Q4/8 w - - 0 1", rating=400)
    bad = [dict(rating=399), dict(rating=3001), dict(color="red"), dict(time_control="fast"),
           dict(fen="not a fen"), dict(fen="8/8/8/8/8/8/8/8 w - - 0 1"),          # no kings
           dict(fen="7k/8/8/8/8/8/6R1/K6R b - - 0 1"),                              # already checkmate
           dict(temperature=-1), dict(top_p=0), dict(unknown=1)]
    for fields in bad:
        with pytest.raises(ValidationError):
            NewGameRequest(**fields)


def test_move_request() -> None:
    assert MoveRequest(move="e4").reply is True
    with pytest.raises(ValidationError):
        MoveRequest(move="")


def test_legal_moves() -> None:
    moves = legal_moves(chess.Board())
    assert len(moves) == 20 and [m.uci for m in moves] == sorted(m.uci for m in moves)
    e4 = next(m for m in moves if m.uci == "e2e4")
    assert e4.san == "e4" and e4.from_square == "e2" and e4.to_square == "e4" and e4.promotion is None
    promo = legal_moves(chess.Board("8/4P1k1/8/8/8/8/8/K7 w - - 0 1"))
    assert {m.promotion for m in promo if m.from_square == "e7"} == {"q", "r", "b", "n"}


def test_bot_move_info_and_outcome() -> None:
    bot = BotPlayer(small_predictor(), 1500, SamplingSettings(temperature=0))
    info = bot_move_info(bot.choose_move(chess.Board()))
    assert len(info.candidates) == 5 and info.candidates[0].uci == info.uci and info.play_prob == 1.0
    assert set(info.wdl) == {"win", "draw", "loss"} and sum(info.wdl.values()) == pytest.approx(1, abs=1e-5)
    assert outcome_info(None) is None
    out = outcome_info(Outcome("0-1", "checkmate", chess.BLACK))
    assert out.winner == "black" and out.description == "Black wins by checkmate (0-1)"
    assert color_name(chess.WHITE) == "white" and color_name(None) is None
