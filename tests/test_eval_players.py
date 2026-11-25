"""Tests for match players (bot, limited Stockfish) and the judge, with a fake engine and real Stockfish."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

from chess_engine.bot.player import BotPlayer
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.eval.players import MATE_CP, BotAgent, Judge, StockfishPlayer
from bot_helpers import small_predictor

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci_engine.py"), "--first-legal"]


def test_bot_agent() -> None:
    agent = BotAgent(BotPlayer(small_predictor(), 1300, SamplingSettings(temperature=0)))
    assert agent.name == "bot 1300" and agent.rating == 1300
    board = chess.Board()
    assert agent.choose(board, opponent_rating=1600) in board.legal_moves
    agent.close()


def test_stockfish_player_with_a_fake_engine() -> None:
    with StockfishPlayer(FAKE, elo=1500) as engine:
        assert engine.name == "Stockfish 1500" and engine.rating == 1500
        board = chess.Board()
        assert engine.choose(board).uci() == min(m.uci() for m in board.legal_moves)
        board.push_san("e4")
        assert engine.choose(board) in board.legal_moves  # the same process keeps answering
    assert engine._engine is None


def test_judge_scores_from_whites_view() -> None:
    with Judge(FAKE + ["--score-cp=150"]) as judge:
        assert judge.evaluate(chess.Board()) == 150                      # White to move: +150 for White
        black_to_move = chess.Board()
        black_to_move.push_san("e4")
        assert judge.evaluate(black_to_move) == -150                     # Black's +150 is White's -150
    mated = chess.Board("7k/8/8/8/8/8/6R1/K6R b - - 0 1")
    assert Judge(FAKE).evaluate(mated) == MATE_CP
    assert Judge(FAKE).evaluate(chess.Board("k7/2K5/1Q6/8/8/8/8/8 b - - 0 1")) == 0  # stalemate
    with Judge(FAKE) as no_score:
        assert no_score.evaluate(chess.Board()) == 0


@pytest.mark.stockfish
def test_real_stockfish_levels_and_judge(stockfish) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    with StockfishPlayer(stockfish.path, elo=1320, move_seconds=0.02) as weak:
        assert weak.choose(board) in board.legal_moves
        assert weak._engine.options["UCI_Elo"] is not None
    with Judge(stockfish.path, nodes=5000) as judge:
        assert judge.evaluate(board) > 50_000                            # mate in one for White
        assert abs(judge.evaluate(chess.Board())) < 100
