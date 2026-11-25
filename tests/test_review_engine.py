"""Tests for the position analyser (fake engine for the plumbing, real Stockfish for the chess)."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

from chess_engine.review.engine import StockfishAnalyzer, finished_eval
from chess_engine.review.scoring import MATE_CP

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci_engine.py"), "--first-legal",
        "--score-cp=40"]


def test_finished_positions() -> None:
    assert finished_eval(chess.Board("7k/8/8/8/8/8/6R1/K6R b - - 0 1")) == -MATE_CP   # Black is checkmated
    assert finished_eval(chess.Board("k7/2K5/1Q6/8/8/8/8/8 b - - 0 1")) == 0          # stalemate
    assert finished_eval(chess.Board()) is None


def test_analyser_with_a_fake_engine_and_its_cache() -> None:
    with StockfishAnalyzer(FAKE, depth=2, multipv=3) as engine:
        board = chess.Board()
        first = engine.analyse(board)
        assert first.best.move.uci() == min(m.uci() for m in board.legal_moves)
        assert first.best_cp == 40 and first.best.san == board.san(first.best.move)
        assert first.line_for(first.best.move) is first.best and first.line_for(chess.Move.from_uci("h2h4")) is None
        assert engine.analyse(chess.Board()) is first                   # cached
        mated = engine.analyse(chess.Board("7k/8/8/8/8/8/6R1/K6R b - - 0 1"))
        assert mated.lines == [] and mated.best_cp == -MATE_CP and mated.best is None
    assert engine._engine is None


@pytest.mark.stockfish
def test_real_stockfish_lines(stockfish) -> None:
    with StockfishAnalyzer(stockfish.path, depth=10, multipv=3) as engine:
        mate_in_one = engine.analyse(chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"))
        assert mate_in_one.best.san == "Rd8#" and mate_in_one.best_cp >= MATE_CP - 1
        assert len(mate_in_one.lines) == 3 and mate_in_one.lines[0].cp >= mate_in_one.lines[1].cp
        start = engine.analyse(chess.Board())
        assert abs(start.best_cp) < 80 and start.best.pv[0] == start.best.move.uci()
        black = chess.Board()
        black.push_san("f3")
        black.push_san("e5")
        black.push_san("g4")
        assert engine.analyse(black).best.san == "Qh4#"                 # Black to move: its own view
