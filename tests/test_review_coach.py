"""Tests for the template explanations of mistakes and level differences."""

from __future__ import annotations

from chess_engine.config import ReviewConfig
from chess_engine.review.analyzer import review_pgn
from chess_engine.review.coach import explain_mistake
from review_helpers import ScriptedAnalyzer, position_key

CFG = ReviewConfig()


def _one_move(fen: str, san: str, script: dict) -> object:
    review = review_pgn(f'[FEN "{fen}"]\n[SetUp "1"]\n[Result "*"]\n\n1. {san} *\n', ScriptedAnalyzer(script), CFG)
    return review.moves[0]


def test_missed_mate() -> None:
    fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    move = _one_move(fen, "h3", {position_key(fen=fen): ("d1d8", 9999, {"h2h3": 50})})
    assert explain_mistake(move).startswith("There was mate in one: Rd8#.")
    assert "dropped from 100% to" in explain_mistake(move)


def test_allowed_mate() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/8/PPPPPP1P/RNBQKBNR w KQkq - 0 2"     # 1. g4 e5, White to play
    move = _one_move(fen, "f3", {position_key(fen=fen): ("f1g2", -30, {"f2f3": -9999})})
    assert explain_mistake(move).startswith("This allows a forced mate.")


def test_hanging_piece() -> None:
    fen = "4k3/8/8/4p3/8/4B3/8/4K3 w - - 0 1"                                # Bd4 walks into the e5 pawn
    move = _one_move(fen, "Bd4", {position_key(fen=fen): ("e3f4", 100, {"e3d4": -150})})
    assert explain_mistake(move).startswith("This leaves your bishop on d4 undefended.")


def test_missed_material() -> None:
    fen = "4k3/8/5n2/8/8/8/1B6/4K3 w - - 0 1"                                # Bxf6 wins a knight
    move = _one_move(fen, "Kd2", {position_key(fen=fen): ("b2f6", 400, {"e1d2": 0})})
    assert explain_mistake(move).startswith("You could have won a knight with Bxf6.")


def test_generic_explanation() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    move = _one_move(fen, "Qh5", {position_key(fen=fen): ("g1f3", 30, {"d1h5": -20})})
    assert explain_mistake(move) == "Nf3 was stronger (+0.3 instead of -0.2). " \
                                    f"Your winning chances dropped from {move.win_before:.0f}% to {move.win_after:.0f}%."
