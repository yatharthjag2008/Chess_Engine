"""Tests for the annotated PGN export of a review."""

from __future__ import annotations

import io

import chess.pgn

from chess_engine.review.analyzer import review_pgn
from chess_engine.review.annotate import annotated_pgn, eval_comment
from chess_engine.review.scoring import MATE_CP
from test_review_analyzer import CFG, PGN, _scripted


def test_eval_comment() -> None:
    assert eval_comment(35) == "[%eval 0.35]" and eval_comment(-120) == "[%eval -1.20]"
    assert eval_comment(MATE_CP - 3) == "[%eval #3]" and eval_comment(-(MATE_CP - 2)) == "[%eval #-2]"


def test_annotated_pgn_reads_back_with_glyphs_evals_and_better_moves() -> None:
    review = review_pgn(PGN, _scripted(), CFG, depth=14)
    text = annotated_pgn(review)
    game = chess.pgn.read_game(io.StringIO(text))
    assert game.headers["Annotator"] == "Stockfish review (depth 14)" and game.headers["White"] == "W"
    nodes = list(game.mainline())
    assert [n.san() for n in nodes] == [m.san for m in review.moves]
    nf6 = nodes[5]
    assert nf6.nags == {4} and "Blunder" in nf6.comment and nf6.eval().white().score(mate_score=MATE_CP) > 9000
    assert nodes[0].nags == set() and nodes[0].eval().white().score() == 30
    alternatives = nf6.parent.variations
    assert len(alternatives) == 2 and alternatives[1].move.uci() == "g7g6"
    assert "g6 was better" in alternatives[1].comment
    assert "Nf6??" in text or "Nf6 $4" in text


def test_custom_start_position() -> None:
    from review_helpers import ScriptedAnalyzer

    fen = "7k/8/6K1/8/8/8/R7/1R6 w - - 0 1"
    review = review_pgn(f'[FEN "{fen}"]\n[SetUp "1"]\n[Result "*"]\n\n1. Ra7 Kg8 2. Rb8# *\n', ScriptedAnalyzer(), CFG)
    game = chess.pgn.read_game(io.StringIO(annotated_pgn(review)))
    assert game.board().fen() == fen and len(list(game.mainline_moves())) == 3
