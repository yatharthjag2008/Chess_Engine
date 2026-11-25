"""Tests for reviewing games move by move (scripted evaluations; real Stockfish in one test)."""

from __future__ import annotations

import chess
import pytest

from chess_engine.config import ReviewConfig
from chess_engine.review.analyzer import review_pgn
from chess_engine.review.classify import MoveClass
from chess_engine.review.scoring import win_percent
from review_helpers import ScriptedAnalyzer, position_key

CFG = ReviewConfig()
PGN = '[White "W"]\n[Black "B"]\n[Result "*"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n'


def _scripted() -> ScriptedAnalyzer:
    return ScriptedAnalyzer({
        position_key(): ("e2e4", 30, {"d2d4": 28}),                                   # 1. e4: the engine's move
        position_key("e4"): ("e7e5", -25, {}),                                       # Black is slightly worse
        position_key("e4", "e5"): ("g1f3", 30, {"d1h5": -20}),                       # 2. Qh5: 50 cp worse
        position_key("e4", "e5", "Qh5"): ("b8c6", 20, {}),
        position_key("e4", "e5", "Qh5", "Nc6"): ("f1c4", 30, {}),
        position_key("e4", "e5", "Qh5", "Nc6", "Bc4"): ("g7g6", -40, {"g8f6": -9997}),  # 3...Nf6?? allows mate
        position_key("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"): ("h5f7", 9999, {}),
    })


def test_every_move_is_judged() -> None:
    review = review_pgn(PGN, _scripted(), CFG, depth=12)
    assert [m.san for m in review.moves] == ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
    assert review.headers["White"] == "W" and review.depth == 12 and review.moves[0].move_number == "1."
    first = review.moves[0]
    assert first.classification is MoveClass.BEST and first.cp_loss == 0 and first.accuracy == 100
    assert first.best_san == "e4" and [a.san for a in first.alternatives] == ["e4", "d4"]
    qh5 = review.moves[2]
    assert qh5.best_san == "Nf3" and qh5.cp_loss == 50 and qh5.played_cp == -20
    assert qh5.win_lost == pytest.approx(win_percent(30) - win_percent(-20))
    assert qh5.classification is MoveClass.GOOD                       # ~4.6 points: imprecise, not an error
    nf6 = review.moves[5]
    assert nf6.color == "black" and nf6.move_number == "3..."
    assert nf6.classification is MoveClass.BLUNDER and nf6.win_after == 0 and nf6.cp_loss == 960
    assert nf6.eval_before == 40 and nf6.eval_after == 9997          # White's view
    mate = review.moves[6]
    assert mate.classification is MoveClass.BEST and mate.eval_after > 9000
    assert len(review.for_color("white")) == 4 and len(review.for_color("black")) == 3


def test_a_move_the_engine_did_not_consider_uses_the_next_position() -> None:
    script = {position_key(): ("e2e4", 30, {}),
              position_key("a3"): ("e7e5", 60, {})}                  # after 1. a3 Black is better by 60
    review = review_pgn('[Result "*"]\n\n1. a3 *\n', ScriptedAnalyzer(script), CFG)
    move = review.moves[0]
    assert move.played_cp == -60 and move.cp_loss == 90 and move.eval_after == -60


def test_errors() -> None:
    with pytest.raises(ValueError, match="no reviewable game"):
        review_pgn("not a game", ScriptedAnalyzer(), CFG)
    with pytest.raises(ValueError, match="number 2"):
        review_pgn(PGN, ScriptedAnalyzer(), CFG, game_index=1)


@pytest.mark.stockfish
def test_real_review_finds_the_scholars_mate_blunder(stockfish) -> None:
    from chess_engine.review.engine import StockfishAnalyzer

    with StockfishAnalyzer(stockfish.path, depth=10, multipv=2) as engine:
        review = review_pgn(PGN, engine, CFG, depth=10)
    nf6 = review.moves[5]
    assert nf6.classification is MoveClass.BLUNDER and nf6.best_san in {"g6", "Qe7", "Qf6", "Nh6"}
    assert review.moves[6].classification is MoveClass.BEST
    assert review.moves[0].classification in (MoveClass.BEST, MoveClass.GOOD)


def test_summary_per_side() -> None:
    review = review_pgn(PGN, _scripted(), CFG)
    curve = review.white_win_curve()
    assert len(curve) == len(review.moves) + 1 and curve[-1] == 100.0
    summary = review.summary()
    white, black = summary["white"], summary["black"]
    assert white.moves == 4 and black.moves == 3
    assert white.counts["best"] + white.counts["good"] == 4 and black.counts["blunder"] == 1
    assert sum(black.counts.values()) == 3
    assert white.accuracy > black.accuracy and 0 <= black.accuracy <= 100
    assert black.average_cp_loss == pytest.approx(sum(m.cp_loss for m in review.for_color("black")) / 3)
    assert "1 blunder" in black.describe() and "no errors" in white.describe()


def test_key_moments() -> None:
    from chess_engine.review.analyzer import key_moments

    script = {
        position_key(): ("e2e4", 20, {}),
        position_key("e4"): ("e7e5", -20, {}),
        position_key("e4", "e5"): ("g1f3", 30, {"f2f3": -150}),      # 2. f3: an inaccuracy-sized loss
        position_key("e4", "e5", "f3"): ("b8c6", 150, {}),
        position_key("e4", "e5", "f3", "Nc6"): ("b1c3", -100, {"e1e2": -600}),   # 3. Ke2: a big error
        position_key("e4", "e5", "f3", "Nc6", "Ke2"): ("d8h4", 600, {"a7a6": 100}),  # 3...a6: a big miss
    }
    review = review_pgn('[Result "*"]\n\n1. e4 e5 2. f3 Nc6 3. Ke2 a6 *\n', ScriptedAnalyzer(script), CFG)
    moments = key_moments(review, count=5)
    assert [m.san for m in moments] == ["f3", "Ke2", "a6"]                 # errors only, in game order
    two = key_moments(review, count=2)
    assert {m.san for m in two} == {"Ke2", "a6"}                            # the two biggest losses
    ke2 = next(m for m in moments if m.san == "Ke2")
    assert ke2.text.startswith("3. Ke2") and "White's chances" in ke2.text and "Better was Nc3." in ke2.text
    a6 = next(m for m in moments if m.san == "a6")
    assert a6.move_number == "3..." and "Better was Qh4" in a6.text
    assert key_moments(review_pgn(PGN, ScriptedAnalyzer(), CFG)) == []      # no evaluations, no errors


def test_graph_and_json_round_trip() -> None:
    import json

    from chess_engine.review.analyzer import eval_graph, review_from_dict, review_to_dict

    review = review_pgn(PGN, _scripted(), CFG, depth=9)
    graph = eval_graph(review)
    assert [p["ply"] for p in graph] == list(range(8)) and graph[-1]["white_win"] == 100
    assert all(-1000 <= p["eval"] <= 1000 for p in graph)
    data = json.loads(json.dumps(review_to_dict(review, key_moment_count=3)))   # JSON-safe
    assert data["depth"] == 9 and data["moves"][5]["classification"] == "blunder"
    assert data["moves"][5]["move_number"] == "3..." and data["summary"]["black"]["counts"]["blunder"] == 1
    assert data["key_moments"][0]["san"] == "Nf6" and len(data["graph"]) == 8
    again = review_from_dict(data)
    assert [m.san for m in again.moves] == [m.san for m in review.moves]
    assert again.moves[5].classification is MoveClass.BLUNDER and again.moves[0].alternatives[0].san == "e4"
    assert review_to_dict(again) == review_to_dict(review)
    board = chess.Board()
    for move in data["moves"]:
        assert move["fen_before"] == board.fen()
        board.push_uci(move["uci"])
        assert move["fen_after"] == board.fen()    # the position after the move (the web page shows it)
    assert eval_graph(review_pgn('[Result "*"]\n\n1. e4 *\n', ScriptedAnalyzer(), CFG))[0]["ply"] == 0


def test_describe_uses_proper_plurals() -> None:
    from chess_engine.review.analyzer import SideSummary

    side = SideSummary("white", 30, 88.0, 40.0, {"best": 20, "good": 5, "inaccuracy": 2, "mistake": 1, "blunder": 2})
    assert side.describe() == "White: accuracy 88%, average loss 40 cp, 2 inaccuracies, 1 mistake, 2 blunders"
