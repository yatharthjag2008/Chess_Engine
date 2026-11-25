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


def _moment(played: str, yours: tuple, stronger: tuple | None, played_win: float = 0.0):
    from chess_engine.review.level_compare import LevelMoment, LevelOption, PositionComparison

    def option(rating, san, uci, prob, win):
        return LevelOption(rating, san, uci, prob, 0, win, [])

    y = option(1300, *yours)
    s = option(1500, *stronger) if stronger else None
    comparison = PositionComparison([y] + ([s] if s else []), s, (s.win - y.win) if s else 0.0, s is not None)
    return LevelMoment(5, "3...", played[0], played[1], played_win, comparison)


def test_explain_level() -> None:
    from chess_engine.review.coach import explain_level

    like_level = _moment(("Nf6", "g8f6"), ("Nf6", "g8f6", 0.4, 0.0), ("g6", "g7g6", 0.35, 46.0))
    text = explain_level(like_level)
    assert text.startswith("Players around 1300 usually play Nf6 here (40%).")
    assert "Players around 1500 prefer g6 (35%), which keeps 46% winning chances instead of 0%." in text
    assert text.endswith("You played Nf6, like most players at your level.")
    found = explain_level(_moment(("g6", "g7g6"), ("Nf6", "g8f6", 0.4, 0.0), ("g6", "g7g6", 0.35, 46.0)))
    assert found.endswith("You found g6: that is the stronger players' move.")
    other = explain_level(_moment(("Qe7", "d8e7"), ("Nf6", "g8f6", 0.4, 0.0), ("g6", "g7g6", 0.35, 46.0), 40.0))
    assert other.endswith("You played Qe7 (40%).")
    agree = explain_level(_moment(("e5", "e7e5"), ("e5", "e7e5", 0.5, 50.0), None))
    assert agree.endswith("Stronger players agree.")


def test_coach_game_merges_mistakes_and_level_notes() -> None:
    import chess

    from chess_engine.config import CoachConfig
    from chess_engine.model.inference import MoveProbability, Prediction
    from chess_engine.review.coach import coach_game
    from test_review_analyzer import PGN, _scripted

    analyzer = _scripted()
    review = review_pgn(PGN, analyzer, CFG)

    class Levels:  # Black's level plays what Black played; stronger players play the engine's move
        def predict_many(self, boards, ratings, **kwargs):
            out = []
            for board, rating in zip(boards, ratings):
                played = next(m for m in review.moves if m.fen_before == board.fen())
                san = played.san if rating == 1300 else (played.best_san or played.san)
                out.append(Prediction([MoveProbability(0.5, board.parse_san(san), san)], None))
            return out

    report = coach_game(review, "black", 1300, Levels(), analyzer, CoachConfig())
    assert report.color == "black" and report.rating == 1300 and report.summary.counts["blunder"] == 1
    assert report.your_level_match == 1.0
    (note,) = report.notes                                     # only 3...Nf6: a blunder AND a level moment
    assert note.san == "Nf6" and note.kinds == ("mistake", "level") and note.classification == "blunder"
    assert note.text.startswith("This allows a forced mate.") and "Players around 1500 prefer g6" in note.text
    white = coach_game(review, "white", 1300, Levels(), analyzer, CoachConfig())
    assert white.notes == []                                  # White made no errors and matched its level


def test_report_json_and_text() -> None:
    import json

    from chess_engine.review.analyzer import SideSummary
    from chess_engine.review.coach import CoachNote, CoachReport, format_report, report_to_dict

    summary = SideSummary("black", 3, 61.0, 330.0, {"best": 2, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 1})
    note = CoachNote(5, "3...", "Nf6", ("mistake", "level"), "blunder", "This allows a forced mate.")
    report = CoachReport("black", 1300, summary, 1.0, 0.6667, [note])
    data = json.loads(json.dumps(report_to_dict(report)))
    assert data["notes"][0]["kinds"] == ["mistake", "level"] and data["summary"]["counts"]["blunder"] == 1
    text = format_report(report)
    assert text.startswith("Coach report for Black (rated about 1300)")
    assert "favourite move 100% of the time, and the 1500-level move 67%" in text
    assert "3... Nf6??  This allows a forced mate." in text
    clean = format_report(CoachReport("white", 1600, summary, 0.5, 0.5, []))
    assert "Well played." in clean
