"""Tests for "your level vs. next level"."""

from __future__ import annotations

import chess
import pytest

from chess_engine.review.level_compare import level_choices, move_values
from bot_helpers import small_predictor

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def test_level_choices_match_single_predictions() -> None:
    predictor = small_predictor()
    board = chess.Board(KIWIPETE)
    choices = level_choices(predictor, board, [1100, 1300, 1500], opp_rating=1250, top=3)
    assert [c.rating for c in choices] == [1100, 1300, 1500]
    assert all(len(c.moves) == 3 for c in choices)
    for choice in choices:
        direct = predictor.predict(board, choice.rating, opp_ratings=1250)
        assert [m.move for m in choice.moves] == [m.move for m in direct.moves[:3]]
        assert choice.top.prob == pytest.approx(direct.moves[0].prob, abs=1e-5)
    assert choices[0].moves[0].prob != pytest.approx(choices[2].moves[0].prob)   # the rating matters
    assert level_choices(predictor, board, []) == []


def test_unknown_opponent_means_same_rating() -> None:
    predictor = small_predictor()
    board = chess.Board()
    (choice,) = level_choices(predictor, board, [1700], top=5)
    direct = predictor.predict(board, 1700, opp_ratings=1700)
    assert [m.move for m in choice.moves] == [m.move for m in direct.moves[:5]]


def test_move_values_reuse_known_lines_and_analyse_the_rest() -> None:
    from review_helpers import ScriptedAnalyzer, position_key

    analyzer = ScriptedAnalyzer({
        position_key(): ("e2e4", 30, {"d2d4": 25}),
        position_key("a3"): ("e7e5", 40, {}),            # after 1. a3, Black is +40 -> -40 for White
    })
    board = chess.Board()
    known = analyzer.analyse(board)
    calls = analyzer.calls
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "a2a3", "e2e4")]
    values = move_values(board, moves, analyzer, known)
    assert values == {"e2e4": 30, "d2d4": 25, "a2a3": -40}
    assert analyzer.calls == calls + 1                    # only a3 needed the engine
    fresh = move_values(board, [chess.Move.from_uci("e2e4")], ScriptedAnalyzer({position_key("e4"): ("e7e5", -30, {})}))
    assert fresh == {"e2e4": 30}


class FixedLevels:
    """A stand-in predictor whose favourite move depends only on the rating (for exact tests)."""

    def __init__(self, by_rating: dict[int, list[tuple[str, float]]]) -> None:
        self.by_rating = by_rating

    def predict_many(self, boards, ratings, **kwargs):
        from chess_engine.model.inference import MoveProbability, Prediction

        out = []
        for board, rating in zip(boards, ratings):
            moves = [MoveProbability(p, board.parse_san(san), san) for san, p in self.by_rating[rating]]
            out.append(Prediction(moves, None))
        return out


def test_compare_position_flags_only_disagreements_that_matter() -> None:
    from chess_engine.config import CoachConfig
    from chess_engine.review.level_compare import compare_position
    from chess_engine.review.scoring import win_percent
    from review_helpers import ScriptedAnalyzer, position_key

    cfg = CoachConfig(level_offsets=[200, 400], min_win_gap=8, top_moves=2)
    analyzer = ScriptedAnalyzer({position_key(): ("e2e4", 40, {"d2d4": 35, "a2a3": -150})})
    board = chess.Board()
    weak = FixedLevels({1200: [("a3", 0.4), ("e4", 0.3)], 1400: [("a3", 0.35), ("e4", 0.3)],
                        1600: [("e4", 0.5), ("d4", 0.2)]})
    result = compare_position(weak, board, 1200, analyzer, cfg, known=analyzer.analyse(board))
    assert [l.rating for l in result.levels] == [1200, 1400, 1600]
    assert result.yours.san == "a3" and result.yours.cp == -150 and result.yours.alternatives == [("e4", 0.3)]
    assert result.stronger.rating == 1600 and result.stronger.san == "e4"     # 1400 agrees with 1200
    assert result.win_gap == pytest.approx(win_percent(40) - win_percent(-150)) and result.matters

    close = FixedLevels({1200: [("d4", 0.4)], 1400: [("e4", 0.4)], 1600: [("e4", 0.5)]})
    small_gap = compare_position(close, board, 1200, analyzer, cfg)
    assert small_gap.stronger.san == "e4" and small_gap.win_gap < 8 and not small_gap.matters

    same = FixedLevels({1200: [("e4", 0.4)], 1400: [("e4", 0.5)], 1600: [("e4", 0.6)]})
    agree = compare_position(same, board, 1200, analyzer, cfg)
    assert agree.stronger is None and agree.win_gap == 0 and not agree.matters


def test_compare_game() -> None:
    from chess_engine.config import CoachConfig, ReviewConfig
    from chess_engine.review.analyzer import review_pgn
    from chess_engine.review.level_compare import compare_game, known_from_review
    from test_review_analyzer import PGN, _scripted

    analyzer = _scripted()
    review = review_pgn(PGN, analyzer, ReviewConfig())
    known = known_from_review(review.moves[5])
    assert known.best.san == "g6" and known.line_for(chess.Move.from_uci("g8f6")).cp == -9997

    # Black's level always prefers the move Black played; the next level (+200) prefers the engine's move.
    class Levels:
        def predict_many(self, boards, ratings, **kwargs):
            from chess_engine.model.inference import MoveProbability, Prediction

            out = []
            for board, rating in zip(boards, ratings):
                played = next(m for m in review.moves if m.fen_before == board.fen())
                san = played.san if rating == 1300 else (played.best_san or played.san)
                out.append(Prediction([MoveProbability(0.5, board.parse_san(san), san)], None))
            return out

    cfg = CoachConfig(level_offsets=[200, 400], min_win_gap=8, max_moments=5)
    calls = analyzer.calls
    result = compare_game(review, "black", 1300, Levels(), analyzer, cfg, opp_rating=1350)
    assert len(result.moments) == 3 and result.your_level_match == 1.0
    assert [m.played_san for m in result.flagged] == ["Nf6"]                 # the only disagreement that matters
    flagged = result.flagged[0]
    from chess_engine.review.scoring import win_percent

    assert flagged.comparison.stronger.san == "g6"
    assert flagged.comparison.win_gap == pytest.approx(win_percent(-40) - 0)   # g6 keeps ~46%, Nf6 loses all
    assert flagged.move_number == "3..." and flagged.played_win == 0
    assert result.next_level_match == pytest.approx(2 / 3)                   # e5 and Nc6 match the next level too
    assert analyzer.calls == calls                                           # the review's lines were enough
