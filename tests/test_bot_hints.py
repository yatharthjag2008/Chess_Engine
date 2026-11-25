"""Tiered hints (step 17): move themes, how vague a hint is, the hint texts, the engine and the per-game limit."""

from __future__ import annotations

import chess
import pytest

from chess_engine.bot.hints import (
    Detail,
    HintBudget,
    HintLimitError,
    HintEngine,
    HintError,
    Theme,
    Tier,
    detail_for,
    find_rate,
    make_hint,
    move_themes,
    nudge_text,
    threatened_piece,
)
from chess_engine.config import HintsConfig
from review_helpers import ScriptedAnalyzer, position_key


def legal(fen: str, uci: str) -> tuple[chess.Board, chess.Move]:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} is not legal in {fen}"
    return board, move


@pytest.mark.parametrize(
    ("fen", "uci", "expected"),
    [
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", [Theme.MATE]),                     # back-rank mate
        ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", "a1a8", [Theme.CHECK]),
        ("4k3/8/8/3r4/8/4N3/8/4K3 w - - 0 1", "e3d5", [Theme.WIN_MATERIAL]),              # an undefended rook
        ("4k3/1p6/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", [Theme.CAPTURE]),               # a defended pawn: a trade
        ("8/P6k/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", [Theme.PROMOTION]),
        ("4k3/8/8/2p5/3N4/8/8/4K3 w - - 0 1", "d4f3", [Theme.SAVE_PIECE]),                # the knight was attacked by a pawn
        ("3rk3/8/8/3N4/8/8/8/R3K3 w Q - 0 1", "a1d1", [Theme.DEFEND]),                    # the rook defends the knight
        ("4k3/8/8/8/r7/8/8/1N2K3 w - - 0 1", "b1c3", [Theme.THREAT]),                     # the knight attacks the rook
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", [Theme.CASTLE]),
        (chess.STARTING_FEN, "e2e4", [Theme.QUIET]),
    ],
)
def test_move_themes(fen: str, uci: str, expected: list[Theme]) -> None:
    board, move = legal(fen, uci)
    assert move_themes(board, move) == expected


def test_themes_combine_in_order() -> None:
    board, move = legal("4k3/8/8/8/8/8/1q6/R3K3 w Q - 0 1", "a1a8")   # check, and the queen on b2 is not taken
    assert move_themes(board, move)[0] is Theme.CHECK
    board, move = legal("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8q")    # takes the rook while promoting, with check
    themes = move_themes(board, move)
    assert themes[:2] == [Theme.CHECK, Theme.WIN_MATERIAL] and Theme.PROMOTION in themes


def test_threatened_piece() -> None:
    board, move = legal("4k3/8/8/8/r7/8/8/1N2K3 w - - 0 1", "b1c3")
    assert threatened_piece(board, move) == chess.ROOK
    board, move = legal(chess.STARTING_FEN, "e2e4")
    assert threatened_piece(board, move) is None


class FixedPredictor:
    """Stands in for the model: a fixed probability per move (UCI) at every rating; remembers the calls."""

    def __init__(self, probs: dict[str, float]) -> None:
        self.probs = probs
        self.calls: list[tuple[str, int, object]] = []

    def predict(self, board: chess.Board, rating: int, **kwargs: object):
        from chess_engine.model.inference import MoveProbability, Prediction

        self.calls.append((board.fen(), rating, kwargs.get("opp_ratings")))
        moves = [MoveProbability(self.probs.get(m.uci(), 0.0), m, board.san(m)) for m in board.legal_moves]
        return Prediction(sorted(moves, reverse=True), None)


def test_find_rate_is_the_models_probability_at_your_rating() -> None:
    predictor = FixedPredictor({"e2e4": 0.4, "d2d4": 0.3})
    board = chess.Board()
    assert find_rate(predictor, board, chess.Move.from_uci("e2e4"), 1300, opp_rating=1500) == 0.4
    assert find_rate(predictor, board, chess.Move.from_uci("h2h4"), 1300) == 0.0
    assert predictor.calls[0][1:] == (1300, 1500)


@pytest.mark.parametrize(("rate", "detail"), [(0.9, Detail.VAGUE), (0.35, Detail.VAGUE), (0.34, Detail.THEME),
                                              (0.10, Detail.THEME), (0.09, Detail.PIECE), (0.0, Detail.PIECE)])
def test_detail_follows_the_find_rate(rate: float, detail: Detail) -> None:
    assert detail_for(rate, HintsConfig()) is detail


def test_nudges_give_more_away_for_rarely_found_moves() -> None:
    themes = [Theme.WIN_MATERIAL]
    vague = nudge_text(themes, chess.KNIGHT, Detail.VAGUE)
    theme = nudge_text(themes, chess.KNIGHT, Detail.THEME)
    piece = nudge_text(themes, chess.KNIGHT, Detail.PIECE)
    assert vague == "There is a forcing move here: look at checks, captures and threats."
    assert theme == "One of your opponent's pieces can be won."
    assert piece == "One of your opponent's pieces can be won. It's a knight move."
    assert "knight" not in vague and "knight" not in theme


def test_nudge_wording_by_theme() -> None:
    assert nudge_text([Theme.SAVE_PIECE], chess.BISHOP, Detail.VAGUE).startswith("Before anything else")
    assert nudge_text([Theme.QUIET], chess.PAWN, Detail.VAGUE).startswith("No need for anything sharp")
    assert nudge_text([Theme.THREAT], chess.KNIGHT, Detail.THEME, threatened=chess.ROOK) == "You can attack their rook."
    assert nudge_text([Theme.MATE], chess.ROOK, Detail.PIECE) == "There is a checkmate! It's a rook move."
    assert nudge_text([Theme.QUIET], chess.PAWN, Detail.THEME, only_move=True).endswith(
        "Careful: only one move keeps your position together.")
    for theme in Theme:      # every theme has words at every detail
        for detail in Detail:
            assert nudge_text([theme], chess.QUEEN, detail)


def test_the_three_tiers() -> None:
    board, move = legal("4k3/8/8/3r4/8/4N3/8/4K3 w - - 0 1", "e3d5")
    cfg = HintsConfig()
    nudge = make_hint(board, move, Tier.NUDGE, 0.2, cfg)
    assert nudge.text == "One of your opponent's pieces can be won." and nudge.from_square is None
    assert nudge.detail is Detail.THEME and nudge.themes == ["win_material"] and nudge.san is None
    point = make_hint(board, move, Tier.POINT, 0.2, cfg)
    assert point.text.endswith("Move your knight on e3.") and point.from_square == "e3" and point.to_square is None
    show = make_hint(board, move, Tier.SHOW, 0.2, cfg)
    assert show.text == "Play Nxd5: it wins material." and (show.from_square, show.to_square) == ("e3", "d5")
    assert show.san == "Nxd5"
    data = show.to_dict()
    assert data["tier"] == "show" and data["detail"] == "theme" and data["find_rate"] == 0.2


def test_show_names_up_to_two_reasons() -> None:
    board, move = legal("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8q")
    assert make_hint(board, move, Tier.SHOW, 0.5, HintsConfig()).text == "Play bxa8=Q+: a check, and it wins material."
    board, move = legal(chess.STARTING_FEN, "e2e4")
    assert make_hint(board, move, Tier.SHOW, 0.5, HintsConfig()).text == "Play e4."


KNIGHT_WINS_ROOK = "4k3/8/8/3r4/8/4N3/8/4K3 w - - 0 1"


def test_engine_hints_about_stockfishs_move_at_your_level() -> None:
    board = chess.Board(KNIGHT_WINS_ROOK)
    analyzer = ScriptedAnalyzer({ScriptedAnalyzer.key(board): ("e3d5", 500, {"e1d2": 0})})
    predictor = FixedPredictor({"e3d5": 0.05})
    engine = HintEngine(predictor, analyzer, HintsConfig())
    nudge = engine.hint(board, Tier.NUDGE, 1200, opp_rating=1400)
    assert nudge.detail is Detail.PIECE and nudge.text.startswith("One of your opponent's pieces can be won. It's a knight")
    assert nudge.only_move, "500 cp against 0 is far more than 15 win-% points"
    show = engine.hint(board, Tier.SHOW, 1200, opp_rating=1400)
    assert show.san == "Nxd5"
    assert analyzer.calls == 1 and len(predictor.calls) == 1, "the next tier reuses the analysis"
    assert predictor.calls[0][1:] == (1200, 1400)
    engine.hint(board, Tier.NUDGE, 1800)
    assert analyzer.calls == 2, "another rating is another question"


def test_engine_easy_move_and_no_only_move() -> None:
    board = chess.Board()
    analyzer = ScriptedAnalyzer({position_key(): ("e2e4", 30, {"d2d4": 25})})
    engine = HintEngine(FixedPredictor({"e2e4": 0.6}), analyzer, HintsConfig())
    hint = engine.hint(board, Tier.NUDGE, 1500)
    assert hint.detail is Detail.VAGUE and not hint.only_move and hint.find_rate == 0.6


def test_engine_cache_is_bounded() -> None:
    engine = HintEngine(FixedPredictor({}), ScriptedAnalyzer(), HintsConfig(), cache_size=2)
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        engine.best(board, 1500)
        board.push_uci(uci)
    assert len(engine._cache) == 2


def test_no_hint_when_the_game_is_over() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    board.push_uci("a1a8")
    engine = HintEngine(FixedPredictor({}), ScriptedAnalyzer(), HintsConfig())
    with pytest.raises(HintError, match="over"):
        engine.hint(board, Tier.NUDGE, 1500)


def test_budget_counts_positions_and_escalates_within_one() -> None:
    budget = HintBudget(limit=2)
    board = chess.Board()
    assert budget.rated and budget.left == 2
    assert [budget.take(board) for _ in range(4)] == [Tier.NUDGE, Tier.POINT, Tier.SHOW, Tier.SHOW]
    assert budget.used == 1 and not budget.rated and budget.left == 1
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert budget.take(board, Tier.SHOW) is Tier.SHOW, "a tier can be asked for directly"
    assert budget.next_tier(board) is Tier.SHOW and budget.left == 0
    board.push_uci("g1f3")
    board.push_uci("b8c6")
    with pytest.raises(HintLimitError, match="no hints left"):
        budget.take(board)
    board.pop()
    board.pop()
    assert budget.take(board) is Tier.SHOW, "a position already hinted can still be shown"
    assert budget.used == 2


def test_budget_off_and_position_identity() -> None:
    with pytest.raises(HintLimitError, match="off"):
        HintBudget(limit=0).take(chess.Board())
    budget = HintBudget(limit=1)
    a = chess.Board()
    budget.take(a)
    same = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:     # back to the start position, move counters differ
        same.push_uci(uci)
    assert budget.next_tier(same) is Tier.POINT
