"""Tiered hints (step 17): move themes, how vague a hint is, the hint texts, the engine and the per-game limit."""

from __future__ import annotations

import chess
import pytest

from chess_engine.bot.hints import (
    Detail,
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
