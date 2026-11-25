"""Tiered hints (step 17): move themes, how vague a hint is, the hint texts, the engine and the per-game limit."""

from __future__ import annotations

import chess
import pytest

from chess_engine.bot.hints import Detail, Theme, detail_for, find_rate, move_themes, threatened_piece
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
