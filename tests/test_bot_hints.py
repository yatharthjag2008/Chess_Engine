"""Tiered hints (step 17): move themes, how vague a hint is, the hint texts, the engine and the per-game limit."""

from __future__ import annotations

import chess
import pytest

from chess_engine.bot.hints import Theme, move_themes, threatened_piece


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
