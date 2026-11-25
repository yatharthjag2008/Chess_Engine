"""Write a reviewed game as an annotated PGN that any chess GUI or Lichess study can show.

Every move gets an ``[%eval]`` comment (White's view, in pawns, ``#n`` for mate, as
Lichess writes it); inaccuracies, mistakes and blunders get the standard glyphs
(``?!``, ``?``, ``??`` as NAGs 6, 2, 4), a short comment, and the engine's better
move as a variation.
"""

from __future__ import annotations

import chess
import chess.pgn

from chess_engine.review.analyzer import GameReview
from chess_engine.review.scoring import MATE_CP


def eval_comment(cp: int) -> str:
    """``[%eval 0.35]`` or ``[%eval #3]`` / ``[%eval #-2]`` (White's view)."""
    if abs(cp) >= MATE_CP - 1000:
        moves = MATE_CP - abs(cp)
        return f"[%eval #{moves if cp > 0 else -moves}]"
    return f"[%eval {cp / 100:.2f}]"


def annotated_pgn(review: GameReview, annotator: str = "Stockfish review") -> str:
    """The game as PGN text with evaluations, glyphs, comments and better moves."""
    game = chess.pgn.Game()
    for key, value in review.headers.items():
        game.headers[key] = value
    if review.start_fen != chess.STARTING_FEN:
        game.setup(chess.Board(review.start_fen))
    game.headers["Annotator"] = annotator + (f" (depth {review.depth})" if review.depth else "")
    node = game
    for m in review.moves:
        board = node.board()
        move = chess.Move.from_uci(m.uci)
        if m.classification.is_error and m.best_uci and m.best_uci != m.uci:
            better = chess.Move.from_uci(m.best_uci)
            if better in board.legal_moves:
                alt = node.add_variation(better)
                alt.comment = f"{m.best_san} was better"
        child = node.add_main_variation(move)
        comment = eval_comment(m.eval_after)
        if m.classification.nag is not None:
            child.nags.add(m.classification.nag)
            comment += (f" {m.classification.value.capitalize()}: {m.win_before:.0f}% -> {m.win_after:.0f}% "
                        "winning chances.")
        child.comment = comment
        node = child
    return str(game) + "\n"
