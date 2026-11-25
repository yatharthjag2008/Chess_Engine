"""Explain mistakes and level differences in plain words, from templates.

Every sentence is built from facts that can be checked on the board (what a move
captures, mates, or leaves hanging; see :mod:`chess_engine.review.features`) and
from the engine's numbers; nothing is generated freely, so an explanation is never
more confident than its evidence.

Mistakes (inaccuracy or worse) are explained by the first rule that applies:

1. the best move mates: "There was mate in one: Qxf7#."
2. the move played allows a forced mate: "This allows a forced mate."
3. the move leaves one of your pieces hanging: "This leaves your bishop on d4 undefended."
4. the best move wins material: "You could have won a knight with Bxf6."
5. otherwise: "Nf3 was stronger (+0.3 instead of -0.2)."

followed by what it cost: "Your winning chances dropped from 55% to 31%."

Level differences (:func:`explain_level`) say what your level and the stronger level
play and what the difference is worth, then what you actually played.
"""

from __future__ import annotations

import chess

from chess_engine.review.analyzer import MoveReview
from chess_engine.review.features import captured_piece, gives_mate, newly_hanging, piece_name
from chess_engine.review.level_compare import LevelMoment
from chess_engine.review.scoring import MATE_CP


def _pawns(cp: int) -> str:
    if abs(cp) >= MATE_CP - 1000:
        return "a forced mate" if cp > 0 else "getting mated"
    return f"{cp / 100:+.1f}"


def explain_mistake(move: MoveReview) -> str:
    """A short explanation of why ``move`` was worse than the engine's choice (see the module docstring)."""
    board = chess.Board(move.fen_before)
    played = chess.Move.from_uci(move.uci)
    best = chess.Move.from_uci(move.best_uci) if move.best_uci else None
    sentences = []
    other_best = best is not None and best != played
    won = captured_piece(board, best) if other_best else None
    if other_best and gives_mate(board, best):
        sentences.append(f"There was mate in one: {move.best_san}.")
    elif move.played_cp <= -(MATE_CP - 1000):
        sentences.append("This allows a forced mate.")
    elif hanging := newly_hanging(board, played):
        sentences.append(f"This leaves your {hanging[0].describe()} undefended.")
    elif won is not None and move.best_cp - move.played_cp >= 150:
        sentences.append(f"You could have won a {piece_name(won)} with {move.best_san}.")
    elif move.best_san and move.best_san != move.san:
        sentences.append(f"{move.best_san} was stronger ({_pawns(move.best_cp)} instead of {_pawns(move.played_cp)}).")
    sentences.append(f"Your winning chances dropped from {move.win_before:.0f}% to {move.win_after:.0f}%.")
    return " ".join(sentences)


def explain_level(moment: LevelMoment) -> str:
    """What players at your rating and at a higher rating play here, and what you played."""
    c = moment.comparison
    yours, stronger = c.yours, c.stronger
    text = f"Players around {yours.rating} usually play {yours.san} here ({yours.prob:.0%})."
    if stronger is None:
        return text + " Stronger players agree."
    text += (f" Players around {stronger.rating} prefer {stronger.san} ({stronger.prob:.0%}), which keeps "
             f"{stronger.win:.0f}% winning chances instead of {yours.win:.0f}%.")
    if moment.played_uci == stronger.uci:
        text += f" You found {stronger.san}: that is the stronger players' move."
    elif moment.played_uci == yours.uci:
        text += f" You played {moment.played_san}, like most players at your level."
    else:
        text += f" You played {moment.played_san} ({moment.played_win:.0f}%)."
    return text
