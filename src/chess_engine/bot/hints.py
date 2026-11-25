"""Tiered hints while playing the bot: nudge -> point -> show.

A hint is about the best move by Stockfish in the current position. It comes in three tiers:

* **nudge**: what kind of move to look for ("there is a way to win material"), from one-move board facts;
* **point**: which piece to move (its square is highlighted);
* **show**: the move itself.

How vague the nudge is depends on how often players at your rating find the move (the model's probability for it
at your rating): a move most players find gets the vaguest nudge, a rarely found one names the piece type too.
"""

from __future__ import annotations

from enum import Enum

import chess

from chess_engine.review.features import (
    PIECE_VALUES,
    after,
    captured_piece,
    gives_mate,
    hanging_pieces,
)


class Theme(str, Enum):
    """What a move does, by one-move board facts (most important first)."""

    MATE = "mate"
    CHECK = "check"
    WIN_MATERIAL = "win_material"     # takes a piece that was hanging (free, or worth more than the taker)
    CAPTURE = "capture"               # any other capture (a trade, or a recapture)
    PROMOTION = "promotion"
    SAVE_PIECE = "save_piece"         # the moved piece was hanging and no longer is
    DEFEND = "defend"                 # another of your hanging pieces is safe afterwards
    THREAT = "threat"                 # attacks an opponent's piece that was safe, leaving it hanging
    CASTLE = "castle"
    QUIET = "quiet"


def move_themes(board: chess.Board, move: chess.Move) -> list[Theme]:
    """The themes of ``move`` in ``board`` (the side to move's move), most important first; at least one."""
    themes: list[Theme] = []
    nxt = after(board, move)
    me, them = board.turn, not board.turn
    if gives_mate(board, move):
        themes.append(Theme.MATE)
    elif nxt.is_check():
        themes.append(Theme.CHECK)

    taken = captured_piece(board, move)
    if taken is not None:
        hanging_before = {h.square for h in hanging_pieces(board, them)}
        target = move.to_square if not board.is_en_passant(move) else chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square))
        if target in hanging_before:
            themes.append(Theme.WIN_MATERIAL)
        else:
            themes.append(Theme.CAPTURE)
    if move.promotion:
        themes.append(Theme.PROMOTION)

    mine_before = {h.square for h in hanging_pieces(board, me)}
    mine_after = {h.square for h in hanging_pieces(nxt, me)}
    if move.from_square in mine_before and move.to_square not in mine_after:
        themes.append(Theme.SAVE_PIECE)
    elif any(sq != move.from_square and sq not in mine_after for sq in mine_before) and not nxt.is_checkmate():
        themes.append(Theme.DEFEND)

    theirs_before = {h.square for h in hanging_pieces(board, them)}
    new_targets = [h for h in hanging_pieces(nxt, them)
                   if h.square not in theirs_before and h.square != move.to_square
                   and move.to_square in nxt.attackers(me, h.square)]
    if new_targets and Theme.MATE not in themes:
        themes.append(Theme.THREAT)
    if board.is_castling(move):
        themes.append(Theme.CASTLE)
    return themes or [Theme.QUIET]


def threatened_piece(board: chess.Board, move: chess.Move) -> chess.PieceType | None:
    """The most valuable opponent piece ``move`` newly leaves hanging (for "attack the queen"), or None."""
    nxt = after(board, move)
    them = not board.turn
    before = {h.square for h in hanging_pieces(board, them)}
    targets = [h for h in hanging_pieces(nxt, them) if h.square not in before
               and move.to_square in nxt.attackers(board.turn, h.square)]
    return max(targets, key=lambda h: PIECE_VALUES[h.piece_type]).piece_type if targets else None
