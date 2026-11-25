"""Simple, checkable facts about moves, used to explain mistakes in plain words.

Nothing here searches: these are one-move facts (what a move captures, whether it
checks or mates, which pieces it leaves hanging) that turn an engine verdict into a
sentence a player recognises, e.g. "this left your knight on f6 undefended".
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def piece_name(piece_type: chess.PieceType) -> str:
    return chess.piece_name(piece_type)


def captured_piece(board: chess.Board, move: chess.Move) -> chess.PieceType | None:
    """The type of piece ``move`` captures in ``board`` (en passant counts as a pawn), or None."""
    if board.is_en_passant(move):
        return chess.PAWN
    piece = board.piece_at(move.to_square)
    return piece.piece_type if piece is not None and board.is_capture(move) else None


def after(board: chess.Board, move: chess.Move) -> chess.Board:
    nxt = board.copy(stack=False)
    nxt.push(move)
    return nxt


def gives_mate(board: chess.Board, move: chess.Move) -> bool:
    return after(board, move).is_checkmate()


@dataclass(frozen=True)
class Hanging:
    """A piece that can be taken for free or for less than it is worth."""

    square: chess.Square
    piece_type: chess.PieceType

    def describe(self) -> str:
        return f"{piece_name(self.piece_type)} on {chess.square_name(self.square)}"


def hanging_pieces(board: chess.Board, color: chess.Color) -> list[Hanging]:
    """``color``'s pieces (not the king) that the opponent attacks and that are undefended,
    or attacked by a cheaper piece; most valuable first."""
    out = []
    for square in chess.SquareSet(board.occupied_co[color]):
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not color, square)
        if not attackers:
            continue
        defended = bool(board.attackers(color, square))
        cheapest = min(PIECE_VALUES[board.piece_at(a).piece_type] for a in attackers)
        if not defended or cheapest < PIECE_VALUES[piece.piece_type]:
            out.append(Hanging(square, piece.piece_type))
    return sorted(out, key=lambda h: -PIECE_VALUES[h.piece_type])


def newly_hanging(board: chess.Board, move: chess.Move) -> list[Hanging]:
    """The mover's pieces hanging after ``move`` that were not hanging before it (the moved piece included)."""
    color = board.turn
    before = {(h.square, h.piece_type) for h in hanging_pieces(board, color)}
    return [h for h in hanging_pieces(after(board, move), color) if (h.square, h.piece_type) not in before]
