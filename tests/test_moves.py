"""Tests for the move vocabulary and move <-> index mapping."""

from __future__ import annotations

import chess

from chess_engine.encoding.moves import MOVE_VOCABULARY, NUM_MOVES, UNDERPROMOTIONS

# ----------------------------------------------------------------------------- vocabulary


def test_vocabulary_size() -> None:
    assert NUM_MOVES == 1858
    plain = [m for m in MOVE_VOCABULARY if m.promotion is None]
    under = [m for m in MOVE_VOCABULARY if m.promotion is not None]
    assert len(plain) == 1792
    assert len(under) == 66


def test_vocabulary_entries_are_unique() -> None:
    assert len(set(MOVE_VOCABULARY)) == NUM_MOVES


def test_queen_and_knight_move_counts_per_square() -> None:
    from_counts = {sq: 0 for sq in chess.SQUARES}
    for move in MOVE_VOCABULARY:
        if move.promotion is None:
            from_counts[move.from_square] += 1
    # A corner square: 21 queen moves + 2 knight moves; a centre square: 27 + 8.
    assert from_counts[chess.A1] == 23
    assert from_counts[chess.D4] == 35
    assert from_counts[chess.H8] == 23


def test_vocabulary_matches_empty_board_attacks() -> None:
    """Plain entries are exactly the squares a queen or knight attacks on an empty board."""
    expected = set()
    for sq in chess.SQUARES:
        board = chess.Board(None)
        board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.WHITE))
        targets = set(board.attacks(sq))
        board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.WHITE))
        targets |= set(board.attacks(sq))
        expected |= {chess.Move(sq, t) for t in targets}
    assert {m for m in MOVE_VOCABULARY if m.promotion is None} == expected


def test_underpromotions_are_seventh_to_eighth_rank() -> None:
    for move in MOVE_VOCABULARY:
        if move.promotion is None:
            continue
        assert move.promotion in UNDERPROMOTIONS
        assert chess.square_rank(move.from_square) == 6
        assert chess.square_rank(move.to_square) == 7
        assert abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) <= 1


def test_castling_moves_are_in_vocabulary() -> None:
    vocab = set(MOVE_VOCABULARY)
    for uci in ["e1g1", "e1c1"]:
        assert chess.Move.from_uci(uci) in vocab
