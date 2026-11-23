"""Tests for the move vocabulary and move <-> index mapping."""

from __future__ import annotations

import chess
import pytest

from chess_engine.encoding.moves import (
    MOVE_VOCABULARY,
    NUM_MOVES,
    UNDERPROMOTIONS,
    index_to_move,
    legal_move_indices,
    legal_move_mask,
    masked_moves,
    move_to_index,
    orient_square,
)
from positions import all_boards, special_boards

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


# ----------------------------------------------------------------------------- move <-> index

BOARDS = all_boards()


def test_every_legal_move_roundtrips() -> None:
    """move -> index -> move gives back the same move, in every test position."""
    checked = 0
    for board in BOARDS:
        for move in board.legal_moves:
            index = move_to_index(move, board.turn)
            assert 0 <= index < NUM_MOVES
            assert index_to_move(index, board) == move, (board.fen(), move.uci())
            checked += 1
    assert checked > 5000


def test_legal_moves_get_distinct_indices() -> None:
    for board in BOARDS:
        indices = [move_to_index(m, board.turn) for m in board.legal_moves]
        assert len(indices) == len(set(indices)), board.fen()


def test_mirrored_position_gives_same_indices() -> None:
    """Black's moves are encoded as if Black were White: board.mirror() must give identical indices."""
    for board in BOARDS:
        mirrored = board.mirror()
        ours = sorted(move_to_index(m, board.turn) for m in board.legal_moves)
        theirs = sorted(move_to_index(m, mirrored.turn) for m in mirrored.legal_moves)
        assert ours == theirs, board.fen()


def test_specific_indices_by_perspective() -> None:
    # e2e4 for White and e7e5 for Black are the same move from the mover's point of view.
    assert move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE) == move_to_index(
        chess.Move.from_uci("e7e5"), chess.BLACK
    )
    assert move_to_index(chess.Move.from_uci("e1g1"), chess.WHITE) == move_to_index(
        chess.Move.from_uci("e8g8"), chess.BLACK
    )
    assert move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE) != move_to_index(
        chess.Move.from_uci("e2e4"), chess.BLACK
    )


def test_queen_promotion_shares_plain_index() -> None:
    board = special_boards()["white_promotes"]
    plain = move_to_index(chess.Move.from_uci("c7c8"), chess.WHITE)
    assert move_to_index(chess.Move.from_uci("c7c8q"), chess.WHITE) == plain
    assert index_to_move(plain, board) == chess.Move.from_uci("c7c8q")
    under = {move_to_index(chess.Move.from_uci(f"c7c8{p}"), chess.WHITE) for p in "nbr"}
    assert len(under) == 3 and plain not in under


def test_black_promotions() -> None:
    board = special_boards()["black_promotes"]
    for uci in ["f2f1q", "f2f1n", "f2g1r", "f2g1b"]:
        move = chess.Move.from_uci(uci)
        assert index_to_move(move_to_index(move, chess.BLACK), board) == move


def test_plain_index_on_non_pawn_does_not_promote() -> None:
    # A rook going from the 7th to the 8th rank uses the same index as a pawn promotion would.
    board = chess.Board("4k3/R7/8/8/8/8/8/4K3 w - - 0 1")
    index = move_to_index(chess.Move.from_uci("a7a8"), chess.WHITE)
    assert index_to_move(index, board) == chess.Move.from_uci("a7a8")


@pytest.mark.parametrize(
    "uci, turn",
    [
        ("a1c4", chess.WHITE),   # neither a queen line nor a knight jump
        ("e2e4n", chess.WHITE),  # "underpromotion" not from the 7th rank
        ("e2e4q", chess.WHITE),  # "queen promotion" not from the 7th rank
        ("e7e8n", chess.BLACK),  # for Black, e7 is (mirrored) its 2nd rank
        ("a7c8n", chess.WHITE),  # promotion "capture" two files away
    ],
)
def test_impossible_moves_raise(uci: str, turn: chess.Color) -> None:
    with pytest.raises(ValueError, match="not in the move vocabulary"):
        move_to_index(chess.Move.from_uci(uci), turn)


def test_king_promotion_raises() -> None:
    with pytest.raises(ValueError):
        move_to_index(chess.Move(chess.A7, chess.A8, promotion=chess.KING), chess.WHITE)


@pytest.mark.parametrize("index", [-1, NUM_MOVES, 10_000])
def test_index_out_of_range(index: int) -> None:
    with pytest.raises(IndexError):
        index_to_move(index, chess.Board())


def test_orient_square() -> None:
    assert orient_square(chess.E2, chess.WHITE) == chess.E2
    assert orient_square(chess.E2, chess.BLACK) == chess.E7
    assert orient_square(chess.A8, chess.BLACK) == chess.A1


# ----------------------------------------------------------------------------- legal-move mask


def test_mask_matches_legal_moves_everywhere() -> None:
    for board in BOARDS:
        mask = legal_move_mask(board)
        assert mask.shape == (NUM_MOVES,)
        assert mask.dtype == bool
        assert mask.sum() == board.legal_moves.count(), board.fen()
        assert set(masked_moves(board, mask)) == set(board.legal_moves), board.fen()


def test_mask_start_position() -> None:
    board = chess.Board()
    mask = legal_move_mask(board)
    assert mask.sum() == 20
    assert mask[move_to_index(chess.Move.from_uci("e2e4"))]
    assert not mask[move_to_index(chess.Move.from_uci("e2e5"))]


def test_mask_special_moves() -> None:
    boards = special_boards()
    kiwi = boards["kiwipete"]
    assert legal_move_mask(kiwi)[move_to_index(chess.Move.from_uci("e1g1"), chess.WHITE)]
    assert legal_move_mask(kiwi)[move_to_index(chess.Move.from_uci("e1c1"), chess.WHITE)]
    kiwi_black = boards["kiwipete_black"]
    assert legal_move_mask(kiwi_black)[move_to_index(chess.Move.from_uci("e8c8"), chess.BLACK)]
    ep = boards["black_en_passant"]
    assert legal_move_mask(ep)[move_to_index(chess.Move.from_uci("b4a3"), chess.BLACK)]
    promo = legal_move_mask(boards["white_promotes"])
    assert sum(promo[move_to_index(chess.Move.from_uci(f"c7c8{p}"))] for p in "qrbn") == 4


@pytest.mark.parametrize("name", ["checkmate", "stalemate"])
def test_mask_no_legal_moves(name: str) -> None:
    board = special_boards()[name]
    assert legal_move_mask(board).sum() == 0
    assert legal_move_indices(board).size == 0


def test_legal_move_indices_sorted() -> None:
    indices = legal_move_indices(special_boards()["queens"])
    assert indices.size == 218
    assert list(indices) == sorted(indices)


def test_chess960_rejected() -> None:
    with pytest.raises(ValueError, match="Chess960"):
        legal_move_mask(chess.Board(chess960=True))
