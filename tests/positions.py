"""Positions shared by the encoding tests: hand-picked edge cases plus seeded random games."""

from __future__ import annotations

import random

import chess

# Positions chosen to exercise every special rule, for both colours.
SPECIAL_FENS: dict[str, str] = {
    "start": chess.STARTING_FEN,
    # "Kiwipete": castling both sides for both colours, pins, lots of captures.
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "kiwipete_black": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    # Promotions with and without capture, both colours.
    "white_promotes": "1r5k/2P5/8/8/8/8/5p2/K5R1 w - - 0 1",
    "black_promotes": "1r5k/2P5/8/8/8/8/5p2/K5R1 b - - 0 1",
    # En passant available: white pawn d5 can take e6; black pawn b4 can take a3 / c3.
    "white_en_passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "black_en_passant": "rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq a3 0 2",
    # Only some castling rights left.
    "partial_castling": "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
    # Black to move and in check from the bishop on b5.
    "in_check": "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3",
    # Fool's mate (White is checkmated) and a stalemate: no legal moves.
    "checkmate": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "stalemate": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    # Many queens: about the most legal moves a position can have.
    "queens": "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1",
}


def special_boards() -> dict[str, chess.Board]:
    """The hand-picked positions as boards."""
    return {name: chess.Board(fen) for name, fen in SPECIAL_FENS.items()}


def random_boards(count: int = 300, seed: int = 1234, max_plies: int = 120) -> list[chess.Board]:
    """Positions from seeded random games, at random depths, so both colours and all phases appear."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            # Prefer promotions and captures a bit so those cases show up often.
            special = [m for m in moves if m.promotion or board.is_capture(m)]
            board.push(rng.choice(special) if special and rng.random() < 0.3 else rng.choice(moves))
        boards.append(board)
    return boards


def all_boards() -> list[chess.Board]:
    """Special positions followed by random ones."""
    return list(special_boards().values()) + random_boards()
