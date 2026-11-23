"""Benchmark the board and move encoders on real positions (step 3).

Replays games from the step-2 Parquet output (or seeded random games if it does
not exist yet), then times each encoding function per position. Useful for
estimating how long step 4's dataset build will take.

Usage:
    python scripts/benchmark_encoding.py
    python scripts/benchmark_encoding.py --games 500 --games-file data/processed/games/lichess_2026-10_sample.parquet
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable
from pathlib import Path

import chess

from chess_engine.config import load_config
from chess_engine.data.game_store import games_output_path, read_games
from chess_engine.encoding import encode_board, legal_move_mask, move_to_index


def positions_from_games(path: Path, max_games: int) -> list[tuple[chess.Board, chess.Move]]:
    """(position, move played) pairs from the first ``max_games`` games of a games Parquet file."""
    df = read_games(path, columns=["moves"]).head(max_games)
    pairs = []
    for moves in df["moves"]:
        board = chess.Board()
        for san in moves:
            move = board.parse_san(san)
            pairs.append((board.copy(stack=False), move))
            board.push(move)
    return pairs


def positions_from_random_games(count: int, seed: int = 0) -> list[tuple[chess.Board, chess.Move]]:
    """(position, random legal move) pairs from seeded random games."""
    rng = random.Random(seed)
    pairs: list[tuple[chess.Board, chess.Move]] = []
    board = chess.Board()
    while len(pairs) < count:
        moves = list(board.legal_moves)
        if not moves or board.ply() > 150:
            board = chess.Board()
            continue
        move = rng.choice(moves)
        pairs.append((board.copy(stack=False), move))
        board.push(move)
    return pairs


def time_per_call(fn: Callable[[], object], calls: int) -> float:
    """Microseconds per call of ``fn`` (``calls`` calls in total)."""
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) / calls * 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-file", type=Path, default=None, help="games Parquet from step 2")
    parser.add_argument("--games", type=int, default=300, help="number of games to replay")
    args = parser.parse_args(argv)

    cfg = load_config()
    path = args.games_file or games_output_path(cfg, cfg.data.lichess.months[0])
    if path.is_file():
        start = time.perf_counter()
        pairs = positions_from_games(path, args.games)
        replay_us = (time.perf_counter() - start) / len(pairs) * 1e6
        print(f"{len(pairs):,} positions from {args.games} games in {path.name}")
        print(f"  replay (parse_san + push + copy) {replay_us:7.1f} us/position")
    else:
        pairs = positions_from_random_games(args.games * 70)
        print(f"{path} not found; using {len(pairs):,} positions from random games")

    n = len(pairs)
    timings = {
        "encode_board": time_per_call(lambda: [encode_board(b) for b, _ in pairs], n),
        "legal_move_mask": time_per_call(lambda: [legal_move_mask(b) for b, _ in pairs], n),
        "move_to_index": time_per_call(lambda: [move_to_index(m, b.turn) for b, m in pairs], n),
    }
    for name, us in timings.items():
        print(f"  {name:<32} {us:7.1f} us/position")
    total = sum(timings.values())
    print(f"  {'total encoding':<32} {total:7.1f} us/position  (~{1e6 / total:,.0f} positions/s on one core)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
