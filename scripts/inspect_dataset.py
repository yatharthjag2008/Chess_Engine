"""Print a built dataset's summary and decode a few random samples back into boards (step 4).

A sanity check that the stored samples mean what they should: each sample is
shown as the board, who is to move, their rating, the clocks and the move the
human actually played.

Usage:
    python scripts/inspect_dataset.py                      # default dataset, 3 samples from train
    python scripts/inspect_dataset.py --split test --samples 5 --seed 7
    python scripts/inspect_dataset.py --dataset data/processed/datasets/lichess_2026-10_sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np

from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import dataset_output_dir, format_summary
from chess_engine.data.shards import read_shard
from chess_engine.encoding.compact import board_from_compact
from chess_engine.encoding.moves import index_to_move


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--dataset", type=Path, default=None, help="dataset directory (default: from config)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=3, help="how many random samples to show")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        directory = args.dataset or dataset_output_dir(load_config(args.config))
        dataset = PositionDataset(directory, args.split, in_memory=False)
    except (ConfigError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(format_summary(dataset.manifest, directory))
    if not dataset.shard_paths or args.samples <= 0:
        return 0

    rng = np.random.default_rng(args.seed)
    shard = read_shard(dataset.shard_paths[int(rng.integers(len(dataset.shard_paths)))])
    labels = dataset.bucket_labels
    for i in rng.choice(len(shard), size=min(args.samples, len(shard)), replace=False):
        board = board_from_compact(shard["pieces"][i], int(shard["flags"][i]), int(shard["ep"][i]),
                                   int(shard["halfmove"][i]))
        move = index_to_move(int(shard["move"][i]), board)
        side = "White" if board.turn == chess.WHITE else "Black"
        result = {1: "won", 0: "drew", -1: "lost"}[int(shard["result"][i])]
        print(f"\n--- {args.split} sample: game {int(shard['game'][i])}, ply {int(shard['ply'][i])} ---")
        print(board)  # ASCII: uppercase = White, lowercase = Black
        print(f"{side} to move, rated {int(shard['rating'][i])} ({labels[int(shard['bucket'][i])]}) vs "
              f"{int(shard['opp_rating'][i])}; clock {_mmss(shard['clock'][i])} vs {_mmss(shard['opp_clock'][i])} "
              f"({int(shard['base_seconds'][i])}+{int(shard['increment'][i])})")
        print(f"played {board.san(move)} after {shard['time_spent'][i]:.0f}s "
              f"(one of {int(shard['legal_count'][i])} legal moves); the mover {result} the game")
    return 0


def _mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
