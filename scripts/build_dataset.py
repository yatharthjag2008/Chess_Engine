"""Turn the filtered games from step 2 into training-sample shards, split by game (step 4).

Reads data/processed/games/lichess_<month>[_sample].parquet for every month in
the config (run scripts/prepare_games.py first) and writes:

    data/processed/datasets/lichess_<months>[_sample]/
        manifest.json, games.parquet, train/*.npz, val/*.npz, test/*.npz

Usage:
    python scripts/build_dataset.py                       # default config, all CPU cores but one
    python scripts/build_dataset.py --max-games 2000      # quick smoke run
    python scripts/build_dataset.py --workers 1 --overwrite
    python scripts/build_dataset.py --games-file a.parquet --games-file b.parquet --output data/processed/datasets/x
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset_builder import DatasetExistsError, build_dataset, format_summary


def main(argv: list[str] | None = None) -> int:
    """Build the dataset; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--games-file", type=Path, action="append", help="games Parquet (repeatable); default: config months")
    parser.add_argument("--output", type=Path, default=None, help="output directory")
    parser.add_argument("--max-games", type=int, default=None, help="use only the first N games")
    parser.add_argument("--workers", type=int, default=None, help="encoding processes (default: from config)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset")
    parser.add_argument("--no-compress", action="store_true", help="write uncompressed shards (bigger, faster to load)")
    parser.add_argument("--no-progress", action="store_true", help="hide the progress bar")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.config)
        result = build_dataset(
            cfg, games_paths=args.games_file, output_dir=args.output, max_games=args.max_games,
            workers=args.workers, overwrite=args.overwrite, compress=not args.no_compress,
            show_progress=not args.no_progress,
        )
    except (ConfigError, FileNotFoundError, DatasetExistsError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print()
    print(format_summary(result.manifest, result.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
