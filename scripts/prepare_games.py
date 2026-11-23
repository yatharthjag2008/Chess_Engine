"""Stream Lichess games, filter them, and save the survivors to Parquet (step 2).

By default each month in the config is read from data/raw/ if it has been
downloaded, otherwise streamed straight from database.lichess.org. In sample mode
(the default config) only the first N accepted games are kept, so this takes a
few minutes and reads only a small slice of each ~30 GB file.

Output (per month):
    data/processed/games/lichess_<month>[_sample].parquet
    data/processed/games/lichess_<month>[_sample].stats.json

Usage:
    python scripts/prepare_games.py                          # months from configs/default.yaml
    python scripts/prepare_games.py --max-games 2000         # quick smoke run
    python scripts/prepare_games.py --month 2026-09 --source path/to/file.pgn.zst
    python scripts/prepare_games.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.game_store import prepare_games


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--month", action="append", help="YYYY-MM to process (repeatable); default: config months")
    parser.add_argument("--source", default=None, help="local .pgn/.pgn.zst file or URL (only with one month)")
    parser.add_argument("--output", type=Path, default=None, help="output .parquet path (only with one month)")
    parser.add_argument("--max-games", type=int, default=None, help="stop after this many accepted games")
    parser.add_argument("--no-progress", action="store_true", help="hide the progress bar")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    months = args.month or cfg.data.lichess.months
    if (args.source or args.output) and len(months) != 1:
        parser.error("--source and --output need exactly one month")

    for month in months:
        result = prepare_games(
            cfg, month, source=args.source, output_path=args.output,
            max_games=args.max_games, show_progress=not args.no_progress,
        )
        summary = json.loads(result.stats_path.read_text(encoding="utf-8"))
        print(f"\n{month}: kept {summary['games_accepted']:,} of {summary['games_read']:,} games "
              f"({summary['acceptance_rate']:.1%}) in {summary['seconds']:.0f}s, "
              f"{summary['compressed_mb_read']:.0f} MB read")
        for reason, count in summary["rejected_by_reason"].items():
            print(f"  rejected {reason:<16} {count:>9,}")
        print(f"  -> {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
