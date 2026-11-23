"""Download full Lichess monthly dumps into data/raw/ (resumable, SHA-256 verified).

Only needed for full-scale runs: sample mode streams games directly from the
server without downloading. Each month is ~30 GB. If the download is interrupted,
run the same command again and it continues where it stopped.

Usage:
    python scripts/download_lichess.py                     # months from configs/default.yaml
    python scripts/download_lichess.py --month 2026-09
    python scripts/download_lichess.py --no-verify         # skip the checksum check
"""

from __future__ import annotations

import argparse
import logging
import sys

from chess_engine.config import ConfigError, load_config
from chess_engine.data.sources import ChecksumError, HTTPStreamError, download_lichess_month


def main(argv: list[str] | None = None) -> int:
    """Download each requested month; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--month", action="append", help="YYYY-MM to download (repeatable); default: config months")
    parser.add_argument("--no-verify", action="store_true", help="do not check against sha256sums.txt")
    parser.add_argument("--no-progress", action="store_true", help="hide the progress bar")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    lichess = cfg.data.lichess
    for month in args.month or lichess.months:
        try:
            path = download_lichess_month(
                lichess.base_url, month, cfg.paths.raw_dir,
                verify_checksum=lichess.verify_checksum and not args.no_verify,
                show_progress=not args.no_progress,
            )
        except (ChecksumError, HTTPStreamError) as exc:
            print(f"{month}: FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"{month}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
