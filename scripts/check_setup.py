"""Check that the development environment is ready.

Prints the Python version, key package versions, the loaded config, and whether
Stockfish was found and responds. Exits with code 1 if something is missing.

Usage:
    python scripts/check_setup.py
    python scripts/check_setup.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from chess_engine.config import ConfigError, load_config
from chess_engine.stockfish import StockfishError, get_stockfish

PACKAGES = ["chess", "torch", "numpy", "pandas", "pyarrow", "zstandard", "pydantic", "fastapi", "pytest"]


def main(argv: list[str] | None = None) -> int:
    """Run all checks and return a process exit code (0 = everything OK)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file to merge on top of configs/default.yaml")
    args = parser.parse_args(argv)
    ok = True

    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 11):
        print("  !! Python 3.11+ is required")
        ok = False

    print("\nPackages:")
    for name in PACKAGES:
        try:
            print(f"  {name:<10} {version(name)}")
        except PackageNotFoundError:
            print(f"  {name:<10} MISSING  (run: pip install -e \".[dev]\")")
            ok = False

    print("\nConfig:")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"  !! {exc}")
        return 1
    print(f"  data dir       {config.paths.data_dir}")
    print(f"  speeds         {config.data.time_control.speeds}")
    print(f"  rating buckets {config.ratings.bucket_labels()}")
    sample = config.data.sample
    print(f"  sample mode    {'ON, first ' + format(sample.max_games, ',') + ' games' if sample.enabled else 'OFF'}")

    print("\nStockfish:")
    try:
        info = get_stockfish(config)
        print(f"  OK  {info.name}  ({info.path}); test move from start: {info.test_move}")
    except StockfishError as exc:
        print("  " + str(exc).replace("\n", "\n  "))
        ok = False

    print("\nAll checks passed." if ok else "\nSome checks failed (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
