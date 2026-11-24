"""Plot a training run's curves (loss, top-1, top-1 per rating bucket, learning rate) to reports/ (step 6).

Works on a run that is still going: the metrics log is written line by line.

Usage:
    python scripts/plot_training.py                       # the most recent run
    python scripts/plot_training.py laptop_64x6           # a run name under checkpoints/
    python scripts/plot_training.py checkpoints/my_run --out reports/my_run.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.training.history import load_history, plot_training_curves, summarize_history
from chess_engine.training.run_log import METRICS_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=None, help="run directory or name (default: most recent run)")
    parser.add_argument("--config", default=None, help="config file (for checkpoints_dir / reports_dir)")
    parser.add_argument("--out", default=None, help="PNG path (default: reports/training/<run>.png)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.run is None:
        logs = sorted(cfg.paths.checkpoints_dir.glob(f"*/{METRICS_NAME}"), key=lambda p: p.stat().st_mtime)
        if not logs:
            print(f"error: no training runs in {cfg.paths.checkpoints_dir}", file=sys.stderr)
            return 1
        run_dir = logs[-1].parent
    else:
        run_dir = Path(args.run) if Path(args.run).is_dir() else cfg.paths.checkpoints_dir / args.run
    history = load_history(run_dir)
    if history.empty:
        print(f"error: {run_dir} has no training records yet", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else cfg.paths.reports_dir / "training" / f"{history.run}.png"
    plot_training_curves(history, out)
    summary = summarize_history(history)
    print(f"{history.run}: {summary['steps']:,} steps, last train loss {summary['last_train_loss']:.4f}")
    if "best_val_step" in summary:
        print(f"  best validation: loss {summary['best_val_loss']:.4f}, top-1 {summary['best_val_top1']:.1%} "
              f"at step {summary['best_val_step']:,}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
