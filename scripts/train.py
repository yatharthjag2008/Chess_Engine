"""Train the rating-conditioned move model on the built dataset (step 6).

Each run gets a directory under ``checkpoints/`` with its config, a metrics log,
checkpoints for resuming, ``best_model.pt`` (lowest validation loss) and
``final_model.pt``. Ctrl+C stops cleanly after the current step.

Usage:
    python scripts/train.py                                  # new run with configs/default.yaml
    python scripts/train.py --max-minutes 60 --threads 4     # laptop: one hour, then stop with a checkpoint
    python scripts/train.py --config configs/full.yaml --run-name full_v1
    python scripts/train.py --resume latest                  # continue the most recent run
    python scripts/train.py --resume full_v1 --max-minutes 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from chess_engine.config import Config, ConfigError, TrainingConfig, load_config
from chess_engine.training.checkpoint import CheckpointError
from chess_engine.training.device import DeviceError
from chess_engine.training.run_log import CHECKPOINTS_DIR, RunDirectory, default_run_name
from chess_engine.training.train import Trainer, TrainingError


def _log(line: str) -> None:
    print(line, flush=True)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "max_steps": args.max_steps, "max_minutes": args.max_minutes, "epochs": args.epochs,
        "device": args.device, "threads": args.threads, "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }
    return {key: value for key, value in values.items() if value is not None}


def _with_training(cfg: Config, overrides: dict[str, Any]) -> Config:
    if not overrides:
        return cfg
    training = TrainingConfig.model_validate({**cfg.training.model_dump(), **overrides})
    return cfg.model_copy(update={"training": training})


def find_run(name: str, checkpoints_dir: Path) -> Path:
    """A run given as a path, a run name under ``checkpoints_dir``, or ``latest`` (newest checkpoint)."""
    if name == "latest":
        runs = [p.parent.parent for p in checkpoints_dir.glob(f"*/{CHECKPOINTS_DIR}/step_*.pt")]
        if not runs:
            raise TrainingError(f"no run with a checkpoint in {checkpoints_dir}")
        return max(set(runs), key=lambda run: max(p.stat().st_mtime for p in (run / CHECKPOINTS_DIR).iterdir()))
    path = Path(name)
    return path if path.is_dir() else checkpoints_dir / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--run-name", default=None, help="run directory name (default: date-time + model size)")
    parser.add_argument("--resume", default=None, metavar="RUN",
                        help="continue a run: its directory, its name, or 'latest'")
    parser.add_argument("--dataset", default=None, help="dataset directory (default: from the config)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None, help="stop this session after N minutes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="peak learning rate")
    args = parser.parse_args(argv)
    if args.resume and (args.config or args.run_name):
        parser.error("--resume uses the run's own config; do not combine it with --config or --run-name")

    try:
        cfg = load_config(args.config)
        overrides = _overrides(args)
        if args.resume:
            run_dir = find_run(args.resume, cfg.paths.checkpoints_dir)
            trainer = Trainer.from_run(run_dir, args.dataset, overrides, log=_log)
            trainer.resume()
        else:
            cfg = _with_training(cfg, overrides)
            run_dir = cfg.paths.checkpoints_dir / (args.run_name or default_run_name(cfg))
            if RunDirectory(run_dir).exists():
                raise TrainingError(f"run {run_dir} already exists; use --resume {run_dir.name} or another --run-name")
            trainer = Trainer(cfg, run_dir, args.dataset, log=_log)
        result = trainer.train()
    except (ConfigError, TrainingError, CheckpointError, DeviceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.stop_reason in ("time_limit", "interrupted", "paused"):
        print(f"\nContinue with: python scripts/train.py --resume {trainer.run.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
