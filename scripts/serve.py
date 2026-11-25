"""Run the API server (step 9): play the human-like bot over HTTP.

Loads the model in the config's `bot.model` once, then serves the endpoints
described in chess_engine/api/app.py. Interactive API docs: http://127.0.0.1:8000/docs

Usage:
    python scripts/serve.py                          # http://127.0.0.1:8000
    python scripts/serve.py --port 9000
    python scripts/serve.py --host 0.0.0.0           # reachable from other devices on your network
    python scripts/serve.py --model checkpoints/my_run/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from chess_engine.api.app import create_app
from chess_engine.config import ConfigError, load_config


def build_app(args: argparse.Namespace):
    cfg = load_config(args.config)
    if args.model:
        cfg = cfg.model_copy(update={"bot": cfg.bot.model_copy(update={"model": Path(args.model).resolve()})})
    if not cfg.bot.model.is_file():
        raise FileNotFoundError(f"model file not found: {cfg.bot.model} (train one with scripts/train.py, "
                                "or pass --model)")
    return cfg, create_app(cfg)


def main(argv: list[str] | None = None, run=uvicorn.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="model file (default: bot.model in the config)")
    parser.add_argument("--host", default=None, help="default: api.host (127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="default: api.port (8000)")
    args = parser.parse_args(argv)
    try:
        cfg, app = build_app(args)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    host, port = args.host or cfg.api.host, args.port or cfg.api.port
    print(f"Serving the bot ({cfg.bot.model.name}) on http://{host}:{port}  (API docs: /docs)", flush=True)
    run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
