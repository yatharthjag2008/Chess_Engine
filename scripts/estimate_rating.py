"""Estimate a player's rating from their games (step 12).

Reads PGN games (a file, or - for standard input), picks the player's moves, and
prints an estimate with a likely range, from all games together and per game.

Usage:
    python scripts/estimate_rating.py my_games.pgn --player MyLichessName
    python scripts/estimate_rating.py game.pgn --color white
    python scripts/estimate_rating.py game.pgn --color black --ignore-opponent-rating

The range width uses the tempering fitted by scripts/evaluate_rating_estimation.py
(read from its results.json) unless --temper or rating_estimation.temper is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset_builder import dataset_output_dir
from chess_engine.eval.evaluate import model_name
from chess_engine.eval.player_games import player_games_from_pgn
from chess_engine.eval.rating_estimation import estimate_rating, make_log_prior
from chess_engine.model.inference import ModelFileError, load_model


def fitted_temper(results: Path) -> float | None:
    try:
        return float(json.loads(results.read_text(encoding="utf-8"))["temper"]["temper"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def main(argv: list[str] | None = None, stdin=sys.stdin) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pgn", help="PGN file with one or more games, or - to read standard input")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--player", help="the player's name as in the White / Black headers")
    who.add_argument("--color", choices=["white", "black"], help="estimate this side of every game")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="model file (default: bot.model)")
    parser.add_argument("--temper", type=float, default=None, help="widen the range (default: the fitted value)")
    parser.add_argument("--ignore-opponent-rating", action="store_true",
                        help="do not give the opponent's rating to the model (assume similar strength)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        r = cfg.rating_estimation
        text = stdin.read() if args.pgn == "-" else Path(args.pgn).read_text(encoding="utf-8")
        color = {"white": chess.WHITE, "black": chess.BLACK}.get(args.color)
        games = player_games_from_pgn(text, player=args.player, color=color)
        if not games:
            raise ValueError("no games with moves by that player in the PGN")
        model_path = Path(args.model).resolve() if args.model else cfg.bot.model
        loaded = load_model(model_path)
        dataset_dir = dataset_output_dir(cfg)
        if r.prior == "data" and not (dataset_dir / "games.parquet").is_file():
            print("note: no dataset found for the rating prior; using a flat prior", file=sys.stderr)
            r = r.model_copy(update={"prior": "flat"})
        log_prior = make_log_prior(r, dataset_dir)
        temper = args.temper or r.temper or fitted_temper(
            cfg.paths.reports_dir / "rating" / model_name(model_path) / "results.json")
        if temper is None:
            print("note: no fitted tempering found (run scripts/evaluate_rating_estimation.py); the range will be "
                  "too narrow", file=sys.stderr)
            temper = 1.0
        result = estimate_rating(loaded.predictor(), games, r, log_prior, temper,
                                 use_opponent_rating=not args.ignore_opponent_rating)
    except (ConfigError, ModelFileError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    name = args.player or f"{args.color} in every game"
    print(f"{name}: {result.overall.describe()}")
    if len(games) > 1:
        print("\nper game:")
    for game, est in zip(games, result.per_game):
        side = "White" if game.color == chess.WHITE else "Black"
        actual = f", rated {game.rating} in the game" if game.rating else ""
        opp = f"vs {game.opp_rating}" if game.opp_rating else "vs ?"
        print(f"  {side:<5} {opp:<8} {len(game):>3} moves: {est.estimate:5.0f}  "
              f"({est.low:.0f}-{est.high:.0f}){actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
