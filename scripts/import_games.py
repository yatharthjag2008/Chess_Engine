"""Import a player's recent games from Lichess or Chess.com, or games from a PGN file (step 13).

Prints the games, saves them as one PGN file (default data/imports/<source>_<user>.pgn),
and with --estimate also estimates the player's rating from their moves (step 12).

Usage:
    python scripts/import_games.py --lichess SomeUser
    python scripts/import_games.py --chesscom SomeUser --max 50 --speed blitz rapid --rated
    python scripts/import_games.py --pgn my_games.pgn --out cleaned.pgn
    python scripts/import_games.py --lichess SomeUser --estimate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset_builder import dataset_output_dir
from chess_engine.data.game_import import GameImportError, ImportCache, import_games, pgn_text
from chess_engine.eval.player_games import player_games_from_pgn
from chess_engine.eval.rating_estimation import estimate_rating, make_log_prior
from chess_engine.model.inference import ModelFileError, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--lichess", metavar="USER")
    where.add_argument("--chesscom", metavar="USER")
    where.add_argument("--pgn", metavar="FILE")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max", type=int, default=None, help="most recent games (default: game_import.max_games)")
    parser.add_argument("--speed", nargs="+", default=None, choices=["ultrabullet", "bullet", "blitz", "rapid",
                                                                     "classical", "correspondence"])
    parser.add_argument("--rated", action="store_true", help="rated games only")
    parser.add_argument("--out", default=None, help="where to save the PGN")
    parser.add_argument("--no-cache", action="store_true", help="always fetch fresh")
    parser.add_argument("--estimate", action="store_true", help="also estimate the player's rating (needs a model)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        source, username, text = "pgn", None, None
        if args.lichess:
            source, username = "lichess", args.lichess
        elif args.chesscom:
            source, username = "chess.com", args.chesscom
        else:
            text = Path(args.pgn).read_text(encoding="utf-8")
        cache = None if args.no_cache else ImportCache(cfg.paths.data_dir / "imports", cfg.game_import.cache_minutes)
        result = import_games(cfg.game_import, source, username, text, args.max, args.speed,
                              args.rated or None, cache=cache)
    except (ConfigError, GameImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    games = result.games
    who = f"{result.username} on {source}" if result.username else args.pgn
    print(f"{len(games)} games from {who}")
    for g in games:
        we = g.white_elo or "?"
        be = g.black_elo or "?"
        print(f"  {g.date or '?':<10}  {g.white} ({we}) - {g.black} ({be})  {g.result:<7}  "
              f"{g.speed or '?':<14} {g.plies:>3} plies")
    if not games:
        return 0
    out = Path(args.out) if args.out else cfg.paths.data_dir / "imports" / f"{source.replace('.', '')}_{result.username or 'pasted'}.pgn"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pgn_text(games), encoding="utf-8")
    print(f"saved {out}")

    if args.estimate:
        if not result.username:
            print("error: --estimate needs --lichess or --chesscom (to know which side is the player)", file=sys.stderr)
            return 1
        try:
            mine = player_games_from_pgn(pgn_text(games), player=result.username)
            loaded = load_model(cfg.bot.model)
            r = cfg.rating_estimation
            if r.prior == "data" and not (dataset_output_dir(cfg) / "games.parquet").is_file():
                r = r.model_copy(update={"prior": "flat"})
            estimate = estimate_rating(loaded.predictor(), mine, r, make_log_prior(r, dataset_output_dir(cfg)))
        except (ModelFileError, FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"rating estimate from {len(mine)} games: {estimate.overall.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
