"""Coach one player's game: their mistakes, and where stronger players would play better (step 15).

Reviews the game with Stockfish (cached), then compares each of the player's moves
with what players of their rating and of +200 / +400 usually play, and explains the
moments that matter.

Usage:
    python scripts/coach_game.py game.pgn --player MyName
    python scripts/coach_game.py game.pgn --color black --rating 1350
    python scripts/coach_game.py --lichess MyName                  # your latest Lichess game
    python scripts/coach_game.py game.pgn --player MyName --json coach.json

Your rating is taken from --rating, else from the game's PGN, else estimated from your moves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.game_import import GameImportError, ImportCache, import_games
from chess_engine.model.inference import ModelFileError
from chess_engine.review.coach import format_report, report_to_dict
from chess_engine.review.run import run_coach
from chess_engine.stockfish import StockfishError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pgn", nargs="?", help="PGN file")
    parser.add_argument("--lichess", metavar="USER", help="coach this player's latest Lichess game")
    parser.add_argument("--chesscom", metavar="USER", help="coach this player's latest Chess.com game")
    parser.add_argument("--player", help="the player to coach (name in the PGN)")
    parser.add_argument("--color", choices=["white", "black"], help="the side to coach")
    parser.add_argument("--rating", type=int, default=None, help="the player's rating")
    parser.add_argument("--index", type=int, default=1, help="which game of the PGN (1 = first)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)
    if sum(x is not None for x in (args.pgn, args.lichess, args.chesscom)) != 1:
        parser.error("give a PGN file, --lichess USER or --chesscom USER")
    player = args.player or args.lichess or args.chesscom
    if player is None and args.color is None:
        parser.error("say who to coach: --player NAME or --color white/black")
    try:
        cfg = load_config(args.config)
        if args.pgn:
            text = Path(args.pgn).read_text(encoding="utf-8")
        else:
            source, user = ("lichess", args.lichess) if args.lichess else ("chess.com", args.chesscom)
            result = import_games(cfg.game_import, source, username=user, max_games=1,
                                  cache=ImportCache(cfg.paths.data_dir / "imports", cfg.game_import.cache_minutes))
            if not result.games:
                raise GameImportError(f"no games found for {user} on {source}")
            text = result.games[0].pgn
        print("Reviewing with Stockfish and comparing levels...", flush=True)
        coached = run_coach(cfg, text, player=None if args.color else player, color=args.color,
                            rating=args.rating, game_index=args.index - 1 if args.pgn else 0)
    except (ConfigError, GameImportError, StockfishError, ModelFileError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if coached.rating_source == "estimated":
        print(f"(rating estimated from this game's moves: about {coached.report.rating}; pass --rating for better advice)")
    print("\n" + format_report(coached.report, cfg.coach.level_offsets[0]))
    if args.json:
        Path(args.json).write_text(json.dumps(report_to_dict(coached.report), indent=1), encoding="utf-8")
        print(f"\ncoach report JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
