"""Train a style (a player's way of playing) or a ghost (you, up to a date) for the bot (step 19).

Usage:
    python scripts/train_style.py --pgn games.pgn --player "Some Player" --name some-player
    python scripts/train_style.py --lichess YourName --name me-may --until 2025-05-31     # ghost: you, to a date
    python scripts/train_style.py --list
    python scripts/train_style.py --delete me-may

Then play it: python scripts/play.py --style some-player (or pick it in the browser).
A style needs at least style.min_games games; with fewer than style.blend_games it is blended with the rating
model, so it only nudges the bot's choices. Training takes seconds to minutes on a CPU.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Callable
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.game_import import GameImportError, ImportCache, import_games
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.style.build import build_style, library_for
from chess_engine.style.store import StyleError, blend_weight


def _date(text: str | None) -> datetime.date | None:
    return datetime.date.fromisoformat(text) if text else None


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pgn", help="a PGN file with the player's games")
    source.add_argument("--lichess", metavar="USER", help="fetch the player's recent Lichess games")
    source.add_argument("--chesscom", metavar="USER", help="fetch the player's recent Chess.com games")
    source.add_argument("--list", action="store_true", help="list the trained styles")
    source.add_argument("--delete", metavar="NAME", help="delete a style")
    parser.add_argument("--player", help="the player's name in the PGN (default: the username)")
    parser.add_argument("--name", help="the style's name (lowercase letters, digits, - and _)")
    parser.add_argument("--until", help="ghost mode: only games up to this date (YYYY-MM-DD)")
    parser.add_argument("--since", help="only games from this date (YYYY-MM-DD)")
    parser.add_argument("--rating", type=int, default=None, help="the player's rating if the games do not say")
    parser.add_argument("--max-games", type=int, default=200, help="games to fetch from a site")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        library = library_for(cfg)
        if args.list:
            styles = library.list()
            if not styles:
                write("No styles yet (train one with --pgn or --lichess).")
            for info in styles:
                m = info.metrics
                how = (f"top-1 {m['base_top1']:.0%} -> {m['styled_top1']:.0%} on held-out games" if m else "")
                when = f", games up to {info.until}" if info.until else ""
                write(f"{info.name:20} {info.kind:5} {info.player} ({info.games} games{when}, ~{info.rating}; "
                      f"weight {blend_weight(info.games, cfg.style):.2f}) {how}")
            return 0
        if args.delete:
            library.delete(args.delete)
            write(f"Deleted {args.delete}.")
            return 0
        if not (args.pgn or args.lichess or args.chesscom):
            parser.error("give --pgn FILE, --lichess USER or --chesscom USER (or --list)")
        if not args.name:
            parser.error("--name is required to train a style")
        if args.pgn:
            if not args.player:
                parser.error("--player is required with --pgn")
            text = Path(args.pgn).read_text(encoding="utf-8")
        else:
            site, user = ("lichess", args.lichess) if args.lichess else ("chess.com", args.chesscom)
            write(f"Fetching up to {args.max_games} games of {user} from {site}...")
            result = import_games(cfg.game_import, site, username=user, max_games=args.max_games,
                                  cache=ImportCache(cfg.paths.data_dir / "imports", cfg.game_import.cache_minutes))
            text = "\n\n".join(g.pgn for g in result.games)
        player = args.player or args.lichess or args.chesscom
        model_path = Path(args.model or cfg.bot.model)
        base = load_model(model_path).model
        info = build_style(cfg, base, text, args.name, player, until=_date(args.until), since=_date(args.since),
                           rating=args.rating, base_model=model_path.name, log=write)
    except (ConfigError, ModelFileError, GameImportError, StyleError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    m = info.metrics
    write(f"Saved {info.kind} {info.name!r}: {info.games} games, ~{info.rating}, "
          f"weight {blend_weight(info.games, cfg.style):.2f}.")
    if m:
        write(f"On held-out games the bot now predicts {info.player}'s move {m['styled_top1']:.1%} of the time "
              f"(the rating model alone: {m['base_top1']:.1%}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
