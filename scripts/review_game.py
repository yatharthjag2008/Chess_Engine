"""Review a game with Stockfish: accuracy, mistakes and the key moments (step 14).

Usage:
    python scripts/review_game.py game.pgn                    # the first game of the file
    python scripts/review_game.py games.pgn --index 3         # the fourth game
    python scripts/review_game.py --lichess SomeUser          # that player's most recent Lichess game
    python scripts/review_game.py game.pgn --depth 18 --annotated reviewed.pgn --json review.json

Reviews are cached in data/reviews/, so showing the same game again is instant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chess_engine.config import ConfigError, load_config
from chess_engine.data.game_import import GameImportError, ImportCache, import_games
from chess_engine.review.analyzer import key_moments, review_to_dict
from chess_engine.review.annotate import annotated_pgn
from chess_engine.review.run import run_review
from chess_engine.stockfish import StockfishError


def format_moves(review) -> str:
    """``1. e4 (0.3)  e5 (0.2)`` lines, errors marked with their glyph."""
    lines = []
    for i in range(0, len(review.moves), 2):
        pair = review.moves[i:i + 2]
        cells = [f"{m.san}{m.classification.symbol if m.classification.is_error else ''} ({m.eval_after / 100:+.1f})"
                 for m in pair]
        lines.append(f"{pair[0].move_number:<5} " + "   ".join(f"{c:<16}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pgn", nargs="?", help="PGN file")
    parser.add_argument("--lichess", metavar="USER", help="review this player's latest Lichess game")
    parser.add_argument("--chesscom", metavar="USER", help="review this player's latest Chess.com game")
    parser.add_argument("--index", type=int, default=1, help="which game of the PGN (1 = first)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--depth", type=int, default=None, help="Stockfish depth (default: stockfish.analysis_depth)")
    parser.add_argument("--annotated", default=None, help="write the annotated PGN here")
    parser.add_argument("--json", default=None, help="write the full review as JSON here")
    parser.add_argument("--quiet", action="store_true", help="no move list")
    args = parser.parse_args(argv)
    if sum(x is not None for x in (args.pgn, args.lichess, args.chesscom)) != 1:
        parser.error("give a PGN file, --lichess USER or --chesscom USER")

    try:
        cfg = load_config(args.config)
        if args.depth:
            cfg = cfg.model_copy(update={"stockfish": cfg.stockfish.model_copy(update={"analysis_depth": args.depth})})
        if args.pgn:
            text = Path(args.pgn).read_text(encoding="utf-8")
        else:
            source, user = ("lichess", args.lichess) if args.lichess else ("chess.com", args.chesscom)
            result = import_games(cfg.game_import, source, username=user, max_games=1,
                                  cache=ImportCache(cfg.paths.data_dir / "imports", cfg.game_import.cache_minutes))
            if not result.games:
                raise GameImportError(f"no games found for {user} on {source}")
            text = result.games[0].pgn
        print("Reviewing with Stockfish (depth %d)..." % cfg.stockfish.analysis_depth, flush=True)
        review = run_review(cfg, text, game_index=args.index - 1 if args.pgn else 0)
    except (ConfigError, GameImportError, StockfishError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    h = review.headers
    print(f"\n{h.get('White', '?')} - {h.get('Black', '?')}  {h.get('Result', '*')}")
    for side in review.summary().values():
        print("  " + side.describe())
    moments = key_moments(review, cfg.review.key_moments)
    if moments:
        print("\nKey moments:")
        for k in moments:
            print("  " + k.text)
    if not args.quiet:
        print("\n" + format_moves(review))
    if args.annotated:
        Path(args.annotated).write_text(annotated_pgn(review), encoding="utf-8")
        print(f"\nannotated PGN: {args.annotated}")
    if args.json:
        Path(args.json).write_text(json.dumps(review_to_dict(review, cfg.review.key_moments), indent=1),
                                   encoding="utf-8")
        print(f"review JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
