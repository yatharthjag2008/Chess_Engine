"""Hints from the command line (step 17): for one position, or for every move of one player in a game.

Usage:
    python scripts/hint.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4" --rating 1200
    python scripts/hint.py --pgn game.pgn --color black --rating 1100      # what each hint would have said

For a position, all three tiers are printed (nudge, point, show). For a game, the nudge for each of the player's
positions, how often players at that rating find Stockfish's move there, and whether the player found it.
Needs Stockfish and the trained model (bot.model in the config).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import chess

from chess_engine.bot.hints import TIERS, HintEngine, HintError, Tier, make_hint
from chess_engine.config import ConfigError, load_config
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.review.engine import Analyzer, StockfishAnalyzer
from chess_engine.review.run import read_game
from chess_engine.stockfish import StockfishError, find_stockfish


def hint_position(engine: HintEngine, board: chess.Board, rating: int, write: Callable[[str], None]) -> None:
    best = engine.best(board, rating)
    write(f"Players rated ~{rating} find {board.san(best.move)} {best.find_rate:.0%} of the time"
          + (" (the only good move)" if best.only_move else "") + ".")
    for tier in TIERS:
        hint = make_hint(board, best.move, tier, best.find_rate, engine.cfg, best.only_move, best.winning)
        write(f"  {tier.value:6} {hint.text}")


def hint_game(engine: HintEngine, game: chess.pgn.Game, color: chess.Color, rating: int,
              write: Callable[[str], None]) -> Counter:
    """The nudge for each of ``color``'s positions; returns counts (positions, found, per detail and theme)."""
    counts: Counter = Counter()
    board = game.board()
    for move in game.mainline_moves():
        if board.turn == color:
            best = engine.best(board, rating)
            hint = make_hint(board, best.move, Tier.NUDGE, best.find_rate, engine.cfg, best.only_move, best.winning)
            found = move == best.move
            counts.update(["positions", f"detail:{hint.detail.name.lower()}", f"theme:{hint.themes[0]}"]
                          + (["found"] if found else []))
            number = f"{board.fullmove_number}{'.' if color == chess.WHITE else '...'}"
            write(f"{number:5} {board.san(move):7} best {board.san(best.move):7} {best.find_rate:4.0%} "
                  f"{'found ' if found else 'missed'}  {hint.text}")
        board.push(move)
    return counts


def make_engine(cfg, analyzer: Analyzer | None = None, model: str | None = None) -> HintEngine:
    predictor = load_model(model or cfg.bot.model).predictor()
    if analyzer is None:
        analyzer = StockfishAnalyzer(str(find_stockfish(cfg.stockfish.path)), cfg.hints.depth, cfg.hints.multipv,
                                     cfg.stockfish.threads, cfg.stockfish.hash_mb)
    return HintEngine(predictor, analyzer, cfg.hints)


def main(argv: list[str] | None = None, write: Callable[[str], None] = print, analyzer: Analyzer | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--fen", help="the position (the side to move gets the hint)")
    where.add_argument("--pgn", help="a PGN file: hints for every position of --color")
    parser.add_argument("--color", choices=["white", "black"], help="the player in --pgn")
    parser.add_argument("--index", type=int, default=1, help="which game of the PGN (1 = first)")
    parser.add_argument("--rating", type=int, default=1500, help="the player's rating")
    parser.add_argument("--model", default=None, help="model file (default: bot.model in the config)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if args.pgn and not args.color:
        parser.error("--pgn needs --color")
    try:
        cfg = load_config(args.config)
        engine = make_engine(cfg, analyzer, args.model)
        if args.fen:
            hint_position(engine, chess.Board(args.fen), args.rating, write)
        else:
            game = read_game(Path(args.pgn).read_text(encoding="utf-8"), args.index - 1)
            color = chess.WHITE if args.color == "white" else chess.BLACK
            counts = hint_game(engine, game, color, args.rating, write)
            n = counts["positions"]
            write(f"\n{n} positions; found Stockfish's move in {counts['found']} ({counts['found'] / max(n, 1):.0%}).")
            write("Nudge detail: " + ", ".join(f"{d} {counts['detail:' + d]}" for d in ("vague", "theme", "piece")))
            themes = sorted(((k[6:], v) for k, v in counts.items() if k.startswith("theme:")), key=lambda kv: -kv[1])
            write("Themes: " + ", ".join(f"{t} {v}" for t, v in themes))
    except (ConfigError, StockfishError, ModelFileError, HintError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if analyzer is None and "engine" in locals() and hasattr(engine.analyzer, "close"):
            engine.analyzer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
