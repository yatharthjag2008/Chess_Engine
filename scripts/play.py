"""Play the human-like bot in your terminal (step 8).

The bot plays the moves humans at the chosen rating play, as often as they play
them (from the trained model in the config's `bot.model`).

Usage:
    python scripts/play.py                                # untimed, you are White, bot rated 1500
    python scripts/play.py --rating 1100 --color black
    python scripts/play.py --rating 1800 --time 3+2 --show-thinking
    python scripts/play.py --temperature 0.5              # the bot sticks closer to the most common moves
    python scripts/play.py --save-pgn my_game.pgn
    python scripts/play.py --hints 5                      # up to 5 hints (type 'hint'; needs Stockfish)

Type moves in SAN (Nf3, exd5, O-O) or UCI (g1f3); type 'help' during the game for commands.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

from chess_engine.bot.cli import TerminalGame, supports_unicode
from chess_engine.bot.clock import ChessClock, TimeControl
from chess_engine.bot.game import Game
from chess_engine.bot.hints import HintEngine
from chess_engine.bot.player import BotError, load_bot, sampling_from_config
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.config import ConfigError, load_config
from chess_engine.review.engine import StockfishAnalyzer
from chess_engine.stockfish import StockfishError, find_stockfish


def main(argv: list[str] | None = None, read=input, write=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--model", default=None, help="model file (default: bot.model in the config)")
    parser.add_argument("--rating", type=int, default=None, help="the bot's rating (default: bot.rating)")
    parser.add_argument("--your-rating", type=int, default=None, help="your rating (the bot is told; default: its own)")
    parser.add_argument("--color", choices=["white", "black", "random"], default="white", help="your side")
    parser.add_argument("--time", default=None, help="time control, e.g. 3+2 or 5+0 (default: untimed)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="reproducible bot moves")
    parser.add_argument("--show-thinking", action="store_true", help="show the bot's likeliest moves after each move")
    parser.add_argument("--ascii", action="store_true", help="letters instead of piece symbols")
    parser.add_argument("--save-pgn", default=None, help="write the game to this PGN file at the end")
    parser.add_argument("--hints", type=int, default=None,
                        help="hints allowed in the game (default: hints.limit in the config; 0 = none)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        base = sampling_from_config(cfg.bot)
        sampling = SamplingSettings(
            args.temperature if args.temperature is not None else base.temperature,
            args.top_k if args.top_k is not None else base.top_k,
            args.top_p if args.top_p is not None else base.top_p,
            base.min_prob,
        )
        control = TimeControl.parse(args.time) if args.time else None
        bot = load_bot(cfg, args.rating, args.model, sampling, args.seed)
    except (ConfigError, BotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    color = {"white": chess.WHITE, "black": chess.BLACK}.get(args.color, random.choice([chess.WHITE, chess.BLACK]))
    you_rating = args.your_rating
    bot_name = f"Bot {bot.rating}"
    game = (Game("You", bot_name, you_rating, bot.rating, event="Game vs. human-like bot") if color == chess.WHITE
            else Game(bot_name, "You", bot.rating, you_rating, event="Game vs. human-like bot"))
    clock = ChessClock(control) if control else None
    if control:
        write(f"Time control {control} (the clocks start after both first moves).")
    hint_limit = cfg.hints.limit if args.hints is None else max(0, min(args.hints, cfg.hints.max_limit))
    engine = None
    if hint_limit:
        try:
            analyzer = StockfishAnalyzer(str(find_stockfish(cfg.stockfish.path)), cfg.hints.depth, cfg.hints.multipv,
                                         cfg.stockfish.threads, cfg.stockfish.hash_mb)
            engine = HintEngine(bot.predictor, analyzer, cfg.hints)
        except StockfishError:
            write("Hints are off: Stockfish was not found (see the README).")
    terminal = TerminalGame(bot, color, game, clock, read=read, write=write,
                            unicode=not args.ascii and supports_unicode(), show_thinking=args.show_thinking,
                            hints=engine, hint_limit=hint_limit)
    try:
        terminal.play()
    finally:
        if engine is not None:
            engine.analyzer.close()

    if args.save_pgn and game.sans:
        path = Path(args.save_pgn)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(game.pgn(), encoding="utf-8")
        write(f"Game saved to {path}")
    elif game.sans:
        write("\n" + game.pgn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
