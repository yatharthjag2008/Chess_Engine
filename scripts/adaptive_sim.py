"""Simulate adaptive difficulty (step 18): a stand-in player of known strength plays the adaptive bot.

The "player" is Stockfish limited to an Elo (or the bot at a fixed rating). Between games, the adaptive bot
should settle near the rating where it scores about half against that player; in live mode, its rating should
move within the game towards whoever is winning.

Usage:
    python scripts/adaptive_sim.py --player stockfish:1500 --mode between --games 30
    python scripts/adaptive_sim.py --player bot:1100 --mode live --games 10 --start 1600

Games alternate colours and are adjudicated by Stockfish (a side 8 pawns up for 8 plies wins; at the move limit
the evaluation decides). Your real profile is not touched: the simulation keeps its own in a temporary file.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import chess

from chess_engine.bot.adaptive import AdaptiveGame, ProfileStore, bot_rating_for
from chess_engine.bot.game import Game, Outcome
from chess_engine.bot.player import BotPlayer
from chess_engine.config import AdaptiveConfig, ConfigError, load_config
from chess_engine.eval.players import BotAgent, Judge, Player, StockfishPlayer
from chess_engine.model.inference import ModelFileError, MovePredictor, load_model
from chess_engine.stockfish import StockfishError, find_stockfish


def play_adaptive_game(bot: BotPlayer, adaptive: AdaptiveGame, player: Player, player_color: chess.Color,
                       judge: Judge | None = None, max_plies: int = 200, adjudicate_cp: int = 800,
                       adjudicate_plies: int = 8) -> Outcome:
    """One game between ``player`` and the adaptive ``bot`` (live changes applied as the bot moves)."""
    game = Game()
    streak_side, streak, evaluation = None, 0, 0
    while not game.is_over:
        if len(game.sans) >= max_plies:
            winner = (chess.WHITE if evaluation > 0 else chess.BLACK) if abs(evaluation) >= adjudicate_cp else None
            return game._end(winner, f"move limit ({max_plies} plies)")  # noqa: SLF001
        if game.turn == player_color:
            game.push(player.choose(game.board, opponent_rating=bot.rating))
        else:
            choice = bot.choose_move(game.board, opp_rating=player.rating)
            game.push(choice.move)
            adaptive.after_bot_move(bot, choice.wdl)
        if judge is None or game.is_over:
            continue
        evaluation = judge.evaluate(game.board)
        side = chess.WHITE if evaluation >= adjudicate_cp else chess.BLACK if evaluation <= -adjudicate_cp else None
        streak = streak + 1 if side is not None and side == streak_side else (1 if side is not None else 0)
        streak_side = side
        if streak >= adjudicate_plies:
            return game._end(side, "adjudication")  # noqa: SLF001
    return game.outcome


def simulate(predictor: MovePredictor, player: Player, cfg: AdaptiveConfig, mode: str, games: int, start: int,
             profile_path: Path, judge: Judge | None = None, max_plies: int = 200,
             write: Callable[[str], None] = print) -> list[int]:
    """Play ``games`` games; returns the bot's starting rating in each game (and its final rating in live mode)."""
    store = ProfileStore(profile_path, cfg)
    store.reset(start)
    ratings = []
    for g in range(games):
        rating = store.load().bot_rating(cfg) if mode == "between" else start
        bot = BotPlayer(predictor, rating, seed=g)
        adaptive = AdaptiveGame(mode, rating, cfg)
        player_color = chess.WHITE if g % 2 == 0 else chess.BLACK
        outcome = play_adaptive_game(bot, adaptive, player, player_color, judge, max_plies)
        profile = adaptive.finish(store, outcome, player_color, rated=True)
        score = 1.0 if outcome.winner == player_color else 0.5 if outcome.winner is None else 0.0
        live = (f" -> {bot.rating} ({len(adaptive.live.changes)} changes)" if adaptive.live else "")
        write(f"game {g + 1:3}: bot {rating}{live}, player scores {score:g} ({outcome.termination}); "
              f"player rating {profile.rating:.0f}")
        ratings.append(bot.rating)
    return ratings


def make_player(spec: str, predictor: MovePredictor, stockfish: str | None) -> Player:
    kind, _, value = spec.partition(":")
    if kind == "stockfish" and stockfish:
        return StockfishPlayer(stockfish, int(value))
    if kind == "bot":
        return BotAgent(BotPlayer(predictor, int(value), seed=12345), name=f"bot {value}")
    raise ValueError(f"--player must be stockfish:ELO (needs Stockfish) or bot:RATING, got {spec!r}")


def main(argv: list[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player", default="stockfish:1500", help="stockfish:ELO or bot:RATING")
    parser.add_argument("--mode", choices=["between", "live"], default="between")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--start", type=int, default=None, help="starting rating (default: adaptive.start_rating)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--no-judge", action="store_true", help="no Stockfish adjudication")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        predictor = load_model(args.model or cfg.bot.model).predictor()
        stockfish = None
        if args.player.startswith("stockfish") or not args.no_judge:
            stockfish = str(find_stockfish(cfg.stockfish.path))
        player = make_player(args.player, predictor, stockfish)
        judge = None if args.no_judge else Judge(stockfish)
        start = args.start or cfg.adaptive.start_rating
        with tempfile.TemporaryDirectory() as tmp:
            ratings = simulate(predictor, player, cfg.adaptive, args.mode, args.games, start,
                               Path(tmp) / "profile.json", judge, args.max_plies, write)
        player.close()
        if judge is not None:
            judge.close()
    except (ConfigError, StockfishError, ModelFileError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    tail = ratings[-10:]
    write(f"\nThe bot's rating over the last {len(tail)} games: mean {sum(tail) / len(tail):.0f} "
          f"(range {min(tail)}-{max(tail)}); next bot {bot_rating_for(tail[-1], cfg.adaptive)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
