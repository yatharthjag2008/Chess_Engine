"""Play bots of different ratings against each other and report the scores (step 8 sanity check).

A working rating dial should make the higher-rated bot score well above 50%.
The full validation (more games, Stockfish levels) is step 11.

Usage:
    python scripts/bot_match.py                              # 1100 vs 1900, 40 games
    python scripts/bot_match.py --ratings 900 1500 2100 --games 60
    python scripts/bot_match.py --temperature 0.5
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

from chess_engine.bot.player import BotError, load_bot, sampling_from_config
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.bot.selfplay import play_match
from chess_engine.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ratings", type=int, nargs="+", default=[1100, 1900], help="ratings to pair up (2+)")
    parser.add_argument("--games", type=int, default=40, help="games per pairing (colours alternate)")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if len(args.ratings) < 2:
        parser.error("give at least two ratings")
    try:
        cfg = load_config(args.config)
        base = sampling_from_config(cfg.bot)
        sampling = SamplingSettings(args.temperature if args.temperature is not None else base.temperature,
                                    base.top_k, base.top_p, base.min_prob)
        bots = {r: load_bot(cfg, r, args.model, sampling, seed=args.seed + i) for i, r in enumerate(args.ratings)}
    except (ConfigError, BotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.games} games per pairing, {bots[args.ratings[0]].describe().split('; ', 1)[1].rstrip(')')}")
    for low, high in itertools.combinations(sorted(args.ratings), 2):
        start = time.perf_counter()
        result = play_match(bots[low], bots[high], args.games, args.max_plies)
        print(f"{result.summary()}  [{time.perf_counter() - start:.0f} s]")
        print("  endings: " + ", ".join(f"{k} {v}" for k, v in result.terminations.most_common()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
