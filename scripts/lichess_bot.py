"""Run the human-like bot on a Lichess BOT account (step 20, optional).

You need a Lichess account upgraded to a BOT account (an account that has never played a game; see
https://lichess.org/api#tag/Bot/operation/botAccountUpgrade) and an API token with the ``bot:play`` scope.
Put the token in the environment, never on the command line:

    set LICHESS_BOT_TOKEN=lip_...            (Windows)      export LICHESS_BOT_TOKEN=lip_...   (macOS / Linux)

Usage:
    python scripts/lichess_bot.py --rating 1500                     # always plays like a 1500
    python scripts/lichess_bot.py --match-opponent --offset -100    # 100 below each opponent's rating
    python scripts/lichess_bot.py --style some-player --casual-only

It accepts standard chess challenges at the chosen speeds (one game at a time) and plays until you stop it
(Ctrl+C). Lichess's bot rules apply: no playing on behalf of a human, and the BOT title is shown to opponents.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from chess_engine.bot.player import BotError, BotPlayer, check_rating, sampling_from_config
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.config import ConfigError, load_config
from chess_engine.lichess_bot import ChallengePolicy, GameSettings, LichessClient, LichessError, run_bot
from chess_engine.model.inference import ModelFileError, load_model
from chess_engine.style.build import library_for
from chess_engine.style.predictor import StylePredictor
from chess_engine.style.store import StyleError, blend_weight

TOKEN_VARIABLE = "LICHESS_BOT_TOKEN"


def main(argv: list[str] | None = None, write: Callable[[str], None] = print, session=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rating", type=int, default=1500, help="the bot's rating (unless --match-opponent)")
    parser.add_argument("--match-opponent", action="store_true", help="play at each opponent's rating")
    parser.add_argument("--offset", type=int, default=0, help="with --match-opponent: points above (+) or below (-)")
    parser.add_argument("--speeds", nargs="+", default=["blitz", "rapid", "classical"],
                        choices=["ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence"])
    rated = parser.add_mutually_exclusive_group()
    rated.add_argument("--casual-only", action="store_true")
    rated.add_argument("--rated-only", action="store_true")
    parser.add_argument("--style", default=None, help="play in a trained style or ghost")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-games", type=int, default=None, help="stop after this many games")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_VARIABLE, "").strip()
    if not token:
        print(f"error: set {TOKEN_VARIABLE} to a Lichess API token with the bot:play scope", file=sys.stderr)
        return 1
    try:
        cfg = load_config(args.config)
        check_rating(args.rating)
        predictor = load_model(args.model or cfg.bot.model).predictor()
        if args.style:
            adapter, info = library_for(cfg).load(args.style, predictor.model)
            predictor = StylePredictor(predictor, adapter, blend_weight(info.games, cfg.style), info)
        base = sampling_from_config(cfg.bot)
        sampling = SamplingSettings(args.temperature if args.temperature is not None else base.temperature,
                                    base.top_k, base.top_p, base.min_prob)
        client = LichessClient(token, session=session)
        account = client.account()
        if account.get("title") != "BOT":
            raise LichessError(f"{account.get('username', account.get('id'))} is not a BOT account "
                               "(upgrade it first; see this script's help)")
        policy = ChallengePolicy(speeds=args.speeds, rated=not args.casual_only, casual=not args.rated_only)
        settings = GameSettings(rating=args.rating, match_opponent=args.match_opponent, offset=args.offset)
        write(f"Playing as {account.get('username', account['id'])}: "
              + (f"each opponent's rating {args.offset:+d}" if args.match_opponent else f"rated {args.rating}")
              + (f", in the style of {predictor.info.player}" if args.style else "")
              + f"; accepting {', '.join(args.speeds)}. Ctrl+C to stop.")
        played = run_bot(client, lambda rating: BotPlayer(predictor, rating, sampling), policy, settings,
                         username=account["id"], max_games=args.max_games, log=write)
    except KeyboardInterrupt:
        write("Stopped.")
        return 0
    except (ConfigError, ModelFileError, BotError, StyleError, LichessError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write(f"Done: {played} game{'s' if played != 1 else ''} played.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
