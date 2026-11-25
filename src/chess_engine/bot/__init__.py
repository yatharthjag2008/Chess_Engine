"""The playing bot and features built on top of it.

Step 8 modules:
    sampling  turn the model's move probabilities into the bot's (temperature, top-k, top-p, min probability)
    player    BotPlayer: sample a move at a target rating; load_bot from the config or a model file
    game      Game: SAN/UCI input, results (incl. automatic draws), resignation, time forfeit, takebacks, PGN
    clock     ChessClock / TimeControl: base time + increment, Lichess-style start, flag fall
    cli       TerminalGame: play the bot in a terminal (scripts/play.py)
    selfplay  bot-vs-bot games and match scores (scripts/bot_match.py)

Planned:
    hints     (step 17) tiered hints (nudge -> point -> show)
    adaptive  (step 18) adaptive difficulty
    styles    (step 19) style imitation and ghost mode
"""

from chess_engine.bot.game import Game, IllegalMoveError, Outcome
from chess_engine.bot.player import BotError, BotMove, BotPlayer, load_bot
from chess_engine.bot.sampling import SamplingSettings

__all__ = ["BotError", "BotMove", "BotPlayer", "Game", "IllegalMoveError", "Outcome", "SamplingSettings", "load_bot"]
