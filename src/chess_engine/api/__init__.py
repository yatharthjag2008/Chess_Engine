"""FastAPI backend (step 9): play the human-like bot over HTTP.

Modules:
    schemas  request / response bodies with validation, converters from chess objects
    store    in-memory games (random ids, idle expiry, size cap)
    service  GameService: the game logic (moves, bot replies, clocks, resign, takeback, PGN)
    app      create_app: the FastAPI app and its endpoints (run with scripts/serve.py)
"""

from chess_engine.api.app import create_app
from chess_engine.api.service import GameRuleError, GameService

__all__ = ["GameRuleError", "GameService", "create_app"]
