"""FastAPI application: play the human-like bot over HTTP.

    uvicorn "chess_engine.api.app:create_app" --factory      # or: python scripts/serve.py

Endpoints (all JSON, under ``/api``; interactive docs at ``/docs``):

=======  ==============================  ===================================================
method   path                            what it does
=======  ==============================  ===================================================
GET      /api/health                     server status, model file, number of games
POST     /api/games                      start a game (rating, colour, time control, ...)
GET      /api/games/{id}                 the full game state (board, moves, legal moves, ...)
=======  ==============================  ===================================================

The model is loaded once when the app starts (from ``bot.model`` in the config),
unless a predictor is passed in (tests, or embedding the app elsewhere).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from chess_engine.api.schemas import GameState, Health, NewGameRequest
from chess_engine.api.service import GameService
from chess_engine.api.store import GameNotFoundError, Session
from chess_engine.config import Config, load_config
from chess_engine.model.inference import MovePredictor, load_model


def create_app(cfg: Config | None = None, predictor: MovePredictor | None = None) -> FastAPI:
    """The API app for ``cfg`` (default: the project config), using ``predictor`` or the configured model."""
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(app.state, "service"):
            loaded = load_model(cfg.bot.model)
            app.state.service = GameService(loaded.predictor(), cfg)
            app.state.model_name = str(cfg.bot.model.name)
        yield

    app = FastAPI(title="Human-like chess bot", version="0.1.0", lifespan=lifespan,
                  description="Play a bot that moves like human players of a chosen rating.")
    if predictor is not None:
        app.state.service = GameService(predictor, cfg)
        app.state.model_name = "in-memory model"

    def service(request: Request) -> GameService:
        return request.app.state.service

    def session(request: Request, game_id: str) -> Session:
        return service(request).store.get(game_id)

    @app.exception_handler(GameNotFoundError)
    async def not_found(request: Request, exc: GameNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "game_not_found",
                                                      "detail": "no such game (unknown id, or it expired)"})

    @app.get("/api/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(status="ok", model=request.app.state.model_name, games=len(service(request).store))

    @app.post("/api/games", response_model=GameState, status_code=201)
    def new_game(body: NewGameRequest, request: Request) -> GameState:
        svc = service(request)
        return svc.state(svc.create(body))

    @app.get("/api/games/{game_id}", response_model=GameState)
    def get_game(game_id: str, request: Request) -> GameState:
        return service(request).state(session(request, game_id))

    return app
