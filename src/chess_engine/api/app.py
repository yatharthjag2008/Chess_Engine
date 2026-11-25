"""FastAPI application: play the human-like bot over HTTP.

    uvicorn "chess_engine.api.app:create_app" --factory      # or: python scripts/serve.py

Endpoints (all JSON, under ``/api``; interactive docs at ``/docs``):

=======  ==============================  ===================================================
method   path                            what it does
=======  ==============================  ===================================================
GET      /api/health                     server status, model file, number of games
POST     /api/games                      start a game (rating, colour, time control, ...)
GET      /api/games/{id}                 the full game state (board, moves, legal moves, ...)
GET      /api/games/{id}/legal-moves     the side to move's legal moves (UCI, SAN, squares)
POST     /api/games/{id}/moves           play your move; the bot replies unless ``reply`` is false
POST     /api/games/{id}/bot-move        the bot plays (after a move sent with ``reply: false``)
POST     /api/games/{id}/resign          you resign
POST     /api/games/{id}/takeback        undo your last move (and the bot's reply)
GET      /api/games/{id}/pgn             the game as PGN text
POST     /api/import/pgn                 import pasted PGN games (summaries + PGN of each)
GET      /api/import/{site}/{username}   a player's recent games from ``lichess`` or ``chess.com``
=======  ==============================  ===================================================

The model is loaded once when the app starts (from ``bot.model`` in the config),
unless a predictor is passed in (tests, or embedding the app elsewhere).

Every error has the same shape, ``{"error": "<code>", "detail": "<message>"}``:
404 ``game_not_found``; 409 ``game_over`` / ``not_your_turn`` / ``not_bots_turn`` /
``nothing_to_take_back`` / ``timed_game``; 422 ``illegal_move`` / ``invalid_request`` /
``invalid_username``; 404 ``user_not_found``; 502 ``source_unavailable``.
Browsers may call the API from the origins in ``api.cors_origins``.

The web UI (``web/``, step 10) is served at ``/`` by the same app, so the page and
the API share one origin; without a web directory, ``/`` redirects to the API docs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dataclasses import asdict

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from chess_engine.api.schemas import (
    GameState,
    Health,
    ImportedGameInfo,
    ImportPgnRequest,
    ImportResponse,
    LegalMove,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    legal_moves,
)
from chess_engine.api.service import GameRuleError, GameService
from chess_engine.api.store import GameNotFoundError, Session
from chess_engine.bot.game import IllegalMoveError
from chess_engine.data.game_import import (
    GameImportError,
    ImportCache,
    ImportResult,
    InvalidUsernameError,
    UserNotFoundError,
    import_games,
)
from chess_engine.config import PROJECT_ROOT, Config, load_config
from chess_engine.model.inference import MovePredictor, load_model


WEB_DIR = PROJECT_ROOT / "web"


def create_app(cfg: Config | None = None, predictor: MovePredictor | None = None,
               web_dir: str | Path | None = WEB_DIR) -> FastAPI:
    """The API app for ``cfg`` (default: the project config), using ``predictor`` or the configured model.

    Args:
        web_dir: Directory served at ``/`` (the web UI); None serves only the API.
    """
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
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.api.cors_origins), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    def service(request: Request) -> GameService:
        return request.app.state.service

    def session(request: Request, game_id: str) -> Session:
        return service(request).store.get(game_id)

    @app.exception_handler(GameNotFoundError)
    async def not_found(request: Request, exc: GameNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "game_not_found",
                                                      "detail": "no such game (unknown id, or it expired)"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            problems.append(f"{where}: {error.get('msg')}" if where else str(error.get("msg")))
        return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": "; ".join(problems)})

    @app.exception_handler(GameRuleError)
    async def rule_error(request: Request, exc: GameRuleError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": exc.code, "detail": str(exc)})

    @app.exception_handler(GameImportError)
    async def import_error(request: Request, exc: GameImportError) -> JSONResponse:
        if isinstance(exc, InvalidUsernameError):
            status, code = 422, "invalid_username"
        elif isinstance(exc, UserNotFoundError):
            status, code = 404, "user_not_found"
        else:
            status, code = 502, "source_unavailable"
        return JSONResponse(status_code=status, content={"error": code, "detail": str(exc)})

    @app.exception_handler(IllegalMoveError)
    async def illegal_move(request: Request, exc: IllegalMoveError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "illegal_move", "detail": str(exc)})

    web = Path(web_dir) if web_dir is not None else None
    if web is None or not (web / "index.html").is_file():
        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse("/docs")

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

    @app.get("/api/games/{game_id}/legal-moves", response_model=list[LegalMove])
    def get_legal_moves(game_id: str, request: Request) -> list[LegalMove]:
        game = session(request, game_id).game
        return [] if game.is_over else legal_moves(game.board)

    @app.post("/api/games/{game_id}/moves", response_model=MoveResponse)
    def play_move(game_id: str, body: MoveRequest, request: Request) -> MoveResponse:
        return service(request).play(session(request, game_id), body.move, body.reply)

    @app.post("/api/games/{game_id}/bot-move", response_model=MoveResponse)
    def bot_move(game_id: str, request: Request) -> MoveResponse:
        return service(request).bot_move(session(request, game_id))

    @app.post("/api/games/{game_id}/resign", response_model=GameState)
    def resign(game_id: str, request: Request) -> GameState:
        return service(request).resign(session(request, game_id))

    @app.post("/api/games/{game_id}/takeback", response_model=GameState)
    def takeback(game_id: str, request: Request) -> GameState:
        return service(request).takeback(session(request, game_id))

    @app.get("/api/games/{game_id}/pgn", response_class=PlainTextResponse)
    def pgn(game_id: str, request: Request) -> str:
        return service(request).pgn(session(request, game_id))

    import_cache = ImportCache(cfg.paths.data_dir / "imports", cfg.game_import.cache_minutes)

    def import_response(result: ImportResult) -> ImportResponse:
        return ImportResponse(source=result.source, username=result.username, count=len(result.games),
                              games=[ImportedGameInfo(**asdict(g)) for g in result.games])

    @app.post("/api/import/pgn", response_model=ImportResponse)
    def import_pgn(body: ImportPgnRequest) -> ImportResponse:
        return import_response(import_games(cfg.game_import, "pgn", text=body.pgn, max_games=body.max_games,
                                            speeds=[], rated_only=False))

    @app.get("/api/import/{site}/{username}", response_model=ImportResponse)
    def import_user(site: str, username: str, max_games: int | None = Query(default=None, ge=1, le=200),
                    speed: list[str] | None = Query(default=None), rated: bool = False) -> ImportResponse:
        return import_response(import_games(cfg.game_import, site, username=username, max_games=max_games,
                                            speeds=speed, rated_only=rated or None, cache=import_cache))

    if web is not None and (web / "index.html").is_file():
        # Mounted last, so every /api route and /docs is matched first.
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
