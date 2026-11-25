"""The game logic behind the API, independent of HTTP (so it is easy to test and reuse).

    service = GameService(predictor, cfg)
    session = service.create(NewGameRequest(rating=1200, color="black"))   # the bot opens as White
    response = service.play(session, "e5")                                  # your move + the bot's reply
    service.state(session)                                                  # everything the UI needs

One model is loaded for the whole server; every game gets its own :class:`BotPlayer`
(its own rating, sampling settings and random generator) on top of it.
"""

from __future__ import annotations

import random

import chess

from chess_engine.api.schemas import (
    GameState,
    MoveResponse,
    NewGameRequest,
    bot_move_info,
    color_name,
    legal_moves,
    outcome_info,
)
from chess_engine.api.store import GameStore, Session
from chess_engine.bot.game import Game, IllegalMoveError
from chess_engine.bot.player import BotPlayer, sampling_from_config
from chess_engine.bot.sampling import SamplingSettings
from chess_engine.config import Config
from chess_engine.model.inference import MovePredictor


class GameRuleError(Exception):
    """A request that the rules do not allow right now (``code`` names the reason for API clients)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise GameRuleError(code, message)


class GameService:
    """Creates and runs games against the bot, stored in a :class:`GameStore`."""

    def __init__(self, predictor: MovePredictor, cfg: Config, store: GameStore | None = None) -> None:
        self.predictor = predictor
        self.cfg = cfg
        self.store = store or GameStore(cfg.api.max_games, cfg.api.idle_minutes * 60)

    # ------------------------------------------------------------------ creating

    def _sampling(self, request: NewGameRequest) -> SamplingSettings:
        base = sampling_from_config(self.cfg.bot)
        return SamplingSettings(
            request.temperature if request.temperature is not None else base.temperature,
            request.top_k if request.top_k is not None else base.top_k,
            request.top_p if request.top_p is not None else base.top_p,
            base.min_prob,
        )

    def create(self, request: NewGameRequest) -> Session:
        """Start a game; if the bot has the first move, it plays it right away."""
        human = {"white": chess.WHITE, "black": chess.BLACK}.get(request.color)
        if human is None:
            human = random.choice([chess.WHITE, chess.BLACK])
        bot = BotPlayer(self.predictor, request.rating, self._sampling(request), request.seed)
        bot_name = f"Bot {bot.rating}"
        white, black = ("You", bot_name) if human == chess.WHITE else (bot_name, "You")
        white_elo, black_elo = ((request.your_rating, bot.rating) if human == chess.WHITE
                                else (bot.rating, request.your_rating))
        game = Game(white, black, white_elo, black_elo, fen=request.fen, event="Game vs. human-like bot")
        session = self.store.add(Session(GameStore.new_id(), game, bot, human, request.your_rating))
        if game.turn != human:
            self._bot_plays(session)
        return session

    # ------------------------------------------------------------------ moves

    def _bot_plays(self, session: Session) -> None:
        game = session.game
        choice = session.bot.choose_move(game.board, opp_rating=session.your_rating or session.bot.rating)
        game.push(choice.move)
        session.last_bot_move = choice

    def play(self, session: Session, move: str, reply: bool = True) -> MoveResponse:
        """Play your move; with ``reply`` the bot answers in the same call.

        Raises:
            GameRuleError: the game is over or it is the bot's turn.
            IllegalMoveError: ``move`` is not a legal move here.
        """
        with session.lock:
            game = session.game
            _require(not game.is_over, "game_over", "the game is over")
            _require(game.turn == session.human_color, "not_your_turn", "it is the bot's turn")
            san = game.push(move)
            bot_reply = None
            if reply and not game.is_over:
                self._bot_plays(session)
                bot_reply = bot_move_info(session.last_bot_move)
            return MoveResponse(played=san, bot_move=bot_reply, state=self._state(session))

    def bot_move(self, session: Session) -> MoveResponse:
        """Let the bot play its move (when you played with ``reply: false``)."""
        with session.lock:
            game = session.game
            _require(not game.is_over, "game_over", "the game is over")
            _require(game.turn != session.human_color, "not_bots_turn", "it is your turn")
            self._bot_plays(session)
            return MoveResponse(played=game.sans[-1], bot_move=bot_move_info(session.last_bot_move),
                                state=self._state(session))

    # ------------------------------------------------------------------ state

    def state(self, session: Session) -> GameState:
        with session.lock:
            return self._state(session)

    def _state(self, session: Session) -> GameState:
        game, board = session.game, session.game.board
        outcome = game.outcome
        return GameState(
            id=session.id,
            fen=board.fen(),
            turn=color_name(board.turn),
            human_color=color_name(session.human_color),
            bot_rating=session.bot.rating,
            bot_description=session.bot.describe(),
            your_rating=session.your_rating,
            status="over" if outcome else "ongoing",
            outcome=outcome_info(outcome),
            check=board.is_check(),
            moves_san=list(game.sans),
            moves_uci=[m.uci() for m in board.move_stack],
            move_list=game.move_list(),
            last_move=board.peek().uci() if board.move_stack else None,
            legal_moves=[] if outcome else legal_moves(board),
            clock=None,
            last_bot_move=bot_move_info(session.last_bot_move) if session.last_bot_move else None,
        )


__all__ = ["GameRuleError", "GameService", "IllegalMoveError"]
