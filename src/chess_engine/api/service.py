"""The game logic behind the API, independent of HTTP (so it is easy to test and reuse).

    service = GameService(predictor, cfg)
    session = service.create(NewGameRequest(rating=1200, color="black"))   # the bot opens as White
    response = service.play(session, "e5")                                  # your move + the bot's reply
    service.state(session)                                                  # everything the UI needs

One model is loaded for the whole server; every game gets its own :class:`BotPlayer`
(its own rating, sampling settings and random generator) on top of it.

Timed games keep their clocks on the server: a flag fall is detected on every
request (the game ends by time forfeit), the bot is told both players' time, and
takebacks are not allowed (they would need the clocks rewound).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import chess

from chess_engine.api.schemas import (
    ClockState,
    GameState,
    HintsState,
    MoveResponse,
    NewGameRequest,
    bot_move_info,
    color_name,
    legal_moves,
    outcome_info,
)
from chess_engine.api.store import GameStore, Session
from chess_engine.bot.clock import ChessClock, TimeControl
from chess_engine.bot.game import Game, IllegalMoveError
from chess_engine.bot.hints import HintBudget
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


def _mover(game: Game, ply: int) -> chess.Color:
    """The side that played the ``ply``-th move of ``game`` (0-based)."""
    first = chess.Board(game.start_fen).turn
    return first if ply % 2 == 0 else not first


class GameService:
    """Creates and runs games against the bot, stored in a :class:`GameStore`."""

    def __init__(self, predictor: MovePredictor, cfg: Config, store: GameStore | None = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.predictor = predictor
        self.cfg = cfg
        self.store = store or GameStore(cfg.api.max_games, cfg.api.idle_minutes * 60)
        self._now = now

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
        clock = ChessClock(TimeControl.parse(request.time_control), self._now) if request.time_control else None
        limit = request.hint_limit if request.hint_limit is not None else self.cfg.hints.limit
        _require(limit <= self.cfg.hints.max_limit, "too_many_hints",
                 f"at most {self.cfg.hints.max_limit} hints per game")
        session = Session(GameStore.new_id(), game, bot, human, request.your_rating, clock, hints=HintBudget(limit))
        session = self.store.add(session)
        if game.turn != human:
            self._bot_plays(session)
        return session

    # ------------------------------------------------------------------ moves

    def _clock_args(self, session: Session) -> dict:
        clock = session.clock
        if clock is None:
            return {}
        bot = not session.human_color
        return {"clock": max(0.0, clock.remaining(bot)), "opp_clock": max(0.0, clock.remaining(session.human_color)),
                "base_seconds": clock.control.base_seconds, "increment": clock.control.increment}

    def _bot_plays(self, session: Session) -> None:
        game = session.game
        choice = session.bot.choose_move(game.board, opp_rating=session.your_rating or session.bot.rating,
                                         **self._clock_args(session))
        game.push(choice.move)
        session.last_bot_move = choice
        self._after_move(session, not session.human_color)

    def _after_move(self, session: Session, mover: chess.Color) -> None:
        """Press the mover's clock (a flag fall before the move is caught by :meth:`_check_time`)."""
        if session.clock is not None:
            session.clock.press(mover)
            if session.game.is_over:
                session.clock.stop()

    def _check_time(self, session: Session) -> None:
        """End the game by time forfeit if a clock has run out."""
        clock = session.clock
        if clock is None or session.game.is_over:
            return
        flagged = clock.flagged()
        if flagged is not None:
            session.game.forfeit_on_time(flagged)
            clock.stop()

    def play(self, session: Session, move: str, reply: bool = True) -> MoveResponse:
        """Play your move; with ``reply`` the bot answers in the same call.

        Raises:
            GameRuleError: the game is over or it is the bot's turn.
            IllegalMoveError: ``move`` is not a legal move here.
        """
        with session.lock:
            game = session.game
            self._check_time(session)
            _require(not game.is_over, "game_over", "the game is over")
            _require(game.turn == session.human_color, "not_your_turn", "it is the bot's turn")
            san = game.push(move)
            self._after_move(session, session.human_color)
            bot_reply = None
            if reply and not game.is_over:
                self._bot_plays(session)
                bot_reply = bot_move_info(session.last_bot_move)
            return MoveResponse(played=san, bot_move=bot_reply, state=self._state(session))

    def bot_move(self, session: Session) -> MoveResponse:
        """Let the bot play its move (when you played with ``reply: false``)."""
        with session.lock:
            game = session.game
            self._check_time(session)
            _require(not game.is_over, "game_over", "the game is over")
            _require(game.turn != session.human_color, "not_bots_turn", "it is your turn")
            self._bot_plays(session)
            return MoveResponse(played=game.sans[-1], bot_move=bot_move_info(session.last_bot_move),
                                state=self._state(session))

    def resign(self, session: Session) -> GameState:
        """You resign."""
        with session.lock:
            self._check_time(session)
            _require(not session.game.is_over, "game_over", "the game is over")
            session.game.resign(session.human_color)
            if session.clock is not None:
                session.clock.stop()
            return self._state(session)

    def takeback(self, session: Session) -> GameState:
        """Undo moves back to your previous turn: your last move, and the bot's reply if it has answered.

        The bot's opening move (when it has White) is never undone on its own.
        """
        with session.lock:
            game = session.game
            _require(session.clock is None, "timed_game", "takebacks are not allowed in timed games")
            _require(not game.is_over, "game_over", "the game is over")
            yours = sum(1 for ply in range(len(game.sans)) if _mover(game, ply) == session.human_color)
            _require(yours > 0, "nothing_to_take_back", "you have not moved yet")
            game.takeback(2 if game.turn == session.human_color else 1)
            session.last_bot_move = None
            return self._state(session)

    def pgn(self, session: Session) -> str:
        """The game as PGN; a game with hints says so (and how many) in its headers."""
        with session.lock:
            game, hints = session.game, session.hints
            if not hints.rated:
                game.extra_headers["Hints"] = str(hints.used)
                game.extra_headers["Annotator"] = "unrated: hints used"
            return game.pgn()

    # ------------------------------------------------------------------ state

    def state(self, session: Session) -> GameState:
        with session.lock:
            self._check_time(session)
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
            clock=self._clock_state(session),
            last_bot_move=bot_move_info(session.last_bot_move) if session.last_bot_move else None,
            hints=HintsState(limit=session.hints.limit, used=session.hints.used, left=session.hints.left,
                             rated=session.hints.rated),
        )


    @staticmethod
    def _clock_state(session: Session) -> ClockState | None:
        clock = session.clock
        if clock is None:
            return None
        return ClockState(white=round(max(0.0, clock.remaining(chess.WHITE)), 2),
                          black=round(max(0.0, clock.remaining(chess.BLACK)), 2),
                          running=color_name(clock.running), base_seconds=clock.control.base_seconds,
                          increment=clock.control.increment)


__all__ = ["GameRuleError", "GameService", "IllegalMoveError"]
