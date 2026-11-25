"""Players for matches: the human-like bot and Stockfish at a limited strength, behind one interface.

    with StockfishPlayer(path, elo=1600) as engine:
        move = engine.choose(board)

    bot = BotAgent(load_bot(cfg, rating=1400))
    move = bot.choose(board, opponent_rating=1600)

A :class:`Judge` (full-strength Stockfish at a fixed node count) evaluates
positions, which matches use to end clearly decided games (the bots never resign).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import chess
import chess.engine

from chess_engine.bot.player import BotPlayer

MATE_CP = 100_000


class Player(Protocol):
    """Anything that can pick a move."""

    name: str
    rating: int | None   # the strength it is set to (the bot's dial, Stockfish's UCI_Elo)

    def choose(self, board: chess.Board, opponent_rating: int | None = None) -> chess.Move: ...

    def close(self) -> None: ...


class BotAgent:
    """The human-like bot as a match player."""

    def __init__(self, bot: BotPlayer, name: str | None = None) -> None:
        self.bot = bot
        self.rating = bot.rating
        self.name = name or f"bot {bot.rating}"

    def choose(self, board: chess.Board, opponent_rating: int | None = None) -> chess.Move:
        return self.bot.choose_move(board, opp_rating=opponent_rating or self.bot.rating).move

    def close(self) -> None:
        pass


def _open(command: str | Sequence[str]) -> chess.engine.SimpleEngine:
    argv = [command] if isinstance(command, str) else list(command)
    return chess.engine.SimpleEngine.popen_uci(argv)


def _configure(engine: chess.engine.SimpleEngine, options: dict) -> None:
    supported = {k: v for k, v in options.items() if k in engine.options}
    if supported:
        engine.configure(supported)


class StockfishPlayer:
    """Stockfish limited to ``elo`` (``UCI_LimitStrength`` + ``UCI_Elo``), thinking ``move_seconds`` per move.

    The engine process starts on first use and stays open until :meth:`close`.
    """

    def __init__(self, command: str | Sequence[str], elo: int, move_seconds: float = 0.05,
                 threads: int = 1, hash_mb: int = 16) -> None:
        self.command = command
        self.rating = elo
        self.name = f"Stockfish {elo}"
        self.move_seconds = move_seconds
        self.options = {"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": threads, "Hash": hash_mb}
        self._engine: chess.engine.SimpleEngine | None = None

    def _get(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = _open(self.command)
            _configure(self._engine, self.options)
        return self._engine

    def choose(self, board: chess.Board, opponent_rating: int | None = None) -> chess.Move:
        result = self._get().play(board, chess.engine.Limit(time=self.move_seconds))
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move in {board.fen()}")
        return result.move

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishPlayer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Judge:
    """Full-strength Stockfish evaluating positions at a fixed node count (centipawns, White's view)."""

    def __init__(self, command: str | Sequence[str], nodes: int = 20_000, threads: int = 1, hash_mb: int = 32) -> None:
        self.command = command
        self.nodes = nodes
        self.options = {"Threads": threads, "Hash": hash_mb}
        self._engine: chess.engine.SimpleEngine | None = None

    def evaluate(self, board: chess.Board) -> int:
        """White's advantage in centipawns (mates count as +-100,000); 0 if the engine gives no score."""
        if board.is_game_over():
            outcome = board.outcome()
            return 0 if outcome.winner is None else (MATE_CP if outcome.winner == chess.WHITE else -MATE_CP)
        if self._engine is None:
            self._engine = _open(self.command)
            _configure(self._engine, self.options)
        info = self._engine.analyse(board, chess.engine.Limit(nodes=self.nodes))
        score = info.get("score")
        if score is None:
            return 0
        return int(score.white().score(mate_score=MATE_CP))

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> Judge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
