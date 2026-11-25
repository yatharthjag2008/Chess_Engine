"""Play one game between two :class:`~chess_engine.eval.players.Player` objects, with adjudication.

The bots imitate humans move by move but never resign, so won positions are often
shuffled for a hundred moves or end in stalemate (step 8 saw many such draws). A
judge settles clearly decided games: when one side has been ahead by at least
``adjudicate_cp`` centipawns for ``adjudicate_plies`` plies in a row, it wins. At
``max_plies`` the judge's last evaluation decides (a draw if neither side is ahead
by that margin). Without a judge, games run to a natural end or to ``max_plies`` (a draw).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import chess

from chess_engine.bot.game import Game
from chess_engine.eval.players import Judge, Player


@dataclass
class MatchGame:
    """The record of one game (plain data, stored as JSON)."""

    white: str
    black: str
    result: str                   # "1-0", "0-1", "1/2-1/2"
    termination: str
    plies: int
    adjudicated: bool = False
    moves: list[str] = field(default_factory=list)   # UCI

    @property
    def white_score(self) -> float:
        return {"1-0": 1.0, "0-1": 0.0}.get(self.result, 0.5)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(winner: chess.Color | None) -> str:
    return "1/2-1/2" if winner is None else ("1-0" if winner == chess.WHITE else "0-1")


def play_match_game(
    white: Player,
    black: Player,
    judge: Judge | None = None,
    max_plies: int = 300,
    adjudicate_cp: int = 800,
    adjudicate_plies: int = 8,
    fen: str | None = None,
) -> MatchGame:
    """Play a game to its end (see the module docstring)."""
    game = Game(white.name, black.name, white.rating, black.rating, fen=fen, event="Elo validation")
    streak_side: chess.Color | None = None
    streak = 0
    evaluation = 0
    while not game.is_over:
        if len(game.sans) >= max_plies:
            margin_winner = None
            if judge is not None and abs(evaluation) >= adjudicate_cp:
                margin_winner = chess.WHITE if evaluation > 0 else chess.BLACK
            return MatchGame(white.name, black.name, _result(margin_winner), f"move limit ({max_plies} plies)",
                             len(game.sans), True, [m.uci() for m in game.moves])
        mover, other = (white, black) if game.turn == chess.WHITE else (black, white)
        game.push(mover.choose(game.board, opponent_rating=other.rating))
        if judge is None or game.is_over:
            continue
        evaluation = judge.evaluate(game.board)
        ahead = chess.WHITE if evaluation >= adjudicate_cp else chess.BLACK if evaluation <= -adjudicate_cp else None
        if ahead is not None and ahead == streak_side:
            streak += 1
        else:
            streak_side, streak = ahead, (1 if ahead is not None else 0)
        if streak_side is not None and streak >= adjudicate_plies:
            return MatchGame(white.name, black.name, _result(streak_side), "adjudicated (decisive advantage)",
                             len(game.sans), True, [m.uci() for m in game.moves])
    outcome = game.outcome
    return MatchGame(white.name, black.name, outcome.result, outcome.termination, len(game.sans), False,
                     [m.uci() for m in game.moves])
