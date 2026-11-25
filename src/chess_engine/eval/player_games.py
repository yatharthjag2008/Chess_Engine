"""One player's moves from PGN games: what rating estimation needs to read.

    games = player_games_from_pgn(pgn_text, player="MagnusFan")    # or color="white"
    for g in games:
        g.boards[i], g.moves[i]          # the position before each of the player's moves, and the move
        g.opp_rating, g.rating           # from the WhiteElo / BlackElo headers (None if absent)
        g.clocks[i], g.opp_clocks[i]     # seconds left before the move, from [%clk] (None if absent)

Games with a different starting position (``FEN`` header) are supported; games
whose moves are illegal are skipped.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import chess
import chess.pgn

from chess_engine.data.pgn_stream import parse_time_control
from chess_engine.data.samples import clocks_before_moves


@dataclass
class PlayerGame:
    """The moves one side played in one game."""

    color: chess.Color
    name: str
    rating: int | None                 # the player's rating in the game, if the PGN has it
    opp_rating: int | None
    result: str
    boards: list[chess.Board] = field(default_factory=list)
    moves: list[chess.Move] = field(default_factory=list)
    clocks: list[float] | None = None
    opp_clocks: list[float] | None = None
    base_seconds: int = 180
    increment: int = 0
    headers: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.moves)


def _elo(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def read_games(text: str) -> list[chess.pgn.Game]:
    """Every game in a PGN text (games python-chess flags as broken are dropped)."""
    stream = io.StringIO(text)
    games = []
    while (game := chess.pgn.read_game(stream)) is not None:
        if not game.errors:
            games.append(game)
    return games


def _sides(game: chess.pgn.Game, player: str | None, color: chess.Color | None) -> list[chess.Color]:
    if player is not None:
        wanted = player.strip().lower()
        return [c for c, h in ((chess.WHITE, "White"), (chess.BLACK, "Black"))
                if game.headers.get(h, "").strip().lower() == wanted]
    if color is not None:
        return [color]
    return [chess.WHITE, chess.BLACK]


def player_games(game: chess.pgn.Game, color: chess.Color) -> PlayerGame:
    """The moves ``color`` played in ``game``, with the positions before them and clocks if recorded."""
    h = game.headers
    tc = parse_time_control(h.get("TimeControl", ""))
    base, increment = (tc.base_seconds, tc.increment_seconds) if tc else (180, 0)
    me, them = ("White", "Black") if color == chess.WHITE else ("Black", "White")
    pg = PlayerGame(color, h.get(me, "?"), _elo(h.get(f"{me}Elo")), _elo(h.get(f"{them}Elo")), h.get("Result", "*"),
                    base_seconds=base, increment=increment, headers=dict(h))
    board = game.board()
    nodes = list(game.mainline())
    after = [node.clock() for node in nodes]
    have_clocks = bool(nodes) and all(c is not None for c in after)
    before_mine, before_theirs = clocks_before_moves(after, base) if have_clocks else (None, None)
    clocks, opp_clocks = [], []
    for ply, node in enumerate(nodes):
        if board.turn == color:
            pg.boards.append(board.copy(stack=False))
            pg.moves.append(node.move)
            if have_clocks:  # before ply i: the mover's clock after ply i-2, the opponent's after ply i-1
                clocks.append(float(before_mine[ply]))
                opp_clocks.append(float(before_theirs[ply]))
        board.push(node.move)
    if have_clocks:
        pg.clocks, pg.opp_clocks = clocks, opp_clocks
    return pg


def player_games_from_pgn(text: str, player: str | None = None,
                          color: chess.Color | None = None) -> list[PlayerGame]:
    """Every game of ``player`` (matched on the White / Black header, ignoring case) in ``text``.

    Without ``player``, ``color`` picks one side of every game; with neither, both sides of
    every game are returned (as separate entries). Sides without a single move are skipped.
    """
    out = []
    for game in read_games(text):
        for side in _sides(game, player, color):
            pg = player_games(game, side)
            if len(pg):
                out.append(pg)
    return out
