"""One game of chess between two players (a human and the bot, or two bots), with its result and PGN.

    game = Game(white="You", black="Bot 1500", black_elo=1500)
    game.push("e4")                  # SAN or UCI; returns the SAN played
    game.push(bot_choice.move)       # or a chess.Move
    game.outcome                     # None while the game goes on, else Outcome("1-0", "checkmate", WHITE)
    game.resign(chess.BLACK)
    print(game.pgn())

Draws by threefold repetition and the fifty-move rule are applied automatically,
as on Lichess, so a game against the bot can never go on forever.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import chess
import chess.pgn


class IllegalMoveError(ValueError):
    """The text is not a legal move in the current position."""


@dataclass(frozen=True)
class Outcome:
    """How a game ended."""

    result: str                      # "1-0", "0-1" or "1/2-1/2"
    termination: str                 # checkmate, stalemate, resignation, time forfeit, ...
    winner: chess.Color | None       # None for a draw

    def describe(self) -> str:
        if self.winner is None:
            return f"Draw by {self.termination} ({self.result})"
        side = "White" if self.winner == chess.WHITE else "Black"
        return f"{side} wins by {self.termination} ({self.result})"


_TERMINATIONS = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
    chess.Termination.SEVENTYFIVE_MOVES: "seventy-five-move rule",
    chess.Termination.FIVEFOLD_REPETITION: "fivefold repetition",
    chess.Termination.FIFTY_MOVES: "fifty-move rule",
    chess.Termination.THREEFOLD_REPETITION: "threefold repetition",
}


def _result(winner: chess.Color | None) -> str:
    return "1/2-1/2" if winner is None else ("1-0" if winner == chess.WHITE else "0-1")


def parse_move(board: chess.Board, text: str) -> chess.Move:
    """A legal move from SAN (``Nf3``, ``exd5``, ``O-O``, ``e8=Q``) or UCI (``g1f3``, ``e7e8q``).

    Raises:
        IllegalMoveError: with a short reason, if ``text`` is not a legal move here.
    """
    text = text.strip()
    if not text:
        raise IllegalMoveError("empty move")
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text.lower())
    except ValueError:
        raise IllegalMoveError(f"{text!r} is not a move (use SAN like Nf3 or UCI like g1f3)") from None
    if move not in board.legal_moves:
        if chess.Move(move.from_square, move.to_square, chess.QUEEN) in board.legal_moves and not move.promotion:
            return chess.Move(move.from_square, move.to_square, chess.QUEEN)  # e7e8 -> e7e8q
        raise IllegalMoveError(f"{text} is not legal in this position")
    return move


class Game:
    """A game from the start position (or ``fen``) with player names, ratings, moves and outcome."""

    def __init__(
        self,
        white: str = "White",
        black: str = "Black",
        white_elo: int | None = None,
        black_elo: int | None = None,
        fen: str | None = None,
        event: str = "Casual game",
    ) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()
        self.start_fen = self.board.fen()
        self.white, self.black = white, black
        self.white_elo, self.black_elo = white_elo, black_elo
        self.event = event
        self.date = datetime.date.today()
        self.sans: list[str] = []
        self._ended: Outcome | None = None  # resignation or time forfeit

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def moves(self) -> list[chess.Move]:
        return list(self.board.move_stack)

    @property
    def outcome(self) -> Outcome | None:
        """How the game ended, or None if it is still going."""
        if self._ended is not None:
            return self._ended
        result = self.board.outcome()
        if result is None:
            # Only a repetition / fifty-move count that has actually happened ends the game
            # (python-chess's claim_draw also counts one the next move could create).
            if self.board.is_repetition(3):
                return Outcome("1/2-1/2", "threefold repetition", None)
            if self.board.is_fifty_moves():
                return Outcome("1/2-1/2", "fifty-move rule", None)
            return None
        return Outcome(result.result(), _TERMINATIONS.get(result.termination, result.termination.name.lower()),
                       result.winner)

    @property
    def is_over(self) -> bool:
        return self.outcome is not None

    def push(self, move: chess.Move | str) -> str:
        """Play a move (``chess.Move``, SAN or UCI) for the side to move; returns its SAN."""
        if self.is_over:
            raise IllegalMoveError("the game is over")
        if isinstance(move, str):
            move = parse_move(self.board, move)
        elif move not in self.board.legal_moves:
            raise IllegalMoveError(f"{move.uci()} is not legal in this position")
        san = self.board.san(move)
        self.board.push(move)
        self.sans.append(san)
        return san

    def takeback(self, plies: int = 1) -> list[str]:
        """Undo the last ``plies`` moves (e.g. 2 = your move and the bot's reply); returns their SANs."""
        if plies < 0 or plies > len(self.sans):
            raise ValueError(f"cannot take back {plies} moves ({len(self.sans)} played)")
        if self._ended is not None:
            raise ValueError("the game has ended")
        undone = []
        for _ in range(plies):
            self.board.pop()
            undone.append(self.sans.pop())
        return undone[::-1]

    def resign(self, color: chess.Color) -> Outcome:
        """``color`` resigns."""
        return self._end(not color, "resignation")

    def forfeit_on_time(self, color: chess.Color) -> Outcome:
        """``color`` ran out of time. A draw if the opponent cannot possibly checkmate (FIDE / Lichess rule)."""
        if self.board.has_insufficient_material(not color):
            return self._end(None, "timeout vs. insufficient material")
        return self._end(not color, "time forfeit")

    def _end(self, winner: chess.Color | None, termination: str) -> Outcome:
        if self.is_over:
            raise ValueError("the game is already over")
        self._ended = Outcome(_result(winner), termination, winner)
        return self._ended

    def move_list(self) -> str:
        """``1. e4 e5 2. Nf3 Nc6`` (starting from the right move number and side)."""
        board = chess.Board(self.start_fen)
        parts = []
        for i, san in enumerate(self.sans):
            if board.turn == chess.WHITE:
                parts.append(f"{board.fullmove_number}. {san}")
            elif i == 0:
                parts.append(f"{board.fullmove_number}... {san}")
            else:
                parts.append(san)
            board.push_san(san)
        return " ".join(parts)

    def pgn(self) -> str:
        """The game as PGN text, with players, ratings, result and termination."""
        game = chess.pgn.Game.from_board(self.board)
        h = game.headers
        h["Event"] = self.event
        h["Site"] = "local"
        h["Date"] = self.date.strftime("%Y.%m.%d")
        h["White"], h["Black"] = self.white, self.black
        if self.white_elo:
            h["WhiteElo"] = str(self.white_elo)
        if self.black_elo:
            h["BlackElo"] = str(self.black_elo)
        outcome = self.outcome
        h["Result"] = outcome.result if outcome else "*"
        if outcome:
            h["Termination"] = outcome.termination
        if self.start_fen != chess.STARTING_FEN:
            h["SetUp"], h["FEN"] = "1", self.start_fen
        return str(game) + "\n"
