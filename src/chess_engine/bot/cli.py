"""Play the bot in a terminal (used by ``scripts/play.py``).

Type moves in SAN (``Nf3``, ``exd5``, ``O-O``) or UCI (``g1f3``). Commands:

    help     this list            board    show the board again      moves   the move list
    undo     take back your last move (and the bot's reply)           flip    turn the board around
    resign   give up              quit     leave without finishing

Input and output are injectable, so the whole loop is tested without a terminal.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import chess

from chess_engine.bot.clock import ChessClock, format_seconds
from chess_engine.bot.game import Game, IllegalMoveError, Outcome
from chess_engine.bot.player import BotMove, BotPlayer

UNICODE_PIECES = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
                  "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}
HELP = __doc__.split("Commands:", 1)[1].split("Input and output", 1)[0].rstrip()


def supports_unicode(stream: object = None) -> bool:
    """Whether ``stream`` (default stdout) can print chess piece symbols."""
    encoding = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        "♔♚".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def render_board(board: chess.Board, perspective: chess.Color = chess.WHITE, unicode: bool = False,
                 last_move: chess.Move | None = None) -> str:
    """The board as text, from ``perspective``'s side, with file and rank labels.

    Pieces are letters (White upper case) or symbols; empty squares are ``.``; the
    squares of ``last_move`` are wrapped in brackets.
    """
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(8)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)
    marked = {last_move.from_square, last_move.to_square} if last_move else set()
    lines = []
    for rank in ranks:
        cells = []
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = "." if piece is None else (UNICODE_PIECES[piece.symbol()] if unicode else piece.symbol())
            cells.append(f"[{symbol}]" if square in marked else f" {symbol} ")
        lines.append(f"{rank + 1} " + "".join(cells))
    lines.append("   " + "  ".join(chess.FILE_NAMES[f] for f in files))
    return "\n".join(lines)


class TerminalGame:
    """One game between a human at the terminal and a bot.

    Args:
        bot: The bot.
        human_color: The human's side.
        game: The game record (players, ratings).
        clock: A running chess clock, or None for an untimed game.
        read: Reads one line of input (default ``input``); raising ``EOFError`` quits.
        write: Prints one block of text (default ``print``).
        unicode: Draw pieces as symbols.
        show_thinking: After each bot move, show its three likeliest moves.
    """

    def __init__(self, bot: BotPlayer, human_color: chess.Color, game: Game, clock: ChessClock | None = None,
                 read: Callable[[str], str] = input, write: Callable[[str], None] = print,
                 unicode: bool = False, show_thinking: bool = False) -> None:
        self.bot, self.human, self.game, self.clock = bot, human_color, game, clock
        self.read, self.write = read, write
        self.unicode, self.show_thinking = unicode, show_thinking
        self.perspective = human_color
        self.human_rating = (game.white_elo if human_color == chess.WHITE else game.black_elo) or bot.rating
        self.quit = False

    # ------------------------------------------------------------------ display

    def show_board(self) -> None:
        board = self.game.board
        last = board.peek() if board.move_stack else None
        text = render_board(board, self.perspective, self.unicode, last)
        if self.clock is not None:
            them, you = (not self.human), self.human
            text = (f"Bot  {format_seconds(self.clock.remaining(them))}\n{text}\n"
                    f"You  {format_seconds(self.clock.remaining(you))}")
        self.write(text)

    def _thinking(self, choice: BotMove) -> str:
        top = ", ".join(f"{c.san} {c.prob:.0%}" for c in choice.candidates[:3])
        return f"  (players rated ~{self.bot.rating} here: {top}; it played a move they choose {choice.prob:.0%} of the time)"

    # ------------------------------------------------------------------ turns

    def _clock_args(self) -> dict:
        if self.clock is None:
            return {}
        c = self.clock.control
        bot_color = not self.human
        return {"clock": max(0.0, self.clock.remaining(bot_color)),
                "opp_clock": max(0.0, self.clock.remaining(self.human)),
                "base_seconds": c.base_seconds, "increment": c.increment}

    def _check_flag(self) -> bool:
        if self.clock is None:
            return False
        flagged = self.clock.flagged()
        if flagged is None:
            return False
        self.game.forfeit_on_time(flagged)
        self.write(("Your" if flagged == self.human else "The bot's") + " time ran out.")
        return True

    def bot_turn(self) -> BotMove:
        choice = self.bot.choose_move(self.game.board, opp_rating=self.human_rating, **self._clock_args())
        san = self.game.push(choice.move)
        if self.clock is not None:
            self.clock.press(not self.human)
        number = self.game.board.fullmove_number - (1 if self.game.turn == chess.WHITE else 0)
        dots = "." if self.game.turn == chess.BLACK else "..."
        self.write(f"Bot plays {number}{dots} {san}")
        if self.show_thinking:
            self.write(self._thinking(choice))
        return choice

    def human_turn(self) -> None:
        """Read input until a move is played (or the human resigns / quits)."""
        while True:
            try:
                text = self.read("Your move: ").strip()
            except EOFError:
                text = "quit"
            command = text.lower()
            if command in ("help", "?"):
                self.write(HELP)
            elif command == "board":
                self.show_board()
            elif command == "moves":
                self.write(self.game.move_list() or "(no moves yet)")
            elif command == "flip":
                self.perspective = not self.perspective
                self.show_board()
            elif command in ("undo", "takeback"):
                self._undo()
            elif command == "resign":
                self.game.resign(self.human)
                return
            elif command in ("quit", "exit"):
                self.quit = True
                return
            elif text:
                try:
                    san = self.game.push(text)
                except IllegalMoveError as exc:
                    self.write(f"{exc}. Type 'help' for commands.")
                    continue
                if self.clock is not None:
                    self.clock.press(self.human)
                self.write(f"You play {san}")
                return

    def _undo(self) -> None:
        plies = 2 if self.game.turn == self.human else 1
        if len(self.game.sans) < plies or (plies == 2 and len(self.game.sans) == 1):
            self.write("Nothing to take back.")
            return
        undone = self.game.takeback(plies)
        self.write("Took back " + " ".join(undone) + (" (clocks keep running)" if self.clock else ""))
        self.show_board()

    def play(self) -> Outcome | None:
        """Play until the game ends (or the human quits); returns the outcome (None if quit)."""
        self.write(f"You play {'White' if self.human == chess.WHITE else 'Black'} against a {self.bot.describe()}.")
        self.write("Type moves like e4 or Nf3; 'help' lists commands.")
        self.show_board()
        while not self.game.is_over and not self.quit:
            if self.game.turn == self.human:
                self.human_turn()
            else:
                self.bot_turn()
            if self._check_flag() or self.game.is_over or self.quit:
                break
            if self.game.turn == self.human:
                self.show_board()
        if self.clock is not None:
            self.clock.stop()
        outcome = self.game.outcome
        if outcome is not None:
            self.show_board()
            self.write(outcome.describe() + ".")
        return outcome
