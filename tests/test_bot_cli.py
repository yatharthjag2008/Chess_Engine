"""Tests for the terminal game (scripted input, captured output)."""

from __future__ import annotations

import chess
import pytest

from chess_engine.bot.cli import HELP, TerminalGame, render_board, supports_unicode
from chess_engine.bot.clock import ChessClock, TimeControl
from chess_engine.bot.game import Game
from chess_engine.bot.player import BotPlayer
from chess_engine.bot.sampling import SamplingSettings
from bot_helpers import small_predictor


class Script:
    """Feeds scripted lines to ``read`` and records everything written."""

    def __init__(self, *lines: str) -> None:
        self.lines = list(lines)
        self.out: list[str] = []

    def read(self, prompt: str) -> str:
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    def write(self, text: str) -> None:
        self.out.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.out)


@pytest.fixture(scope="module")
def bot() -> BotPlayer:
    return BotPlayer(small_predictor(), 1500, SamplingSettings(temperature=0), seed=0)


def test_render_board() -> None:
    board = chess.Board()
    text = render_board(board)
    lines = text.splitlines()
    assert lines[0].startswith("8  r  n  b  q  k  b  n  r")
    assert lines[-1].strip() == "a  b  c  d  e  f  g  h"
    black_view = render_board(board, chess.BLACK).splitlines()
    assert black_view[0].startswith("1  R  N  B  K") and black_view[-1].strip().startswith("h")
    board.push_san("e4")
    marked = render_board(board, last_move=board.peek())
    assert "[P]" in marked and "[.]" in marked
    assert "♔" in render_board(chess.Board(), unicode=True)


def test_supports_unicode() -> None:
    class Stream:
        def __init__(self, encoding):
            self.encoding = encoding

    assert supports_unicode(Stream("utf-8")) and not supports_unicode(Stream("ascii"))
    assert not supports_unicode(Stream("no-such-codec"))


def test_human_plays_white_and_resigns(bot: BotPlayer) -> None:
    io = Script("e4", "xx", "e2e5", "help", "moves", "Nf3", "resign")
    game = Game("You", "Bot", black_elo=1500)
    outcome = TerminalGame(bot, chess.WHITE, game, read=io.read, write=io.write).play()
    assert outcome.result == "0-1" and outcome.termination == "resignation"
    assert "You play e4" in io.text and "Bot plays 1... " in io.text
    assert "is not a move" in io.text and "not legal" in io.text
    assert HELP.strip().splitlines()[0].strip() in io.text and "1. e4 " in io.text
    assert len(game.sans) == 4  # e4, bot, Nf3, bot


def test_human_plays_black_quits_with_eof(bot: BotPlayer) -> None:
    io = Script("flip", "board")
    game = Game("Bot", "You", white_elo=1500)
    terminal = TerminalGame(bot, chess.BLACK, game, read=io.read, write=io.write, show_thinking=True)
    assert terminal.play() is None and terminal.quit
    assert io.text.count("Bot plays 1. ") == 1 and "players rated ~1500 here:" in io.text
    assert len(game.sans) == 1


def test_undo_takes_back_the_pair(bot: BotPlayer) -> None:
    io = Script("undo", "d4", "undo", "quit")
    game = Game("You", "Bot")
    TerminalGame(bot, chess.WHITE, game, read=io.read, write=io.write).play()
    assert "Nothing to take back." in io.text and "Took back d4 " in io.text
    assert game.sans == []


def test_game_to_the_end_and_time_forfeit(bot: BotPlayer) -> None:
    # The human mates in two from a set-up position (Black's only move is forced).
    io = Script("Ra7", "Rb8#")
    game = Game("You", "Bot", fen="7k/8/6K1/8/8/8/R7/1R6 w - - 0 1")
    outcome = TerminalGame(bot, chess.WHITE, game, read=io.read, write=io.write).play()
    assert outcome is not None and outcome.termination == "checkmate" and "White wins by checkmate" in io.text

    now = [0.0]
    clock = ChessClock(TimeControl(60, 0), lambda: now[0])
    moves = iter(["a3", "h3", "b3", "g3", "c3"])

    def slow_read(prompt: str) -> str:
        now[0] += 50  # the human thinks 50 s per move; the first moves are free, the third one flags
        return next(moves)

    game2 = Game("You", "Bot")
    outcome = TerminalGame(bot, chess.WHITE, game2, clock=clock, read=slow_read, write=lambda _: None).play()
    assert outcome.termination == "time forfeit" and outcome.winner == chess.BLACK
    assert len(game2.sans) == 5  # a3 (free), bot, h3 (10 s left), bot, b3 (flag)


def test_hints_in_the_terminal(bot: BotPlayer) -> None:
    from chess_engine.bot.hints import HintEngine
    from chess_engine.config import HintsConfig
    from review_helpers import ScriptedAnalyzer

    engine = HintEngine(bot.predictor, ScriptedAnalyzer(), HintsConfig())
    io = Script("hint", "hint", "hint", "a3", "hint show", "hint", "quit")
    game = Game("You", "Bot")
    TerminalGame(bot, chess.WHITE, game, read=io.read, write=io.write, hints=engine, hint_limit=2).play()
    hints = [line for line in io.out if line.startswith("Hint (")]
    assert [h.split(")")[0] for h in hints] == ["Hint (nudge", "Hint (point", "Hint (show", "Hint (show", "Hint (show"]
    assert "Play a3." in hints[2] and "[1 hint left; this game is now unrated]" in hints[0]
    assert "[0 hints left" in hints[3]
    assert game.extra_headers == {"Hints": "2", "Annotator": "unrated: hints used"}
    assert '[Hints "2"]' in game.pgn()


def test_hints_off_and_limit(bot: BotPlayer) -> None:
    io = Script("hint", "quit")
    TerminalGame(bot, chess.WHITE, Game("You", "Bot"), read=io.read, write=io.write).play()
    assert "Hints are off" in io.text
    from chess_engine.bot.hints import HintEngine
    from chess_engine.config import HintsConfig
    from review_helpers import ScriptedAnalyzer

    io = Script("hint", "e4", "hint", "quit")
    game = Game("You", "Bot")
    engine = HintEngine(bot.predictor, ScriptedAnalyzer(), HintsConfig())
    TerminalGame(bot, chess.WHITE, game, read=io.read, write=io.write, hints=engine, hint_limit=1).play()
    assert "No hints left in this game." in io.text and game.extra_headers["Hints"] == "1"
    assert "hint" in HELP


def test_adaptive_in_the_terminal(tmp_path) -> None:
    from chess_engine.bot.adaptive import AdaptiveGame, ProfileStore
    from chess_engine.config import AdaptiveConfig

    from test_api_service import LosingBotPredictor

    cfg = AdaptiveConfig(live_after=0, live_every=1)
    store = ProfileStore(tmp_path / "a.json", cfg)
    bot = BotPlayer(LosingBotPredictor(), 1500, SamplingSettings(temperature=0), seed=0)
    adaptive = AdaptiveGame("live", 1500, cfg)
    io = Script("e4", "d4", "resign")
    TerminalGame(bot, chess.WHITE, Game("You", "Bot"), read=io.read, write=io.write, adaptive=adaptive,
                 profiles=store).play()
    adapts = [line for line in io.out if "the bot adapts" in line]
    assert len(adapts) == len(adaptive.live.changes) == 2, "the bot thinks it is losing after each of its moves"
    assert adapts[0] == "  (the bot adapts: it now plays like a 1550 player, stronger)" and bot.rating == 1600
    assert "Your adaptive rating: 1200 -> " in io.text and store.load().games == 1


def test_adaptive_skips_hinted_games(tmp_path) -> None:
    from chess_engine.bot.adaptive import AdaptiveGame, ProfileStore
    from chess_engine.bot.hints import HintEngine
    from chess_engine.config import AdaptiveConfig, HintsConfig
    from review_helpers import ScriptedAnalyzer

    store = ProfileStore(tmp_path / "a.json", AdaptiveConfig())
    bot = BotPlayer(small_predictor(), 1500, SamplingSettings(temperature=0), seed=0)
    io = Script("hint", "resign")
    TerminalGame(bot, chess.WHITE, Game("You", "Bot"), read=io.read, write=io.write,
                 hints=HintEngine(bot.predictor, ScriptedAnalyzer(), HintsConfig()), hint_limit=1,
                 adaptive=AdaptiveGame("between", 1500, AdaptiveConfig()), profiles=store).play()
    assert "does not change your adaptive rating" in io.text and store.load().games == 0
