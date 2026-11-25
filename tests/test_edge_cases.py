"""Edge cases found by the coverage pass (step 20): clocks at the end of games, failed hints, file versions."""

from __future__ import annotations

import chess
import pytest
import torch

from bot_helpers import small_predictor
from chess_engine.api.schemas import NewGameRequest
from chess_engine.api.service import GameRuleError, GameService
from chess_engine.bot.adaptive import ProfileStore
from chess_engine.config import Config


@pytest.fixture()
def service(default_config: Config, tmp_path) -> GameService:
    return GameService(small_predictor(), default_config,
                       profiles=ProfileStore(tmp_path / "a.json", default_config.adaptive))


def test_a_timed_game_ending_by_mate_or_resignation_stops_the_clock(service: GameService) -> None:
    mate = service.create(NewGameRequest(fen="7k/8/6K1/8/8/8/R7/1R6 w - - 0 1", time_control="3+0", seed=1))
    service.play(mate, "a2a7")
    service.play(mate, "b1b8")
    state = service.state(mate)
    assert state.status == "over" and state.outcome.termination == "checkmate"
    assert state.clock.running is None, "the clock stops at checkmate"
    resigned = service.create(NewGameRequest(time_control="3+0", seed=2))
    service.play(resigned, "e4")
    state = service.resign(resigned)
    assert state.status == "over" and state.clock.running is None


class _NoHint:
    """A hint engine that cannot give a hint (as when the engine finds no move)."""

    def hint(self, *args, **kwargs):
        from chess_engine.bot.hints import HintError

        raise HintError("the engine found no move")


def test_a_failed_hint_is_reported_and_not_counted(service: GameService) -> None:
    session = service.create(NewGameRequest(time_control="5+0", hint_limit=1, seed=3))
    with pytest.raises(GameRuleError) as err:
        service.hint(session, _NoHint())
    assert err.value.code == "no_hint" and session.hints.used == 0


def test_terminal_hints_in_a_timed_game_and_a_failed_hint() -> None:
    from chess_engine.bot.cli import TerminalGame
    from chess_engine.bot.clock import ChessClock, TimeControl
    from chess_engine.bot.game import Game
    from chess_engine.bot.hints import HintEngine
    from chess_engine.bot.player import BotPlayer
    from chess_engine.config import HintsConfig
    from review_helpers import ScriptedAnalyzer

    bot = BotPlayer(small_predictor(), 1500, seed=0)
    out: list[str] = []
    lines = iter(["hint", "quit"])
    clock = ChessClock(TimeControl.parse("3+2"))
    engine = HintEngine(bot.predictor, ScriptedAnalyzer(), HintsConfig())
    TerminalGame(bot, chess.WHITE, Game("You", "Bot"), clock=clock, read=lambda p: next(lines), write=out.append,
                 hints=engine, hint_limit=2).play()
    assert any(line.startswith("Hint (nudge)") for line in out), out
    out.clear()
    lines = iter(["hint", "quit"])
    TerminalGame(bot, chess.WHITE, Game("You", "Bot"), read=lambda p: next(lines), write=out.append,
                 hints=_NoHint(), hint_limit=2).play()
    assert "No hint: the engine found no move." in out


def test_model_files_wrong_version_wrong_buckets_and_failed_saves(tmp_path, monkeypatch) -> None:
    from chess_engine.config import RatingsConfig
    from chess_engine.model.inference import ModelFileError, load_model, save_model

    predictor = small_predictor()
    with pytest.raises(ValueError, match="number of buckets"):
        save_model(predictor.model, tmp_path / "m.pt", RatingsConfig(bucket_edges=[0, 1500]))
    path = save_model(predictor.model, tmp_path / "m.pt", predictor.ratings)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, tmp_path / "old.pt")
    with pytest.raises(ModelFileError, match="model format 99"):
        load_model(tmp_path / "old.pt")

    def broken(obj, f):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError, match="disk full"):
        save_model(predictor.model, tmp_path / "new.pt", predictor.ratings)
    assert not list(tmp_path.glob("*.tmp")), "a failed save leaves no temporary file"


def test_style_file_from_another_version(tmp_path) -> None:
    from test_style import _info

    from chess_engine.style.adapter import StyleAdapter
    from chess_engine.style.store import StyleError, StyleLibrary

    base = small_predictor().model
    library = StyleLibrary(tmp_path)
    path = library.save(StyleAdapter.for_model(base), _info(), base)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 0
    torch.save(payload, path)
    with pytest.raises(StyleError, match="style format 0"):
        library.info("ann")
