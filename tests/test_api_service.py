"""Tests for the API's game service (no HTTP)."""

from __future__ import annotations

import pytest

from chess_engine.api.schemas import NewGameRequest
from chess_engine.api.service import GameRuleError, GameService
from chess_engine.bot.game import IllegalMoveError
from chess_engine.config import Config
from bot_helpers import small_predictor


@pytest.fixture(scope="module")
def service(default_config: Config) -> GameService:
    return GameService(small_predictor(), default_config)


def test_new_game_as_white(service: GameService) -> None:
    session = service.create(NewGameRequest(rating=1300, seed=1))
    state = service.state(session)
    assert state.turn == "white" == state.human_color and state.status == "ongoing"
    assert state.moves_san == [] and len(state.legal_moves) == 20 and state.last_bot_move is None
    assert state.bot_rating == 1300 and "1200-1399" in state.bot_description
    assert service.store.get(session.id) is session


def test_new_game_as_black_bot_opens(service: GameService) -> None:
    session = service.create(NewGameRequest(color="black", your_rating=1100, temperature=0))
    state = service.state(session)
    assert state.turn == "black" and len(state.moves_san) == 1 and state.last_bot_move is not None
    assert state.last_move == state.moves_uci[0] == state.last_bot_move.uci
    assert session.game.white == "Bot 1500" and session.game.black_elo == 1100
    assert session.bot.sampling.greedy


def test_random_colour_and_custom_position(service: GameService) -> None:
    colours = {service.create(NewGameRequest(color="random")).human_color for _ in range(20)}
    assert colours == {True, False}
    custom = service.create(NewGameRequest(fen="7k/8/6K1/8/8/8/R7/1R6 b - - 0 1", color="white"))
    assert len(custom.game.sans) == 1  # Black (the bot) was to move, so it moved first


def test_play_with_and_without_reply(service: GameService) -> None:
    session = service.create(NewGameRequest(seed=2))
    response = service.play(session, "e4")
    assert response.played == "e4" and response.bot_move is not None
    assert response.state.turn == "white" and len(response.state.moves_san) == 2
    quiet = service.play(session, "g1f3", reply=False)
    assert quiet.played == "Nf3" and quiet.bot_move is None and quiet.state.turn == "black"
    with pytest.raises(GameRuleError) as error:
        service.play(session, "d4")
    assert error.value.code == "not_your_turn"
    answer = service.bot_move(session)
    assert answer.bot_move.san == answer.played == answer.state.moves_san[-1]
    with pytest.raises(GameRuleError) as error:
        service.bot_move(session)
    assert error.value.code == "not_bots_turn"
    with pytest.raises(IllegalMoveError):
        service.play(session, "Ke5")


def test_game_over_is_reported(service: GameService) -> None:
    session = service.create(NewGameRequest(fen="7k/8/6K1/8/8/8/R7/1R6 w - - 0 1"))
    service.play(session, "Ra7")  # Black's only reply is Kg8
    response = service.play(session, "Rb8#")
    assert response.bot_move is None and response.state.status == "over"
    assert response.state.outcome.termination == "checkmate" and response.state.outcome.winner == "white"
    assert response.state.legal_moves == []
    with pytest.raises(GameRuleError) as error:
        service.play(session, "Kf7")
    assert error.value.code == "game_over"


class FakeTime:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_timed_game_clocks_and_flag(default_config: Config) -> None:
    now = FakeTime()
    service = GameService(small_predictor(), default_config, now=now)
    session = service.create(NewGameRequest(time_control="1+2", seed=4))
    state = service.state(session)
    assert state.clock.white == 60 and state.clock.black == 60 and state.clock.running is None
    assert state.clock.base_seconds == 60 and state.clock.increment == 2
    service.play(session, "e4")             # first moves are free; then White's clock runs
    assert service.state(session).clock.running == "white"
    now.t += 20
    response = service.play(session, "d4")
    assert response.state.clock.white == pytest.approx(42)  # 60 - 20 + 2
    with pytest.raises(GameRuleError) as error:
        service.takeback(session)
    assert error.value.code == "timed_game"
    now.t += 100                            # White's clock runs out while thinking
    state = service.state(session)
    assert state.status == "over" and state.outcome.termination == "time forfeit"
    assert state.outcome.winner == "black" and state.clock.running is None and state.clock.white == 0
    with pytest.raises(GameRuleError):
        service.play(session, "Nf3")


def test_untimed_game_has_no_clock(service: GameService) -> None:
    assert service.state(service.create(NewGameRequest())).clock is None


def test_hint_limit_and_rated_state(service: GameService, default_config: Config) -> None:
    session = service.create(NewGameRequest(seed=3))
    hints = service.state(session).hints
    assert (hints.limit, hints.used, hints.left, hints.rated) == (default_config.hints.limit, 0, hints.limit, True)
    assert "[Hints" not in service.pgn(session)
    off = service.create(NewGameRequest(hint_limit=0))
    assert service.state(off).hints.limit == 0 and service.state(off).hints.rated
    with pytest.raises(GameRuleError) as err:
        service.create(NewGameRequest(hint_limit=default_config.hints.max_limit + 1))
    assert err.value.code == "too_many_hints"


def test_a_game_with_hints_is_unrated_and_says_so_in_the_pgn(service: GameService) -> None:
    session = service.create(NewGameRequest(seed=4, hint_limit=2))
    session.hints.take(session.game.board)          # what the hint endpoint does
    state = service.state(session)
    assert (state.hints.used, state.hints.left, state.hints.rated) == (1, 1, False)
    pgn = service.pgn(session)
    assert '[Hints "1"]' in pgn and '[Annotator "unrated: hints used"]' in pgn
    service.play(session, "e4")
    service.takeback(session)
    assert service.state(session).hints.used == 1, "a takeback does not give the hint back"
