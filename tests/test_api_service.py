"""Tests for the API's game service (no HTTP)."""

from __future__ import annotations

import pytest

from chess_engine.api.schemas import NewGameRequest
from chess_engine.api.service import GameRuleError, GameService
from chess_engine.bot.adaptive import ProfileStore
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


@pytest.fixture()
def adaptive_service(default_config: Config, tmp_path) -> GameService:
    return GameService(small_predictor(), default_config,
                       profiles=ProfileStore(tmp_path / "adaptive.json", default_config.adaptive))


def test_adaptive_off_by_default(adaptive_service: GameService) -> None:
    session = adaptive_service.create(NewGameRequest(rating=1700))
    a = adaptive_service.state(session).adaptive
    assert (a.mode, a.start_rating, a.changes, a.your_rating, a.result) == ("off", 1700, [], None, None)


def test_between_games_uses_your_profile_rating(adaptive_service: GameService) -> None:
    adaptive_service.profiles.reset(1433)
    session = adaptive_service.create(NewGameRequest(rating=2000, adaptive="between"))
    state = adaptive_service.state(session)
    assert state.bot_rating == 1450 and state.adaptive.start_rating == 1450, "the chosen rating is ignored"
    assert state.your_rating == 1433 and state.adaptive.your_rating == 1433
    live = adaptive_service.create(NewGameRequest(rating=1800, adaptive="live", your_rating=1600))
    assert adaptive_service.state(live).bot_rating == 1800 and live.your_rating == 1600
    assert live.adaptive.live is not None and live.adaptive.current == 1800


class LosingBotPredictor:
    """The small model, but the bot always thinks it is losing (for live-mode tests)."""

    def __init__(self) -> None:
        self.inner = small_predictor()
        self.ratings = self.inner.ratings

    def predict(self, board, rating, **kwargs):
        from chess_engine.model.inference import Prediction

        prediction = self.inner.predict(board, rating, **kwargs)
        return Prediction(prediction.moves, (0.05, 0.1, 0.85))

    def predict_many(self, *args, **kwargs):
        return self.inner.predict_many(*args, **kwargs)


def test_live_mode_raises_the_rating_and_records_the_result(default_config: Config, tmp_path) -> None:
    profiles = ProfileStore(tmp_path / "a.json", default_config.adaptive)
    service = GameService(LosingBotPredictor(), default_config, profiles=profiles)
    session = service.create(NewGameRequest(rating=1500, adaptive="live", your_rating=1500, seed=1))
    a = default_config.adaptive
    for _ in range(a.live_after + 1):            # we play the first legal move; the bot "is losing" every time
        board = session.game.board
        service.play(session, min(m.uci() for m in board.legal_moves))
    state = service.state(session)
    assert state.status == "ongoing" and len(state.moves_san) == 2 * (a.live_after + 1)
    assert state.bot_rating == 1550 and [c.rating for c in state.adaptive.changes] == [1550]
    assert state.adaptive.changes[0].bot_move == a.live_after + 1
    service.resign(session)
    result = service.state(session).adaptive.result
    assert result is not None and result.score == 0 and result.rating_after < result.rating_before
    assert result.bot_rating == 1500, "measured against the bot's mean rating (every move so far at 1500)"
    assert profiles.load().games == 1 and profiles.load().history[-1].mode == "live"
    pgn = service.pgn(session)
    assert '[Adaptive "live"]' in pgn and '[BotRatings "1500 1550"]' in pgn


def test_between_games_records_rated_games_only(adaptive_service: GameService) -> None:
    profiles = adaptive_service.profiles
    profiles.reset(1500)
    rated = adaptive_service.create(NewGameRequest(adaptive="between", seed=2))
    adaptive_service.resign(rated)
    result = adaptive_service.state(rated).adaptive.result
    assert result.rating_before == 1500 and result.rating_after == 1500 - 80 / 2 and result.next_bot_rating == 1450
    hinted = adaptive_service.create(NewGameRequest(adaptive="between", seed=3))
    hinted.hints.take(hinted.game.board)
    adaptive_service.resign(hinted)
    assert adaptive_service.state(hinted).adaptive.result is None, "a game with hints does not count"
    off = adaptive_service.create(NewGameRequest(adaptive="off", seed=4))
    adaptive_service.resign(off)
    assert profiles.load().games == 1
