"""The Lichess BOT account integration (step 20), against a fake Lichess."""

from __future__ import annotations

import pytest

from chess_engine.lichess_bot import USER_AGENT, ChallengePolicy, LichessClient, LichessError, challenge_decision
from lichess_fake import FakeLichess, FakeResponse

TOKEN = "lip_secret_token"


def test_client_calls_and_headers() -> None:
    fake = FakeLichess({"/api/account": FakeResponse(body={"id": "humanbot", "title": "BOT"}),
                        "/api/stream/event": FakeResponse(lines=[{"type": "gameStart", "game": {"id": "g1"}}])})
    client = LichessClient(TOKEN, session=fake)
    assert client.account()["id"] == "humanbot"
    assert list(client.stream_events()) == [{"type": "gameStart", "game": {"id": "g1"}}]
    client.accept("c1")
    client.decline("c2", "later")
    client.move("g1", "e2e4")
    assert fake.posts == [("/api/challenge/c1/accept", {}), ("/api/challenge/c2/decline", {"reason": "later"}),
                          ("/api/bot/game/g1/move/e2e4", {})]
    assert all(h == {"Authorization": f"Bearer {TOKEN}", "User-Agent": USER_AGENT} for h in fake.headers)
    assert TOKEN not in repr(client)


def test_client_errors() -> None:
    with pytest.raises(LichessError, match="token is required"):
        LichessClient("", session=FakeLichess())
    fake = FakeLichess()
    client = LichessClient(TOKEN, session=fake)
    with pytest.raises(LichessError, match=r"GET /api/account: HTTP 404 \(Not found\)"):
        client.account()
    fake.fail_posts["/api/bot/game/g1/move/e2e5"] = FakeResponse(400, {"error": "Not your turn, or game already over"})
    with pytest.raises(LichessError, match="Not your turn") as err:
        client.move("g1", "e2e5")
    assert TOKEN not in str(err.value)


def _challenge(**kw) -> dict:
    return {"id": "c1", "variant": {"key": kw.get("variant", "standard")}, "speed": kw.get("speed", "blitz"),
            "rated": kw.get("rated", True), "challenger": {"name": "someone", "rating": 1500}}


def test_challenge_decisions() -> None:
    policy = ChallengePolicy()
    assert challenge_decision(_challenge(), policy, 0) is None
    assert challenge_decision(_challenge(variant="chess960"), policy, 0) == "variant"
    assert challenge_decision(_challenge(speed="bullet"), policy, 0) == "timeControl"
    assert challenge_decision(_challenge(speed="correspondence"), policy, 0) == "timeControl"
    assert challenge_decision(_challenge(), policy, 1) == "later"
    assert challenge_decision(_challenge(rated=True), ChallengePolicy(rated=False), 0) == "casual"
    assert challenge_decision(_challenge(rated=False), ChallengePolicy(casual=False), 0) == "rated"


class _Choice:
    def __init__(self, uci: str) -> None:
        import chess

        self.move = chess.Move.from_uci(uci)


class ScriptedBot:
    """Plays the first legal move (UCI order) and remembers what it was told."""

    def __init__(self, rating: int) -> None:
        self.rating = rating
        self.calls: list[dict] = []

    def choose_move(self, board, **kwargs):
        self.calls.append({"fen": board.fen(), **kwargs})
        return _Choice(min(m.uci() for m in board.legal_moves))


def _game(bot_white: bool, move_states: list[str], status_end: str = "mate", opp_rating: int = 1720) -> list[dict]:
    me = {"id": "humanbot", "name": "HumanBot", "rating": 1500}
    them = {"id": "opp", "name": "x", "rating": opp_rating}
    full = {"type": "gameFull", "id": "g1", "initialFen": "startpos",
            "white": me if bot_white else them, "black": them if bot_white else me,
            "clock": {"initial": 300000, "increment": 3000},
            "state": {"type": "gameState", "moves": "", "wtime": 300000, "btime": 300000, "status": "started"}}
    states = [{"type": "gameState", "moves": m, "wtime": 290000, "btime": 280000, "status": "started"}
              for m in move_states]
    states.append({"type": "gameState", "moves": move_states[-1] if move_states else "", "status": status_end})
    return [full, {"type": "chatLine", "text": "hi"}] + states


def test_play_game_as_white() -> None:
    from chess_engine.lichess_bot import GameSettings, play_game

    fake = FakeLichess({"/api/bot/game/stream/g1": FakeResponse(lines=_game(True, ["a2a3", "a2a3 e7e5"]))})
    bots: list[ScriptedBot] = []

    def make(rating: int) -> ScriptedBot:
        bots.append(ScriptedBot(rating))
        return bots[-1]

    logs: list[str] = []
    status = play_game(LichessClient(TOKEN, session=fake), "g1", "HumanBot", make,
                       GameSettings(match_opponent=True, offset=-100), logs.append)
    assert status == "mate"
    assert bots[0].rating == 1620, "the opponent's 1720 minus the offset"
    assert [p for p, _ in fake.posts] == ["/api/bot/game/g1/move/a2a3", "/api/bot/game/g1/move/a1a2"], \
        "moves at the start and after Black's reply, never on Black's turn"
    second = bots[0].calls[1]
    assert second["clock"] == 290 and second["opp_clock"] == 280 and second["base_seconds"] == 300
    assert second["increment"] == 3 and second["opp_rating"] == 1720
    assert logs[0] == "game g1: playing White against x (1720) at 1620" and logs[-1] == "game g1: over (mate)"


def test_play_game_as_black_and_refused_moves() -> None:
    from chess_engine.lichess_bot import GameSettings, play_game

    fake = FakeLichess({"/api/bot/game/stream/g1": FakeResponse(lines=_game(False, ["e2e4"], "resign"))})
    fake.fail_posts["/api/bot/game/g1/move/a7a5"] = FakeResponse(400, {"error": "game over"})
    logs: list[str] = []
    status = play_game(LichessClient(TOKEN, session=fake), "g1", "humanbot", ScriptedBot, GameSettings(rating=1300),
                       logs.append)
    assert status == "resign" and [p for p, _ in fake.posts] == ["/api/bot/game/g1/move/a7a5"]
    assert any("refused" in line for line in logs)
    broken = FakeLichess({"/api/bot/game/stream/g2": FakeResponse(lines=[{"type": "gameState"}])})
    with pytest.raises(LichessError, match="did not start"):
        play_game(LichessClient(TOKEN, session=broken), "g2", "humanbot", ScriptedBot)
    ended = FakeLichess({"/api/bot/game/stream/g3": FakeResponse(lines=_game(False, ["e2e4"])[:1])})
    assert play_game(LichessClient(TOKEN, session=ended), "g3", "humanbot", ScriptedBot) == "aborted"


def test_rating_settings() -> None:
    from chess_engine.lichess_bot import GameSettings

    assert GameSettings(rating=1300).rating_for(2000) == 1300
    assert GameSettings(match_opponent=True).rating_for(None) == 1500
    assert GameSettings(match_opponent=True).rating_for(2900) == 2400
    assert GameSettings(match_opponent=True, offset=-300).rating_for(900) == 800
