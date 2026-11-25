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
