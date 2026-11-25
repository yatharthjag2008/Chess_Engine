"""Run the human-like bot on a Lichess BOT account (step 20, optional).

Lichess's Bot API (https://lichess.org/api#tag/Bot): the account streams events (challenges, games starting),
accepts or declines challenges, then streams each game and posts its moves. This module is small and
single-threaded: one game at a time, the bot answering with the same move model as everywhere else.

    client = LichessClient(token)                     # a token of a BOT account, with the bot:play scope
    run_bot(client, make_bot, ChallengePolicy())      # until interrupted

The token is only sent to Lichess (in the Authorization header) and never printed or logged. Nothing here
runs in the tests against the real site: the client takes any object with ``get`` / ``post`` like
``requests.Session``, and the tests pass a fake one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import chess

USER_AGENT = "human-chess-engine/0.1 (lichess bot)"
LICHESS = "https://lichess.org"


class LichessError(Exception):
    """Lichess refused a request (bad token, not a BOT account, game over, ...)."""


class HttpSession(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...


def _ndjson(lines: Iterator[bytes | str]) -> Iterator[dict]:
    """Objects from a newline-delimited JSON stream (keep-alive blank lines are skipped)."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if line:
            yield json.loads(line)


class LichessClient:
    """The few Bot API calls the bot needs."""

    def __init__(self, token: str, session: HttpSession | None = None, base_url: str = LICHESS,
                 timeout: float = 30.0) -> None:
        if not token:
            raise LichessError("a Lichess API token is required (set LICHESS_BOT_TOKEN)")
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}

    def __repr__(self) -> str:          # never show the token
        return f"LichessClient({self.base_url!r})"

    def _check(self, response: Any, what: str) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", "")
            except Exception:  # noqa: BLE001 - any non-JSON body
                detail = ""
            raise LichessError(f"{what}: HTTP {response.status_code}" + (f" ({detail})" if detail else ""))
        return response

    def _get(self, path: str, stream: bool = False) -> Any:
        response = self.session.get(self.base_url + path, headers=self._headers, stream=stream,
                                    timeout=None if stream else self.timeout)
        return self._check(response, f"GET {path}")

    def _post(self, path: str, data: dict | None = None) -> None:
        response = self.session.post(self.base_url + path, headers=self._headers, data=data or {},
                                     timeout=self.timeout)
        self._check(response, f"POST {path}")

    def account(self) -> dict:
        return self._get("/api/account").json()

    def stream_events(self) -> Iterator[dict]:
        return _ndjson(self._get("/api/stream/event", stream=True).iter_lines())

    def stream_game(self, game_id: str) -> Iterator[dict]:
        return _ndjson(self._get(f"/api/bot/game/stream/{game_id}", stream=True).iter_lines())

    def accept(self, challenge_id: str) -> None:
        self._post(f"/api/challenge/{challenge_id}/accept")

    def decline(self, challenge_id: str, reason: str = "generic") -> None:
        self._post(f"/api/challenge/{challenge_id}/decline", {"reason": reason})

    def move(self, game_id: str, uci: str) -> None:
        self._post(f"/api/bot/game/{game_id}/move/{uci}")


# --------------------------------------------------------------------------- challenges


@dataclass
class ChallengePolicy:
    """Which challenges to accept (the rest are declined with Lichess's reason codes)."""

    speeds: list[str] = field(default_factory=lambda: ["blitz", "rapid", "classical"])
    rated: bool = True                 # accept rated games
    casual: bool = True                # accept casual games
    max_games: int = 1                 # games at the same time (this bot plays one at a time)


def challenge_decision(challenge: dict, policy: ChallengePolicy, active_games: int) -> str | None:
    """None to accept ``challenge`` (from the event stream), else Lichess's decline reason."""
    if challenge.get("variant", {}).get("key", "standard") != "standard":
        return "variant"
    if challenge.get("speed") not in policy.speeds:
        return "timeControl"
    rated = bool(challenge.get("rated"))
    if rated and not policy.rated:
        return "casual"
    if not rated and not policy.casual:
        return "rated"
    if active_games >= policy.max_games:
        return "later"
    return None
