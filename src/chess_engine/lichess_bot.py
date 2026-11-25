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


# --------------------------------------------------------------------------- one game


@dataclass
class GameSettings:
    """The bot's rating in a game: fixed, or the opponent's Lichess rating (plus ``offset``), kept in range."""

    rating: int = 1500
    match_opponent: bool = False
    offset: int = 0
    min_rating: int = 800
    max_rating: int = 2400

    def rating_for(self, opponent: int | None) -> int:
        if not self.match_opponent or opponent is None:
            return self.rating
        return min(max(opponent + self.offset, self.min_rating), self.max_rating)


BotFactory = Callable[[int], Any]       # rating -> a BotPlayer (anything with ``choose_move`` and ``rating``)


def _position(initial_fen: str, moves: str) -> chess.Board:
    board = chess.Board() if initial_fen in ("", "startpos") else chess.Board(initial_fen)
    for uci in moves.split():
        board.push_uci(uci)
    return board


def play_game(client: LichessClient, game_id: str, username: str, make_bot: BotFactory,
              settings: GameSettings | None = None, log: Callable[[str], None] = lambda _: None) -> str:
    """Play one game to its end; returns Lichess's final status (``mate``, ``resign``, ``outoftime``, ...)."""
    settings = settings or GameSettings()
    events = client.stream_game(game_id)
    full = next(events, None)
    if full is None or full.get("type") != "gameFull":
        raise LichessError(f"game {game_id}: the stream did not start with the game")
    white, black = full.get("white", {}), full.get("black", {})
    color = chess.WHITE if str(white.get("id", "")).lower() == username.lower() else chess.BLACK
    opponent = black if color == chess.WHITE else white
    bot = make_bot(settings.rating_for(opponent.get("rating")))
    clock = full.get("clock") or {}
    base = int(clock.get("initial", 180_000) / 1000)
    increment = int(clock.get("increment", 0) / 1000)
    log(f"game {game_id}: playing {'White' if color == chess.WHITE else 'Black'} against "
        f"{opponent.get('name', '?')} ({opponent.get('rating', '?')}) at {bot.rating}")
    initial = full.get("initialFen", "startpos")
    mine, theirs = ("wtime", "btime") if color == chess.WHITE else ("btime", "wtime")

    def handle(state: dict) -> str | None:
        status = state.get("status", "started")
        if status not in ("created", "started"):
            return status
        board = _position(initial, state.get("moves", ""))
        if board.turn != color or board.is_game_over():
            return None
        choice = bot.choose_move(board, opp_rating=opponent.get("rating"),
                                 clock=state.get(mine, base * 1000) / 1000,
                                 opp_clock=state.get(theirs, base * 1000) / 1000,
                                 base_seconds=base, increment=increment)
        try:
            client.move(game_id, choice.move.uci())
        except LichessError as exc:     # e.g. the opponent resigned meanwhile: the next state says so
            log(f"game {game_id}: move {choice.move.uci()} refused ({exc})")
        return None

    result = handle(full.get("state", {}))
    if result is None:
        for event in events:
            if event.get("type") == "gameState" and (result := handle(event)) is not None:
                break
    result = result or "aborted"
    log(f"game {game_id}: over ({result})")
    return result


# --------------------------------------------------------------------------- the account


def run_bot(client: LichessClient, make_bot: BotFactory, policy: ChallengePolicy | None = None,
            settings: GameSettings | None = None, username: str | None = None, max_games: int | None = None,
            log: Callable[[str], None] = print) -> int:
    """Answer challenges and play games until the event stream ends or ``max_games`` games were played.

    Games are played one at a time (a challenge arriving meanwhile waits in the stream and is answered after
    the game). Returns the number of games played.
    """
    policy = policy or ChallengePolicy()
    username = username or client.account()["id"]
    played = 0
    for event in client.stream_events():
        kind = event.get("type")
        if kind == "challenge":
            challenge = event.get("challenge", {})
            if str(challenge.get("challenger", {}).get("id", "")).lower() == username.lower():
                continue                                   # our own challenge to someone else
            who = challenge.get("challenger", {}).get("name", "?")
            reason = challenge_decision(challenge, policy, active_games=0)
            if reason is None:
                log(f"accepting {challenge.get('speed')} challenge from {who}")
                client.accept(challenge["id"])
            else:
                log(f"declining challenge from {who} ({reason})")
                client.decline(challenge["id"], reason)
        elif kind == "gameStart":
            game_id = event.get("game", {}).get("gameId") or event.get("game", {}).get("id")
            try:
                play_game(client, game_id, username, make_bot, settings, log)
            except LichessError as exc:
                log(f"game {game_id}: stopped ({exc})")
            played += 1
            if max_games is not None and played >= max_games:
                break
    return played
