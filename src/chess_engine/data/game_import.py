"""Import games from pasted PGN or the Lichess / Chess.com public APIs.

    games = parse_pgn_text(pasted_text)                  # list[ImportedGame]
    games[0].white, games[0].speed, games[0].pgn
    games = fetch_lichess("someone", max_games=20)       # the most recent games, newest first
    games = fetch_chesscom("someone", max_games=20)
    result = import_games(cfg.game_import, "lichess", username="someone")   # one entry point, config defaults

Every imported game keeps its PGN text plus a summary (players, ratings, result,
date, time control, speed, a stable id). Only standard chess is kept: the model
does not know variants such as Chess960, so those games are skipped.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess.pgn
import requests

from chess_engine.config import GameImportConfig
from chess_engine.data.pgn_stream import game_id_from_site, speed_from_seconds

#: Sent with every request. Deliberately generic: it identifies the software, never the person using it.
USER_AGENT = "human-chess-engine/0.1 (game import)"
STANDARD_VARIANTS = {"", "standard", "from position"}
SOURCES = ("lichess", "chess.com")

# Lichess: 2-30 letters, digits, "_" or "-", starting and ending with a letter or digit.
# Chess.com: 3-25 letters, digits, "_" or "-". Checking also guarantees a username can never
# change the API URL it is put into (no "/", "?", "..", spaces).
_USERNAME = {
    "lichess": re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,28}[A-Za-z0-9])?$"),
    "chess.com": re.compile(r"^[A-Za-z0-9_-]{3,25}$"),
}


class GameImportError(Exception):
    """A game import failed (bad username, unknown user, network trouble); the message is shown to the user."""


class InvalidUsernameError(GameImportError):
    """The username cannot exist on that site."""


def check_username(source: str, username: str) -> str:
    """The username without surrounding spaces, if it is valid for ``source``.

    Raises:
        InvalidUsernameError: unknown source, or a name the site would not allow.
    """
    if source not in _USERNAME:
        raise InvalidUsernameError(f"unknown source {source!r} (use one of {', '.join(SOURCES)})")
    name = username.strip()
    if not _USERNAME[source].fullmatch(name) or (source == "lichess" and len(name) < 2):
        raise InvalidUsernameError(f"{username!r} is not a valid {source} username")
    return name


@dataclass(frozen=True)
class ImportedGame:
    """One imported game: its PGN and a summary of its headers."""

    source: str                 # "pgn", "lichess" or "chess.com"
    id: str
    pgn: str
    white: str
    black: str
    white_elo: int | None
    black_elo: int | None
    result: str
    date: str | None
    time_control: str | None
    speed: str | None           # ultrabullet / bullet / blitz / rapid / classical / correspondence
    rated: bool | None          # None when the source does not say
    url: str | None
    plies: int

    def summary(self) -> dict:
        """Everything but the PGN text (for listings)."""
        data = asdict(self)
        data.pop("pgn")
        return data


def _elo(value: str | None) -> int | None:
    try:
        return int(value) if value and value.strip().isdigit() else None
    except ValueError:
        return None


def speed_of_time_control(value: str | None) -> str | None:
    """Speed category of a PGN ``TimeControl`` such as ``180+2`` or ``600`` (Chess.com omits ``+0``).

    Daily games (``1/86400``) and ``-`` are correspondence.
    """
    if not value or value in ("-", "?"):
        return None
    if "/" in value:
        return "correspondence"
    base, _, increment = value.partition("+")
    if not base.isdigit() or (increment and not increment.isdigit()):
        return None
    return speed_from_seconds(int(base) + 40 * int(increment or 0))


def _game_id(headers: chess.pgn.Headers, movetext: str) -> str:
    site = headers.get("Site", "")
    if "lichess.org/" in site:
        return game_id_from_site(site)
    link = headers.get("Link", "")
    if "chess.com/" in link:
        return link.rstrip("/").rsplit("/", 1)[-1]
    key = "|".join([headers.get("White", ""), headers.get("Black", ""), headers.get("Date", ""),
                    headers.get("UTCTime", ""), movetext])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _rated(headers: chess.pgn.Headers) -> bool | None:
    event = headers.get("Event", "").lower()
    if "rated" in event:
        return not event.startswith("casual") and "unrated" not in event
    if "casual" in event:
        return False
    return None


def to_imported(game: chess.pgn.Game, source: str, rated: bool | None = None, url: str | None = None) -> ImportedGame:
    """An :class:`ImportedGame` from a parsed game (``rated`` / ``url`` override what the headers say)."""
    h = game.headers
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    movetext = game.accept(exporter)
    full = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True))
    date = h.get("UTCDate") or h.get("Date")
    return ImportedGame(
        source=source, id=_game_id(h, movetext), pgn=full.strip() + "\n",
        white=h.get("White", "?"), black=h.get("Black", "?"),
        white_elo=_elo(h.get("WhiteElo")), black_elo=_elo(h.get("BlackElo")),
        result=h.get("Result", "*"), date=None if not date or "?" in date else date,
        time_control=h.get("TimeControl"), speed=speed_of_time_control(h.get("TimeControl")),
        rated=rated if rated is not None else _rated(h),
        url=url or (h.get("Site") if h.get("Site", "").startswith("http") else h.get("Link")),
        plies=sum(1 for _ in game.mainline_moves()),
    )


def parse_pgn_text(text: str, source: str = "pgn") -> list[ImportedGame]:
    """Every standard-chess game with at least one move in ``text`` (broken games and variants are skipped)."""
    stream = io.StringIO(text)
    games = []
    while (game := chess.pgn.read_game(stream)) is not None:
        if game.errors or game.headers.get("Variant", "").strip().lower() not in STANDARD_VARIANTS:
            continue
        if next(iter(game.mainline_moves()), None) is None:
            continue
        games.append(to_imported(game, source))
    return dedupe(games)


def dedupe(games: list[ImportedGame]) -> list[ImportedGame]:
    """Drop repeated games (same id), keeping the first."""
    seen: set[str] = set()
    out = []
    for game in games:
        if game.id not in seen:
            seen.add(game.id)
            out.append(game)
    return out


# --------------------------------------------------------------------------- HTTP


class UserNotFoundError(GameImportError):
    """The site has no player with that name."""


class SourceUnavailableError(GameImportError):
    """The site could not be reached or refused the request (after retries)."""


class HttpClient:
    """GET requests with retries.

    Network errors, HTTP 429 (rate limited) and 5xx answers are retried up to ``retries``
    times, waiting ``backoff * 2**attempt`` seconds or the server's ``Retry-After`` (at most
    60 s). A 404 means the user does not exist.
    """

    def __init__(self, timeout: float = 30, retries: int = 3, backoff: float = 1.0,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = session or requests.Session()
        self.sleep = sleep

    def get(self, url: str, params: dict[str, Any] | None = None, accept: str | None = None,
            what: str = "the site") -> requests.Response:
        headers = {"User-Agent": USER_AGENT}
        if accept:
            headers["Accept"] = accept
        problem = ""
        for attempt in range(self.retries + 1):
            wait = self.backoff * 2**attempt
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                problem = f"cannot reach {what}: {type(exc).__name__}"
            else:
                if response.status_code == 404:
                    raise UserNotFoundError(f"{what} has no such player")
                if response.status_code == 429 or response.status_code >= 500:
                    problem = f"{what} answered HTTP {response.status_code}"
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        wait = float(retry_after)
                elif response.status_code >= 400:
                    raise SourceUnavailableError(f"{what} refused the request (HTTP {response.status_code})")
                else:
                    return response
            if attempt < self.retries:
                self.sleep(min(wait, 60.0))
        raise SourceUnavailableError(f"{problem} (gave up after {self.retries + 1} attempts)")


# --------------------------------------------------------------------------- Lichess

_LICHESS_PERF = {"ultrabullet": "ultraBullet", "bullet": "bullet", "blitz": "blitz", "rapid": "rapid",
                 "classical": "classical", "correspondence": "correspondence"}


def fetch_lichess(
    username: str,
    max_games: int = 20,
    speeds: list[str] | None = None,
    rated_only: bool = False,
    base_url: str = "https://lichess.org",
    client: HttpClient | None = None,
) -> list[ImportedGame]:
    """The player's most recent games on Lichess, newest first (``GET /api/games/user/{name}``, as PGN).

    Raises:
        InvalidUsernameError, UserNotFoundError, SourceUnavailableError.
    """
    name = check_username("lichess", username)
    params: dict[str, Any] = {"max": max_games, "clocks": "true", "opening": "true", "moves": "true",
                              "tags": "true", "evals": "false"}
    if speeds:
        params["perfType"] = ",".join(_LICHESS_PERF[s] for s in speeds if s in _LICHESS_PERF)
    if rated_only:
        params["rated"] = "true"
    response = (client or HttpClient()).get(f"{base_url.rstrip('/')}/api/games/user/{name}", params,
                                            accept="application/x-chess-pgn", what="Lichess")
    return parse_pgn_text(response.text, source="lichess")[:max_games]


# --------------------------------------------------------------------------- Chess.com

_CHESSCOM_SPEED = {"bullet": "bullet", "blitz": "blitz", "rapid": "rapid", "daily": "correspondence"}


def fetch_chesscom(
    username: str,
    max_games: int = 20,
    speeds: list[str] | None = None,
    rated_only: bool = False,
    base_url: str = "https://api.chess.com",
    client: HttpClient | None = None,
    max_archives: int = 12,
) -> list[ImportedGame]:
    """The player's most recent standard-chess games on Chess.com, newest first.

    Chess.com publishes games in monthly archives: the list is read
    (``/pub/player/{name}/games/archives``), then archives from the newest month back
    (at most ``max_archives``) until ``max_games`` games pass the filters. Only archive
    links on ``base_url`` are followed.

    Raises:
        InvalidUsernameError, UserNotFoundError, SourceUnavailableError.
    """
    name = check_username("chess.com", username).lower()
    client = client or HttpClient()
    base = base_url.rstrip("/")
    archives = client.get(f"{base}/pub/player/{name}/games/archives", what="Chess.com").json().get("archives", [])
    games: list[ImportedGame] = []
    for url in reversed(archives[-max_archives:]):
        if not url.startswith(base + "/"):
            continue  # never follow a link to another host
        month = client.get(url, what="Chess.com").json().get("games", [])
        for entry in reversed(month):  # newest first
            if entry.get("rules", "chess") != "chess" or not entry.get("pgn"):
                continue
            speed = _CHESSCOM_SPEED.get(entry.get("time_class", ""))
            if speeds and speed not in speeds:
                continue
            if rated_only and not entry.get("rated", False):
                continue
            parsed = parse_pgn_text(entry["pgn"], source="chess.com")
            if not parsed:
                continue
            game = parsed[0]
            games.append(ImportedGame(**{**game.__dict__, "rated": bool(entry.get("rated")) if "rated" in entry
                                         else None, "url": entry.get("url") or game.url,
                                         "speed": speed or game.speed}))
            if len(games) >= max_games:
                return dedupe(games)
    return dedupe(games)


# --------------------------------------------------------------------------- one entry point


@dataclass(frozen=True)
class ImportResult:
    """Games from one import, newest first for the online sources."""

    source: str                 # "pgn", "lichess" or "chess.com"
    username: str | None
    games: list[ImportedGame]
    fetched_at: float           # time.time() of the fetch


def filter_games(games: list[ImportedGame], speeds: list[str] | None = None, rated_only: bool = False,
                 max_games: int | None = None) -> list[ImportedGame]:
    """Keep the games of the given speeds (all if empty), rated only if asked, at most ``max_games``."""
    kept = [g for g in games if (not speeds or g.speed in speeds) and (not rated_only or g.rated)]
    return kept[:max_games] if max_games is not None else kept


class ImportCache:
    """Recent online imports on disk (one JSON file per request), reused for ``ttl_minutes``."""

    def __init__(self, directory: str | Path, ttl_minutes: float, now: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory)
        self.ttl = ttl_minutes * 60
        self.now = now

    def _path(self, source: str, username: str, max_games: int, speeds: list[str], rated_only: bool) -> Path:
        key = json.dumps([source, username.lower(), max_games, sorted(speeds), rated_only])
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.directory / source.replace(".", "_") / f"{username.lower()}_{digest}.json"

    def get(self, source: str, username: str, max_games: int, speeds: list[str], rated_only: bool) -> ImportResult | None:
        if self.ttl <= 0:
            return None
        path = self._path(source, username, max_games, speeds, rated_only)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if self.now() - data["cached_at"] > self.ttl:
            return None
        return ImportResult(data["source"], data["username"], [ImportedGame(**g) for g in data["games"]],
                            data["fetched_at"])

    def put(self, result: ImportResult, max_games: int, speeds: list[str], rated_only: bool) -> None:
        if self.ttl <= 0 or result.username is None:
            return
        path = self._path(result.source, result.username, max_games, speeds, rated_only)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": result.source, "username": result.username, "fetched_at": result.fetched_at,
                   "cached_at": self.now(),  # expiry is judged on the cache's own clock
                   "games": [asdict(g) for g in result.games]}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)


def import_games(
    cfg: GameImportConfig,
    source: str,
    username: str | None = None,
    text: str | None = None,
    max_games: int | None = None,
    speeds: list[str] | None = None,
    rated_only: bool | None = None,
    client: HttpClient | None = None,
    cache: ImportCache | None = None,
) -> ImportResult:
    """Import from ``source``: ``"pgn"`` (needs ``text``), ``"lichess"`` or ``"chess.com"`` (need ``username``).

    Unset limits and filters come from ``cfg``. With a ``cache``, a recent identical online
    import is returned without contacting the site.
    """
    max_games = max_games or cfg.max_games
    speeds = cfg.speeds if speeds is None else speeds
    rated_only = cfg.rated_only if rated_only is None else rated_only
    if source == "pgn":
        if text is None:
            raise GameImportError("paste some PGN to import")
        return ImportResult("pgn", None, filter_games(parse_pgn_text(text), speeds, rated_only, max_games), time.time())
    if not username:
        raise GameImportError(f"a {source} username is needed")
    if source in SOURCES:
        username = check_username(source, username)
    if cache is not None and (cached := cache.get(source, username, max_games, speeds, rated_only)) is not None:
        return cached
    client = client or HttpClient(cfg.timeout_seconds, cfg.retries)
    if source == "lichess":
        games = fetch_lichess(username, max_games, speeds, rated_only, cfg.lichess_url, client)
    elif source == "chess.com":
        games = fetch_chesscom(username, max_games, speeds, rated_only, cfg.chesscom_url, client)
    else:
        raise InvalidUsernameError(f"unknown source {source!r} (use pgn, lichess or chess.com)")
    result = ImportResult(source, username, filter_games(games, speeds, rated_only, max_games), time.time())
    if cache is not None:
        cache.put(result, max_games, speeds, rated_only)
    return result
