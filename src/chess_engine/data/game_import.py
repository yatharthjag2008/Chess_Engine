"""Import games from pasted PGN or the Lichess / Chess.com public APIs.

    games = parse_pgn_text(pasted_text)                  # list[ImportedGame]
    games[0].white, games[0].speed, games[0].pgn

Every imported game keeps its PGN text plus a summary (players, ratings, result,
date, time control, speed, a stable id). Only standard chess is kept: the model
does not know variants such as Chess960, so those games are skipped.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict, dataclass

import chess.pgn

from chess_engine.data.pgn_stream import game_id_from_site, speed_from_seconds

STANDARD_VARIANTS = {"", "standard", "from position"}
SOURCES = ("lichess", "chess.com")

# Lichess: 2-30 letters, digits, "_" or "-", starting and ending with a letter or digit.
# Chess.com: 3-25 letters, digits, "_" or "-". Checking also guarantees a username can never
# change the API URL it is put into (no "/", "?", "..", spaces).
_USERNAME = {
    "lichess": re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,28}[A-Za-z0-9])?$"),
    "chess.com": re.compile(r"^[A-Za-z0-9_-]{3,25}$"),
}


class ImportError_(Exception):
    """A game import failed (bad username, unknown user, network trouble); the message is shown to the user."""


class InvalidUsernameError(ImportError_):
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
