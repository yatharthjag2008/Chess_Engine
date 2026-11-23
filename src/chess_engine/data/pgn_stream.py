"""Stream Lichess PGN games and filter them by time control, rating and length.

The Lichess monthly dumps are ~30 GB compressed, so nothing here loads a whole
file: games are read one at a time from a text stream, filtered on their headers
first (cheap), and only then is the movetext parsed.

Typical use::

    game_filter = GameFilter.from_config(cfg.data)
    stats = FilterStats()
    for record in filter_games(iter_raw_games(lines), game_filter, stats):
        ...
    print(stats.as_dict())
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_engine.config import DataConfig

# --------------------------------------------------------------------------- time control

# Lichess speed categories by estimated game duration (base + 40 * increment), in seconds.
# Each entry is (upper bound, exclusive) -> speed. Anything >= the last bound is classical.
# Source: lichess.org/faq#time-controls
_SPEED_BOUNDS: tuple[tuple[int, str], ...] = (
    (30, "ultrabullet"),
    (180, "bullet"),
    (480, "blitz"),
    (1500, "rapid"),
)
CORRESPONDENCE = "correspondence"


@dataclass(frozen=True, slots=True)
class TimeControl:
    """A parsed ``TimeControl`` header such as ``"180+2"`` (3 minutes + 2 seconds per move)."""

    base_seconds: int
    increment_seconds: int

    @property
    def estimated_seconds(self) -> int:
        """Lichess's estimated game duration: base + 40 moves' worth of increment."""
        return self.base_seconds + 40 * self.increment_seconds

    @property
    def speed(self) -> str:
        """Lichess speed category: ultrabullet, bullet, blitz, rapid or classical."""
        return speed_from_seconds(self.estimated_seconds)


def speed_from_seconds(estimated_seconds: int) -> str:
    """Map an estimated game duration (base + 40 * increment) to a Lichess speed category."""
    for bound, speed in _SPEED_BOUNDS:
        if estimated_seconds < bound:
            return speed
    return "classical"


def parse_time_control(value: str) -> TimeControl | None:
    """Parse a PGN ``TimeControl`` value like ``"300+0"``.

    Returns ``None`` for correspondence games (``"-"``) and for anything malformed.
    """
    base, sep, increment = value.strip().partition("+")
    if not sep or not base.isdigit() or not increment.isdigit():
        return None
    return TimeControl(int(base), int(increment))


def speed_of(time_control: str) -> str | None:
    """Speed category for a raw ``TimeControl`` header value.

    ``"-"`` gives ``"correspondence"``; a malformed value gives ``None``.
    """
    if time_control.strip() == "-":
        return CORRESPONDENCE
    parsed = parse_time_control(time_control)
    return parsed.speed if parsed else None


# --------------------------------------------------------------------------- splitting a stream into games

# A tag pair like  [WhiteElo "1825"]  -- values may contain escaped quotes (\") and backslashes (\\).
_HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$')
_UNESCAPE_RE = re.compile(r"\\(.)")
_BOM = chr(0xFEFF)


@dataclass(slots=True)
class RawGame:
    """One game exactly as read from the PGN: its tag pairs and its unparsed movetext."""

    headers: dict[str, str] = field(default_factory=dict)
    movetext: str = ""

    def header(self, name: str, default: str = "") -> str:
        """Return a header value, or ``default`` if the tag is missing."""
        return self.headers.get(name, default)


def parse_header_line(line: str) -> tuple[str, str] | None:
    """Parse a tag-pair line into ``(name, value)``, or return ``None`` if it isn't one."""
    match = _HEADER_RE.match(line)
    if match is None:
        return None
    name, value = match.groups()
    if "\\" in value:
        value = _UNESCAPE_RE.sub(r"\1", value)
    return name, value


def iter_raw_games(lines: Iterable[str]) -> Iterator[RawGame]:
    """Split a stream of PGN lines into :class:`RawGame` objects, one game at a time.

    Works on any iterable of lines (an open text file, a decompressing stream, a
    list), so memory use stays constant no matter how big the input is. A new game
    starts at the first tag-pair line after movetext (or after a blank line that
    ends a header block). Text before the first header is ignored.
    """
    headers: dict[str, str] = {}
    movetext: list[str] = []
    after_headers = False  # saw the blank line (or movetext) that ends the header block

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith(_BOM):  # UTF-8 byte-order mark at the start of a file
            line = line[1:]
        if not line:
            if headers:
                after_headers = True
            continue

        tag = parse_header_line(line) if line[0] == "[" else None
        if tag is not None:
            if headers and after_headers:
                yield RawGame(headers, " ".join(movetext))
                headers, movetext = {}, []
            after_headers = False
            headers[tag[0]] = tag[1]
        elif headers:
            movetext.append(line)
            after_headers = True

    if headers:
        yield RawGame(headers, " ".join(movetext))


# --------------------------------------------------------------------------- movetext

# Tokens, in priority order: {comment}, ;line comment, variation brackets, $NAG,
# move numbers ("12." / "12..."), and anything else (SAN moves and the result).
_TOKEN_RE = re.compile(r"\{[^}]*\}|;[^\n]*|[()]|\$\d+|\d+\.+|[^\s{}();$]+")
_CLOCK_RE = re.compile(r"\[%clk\s+(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\]")
RESULTS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


@dataclass(frozen=True, slots=True)
class Movetext:
    """The mainline of a game.

    Attributes:
        moves: SAN moves in order (annotation glyphs like ``?!`` removed), e.g. ``["e4", "e5", "Nf3"]``.
        clocks: Seconds left on the mover's clock after each move (from ``[%clk]`` comments).
            Either one entry per move, or empty if any move lacks a clock.
        result: The game termination marker (``"1-0"``, ``"0-1"``, ``"1/2-1/2"``, ``"*"``) or ``None``.
    """

    moves: list[str]
    clocks: list[float]
    result: str | None

    @property
    def num_plies(self) -> int:
        """Number of half-moves in the mainline."""
        return len(self.moves)


def parse_clock(comment: str) -> float | None:
    """Extract the ``[%clk H:MM:SS(.f)]`` value from a comment, in seconds."""
    match = _CLOCK_RE.search(comment)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_movetext(text: str) -> Movetext:
    """Parse PGN movetext into mainline SAN moves, per-move clocks and the result.

    Variations in parentheses, NAGs and move numbers are skipped. Moves are not
    checked for legality here (that needs a board replay, done when building the
    dataset in step 4); this keeps streaming fast.
    """
    moves: list[str] = []
    clocks: list[float | None] = []
    result: str | None = None
    depth = 0  # variation nesting level; only depth 0 is the mainline

    for match in _TOKEN_RE.finditer(text):
        token = match.group()
        first = token[0]
        if first == "(":
            depth += 1
        elif first == ")":
            depth = max(depth - 1, 0)
        elif depth:
            continue
        elif first == "{":
            if moves and clocks[-1] is None:
                clocks[-1] = parse_clock(token)
        elif first in ";$" or (first.isdigit() and token.endswith(".")):
            continue
        elif token in RESULTS:
            result = token
        else:
            moves.append(token.rstrip("!?"))
            clocks.append(None)

    complete = [c for c in clocks if c is not None]
    return Movetext(moves, complete if len(complete) == len(moves) else [], result)


# --------------------------------------------------------------------------- filtering


class Reject(StrEnum):
    """Why a game was filtered out. Values are the keys used in the stats report."""

    NOT_RATED = "not_rated"
    VARIANT = "variant"
    SPEED = "speed"
    TERMINATION = "termination"
    TITLE = "title"
    RATING_MISSING = "rating_missing"
    RATING_TOO_LOW = "rating_too_low"
    RATING_TOO_HIGH = "rating_too_high"
    TOO_SHORT = "too_short"
    NO_CLOCKS = "no_clocks"


@dataclass(frozen=True, slots=True)
class GameFilter:
    """Game-level filters. Build it from config with :meth:`from_config`.

    Header checks run first because they are cheap; the movetext is only parsed
    for games that pass them. Each check returns the first reason that applies.
    """

    speeds: frozenset[str]
    rated_only: bool = True
    min_rating: int = 0
    max_rating: int | None = None
    min_plies: int = 0
    exclude_terminations: frozenset[str] = frozenset()
    exclude_titles: frozenset[str] = frozenset()
    require_clocks: bool = False

    @classmethod
    def from_config(cls, data: DataConfig) -> GameFilter:
        """Create a filter from the ``data`` section of the project config."""
        f = data.filters
        return cls(
            speeds=frozenset(data.time_control.speeds),
            rated_only=f.rated_only,
            min_rating=f.min_rating,
            max_rating=f.max_rating,
            min_plies=f.min_plies,
            exclude_terminations=frozenset(f.exclude_terminations),
            exclude_titles=frozenset(f.exclude_titles),
            require_clocks=f.require_clocks,
        )

    def check_headers(self, headers: Mapping[str, str]) -> Reject | None:
        """Return why the game should be rejected based on its headers, or ``None`` to keep it."""
        if self.rated_only and not is_rated(headers):
            return Reject.NOT_RATED
        if headers.get("Variant", "Standard") != "Standard":
            return Reject.VARIANT
        if speed_of(headers.get("TimeControl", "")) not in self.speeds:
            return Reject.SPEED
        if headers.get("Termination", "") in self.exclude_terminations:
            return Reject.TERMINATION
        if self.exclude_titles and (
            headers.get("WhiteTitle") in self.exclude_titles or headers.get("BlackTitle") in self.exclude_titles
        ):
            return Reject.TITLE

        white_elo, black_elo = _to_int(headers.get("WhiteElo")), _to_int(headers.get("BlackElo"))
        if white_elo is None or black_elo is None:
            return Reject.RATING_MISSING
        if min(white_elo, black_elo) < self.min_rating:
            return Reject.RATING_TOO_LOW
        if self.max_rating is not None and max(white_elo, black_elo) > self.max_rating:
            return Reject.RATING_TOO_HIGH
        return None

    def check_movetext(self, movetext: Movetext) -> Reject | None:
        """Return why the game should be rejected based on its moves, or ``None`` to keep it."""
        if movetext.num_plies < self.min_plies:
            return Reject.TOO_SHORT
        if self.require_clocks and not movetext.clocks:
            return Reject.NO_CLOCKS
        return None


def is_rated(headers: Mapping[str, str]) -> bool:
    """True if the game was rated.

    Arena and casual games say so in the event name ("Rated Blitz game", "Casual
    Rapid game"), but Swiss events do not ("Blitz swiss https://..."). Rated games
    carry rating changes, so a ``WhiteRatingDiff``/``BlackRatingDiff`` header
    also counts as rated.
    """
    event = headers.get("Event", "")
    if event.startswith("Rated"):
        return True
    if event.startswith("Casual"):
        return False
    return "WhiteRatingDiff" in headers or "BlackRatingDiff" in headers


def _to_int(value: str | None) -> int | None:
    """Parse a header like ``"1825"``; ``None`` for missing or ``"?"`` values."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------- accepted games


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A game that passed every filter, in the flat form that gets saved to Parquet."""

    game_id: str
    event: str
    utc_date: str          # "YYYY-MM-DD"
    utc_time: str          # "HH:MM:SS"
    white: str
    black: str
    white_elo: int
    black_elo: int
    white_title: str       # "" when the player has no title
    black_title: str
    result: str            # "1-0", "0-1", "1/2-1/2"
    termination: str       # "Normal", "Time forfeit", ...
    base_seconds: int
    increment_seconds: int
    speed: str
    eco: str
    opening: str
    moves: list[str]       # SAN
    clocks: list[float]    # seconds left after each move (empty if the game has no clocks)

    @property
    def num_plies(self) -> int:
        """Number of half-moves."""
        return len(self.moves)

    @classmethod
    def from_parts(cls, headers: Mapping[str, str], movetext: Movetext) -> GameRecord:
        """Build a record from a game's headers and parsed movetext."""
        tc = parse_time_control(headers.get("TimeControl", "")) or TimeControl(0, 0)
        return cls(
            game_id=game_id_from_site(headers.get("Site", "")),
            event=headers.get("Event", ""),
            utc_date=headers.get("UTCDate", headers.get("Date", "")).replace(".", "-"),
            utc_time=headers.get("UTCTime", ""),
            white=headers.get("White", ""),
            black=headers.get("Black", ""),
            white_elo=_to_int(headers.get("WhiteElo")) or 0,
            black_elo=_to_int(headers.get("BlackElo")) or 0,
            white_title=headers.get("WhiteTitle", ""),
            black_title=headers.get("BlackTitle", ""),
            result=headers.get("Result", movetext.result or "*"),
            termination=headers.get("Termination", ""),
            base_seconds=tc.base_seconds,
            increment_seconds=tc.increment_seconds,
            speed=tc.speed,
            eco=headers.get("ECO", ""),
            opening=headers.get("Opening", ""),
            moves=movetext.moves,
            clocks=movetext.clocks,
        )

    def to_dict(self) -> dict[str, Any]:
        """All fields as a plain dict (lists are shared, not copied)."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def game_id_from_site(site: str) -> str:
    """Lichess game id from a ``Site`` header.

    Handles both ``https://lichess.org/abcd1234`` and ``https://lichess.org/game/abcd1234``.
    """
    return site.rstrip("/").rsplit("/", 1)[-1]


@dataclass
class FilterStats:
    """Counts of games read, accepted, and rejected (by reason)."""

    games_read: int = 0
    games_accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    @property
    def games_rejected(self) -> int:
        """Total number of rejected games."""
        return sum(self.rejected.values())

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary; rejection reasons are sorted by count."""
        return {
            "games_read": self.games_read,
            "games_accepted": self.games_accepted,
            "games_rejected": self.games_rejected,
            "acceptance_rate": round(self.games_accepted / self.games_read, 4) if self.games_read else 0.0,
            "rejected_by_reason": dict(self.rejected.most_common()),
        }


def filter_games(
    raw_games: Iterable[RawGame],
    game_filter: GameFilter,
    stats: FilterStats | None = None,
    max_games: int | None = None,
) -> Iterator[GameRecord]:
    """Yield a :class:`GameRecord` for each game that passes ``game_filter``.

    Args:
        raw_games: Games from :func:`iter_raw_games`.
        game_filter: The filters to apply.
        stats: Optional counter object, updated in place as games are read.
        max_games: Stop after this many *accepted* games (sample mode). ``None`` = no limit.
    """
    if max_games is not None and max_games <= 0:
        return
    stats = stats if stats is not None else FilterStats()
    for raw in raw_games:
        stats.games_read += 1
        reason = game_filter.check_headers(raw.headers)
        if reason is None:
            movetext = parse_movetext(raw.movetext)
            reason = game_filter.check_movetext(movetext)
        if reason is not None:
            stats.rejected[reason.value] += 1
            continue
        stats.games_accepted += 1
        yield GameRecord.from_parts(raw.headers, movetext)
        if max_games is not None and stats.games_accepted >= max_games:
            return
