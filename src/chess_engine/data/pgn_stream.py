"""Stream Lichess PGN games and filter them by time control, rating and length.

The Lichess monthly dumps are ~30 GB compressed, so nothing here loads a whole
file: games are read one at a time from a text stream, filtered on their headers
first (cheap), and only then is the movetext parsed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

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
