"""Stream Lichess PGN games and filter them by time control, rating and length.

The Lichess monthly dumps are ~30 GB compressed, so nothing here loads a whole
file: games are read one at a time from a text stream, filtered on their headers
first (cheap), and only then is the movetext parsed.
"""

from __future__ import annotations

from dataclasses import dataclass

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
