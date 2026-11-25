"""A chess clock with base time and increment, as on Lichess.

    clock = ChessClock(TimeControl.parse("3+2"))
    clock.press(chess.WHITE)        # White has moved: charge White's thinking time, add the increment
    clock.remaining(chess.BLACK)    # Black's time, counting down live
    clock.flagged()                 # the side whose time ran out, or None

Like Lichess, the clocks only start once both sides have made their first move,
and the increment is added after every move. The time source is injectable, so
tests (and replays) do not have to wait.

The clock also feeds the model: the bot is told how much time both sides have
left, so it plays like a human with that much time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class TimeControl:
    """Base time and increment in seconds."""

    base_seconds: int
    increment: int = 0

    def __post_init__(self) -> None:
        if self.base_seconds <= 0 or self.increment < 0:
            raise ValueError("base time must be positive and the increment non-negative")

    @classmethod
    def parse(cls, text: str) -> TimeControl:
        """``"3+2"`` -> 3 minutes + 2 seconds; ``"180+2"`` also works (values over 60 are seconds)."""
        try:
            base_text, _, inc_text = text.strip().partition("+")
            base = float(base_text)
            increment = int(inc_text) if inc_text else 0
        except ValueError:
            raise ValueError(f"time control must look like 3+2 (minutes + seconds), got {text!r}") from None
        return cls(int(base * 60) if base <= 60 else int(base), increment)

    def __str__(self) -> str:
        minutes = self.base_seconds / 60
        return f"{minutes:g}+{self.increment}"


def format_seconds(seconds: float) -> str:
    """``m:ss`` (and tenths under 10 seconds), never negative."""
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"0:{seconds:04.1f}"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


class ChessClock:
    """Two countdown clocks; the side to move's clock runs once both sides have moved."""

    def __init__(self, control: TimeControl, now: Callable[[], float] = time.monotonic) -> None:
        self.control = control
        self._now = now
        self._left = {chess.WHITE: float(control.base_seconds), chess.BLACK: float(control.base_seconds)}
        self._moves = 0
        self._running: chess.Color | None = None
        self._since = 0.0

    @property
    def running(self) -> chess.Color | None:
        """The side whose clock is running (None before both sides have moved, or after :meth:`stop`)."""
        return self._running

    def remaining(self, color: chess.Color) -> float:
        """Seconds left for ``color`` right now (may be negative once flagged)."""
        left = self._left[color]
        if self._running == color:
            left -= self._now() - self._since
        return left

    def press(self, color: chess.Color) -> float:
        """``color`` has just moved: charge its thinking time, add the increment, start the opponent's clock.

        Returns the seconds ``color`` spent on the move (0 before the clocks start).
        """
        spent = 0.0
        if self._running is not None:
            if self._running != color:
                raise ValueError("pressed by the side whose clock is not running")
            spent = self._now() - self._since
            self._left[color] -= spent
        self._moves += 1
        if self._moves > 2 or self._running is not None:
            self._left[color] += self.control.increment
        if self._moves >= 2:
            self._running, self._since = (not color), self._now()
        return spent

    def flagged(self) -> chess.Color | None:
        """The side whose time has run out, if any."""
        for color in (chess.WHITE, chess.BLACK):
            if self.remaining(color) <= 0:
                return color
        return None

    def stop(self) -> None:
        """Freeze both clocks (game over)."""
        if self._running is not None:
            self._left[self._running] = self.remaining(self._running)
        self._running = None
