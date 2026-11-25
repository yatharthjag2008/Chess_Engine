"""Classify each move by how much of the mover's win chance it gave away.

=============  ===================================================  ======
class          rule (win-% points lost, from Stockfish's evaluation)  symbol
=============  ===================================================  ======
best           the engine's move, or lost at most ``good_drop``     !
good           lost less than ``inaccuracy_drop``
inaccuracy     lost at least ``inaccuracy_drop``                    ?!
mistake        lost at least ``mistake_drop``                       ?
blunder        lost at least ``blunder_drop``                       ??
=============  ===================================================  ======

Because the measure is win chance rather than centipawns, dropping 300 cp in a
position that stays completely winning is not called a blunder, while dropping
from equal to clearly worse is (Lichess judges moves the same way).
"""

from __future__ import annotations

from enum import StrEnum

from chess_engine.config import ReviewConfig


class MoveClass(StrEnum):
    BEST = "best"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"

    @property
    def symbol(self) -> str:
        return {"best": "!", "inaccuracy": "?!", "mistake": "?", "blunder": "??"}.get(self.value, "")

    @property
    def nag(self) -> int | None:
        """The PGN annotation glyph (NAG) number: 6 ?!, 2 ?, 4 ??; none for best / good moves."""
        return {"inaccuracy": 6, "mistake": 2, "blunder": 4}.get(self.value)

    @property
    def is_error(self) -> bool:
        return self in (MoveClass.INACCURACY, MoveClass.MISTAKE, MoveClass.BLUNDER)


#: Classes from best to worst (for counting and display order).
ORDER: tuple[MoveClass, ...] = tuple(MoveClass)


def classify(win_lost: float, is_engine_move: bool, cfg: ReviewConfig) -> MoveClass:
    """The class of a move that lost ``win_lost`` win-% points (negative = gained)."""
    if is_engine_move or win_lost <= cfg.good_drop:
        return MoveClass.BEST
    if win_lost >= cfg.blunder_drop:
        return MoveClass.BLUNDER
    if win_lost >= cfg.mistake_drop:
        return MoveClass.MISTAKE
    if win_lost >= cfg.inaccuracy_drop:
        return MoveClass.INACCURACY
    return MoveClass.GOOD
