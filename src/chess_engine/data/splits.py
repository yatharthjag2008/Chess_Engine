"""Assign each game to train / val / test, deterministically and by game (never by position).

Every position of a game lands in the same split, so the model is never evaluated
on a position from a game it saw during training (positions from one game are
strongly correlated: same players, same opening, same structure).

The split is a pure function of ``(seed, game_id)``: a hash of the two is mapped
to a number in ``[0, 1)`` and compared with the configured fractions. So

* re-running the dataset build always gives the same split,
* adding more games (another month, a bigger sample) never moves an existing game
  to a different split, and
* later steps (e.g. rating estimation on held-out games, step 12) can find the
  test games again from their ids alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

from chess_engine.config import SplitConfig

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "val", "test")

_HASH_BYTES = 8
_HASH_SCALE = float(2 ** (8 * _HASH_BYTES))


def split_value(game_id: str, seed: int) -> float:
    """A uniform-looking number in ``[0, 1)`` that depends only on ``seed`` and ``game_id``."""
    digest = hashlib.blake2b(f"{seed}:{game_id}".encode(), digest_size=_HASH_BYTES).digest()
    return int.from_bytes(digest, "big") / _HASH_SCALE


def assign_split(game_id: str, split: SplitConfig) -> SplitName:
    """Which split ``game_id`` belongs to: ``train`` if its value < train, then ``val``, then ``test``."""
    value = split_value(game_id, split.seed)
    if value < split.train:
        return "train"
    if value < split.train + split.val:
        return "val"
    return "test"


def assign_splits(game_ids: Iterable[str], split: SplitConfig) -> list[SplitName]:
    """:func:`assign_split` for many games, in order."""
    return [assign_split(game_id, split) for game_id in game_ids]
