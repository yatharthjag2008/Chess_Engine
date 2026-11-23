"""Replay a game and turn every position into a training sample (step 4).

One sample = the position before a move + the move a human played there, plus
who played it and under what conditions:

==============  ========  =====================================================
column          dtype     content
==============  ========  =====================================================
pieces          uint64x12 board, compact form (:mod:`chess_engine.encoding.compact`)
flags           uint8     side to move + castling rights (compact form)
ep              int8      legal en passant square, mover-oriented, or -1
halfmove        uint8     halfmove clock (capped at 255)
move            int16     policy index of the move played (the training target)
legal_count     uint8     how many legal moves the position has (see ``legal_moves``)
rating          int16     the mover's rating
bucket          int8      the mover's rating bucket (``cfg.ratings``)
opp_rating      int16     the opponent's rating
clock           float32   the mover's remaining time before the move, seconds
opp_clock       float32   the opponent's remaining time, seconds
time_spent      float32   seconds the mover spent on this move (from the clocks; 0 on move 1)
base_seconds    int16     time control, e.g. 180 for 3+2
increment       int16     time control increment, e.g. 2 for 3+2
ply             int16     0 = White's first move
result          int8      game result for the mover: +1 win, 0 draw, -1 loss
game            int32     row of the game in the dataset's ``games.parquet``
==============  ========  =====================================================

The legal moves of all samples are stored back to back in one ``legal_moves``
int16 array (``legal_count`` entries per sample, sorted), which is about 3x
smaller than a 1858-entry mask per position. :meth:`Samples.legal_mask` rebuilds
the masks for a batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import chess
import numpy as np

from chess_engine.config import DatasetConfig, RatingsConfig
from chess_engine.encoding.compact import compact_board
from chess_engine.encoding.moves import NUM_MOVES, legal_move_indices, move_to_index

#: (dtype, per-sample shape) of every fixed-size column, in storage order.
SAMPLE_COLUMNS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "pieces": (np.dtype(np.uint64), (12,)),
    "flags": (np.dtype(np.uint8), ()),
    "ep": (np.dtype(np.int8), ()),
    "halfmove": (np.dtype(np.uint8), ()),
    "move": (np.dtype(np.int16), ()),
    "legal_count": (np.dtype(np.uint8), ()),
    "rating": (np.dtype(np.int16), ()),
    "bucket": (np.dtype(np.int8), ()),
    "opp_rating": (np.dtype(np.int16), ()),
    "clock": (np.dtype(np.float32), ()),
    "opp_clock": (np.dtype(np.float32), ()),
    "time_spent": (np.dtype(np.float32), ()),
    "base_seconds": (np.dtype(np.int16), ()),
    "increment": (np.dtype(np.int16), ()),
    "ply": (np.dtype(np.int16), ()),
    "result": (np.dtype(np.int8), ()),
    "game": (np.dtype(np.int32), ()),
}
LEGAL_MOVES_DTYPE = np.dtype(np.int16)

_RESULT_FOR_WHITE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


class GameRejected(Exception):
    """A game that cannot be turned into samples; ``reason`` is a short machine-readable key."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


# --------------------------------------------------------------------------- the sample container


class Samples:
    """A columnar batch of samples: one numpy array per column plus the ragged ``legal_moves``."""

    def __init__(self, columns: Mapping[str, np.ndarray], legal_moves: np.ndarray) -> None:
        missing = set(SAMPLE_COLUMNS) - set(columns)
        extra = set(columns) - set(SAMPLE_COLUMNS)
        if missing or extra:
            raise ValueError(f"sample columns mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}")
        n = len(columns["move"])
        self.columns: dict[str, np.ndarray] = {}
        for name, (dtype, shape) in SAMPLE_COLUMNS.items():
            array = np.asarray(columns[name], dtype=dtype)
            if array.shape != (n, *shape):
                raise ValueError(f"column {name!r} has shape {array.shape}, expected {(n, *shape)}")
            self.columns[name] = array
        self.legal_moves = np.asarray(legal_moves, dtype=LEGAL_MOVES_DTYPE)
        if len(self.legal_moves) != int(self.columns["legal_count"].sum(dtype=np.int64)):
            raise ValueError("legal_moves length does not match the sum of legal_count")

    def __len__(self) -> int:
        return len(self.columns["move"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @classmethod
    def empty(cls) -> Samples:
        """No samples."""
        columns = {name: np.empty((0, *shape), dtype=dtype) for name, (dtype, shape) in SAMPLE_COLUMNS.items()}
        return cls(columns, np.empty(0, dtype=LEGAL_MOVES_DTYPE))

    @classmethod
    def concat(cls, parts: Sequence[Samples]) -> Samples:
        """All samples of ``parts``, in order."""
        if not parts:
            return cls.empty()
        if len(parts) == 1:
            return parts[0]
        columns = {name: np.concatenate([p.columns[name] for p in parts]) for name in SAMPLE_COLUMNS}
        return cls(columns, np.concatenate([p.legal_moves for p in parts]))

    def legal_offsets(self) -> np.ndarray:
        """``offsets[i]:offsets[i + 1]`` is sample ``i``'s slice of ``legal_moves`` (length N + 1)."""
        offsets = np.zeros(len(self) + 1, dtype=np.int64)
        np.cumsum(self.columns["legal_count"], out=offsets[1:])
        return offsets

    def take(self, indices: np.ndarray | Sequence[int]) -> Samples:
        """The samples at ``indices`` (any order, repeats allowed), legal moves included."""
        indices = np.asarray(indices, dtype=np.int64)
        columns = {name: array[indices] for name, array in self.columns.items()}
        return Samples(columns, self.legal_moves[_ragged_positions(self.legal_offsets(), indices)])

    def legal_moves_of(self, index: int) -> np.ndarray:
        """Sorted policy indices of the legal moves of sample ``index``."""
        offsets = self.legal_offsets()
        return self.legal_moves[offsets[index]:offsets[index + 1]]

    def legal_mask(self, indices: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        """Boolean ``(len(indices), NUM_MOVES)`` legal-move masks (all samples if ``indices`` is None)."""
        offsets = self.legal_offsets()
        indices = np.arange(len(self)) if indices is None else np.asarray(indices, dtype=np.int64)
        counts = self.columns["legal_count"][indices].astype(np.int64)
        rows = np.repeat(np.arange(len(indices)), counts)
        mask = np.zeros((len(indices), NUM_MOVES), dtype=bool)
        mask[rows, self.legal_moves[_ragged_positions(offsets, indices)]] = True
        return mask


def _ragged_positions(offsets: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Flat positions in a ragged array of the rows ``indices`` (rows given by ``offsets``), in order."""
    counts = offsets[indices + 1] - offsets[indices]
    out_starts = np.cumsum(counts) - counts
    return np.repeat(offsets[indices] - out_starts, counts) + np.arange(int(counts.sum()))


# --------------------------------------------------------------------------- game -> samples


@dataclass
class SampleStats:
    """Counts from turning games into samples (summed over workers)."""

    games_in: int = 0
    games_used: int = 0
    positions: int = 0
    skipped_opening: int = 0
    skipped_low_clock: int = 0
    games_rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.games_rejected[reason] = self.games_rejected.get(reason, 0) + 1

    def merge(self, other: SampleStats) -> None:
        """Add ``other``'s counts to these."""
        for f in fields(self):
            if f.name == "games_rejected":
                for reason, count in other.games_rejected.items():
                    self.games_rejected[reason] = self.games_rejected.get(reason, 0) + count
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def clocks_before_moves(clocks: Sequence[float], base_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """(mover's clock, opponent's clock) before every ply, from Lichess's after-move clocks.

    ``clocks[i]`` is the mover's time left *after* ply ``i`` (as in ``[%clk]``), so
    before ply ``i`` the mover has ``clocks[i - 2]`` and the opponent ``clocks[i - 1]``;
    both start with the full base time.
    """
    after = np.asarray(clocks, dtype=np.float32)
    padded = np.concatenate([np.full(2, base_seconds, dtype=np.float32), after])
    return padded[:-2], padded[1:-1]


def game_to_samples(
    game: Mapping[str, Any],
    game_index: int,
    ratings: RatingsConfig,
    dataset: DatasetConfig,
    stats: SampleStats | None = None,
) -> Samples:
    """Replay ``game`` (a row of the step-2 games table) into one sample per kept position.

    Raises:
        GameRejected: the game has an illegal or unparsable move (``illegal_move``),
            clocks that do not line up with the moves (``bad_clocks``), a rating
            below the lowest bucket (``rating_below_buckets``) or an unknown result.
    """
    moves = list(game["moves"])
    clocks = list(game["clocks"])
    if len(clocks) != len(moves):
        raise GameRejected("bad_clocks", f"{len(clocks)} clocks for {len(moves)} moves")
    elos = (int(game["white_elo"]), int(game["black_elo"]))
    if min(elos) < ratings.bucket_edges[0]:
        raise GameRejected("rating_below_buckets", f"ratings {elos} below {ratings.bucket_edges[0]}")
    if game["result"] not in _RESULT_FOR_WHITE:
        raise GameRejected("unknown_result", repr(game["result"]))
    buckets = (ratings.bucket_index(elos[0]), ratings.bucket_index(elos[1]))
    white_result = _RESULT_FOR_WHITE[game["result"]]
    base, increment = int(game["base_seconds"]), int(game["increment_seconds"])
    clock_before, opp_clock_before = clocks_before_moves(clocks, base)

    rows: dict[str, list[Any]] = {name: [] for name in SAMPLE_COLUMNS}
    legal_parts: list[np.ndarray] = []
    skipped_opening = skipped_clock = 0
    board = chess.Board()
    for ply, san in enumerate(moves):
        try:
            move = board.parse_san(san)
        except ValueError as exc:
            raise GameRejected("illegal_move", f"ply {ply} {san!r}: {exc}") from exc
        if ply < dataset.skip_first_plies:
            skipped_opening += 1
        elif clock_before[ply] < dataset.min_clock_seconds:
            skipped_clock += 1
        else:
            mover = 0 if board.turn == chess.WHITE else 1
            compact = compact_board(board)
            legal = legal_move_indices(board)
            rows["pieces"].append(compact.pieces)
            rows["flags"].append(compact.flags)
            rows["ep"].append(compact.ep)
            rows["halfmove"].append(compact.halfmove)
            rows["move"].append(move_to_index(move, board.turn))
            rows["legal_count"].append(len(legal))
            rows["rating"].append(elos[mover])
            rows["bucket"].append(buckets[mover])
            rows["opp_rating"].append(elos[1 - mover])
            rows["clock"].append(clock_before[ply])
            rows["opp_clock"].append(opp_clock_before[ply])
            # Lichess does not run the clock or add the increment on each side's first move.
            bonus = increment if ply >= 2 else 0
            rows["time_spent"].append(max(0.0, float(clock_before[ply]) + bonus - clocks[ply]))
            rows["base_seconds"].append(base)
            rows["increment"].append(increment)
            rows["ply"].append(ply)
            rows["result"].append(white_result if mover == 0 else -white_result)
            rows["game"].append(game_index)
            legal_parts.append(legal)
        board.push(move)

    if stats is not None:
        stats.skipped_opening += skipped_opening
        stats.skipped_low_clock += skipped_clock
    if not rows["move"]:
        return Samples.empty()
    columns = {name: np.array(values) for name, values in rows.items()}
    columns["pieces"] = np.stack(rows["pieces"])
    return Samples(columns, np.concatenate(legal_parts))


def games_to_samples(
    games: Sequence[Mapping[str, Any]],
    first_index: int,
    ratings: RatingsConfig,
    dataset: DatasetConfig,
) -> tuple[Samples, SampleStats, list[int]]:
    """Samples of many games (game ``i`` gets index ``first_index + i``).

    Rejected games are counted and skipped. Returns the samples, the stats and the
    number of samples of each game (0 for rejected games), in order.
    """
    stats = SampleStats()
    parts: list[Samples] = []
    per_game: list[int] = []
    for offset, game in enumerate(games):
        stats.games_in += 1
        try:
            samples = game_to_samples(game, first_index + offset, ratings, dataset, stats)
        except GameRejected as exc:
            stats.reject(exc.reason)
            per_game.append(0)
            continue
        stats.games_used += 1
        stats.positions += len(samples)
        per_game.append(len(samples))
        parts.append(samples)
    return Samples.concat(parts), stats, per_game
