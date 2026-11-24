"""Stockfish baseline: how often is the engine's best move the move the human played?

A strong engine is a poor model of human play, and more so at low ratings,
which is the gap a human-like model has to show. Stockfish is slow compared
with a batched network, so it analyses a random sample of the test positions
(``evaluation.stockfish_positions``, at ``evaluation.stockfish_depth``), and the
model is compared with it on exactly the same sample.

Engine answers are cached in a small JSON file keyed by depth + FEN, so
re-running an evaluation does not re-analyse positions. If Stockfish is not
installed, the baseline is skipped and the report says so.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

from chess_engine.data.samples import Samples
from chess_engine.encoding.moves import move_to_index
from chess_engine.eval.baselines import BaselineResult, summarize_baseline
from chess_engine.eval.positions import board_of
from chess_engine.eval.predictions import PredictionTable


def sample_rows(num_positions: int, count: int, seed: int = 0) -> np.ndarray:
    """``count`` distinct row indices (sorted), or every row if there are fewer."""
    if count >= num_positions:
        return np.arange(num_positions)
    return np.sort(np.random.default_rng(seed).choice(num_positions, count, replace=False))


class MoveCache:
    """``{"depth:fen": uci}`` stored as JSON; written back by :meth:`save`."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.moves: dict[str, str] = {}
        if self.path is not None and self.path.is_file():
            self.moves = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(board: chess.Board, depth: int) -> str:
        return f"{depth}:{board.fen()}"

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.moves), encoding="utf-8")
        tmp.replace(self.path)


def engine_moves(
    boards: Sequence[chess.Board],
    command: str | Sequence[str],
    depth: int,
    threads: int = 1,
    hash_mb: int = 128,
    cache_path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[chess.Move]:
    """The engine's best move in each position (fixed depth), using and updating the cache."""
    cache = MoveCache(cache_path)
    todo = [i for i, board in enumerate(boards) if MoveCache.key(board, depth) not in cache.moves]
    if todo:
        argv = [command] if isinstance(command, str) else list(command)
        with chess.engine.SimpleEngine.popen_uci(argv) as engine:
            options = {name: value for name, value in (("Threads", threads), ("Hash", hash_mb))
                       if name in engine.options}
            if options:
                engine.configure(options)
            for done, i in enumerate(todo, 1):
                result = engine.play(boards[i], chess.engine.Limit(depth=depth))
                if result.move is None:
                    raise RuntimeError(f"engine returned no move for {boards[i].fen()}")
                cache.moves[MoveCache.key(boards[i], depth)] = result.move.uci()
                if progress is not None:
                    progress(done, len(todo))
                if done % 200 == 0:
                    cache.save()  # keep work if the run is interrupted
        cache.save()
    return [chess.Move.from_uci(cache.moves[MoveCache.key(board, depth)]) for board in boards]


def stockfish_baseline(
    table: PredictionTable,
    samples: Samples,
    rows: np.ndarray,
    command: str | Sequence[str],
    depth: int,
    labels: Sequence[str] | None = None,
    cache_path: str | Path | None = None,
    threads: int = 1,
    hash_mb: int = 128,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Engine top-move accuracy on ``rows`` of ``table``, next to the model's on the same rows.

    ``samples`` are the split's samples (``table`` row ``i`` is sample ``i``).
    """
    boards = [board_of(samples, int(i)) for i in rows]
    moves = engine_moves(boards, command, depth, threads, hash_mb, cache_path, progress)
    predicted = np.array([move_to_index(move, board.turn) for move, board in zip(moves, boards)])
    subset = table.take(rows)
    hits = (predicted == subset["target"].astype(np.int64)).astype(np.float64)
    summary = summarize_baseline(BaselineResult(f"Stockfish depth {depth}", hits, np.ones(len(rows), dtype=bool)),
                                 subset, labels)
    summary["depth"] = depth
    summary["rows"] = [int(i) for i in rows]
    summary["hits"] = hits.tolist()
    return summary
