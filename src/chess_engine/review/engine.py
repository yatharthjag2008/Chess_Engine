"""Stockfish analysis of single positions for the game review.

    with StockfishAnalyzer(path, depth=14, multipv=3) as engine:
        analysis = engine.analyse(board)
        analysis.best.san, analysis.best_cp          # the engine's move and its evaluation (side to move)
        analysis.lines[1:]                           # the next-best moves

Evaluations are in centipawns from the side to move's point of view (mates as
``+-(MATE_CP - n)``, see :mod:`chess_engine.review.scoring`). Positions already
analysed are answered from a cache (repetitions, and the same game reviewed twice).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chess
import chess.engine

from chess_engine.review.scoring import MATE_CP, score_to_cp


@dataclass(frozen=True)
class EngineLine:
    """One candidate move with the evaluation after it (side to move's view) and its main line."""

    move: chess.Move
    san: str
    cp: int
    pv: list[str] = field(default_factory=list)   # UCI moves, starting with ``move``


@dataclass(frozen=True)
class PositionAnalysis:
    """The engine's view of one position: lines best first, and the evaluation of the best one."""

    fen: str
    lines: list[EngineLine]
    best_cp: int                                   # side to move's view; for finished games from the result

    @property
    def best(self) -> EngineLine | None:
        return self.lines[0] if self.lines else None

    def line_for(self, move: chess.Move) -> EngineLine | None:
        return next((line for line in self.lines if line.move == move), None)


class Analyzer(Protocol):
    def analyse(self, board: chess.Board) -> PositionAnalysis: ...


def finished_eval(board: chess.Board) -> int | None:
    """The evaluation of a finished game for the side to move (checkmated: -MATE_CP; draws 0), else None."""
    if board.is_checkmate():
        return -MATE_CP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return None


class StockfishAnalyzer:
    """Stockfish at a fixed ``depth`` with ``multipv`` lines; the engine starts on first use."""

    def __init__(self, command: str | Sequence[str], depth: int = 14, multipv: int = 3, threads: int = 1,
                 hash_mb: int = 128) -> None:
        self.command = command
        self.depth = depth
        self.multipv = multipv
        self.options = {"Threads": threads, "Hash": hash_mb}
        self._engine: chess.engine.SimpleEngine | None = None
        self._cache: dict[str, PositionAnalysis] = {}

    def _get(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            argv = [self.command] if isinstance(self.command, str) else list(self.command)
            self._engine = chess.engine.SimpleEngine.popen_uci(argv)
            supported = {k: v for k, v in self.options.items() if k in self._engine.options}
            if supported:
                self._engine.configure(supported)
        return self._engine

    def analyse(self, board: chess.Board) -> PositionAnalysis:
        key = board.fen()
        if key in self._cache:
            return self._cache[key]
        done = finished_eval(board)
        if done is not None or not any(board.legal_moves):
            result = PositionAnalysis(key, [], done if done is not None else 0)
        else:
            engine = self._get()
            multipv = self.multipv if "MultiPV" in engine.options else None  # engines without it give one line
            infos = engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=multipv)
            if isinstance(infos, dict):
                infos = [infos]
            lines = []
            for info in infos:
                pv = info.get("pv") or []
                score = info.get("score")
                if not pv or score is None or pv[0] not in board.legal_moves:
                    continue
                lines.append(EngineLine(pv[0], board.san(pv[0]), score_to_cp(score.pov(board.turn)),
                                        [m.uci() for m in pv[:8]]))
            lines.sort(key=lambda line: -line.cp)
            result = PositionAnalysis(key, lines, lines[0].cp if lines else 0)
        self._cache[key] = result
        return result

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishAnalyzer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
