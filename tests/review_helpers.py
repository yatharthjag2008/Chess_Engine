"""A scripted analyser for review tests: evaluations chosen by the test, no engine needed."""

from __future__ import annotations

import chess

from chess_engine.review.engine import EngineLine, PositionAnalysis, finished_eval


class ScriptedAnalyzer:
    """Evaluates positions from a dict ``{fen_prefix: (best_uci, best_cp, {uci: cp, ...})}`` (side to move's view).

    Positions not in the script: best move = the first legal move (UCI order), evaluation 0.
    The key is the FEN's piece placement + side to move (move counters ignored).
    """

    def __init__(self, script: dict[str, tuple[str, int, dict[str, int]]] | None = None) -> None:
        self.script = script or {}
        self.calls = 0

    @staticmethod
    def key(board: chess.Board) -> str:
        return " ".join(board.fen().split()[:2])

    def analyse(self, board: chess.Board) -> PositionAnalysis:
        self.calls += 1
        done = finished_eval(board)
        if done is not None:
            return PositionAnalysis(board.fen(), [], done)
        best_uci, best_cp, others = self.script.get(self.key(board),
                                                    (min(m.uci() for m in board.legal_moves), 0, {}))
        lines = [EngineLine(chess.Move.from_uci(best_uci), board.san(chess.Move.from_uci(best_uci)), best_cp,
                            [best_uci])]
        for uci, cp in others.items():
            move = chess.Move.from_uci(uci)
            lines.append(EngineLine(move, board.san(move), cp, [uci]))
        lines.sort(key=lambda line: -line.cp)
        return PositionAnalysis(board.fen(), lines, lines[0].cp)


def position_key(*sans: str, fen: str | None = None) -> str:
    """The script key of the position after ``sans`` (from ``fen`` or the start)."""
    board = chess.Board(fen) if fen else chess.Board()
    for san in sans:
        board.push_san(san)
    return ScriptedAnalyzer.key(board)
