"""Review a game with Stockfish: every move evaluated, classified and scored.

    with StockfishAnalyzer(stockfish_path, depth=14, multipv=3) as engine:
        review = review_pgn(pgn_text, engine, cfg.review)
    for m in review.moves:
        m.ply, m.san, m.classification, m.win_lost, m.best_san

How a move is judged: the position before it is analysed (the engine's best move
and evaluation), and so is the position after it. The mover's win chance before
the move (best play) against after it (the move played) gives the win % lost,
the move's accuracy and its class (:mod:`chess_engine.review.classify`).
Centipawn loss is the same comparison in centipawns, each side capped at 1000.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import Any

import chess
import chess.pgn

from chess_engine.config import ReviewConfig
from chess_engine.review.classify import ORDER, MoveClass, classify
from chess_engine.review.engine import Analyzer
from chess_engine.review.scoring import CP_CAP, game_accuracy, move_accuracy, win_percent


@dataclass
class Alternative:
    san: str
    uci: str
    cp: int          # mover's view, after the move


@dataclass
class MoveReview:
    """One move of the game, judged."""

    ply: int
    color: str                       # "white" / "black"
    san: str
    uci: str
    fen_before: str
    eval_before: int                 # White's view: the position before the move, best play
    eval_after: int                  # White's view: after the move played
    best_san: str | None
    best_uci: str | None
    best_cp: int                     # mover's view: what the best move keeps
    played_cp: int                   # mover's view: what the move played keeps
    cp_loss: int
    win_before: float                # mover's win % with best play
    win_after: float                 # mover's win % after the move played
    win_lost: float
    accuracy: float
    classification: MoveClass
    alternatives: list[Alternative] = field(default_factory=list)

    @property
    def move_number(self) -> str:
        number = self.ply // 2 + 1
        return f"{number}." if self.color == "white" else f"{number}..."


@dataclass
class GameReview:
    headers: dict[str, str]
    start_fen: str
    moves: list[MoveReview]
    depth: int | None = None

    def for_color(self, color: str) -> list[MoveReview]:
        return [m for m in self.moves if m.color == color]

    def white_win_curve(self) -> list[float]:
        """White's win % before every move (best play) and after the last one."""
        if not self.moves:
            return []
        curve = [win_percent(m.eval_before) for m in self.moves]
        curve.append(win_percent(self.moves[-1].eval_after))
        return curve

    def summary(self) -> dict[str, SideSummary]:
        """Accuracy, average centipawn loss and move classes for each side."""
        curve = self.white_win_curve()
        out = {}
        for color in ("white", "black"):
            moves = self.for_color(color)
            counts = {c.value: sum(m.classification is c for m in moves) for c in ORDER}
            accuracy = game_accuracy([m.accuracy for m in moves], curve, [m.ply for m in moves]) if moves else None
            acpl = sum(m.cp_loss for m in moves) / len(moves) if moves else None
            out[color] = SideSummary(color, len(moves), accuracy, acpl, counts)
        return out


@dataclass
class SideSummary:
    """One player's numbers for a reviewed game."""

    color: str
    moves: int
    accuracy: float | None             # 0-100 (Lichess's method)
    average_cp_loss: float | None
    counts: dict[str, int]             # moves per class: best, good, inaccuracy, mistake, blunder

    def describe(self) -> str:
        if not self.moves:
            return f"{self.color}: no moves"
        errors = ", ".join(f"{self.counts[k]} {k}{'s' if self.counts[k] != 1 else ''}"
                           for k in ("inaccuracy", "mistake", "blunder") if self.counts[k])
        return (f"{self.color.capitalize()}: accuracy {self.accuracy:.0f}%, average loss {self.average_cp_loss:.0f} cp"
                + (f", {errors}" if errors else ", no errors"))


def _cap(cp: int) -> int:
    return max(-CP_CAP, min(CP_CAP, cp))


def review_moves(start: chess.Board, moves: list[chess.Move], analyzer: Analyzer, cfg: ReviewConfig,
                 headers: dict[str, str] | None = None, depth: int | None = None) -> GameReview:
    """Review ``moves`` played from ``start`` (see the module docstring)."""
    board = start.copy(stack=False)
    start_fen = board.fen()
    boards = [board.copy(stack=False)]
    for move in moves:
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
        board.push(move)
        boards.append(board.copy(stack=False))
    analyses = [analyzer.analyse(b) for b in boards]

    reviews = []
    for ply, move in enumerate(moves):
        before, after = boards[ply], analyses[ply + 1]
        mover = before.turn
        analysis = analyses[ply]
        line = analysis.line_for(move)
        best_cp = analysis.best_cp
        played_cp = line.cp if line is not None else -after.best_cp
        best = analysis.best
        is_engine_move = best is not None and best.move == move
        if is_engine_move:
            played_cp = best_cp
        win_before, win_after = win_percent(best_cp), win_percent(played_cp)
        lost = win_before - win_after
        sign = 1 if mover == chess.WHITE else -1
        reviews.append(MoveReview(
            ply=ply, color="white" if mover == chess.WHITE else "black", san=before.san(move), uci=move.uci(),
            fen_before=before.fen(), eval_before=sign * best_cp, eval_after=sign * played_cp,
            best_san=best.san if best else None, best_uci=best.move.uci() if best else None,
            best_cp=best_cp, played_cp=played_cp, cp_loss=max(0, _cap(best_cp) - _cap(played_cp)),
            win_before=win_before, win_after=win_after, win_lost=lost,
            accuracy=move_accuracy(win_before, win_after),
            classification=classify(lost, is_engine_move, cfg),
            alternatives=[Alternative(alt.san, alt.move.uci(), alt.cp) for alt in analysis.lines],
        ))
    return GameReview(dict(headers or {}), start_fen, reviews, depth)


def review_game(game: chess.pgn.Game, analyzer: Analyzer, cfg: ReviewConfig, depth: int | None = None) -> GameReview:
    return review_moves(game.board(), list(game.mainline_moves()), analyzer, cfg, dict(game.headers), depth)


def review_pgn(text: str, analyzer: Analyzer, cfg: ReviewConfig, game_index: int = 0,
               depth: int | None = None) -> GameReview:
    """Review game number ``game_index`` (0-based) of a PGN text."""
    stream = io.StringIO(text)
    for _ in range(game_index):
        if chess.pgn.read_game(stream) is None:
            break
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or next(iter(game.mainline_moves()), None) is None:
        raise ValueError(f"no reviewable game number {game_index + 1} in the PGN")
    return review_game(game, analyzer, cfg, depth)


# --------------------------------------------------------------------------- key moments


@dataclass
class KeyMoment:
    """A turning point: a move that gave away a lot of the mover's winning chances."""

    ply: int
    move_number: str
    color: str
    san: str
    classification: MoveClass
    win_before: float
    win_after: float
    best_san: str | None
    text: str


def _side(color: str) -> str:
    return "White" if color == "white" else "Black"


def describe_moment(m: MoveReview) -> str:
    """A one-line explanation, e.g. ``3... Nf6?? dropped Black's chances from 48% to 0%. Better was g6.``"""
    text = (f"{m.move_number} {m.san}{m.classification.symbol} dropped {_side(m.color)}'s chances from "
            f"{m.win_before:.0f}% to {m.win_after:.0f}%.")
    if m.best_san and m.best_san != m.san:
        text += f" Better was {m.best_san}."
    return text


def key_moments(review: GameReview, count: int = 5) -> list[KeyMoment]:
    """The ``count`` moves that lost the most win %, errors (inaccuracy or worse) only, in game order."""
    errors = [m for m in review.moves if m.classification.is_error]
    worst = sorted(errors, key=lambda m: -m.win_lost)[:count]
    return [KeyMoment(m.ply, m.move_number, m.color, m.san, m.classification, m.win_before, m.win_after,
                      m.best_san, describe_moment(m)) for m in sorted(worst, key=lambda m: m.ply)]


# --------------------------------------------------------------------------- graph and JSON


def eval_graph(review: GameReview) -> list[dict[str, float]]:
    """Per ply (0 = the start position): White's win % and evaluation in centipawns (capped at +-1000)."""
    if not review.moves:
        return []
    points = [{"ply": m.ply, "white_win": win_percent(m.eval_before), "eval": float(_cap(m.eval_before))}
              for m in review.moves]
    last = review.moves[-1]
    points.append({"ply": last.ply + 1, "white_win": win_percent(last.eval_after), "eval": float(_cap(last.eval_after))})
    return points


def review_to_dict(review: GameReview, key_moment_count: int = 5) -> dict[str, Any]:
    """Everything about a review as plain JSON-ready data (moves, summary, key moments, graph)."""
    return {
        "headers": review.headers,
        "start_fen": review.start_fen,
        "depth": review.depth,
        "moves": [{**asdict(m), "classification": m.classification.value, "move_number": m.move_number}
                  for m in review.moves],
        "summary": {color: asdict(side) for color, side in review.summary().items()},
        "key_moments": [{**asdict(k), "classification": k.classification.value}
                        for k in key_moments(review, key_moment_count)],
        "graph": eval_graph(review),
    }


def review_from_dict(data: dict[str, Any]) -> GameReview:
    """Rebuild a :class:`GameReview` from :func:`review_to_dict`'s output."""
    moves = []
    for m in data["moves"]:
        fields = {k: v for k, v in m.items() if k != "move_number"}
        fields["classification"] = MoveClass(fields["classification"])
        fields["alternatives"] = [Alternative(**a) for a in fields.get("alternatives", [])]
        moves.append(MoveReview(**fields))
    return GameReview(dict(data["headers"]), data["start_fen"], moves, data.get("depth"))
