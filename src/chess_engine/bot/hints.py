"""Tiered hints while playing the bot: nudge -> point -> show.

A hint is about the best move by Stockfish in the current position. It comes in three tiers:

* **nudge**: what kind of move to look for ("there is a way to win material"), from one-move board facts;
* **point**: which piece to move (its square is highlighted);
* **show**: the move itself.

How vague the nudge is depends on how often players at your rating find the move (the model's probability for it
at your rating): a move most players find gets the vaguest nudge, a rarely found one names the piece type too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from collections import OrderedDict
from typing import Any

import chess

from chess_engine.config import HintsConfig
from chess_engine.model.inference import MovePredictor
from chess_engine.review.scoring import win_percent

from chess_engine.review.engine import Analyzer
from chess_engine.review.features import (
    PIECE_VALUES,
    after,
    captured_piece,
    gives_mate,
    hanging_pieces,
    piece_name,
)


class Theme(str, Enum):
    """What a move does, by one-move board facts (most important first)."""

    MATE = "mate"
    CHECK = "check"
    WIN_MATERIAL = "win_material"     # takes a piece that was hanging (free, or worth more than the taker)
    CAPTURE = "capture"               # any other capture (a trade, or a recapture)
    PROMOTION = "promotion"
    SAVE_PIECE = "save_piece"         # the moved piece was hanging and no longer is
    DEFEND = "defend"                 # another of your hanging pieces is safe afterwards
    THREAT = "threat"                 # attacks an opponent's piece that was safe, leaving it hanging
    CASTLE = "castle"
    QUIET = "quiet"


def move_themes(board: chess.Board, move: chess.Move) -> list[Theme]:
    """The themes of ``move`` in ``board`` (the side to move's move), most important first; at least one."""
    themes: list[Theme] = []
    nxt = after(board, move)
    me, them = board.turn, not board.turn
    if gives_mate(board, move):
        themes.append(Theme.MATE)
    elif nxt.is_check():
        themes.append(Theme.CHECK)

    taken = captured_piece(board, move)
    if taken is not None:
        hanging_before = {h.square for h in hanging_pieces(board, them)}
        target = move.to_square if not board.is_en_passant(move) else chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square))
        if target in hanging_before:
            themes.append(Theme.WIN_MATERIAL)
        else:
            themes.append(Theme.CAPTURE)
    if move.promotion:
        themes.append(Theme.PROMOTION)

    mine_before = {h.square for h in hanging_pieces(board, me)}
    mine_after = {h.square for h in hanging_pieces(nxt, me)}
    if move.from_square in mine_before and move.to_square not in mine_after:
        themes.append(Theme.SAVE_PIECE)
    elif any(sq != move.from_square and sq not in mine_after for sq in mine_before) and not nxt.is_checkmate():
        themes.append(Theme.DEFEND)

    theirs_before = {h.square for h in hanging_pieces(board, them)}
    new_targets = [h for h in hanging_pieces(nxt, them)
                   if h.square not in theirs_before and h.square != move.to_square
                   and move.to_square in nxt.attackers(me, h.square)]
    if new_targets and Theme.MATE not in themes:
        themes.append(Theme.THREAT)
    if board.is_castling(move):
        themes.append(Theme.CASTLE)
    return themes or [Theme.QUIET]


def threatened_piece(board: chess.Board, move: chess.Move) -> chess.PieceType | None:
    """The most valuable opponent piece ``move`` newly leaves hanging (for "attack the queen"), or None."""
    nxt = after(board, move)
    them = not board.turn
    before = {h.square for h in hanging_pieces(board, them)}
    targets = [h for h in hanging_pieces(nxt, them) if h.square not in before
               and move.to_square in nxt.attackers(board.turn, h.square)]
    return max(targets, key=lambda h: PIECE_VALUES[h.piece_type]).piece_type if targets else None


# --------------------------------------------------------------------------- how vague


class Detail(IntEnum):
    """How much the nudge gives away (more for moves that players at your rating rarely find)."""

    VAGUE = 0       # usually found: only the kind of move ("there is a forcing move")
    THEME = 1       # the theme ("there is a way to win material")
    PIECE = 2       # rarely found: the theme and the piece type ("a knight move wins material")


def find_rate(predictor: MovePredictor, board: chess.Board, move: chess.Move, rating: int,
              opp_rating: int | None = None, **clock: Any) -> float:
    """How often players rated ``rating`` play ``move`` here: the model's probability for it."""
    prediction = predictor.predict(board, rating, opp_ratings=opp_rating, **clock)
    return prediction.as_dict().get(move.uci(), 0.0)


def detail_for(rate: float, cfg: HintsConfig) -> Detail:
    """The nudge's detail for a move found with probability ``rate`` at your rating."""
    if rate >= cfg.easy_find_rate:
        return Detail.VAGUE
    if rate < cfg.hard_find_rate:
        return Detail.PIECE
    return Detail.THEME


# --------------------------------------------------------------------------- the texts


class Tier(str, Enum):
    NUDGE = "nudge"     # what kind of move to look for
    POINT = "point"     # which piece
    SHOW = "show"       # the move


TIERS = [Tier.NUDGE, Tier.POINT, Tier.SHOW]

_FORCING = {Theme.MATE, Theme.CHECK, Theme.WIN_MATERIAL, Theme.CAPTURE, Theme.PROMOTION}

_THEME_NUDGE = {
    Theme.MATE: "There is a checkmate!",
    Theme.CHECK: "Look for a strong check.",
    Theme.WIN_MATERIAL: "One of your opponent's pieces can be won.",
    Theme.CAPTURE: "Look at the captures.",
    Theme.PROMOTION: "One of your pawns can promote.",
    Theme.SAVE_PIECE: "One of your pieces is in danger: save it.",
    Theme.DEFEND: "One of your pieces is loose: protect it.",
    Theme.THREAT: "You can attack something your opponent has left loose.",
    Theme.CASTLE: "Think about your king's safety.",
    Theme.QUIET: "Look for a quiet move that improves your position.",
}

_SHOW_REASON = {
    Theme.MATE: "checkmate", Theme.CHECK: "a check", Theme.WIN_MATERIAL: "it wins material",
    Theme.CAPTURE: "a capture", Theme.PROMOTION: "it promotes", Theme.SAVE_PIECE: "it saves the piece",
    Theme.DEFEND: "it protects your loose piece", Theme.THREAT: "it creates a threat",
    Theme.CASTLE: "it gets your king to safety",
}


def nudge_text(themes: list[Theme], piece_type: chess.PieceType, detail: Detail, only_move: bool = False,
               threatened: chess.PieceType | None = None) -> str:
    """The first hint: what kind of move to look for, giving more away for rarely found moves."""
    main = themes[0]
    if detail is Detail.VAGUE:
        if main in _FORCING:
            text = "There is a forcing move here: look at checks, captures and threats."
        elif main in (Theme.SAVE_PIECE, Theme.DEFEND):
            text = "Before anything else, check that all your pieces are safe."
        elif main is Theme.THREAT:
            text = "You can put your opponent under pressure."
        else:
            text = "No need for anything sharp: look for a calm, useful move."
    else:
        text = _THEME_NUDGE[main]
        if main is Theme.THREAT and threatened is not None:
            text = f"You can attack their {piece_name(threatened)}."
        if detail is Detail.PIECE:
            text += f" It's a {piece_name(piece_type)} move."
    if only_move:
        text += " Careful: only one move keeps your position together."
    return text


def point_text(board: chess.Board, move: chess.Move) -> str:
    piece = board.piece_at(move.from_square)
    return f"Move your {piece_name(piece.piece_type)} on {chess.square_name(move.from_square)}."


def show_text(board: chess.Board, move: chess.Move, themes: list[Theme]) -> str:
    reasons = [_SHOW_REASON[t] for t in themes if t in _SHOW_REASON]
    san = board.san(move)
    return f"Play {san}" + (f": {', and '.join(reasons[:2])}." if reasons else ".")


@dataclass(frozen=True)
class Hint:
    """One hint about the best move in a position."""

    tier: Tier
    text: str
    from_square: str | None        # highlighted from the "point" tier on
    to_square: str | None          # and this one at "show"
    san: str | None                # the move, at "show" only
    themes: list[str]
    find_rate: float               # how often players at your rating play the move
    detail: Detail
    only_move: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tier"] = self.tier.value
        data["detail"] = self.detail.name.lower()
        return data


def make_hint(board: chess.Board, move: chess.Move, tier: Tier, rate: float, cfg: HintsConfig,
              only_move: bool = False) -> Hint:
    """The hint of ``tier`` about ``move`` (the best move) in ``board``."""
    themes = move_themes(board, move)
    detail = detail_for(rate, cfg)
    piece = board.piece_at(move.from_square).piece_type
    nudge = nudge_text(themes, piece, detail, only_move, threatened_piece(board, move))
    if tier is Tier.NUDGE:
        text, frm, to, san = nudge, None, None, None
    elif tier is Tier.POINT:
        text, frm, to, san = f"{nudge} {point_text(board, move)}", chess.square_name(move.from_square), None, None
    else:
        text = show_text(board, move, themes)
        frm, to, san = chess.square_name(move.from_square), chess.square_name(move.to_square), board.san(move)
    return Hint(tier, text, frm, to, san, [t.value for t in themes], rate, detail, only_move)


# --------------------------------------------------------------------------- the engine


class HintError(ValueError):
    """No hint is possible (the game is over, or the engine found no move)."""


@dataclass(frozen=True)
class BestMove:
    """What a hint is about: Stockfish's best move, how often your level finds it, and whether it is the only one."""

    move: chess.Move
    find_rate: float
    only_move: bool


class HintEngine:
    """Hints from Stockfish (the best move) and the model (how often your level finds it).

    The analysis of a position is kept (a small LRU cache), so asking for the next tier is instant.
    """

    def __init__(self, predictor: MovePredictor, analyzer: Analyzer, cfg: HintsConfig, cache_size: int = 256) -> None:
        self.predictor = predictor
        self.analyzer = analyzer
        self.cfg = cfg
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int], BestMove] = OrderedDict()

    def best(self, board: chess.Board, rating: int, opp_rating: int | None = None, **clock: Any) -> BestMove:
        key = (board.fen(), rating)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if board.is_game_over():
            raise HintError("the game is over")
        analysis = self.analyzer.analyse(board)
        if not analysis.lines:
            raise HintError("the engine found no move")
        top = analysis.lines[0]
        only = (len(analysis.lines) > 1
                and win_percent(top.cp) - win_percent(analysis.lines[1].cp) >= self.cfg.only_move_gap)
        found = BestMove(top.move, find_rate(self.predictor, board, top.move, rating, opp_rating, **clock), only)
        self._cache[key] = found
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return found

    def hint(self, board: chess.Board, tier: Tier, rating: int, opp_rating: int | None = None, **clock: Any) -> Hint:
        """The hint of ``tier`` for the side to move in ``board``, a player rated ``rating``."""
        best = self.best(board, rating, opp_rating, **clock)
        return make_hint(board, best.move, tier, best.find_rate, self.cfg, best.only_move)
