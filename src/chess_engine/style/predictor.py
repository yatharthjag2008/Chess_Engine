"""A predictor that plays in a style: the styled network blended with the base model.

``StylePredictor`` has the same interface as :class:`~chess_engine.model.inference.MovePredictor`
(``predict``, ``predict_many``, ``ratings``), so the bot, hints and the API use it unchanged. Its move
probabilities are ``weight * styled + (1 - weight) * base``; the weight grows with the number of games the style
was trained on (see :func:`chess_engine.style.store.blend_weight`), so a style from a handful of games only
nudges the rating model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chess

from chess_engine.model.inference import MoveProbability, MovePredictor, Prediction
from chess_engine.style.adapter import StyleAdapter, StyledNet
from chess_engine.style.store import StyleInfo


class StylePredictor:
    """``base`` (a MovePredictor) with a style adapter, blended with weight ``weight``."""

    def __init__(self, base: MovePredictor, adapter: StyleAdapter, weight: float, info: StyleInfo | None = None) -> None:
        if not 0 <= weight <= 1:
            raise ValueError(f"weight must be between 0 and 1, got {weight}")
        self.base = base
        self.styled = MovePredictor(StyledNet(base.model, adapter), base.ratings, base.device)
        self.weight = weight
        self.info = info
        self.ratings = base.ratings
        self.model = self.styled.model

    def predict_many(self, boards: Sequence[chess.Board], ratings: Sequence[int] | int, **kwargs: Any) -> list[Prediction]:
        styled = self.styled.predict_many(boards, ratings, **kwargs)
        if self.weight >= 1:
            return styled
        base = self.base.predict_many(boards, ratings, **kwargs)
        return [self._blend(b, s) for b, s in zip(base, styled)]

    def predict(self, board: chess.Board, rating: int, **kwargs: Any) -> Prediction:
        return self.predict_many([board], rating, **kwargs)[0]

    def _blend(self, base: Prediction, styled: Prediction) -> Prediction:
        w = self.weight
        theirs = {p.move: p.prob for p in styled.moves}
        moves = [MoveProbability(w * theirs.get(p.move, 0.0) + (1 - w) * p.prob, p.move, p.san) for p in base.moves]
        moves.sort(reverse=True)
        wdl = None
        if base.wdl is not None and styled.wdl is not None:
            wdl = tuple(w * s + (1 - w) * b for s, b in zip(styled.wdl, base.wdl))
        return Prediction(moves, wdl)
