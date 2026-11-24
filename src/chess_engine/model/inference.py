"""Use a model on real positions, and save / load trained models.

    predictor = MovePredictor(model, cfg.ratings)
    for p in predictor.predict(board, rating=1500)[:3]:
        print(p.san, f"{p.prob:.1%}")        # e.g. "e4 31.2%", "d4 24.0%", "Nf3 9.8%"

A saved model (:func:`save_model`) is one ``.pt`` file holding the weights plus
everything needed to rebuild and use it: the model config, the rating bucket
edges and the encoding it was trained with. :func:`load_model` refuses files
whose encoding does not match the current code. Files are loaded with
``weights_only=True``, so opening an untrusted file cannot run code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_engine.config import ModelConfig, RatingsConfig
from chess_engine.encoding.board import NUM_PLANES, PLANE_NAMES, encode_boards
from chess_engine.encoding.moves import NUM_MOVES, index_to_move, legal_move_mask
from chess_engine.model.features import buckets_from_ratings, context_features, make_inputs
from chess_engine.model.masking import legal_probs
from chess_engine.model.network import RatingConditionedNet

MODEL_FORMAT_VERSION = 1


class ModelFileError(Exception):
    """A saved model is unreadable or was trained with a different encoding."""


@dataclass(frozen=True, order=True)
class MoveProbability:
    """One legal move and the probability the model gives it (sorts by probability)."""

    prob: float
    move: chess.Move = field(compare=False)
    san: str = field(compare=False)


@dataclass(frozen=True)
class Prediction:
    """The model's view of one position: every legal move, most likely first, plus win/draw/loss."""

    moves: list[MoveProbability]
    wdl: tuple[float, float, float] | None  # (win, draw, loss) for the side to move, if the model has a value head

    def __getitem__(self, index: int) -> MoveProbability:
        return self.moves[index]

    def __len__(self) -> int:
        return len(self.moves)

    def as_dict(self) -> dict[str, float]:
        """``{uci: probability}``."""
        return {p.move.uci(): p.prob for p in self.moves}


class MovePredictor:
    """Wraps a model for positions given as ``chess.Board`` objects (puts it in eval mode)."""

    def __init__(self, model: RatingConditionedNet, ratings: RatingsConfig, device: str | torch.device = "cpu") -> None:
        if ratings.num_buckets != model.num_buckets:
            raise ValueError(f"model has {model.num_buckets} rating buckets, ratings config has {ratings.num_buckets}")
        self.model = model.to(device).eval()
        self.ratings = ratings
        self.device = torch.device(device)

    @torch.no_grad()
    def predict_many(
        self,
        boards: Sequence[chess.Board],
        ratings: Sequence[int] | int,
        opp_ratings: Sequence[int] | int | None = None,
        clocks: Sequence[float] | float | None = None,
        opp_clocks: Sequence[float] | float | None = None,
        base_seconds: int = 180,
        increment: int = 0,
        temperature: float = 1.0,
    ) -> list[Prediction]:
        """Predictions for several positions in one batch.

        Args:
            boards: Positions (each must have a legal move).
            ratings: Rating of the player to move (one per board, or one for all).
            opp_ratings: The opponent's rating; default: the same as ``ratings``.
            clocks / opp_clocks: Seconds left for the mover / opponent; default: ``base_seconds``.
            base_seconds / increment: The time control (blitz 3+0 by default).
            temperature: Below 1 favours the likeliest moves, above 1 spreads probability out.
        """
        n = len(boards)
        if n == 0:
            return []
        for board in boards:
            if not any(board.legal_moves):
                raise ValueError(f"no legal moves (checkmate or stalemate) in {board.fen()}")
        mover = _per_board(ratings, n)
        opponent = _per_board(opp_ratings if opp_ratings is not None else ratings, n)
        clock = _per_board(clocks if clocks is not None else base_seconds, n)
        opp_clock = _per_board(opp_clocks if opp_clocks is not None else base_seconds, n)

        inputs = make_inputs(
            encode_boards(list(boards)),
            buckets_from_ratings(mover, self.ratings),
            buckets_from_ratings(opponent, self.ratings),
            context_features(clock, opp_clock, np.full(n, base_seconds), np.full(n, increment)),
            self.device,
        )
        mask = torch.as_tensor(np.stack([legal_move_mask(b) for b in boards]), device=self.device)
        out = self.model(inputs)
        probs = legal_probs(out.policy, mask, temperature).cpu().numpy()
        wdl = torch.softmax(out.value.float(), dim=-1).cpu().numpy() if out.value is not None else None

        predictions = []
        for i, board in enumerate(boards):
            moves = []
            for index in np.flatnonzero(mask[i].cpu().numpy()):
                move = index_to_move(int(index), board)
                moves.append(MoveProbability(float(probs[i, index]), move, board.san(move)))
            moves.sort(reverse=True)
            # value head classes are (loss, draw, win); report (win, draw, loss) like engines do
            predictions.append(Prediction(moves, (float(wdl[i, 2]), float(wdl[i, 1]), float(wdl[i, 0]))
                                          if wdl is not None else None))
        return predictions

    def predict(self, board: chess.Board, rating: int, **kwargs: Any) -> Prediction:
        """:meth:`predict_many` for one position."""
        return self.predict_many([board], rating, **kwargs)[0]


def _per_board(value: Sequence[float] | float, n: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n, float(array))
    if len(array) != n:
        raise ValueError(f"expected {n} values (one per board), got {len(array)}")
    return array


# --------------------------------------------------------------------------- saving / loading


def _encoding_info() -> dict[str, Any]:
    return {"num_planes": NUM_PLANES, "plane_names": list(PLANE_NAMES), "num_moves": NUM_MOVES}


def save_model(
    model: RatingConditionedNet, path: str | Path, ratings: RatingsConfig, extra: dict[str, Any] | None = None
) -> Path:
    """Save weights + config + bucket edges + encoding info to ``path`` (written atomically).

    ``extra`` is for small plain metadata (training step, metrics); it must contain
    only numbers, strings, lists and dicts so the file stays loadable with ``weights_only``.
    """
    if ratings.num_buckets != model.num_buckets:
        raise ValueError("ratings config does not match the model's number of buckets")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_config": model.config.model_dump(),
        "bucket_edges": list(ratings.bucket_edges),
        "encoding": _encoding_info(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


@dataclass(frozen=True)
class LoadedModel:
    """A model read back by :func:`load_model`."""

    model: RatingConditionedNet
    ratings: RatingsConfig
    extra: dict[str, Any]

    def predictor(self, device: str | torch.device = "cpu") -> MovePredictor:
        return MovePredictor(self.model, self.ratings, device)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> LoadedModel:
    """Rebuild a model saved with :func:`save_model` (in eval mode, on ``device``)."""
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - torch raises many different errors for bad files
        raise ModelFileError(f"{path} is not a readable model file: {exc}") from exc
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ModelFileError(f"{path}: model format {payload.get('format_version')}, expected {MODEL_FORMAT_VERSION}")
    if payload.get("encoding") != _encoding_info():
        raise ModelFileError(f"{path} was trained with a different board/move encoding than this code")
    ratings = RatingsConfig(bucket_edges=payload["bucket_edges"])
    model = RatingConditionedNet(ModelConfig(**payload["model_config"]), ratings.num_buckets)
    model.load_state_dict(payload["state_dict"])
    return LoadedModel(model.to(device).eval(), ratings, payload.get("extra", {}))
