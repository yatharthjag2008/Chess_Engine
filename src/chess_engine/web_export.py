"""Export a trained model for the static website (GitHub Pages), where it runs in the visitor's browser.

The browser has no PyTorch, so the network is written out as plain numbers and recomputed in JavaScript
(``site/js/engine.js``):

* ``model.bin``: every weight as little-endian float32, one tensor after another;
* ``model.json``: where each tensor starts and its shape, plus everything else the JavaScript needs (the move
  vocabulary, the policy gather index, the rating buckets, the clock scaling) and a few **test vectors**: positions
  with the PyTorch model's probabilities and win/draw/loss, so the page can check its own arithmetic.

Batch normalisation is folded into the convolution before it (inference only), so the JavaScript needs just
convolutions, ReLU, FiLM, two small dense layers and a softmax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch import nn

from chess_engine.config import RatingsConfig
from chess_engine.encoding.moves import MOVE_VOCABULARY, NUM_MOVES
from chess_engine.model.features import _LOG_CLOCK_SCALE, NUM_CONTEXT_FEATURES
from chess_engine.model.inference import MovePredictor
from chess_engine.model.network import RatingConditionedNet
from chess_engine.model.policy_map import policy_gather_index

EXPORT_FORMAT = 1

#: Positions whose predictions go into the export, so the browser can check it computes the same thing.
TEST_POSITIONS: tuple[tuple[str, int, int], ...] = (
    (chess.STARTING_FEN, 1500, 1500),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", 1100, 1300),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 2200, 2100),
    ("8/P6k/8/8/8/8/6K1/8 w - - 0 60", 900, 900),                        # promotions
    ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 5 8", 1700, 1650),   # castling both ways
    ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 1300, 1300),           # en passant
)


class ExportError(ValueError):
    """The model has a shape the website's JavaScript does not implement."""


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """Weights and bias of ``bn(conv(x))`` as one convolution (eval-mode statistics)."""
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    weight = conv.weight.detach() * scale[:, None, None, None]
    bias = bn.bias.detach() - bn.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias.detach() * scale
    return weight.numpy(), bias.numpy()


def _check(model: RatingConditionedNet) -> None:
    c = model.config
    problems = [what for what, ok in [
        ("architecture resnet", c.architecture == "resnet"), ("FiLM conditioning", c.conditioning == "film"),
        ("a convolutional policy head", c.policy_head == "conv"), ("a value head", c.value_head),
        ("the opponent's rating", c.condition_on_opponent), ("the clocks", c.use_clock),
    ] if not ok]
    if problems:
        raise ExportError("the website supports models with " + ", ".join(problems))


def model_tensors(model: RatingConditionedNet) -> dict[str, np.ndarray]:
    """Every tensor the JavaScript forward pass uses, by name (batch norm folded)."""
    _check(model)
    model = model.eval()
    t: dict[str, np.ndarray] = {}
    cond = model.condition
    t["cond.rating"] = cond.rating.weight.detach().numpy()
    t["cond.opponent"] = cond.opponent.weight.detach().numpy()
    t["cond.clock.w"], t["cond.clock.b"] = cond.clock.weight.detach().numpy(), cond.clock.bias.detach().numpy()
    t["cond.fc1.w"], t["cond.fc1.b"] = cond.mlp[0].weight.detach().numpy(), cond.mlp[0].bias.detach().numpy()
    t["cond.fc2.w"], t["cond.fc2.b"] = cond.mlp[2].weight.detach().numpy(), cond.mlp[2].bias.detach().numpy()
    t["stem.w"], t["stem.b"] = _fold(model.stem[0], model.stem[1])
    for i, block in enumerate(model.blocks):
        t[f"block{i}.conv1.w"], t[f"block{i}.conv1.b"] = _fold(block.conv1, block.bn1)
        t[f"block{i}.conv2.w"], t[f"block{i}.conv2.b"] = _fold(block.conv2, block.bn2)
        t[f"block{i}.mod.w"] = block.modulation.weight.detach().numpy()
        t[f"block{i}.mod.b"] = block.modulation.bias.detach().numpy()
    t["policy.conv.w"], t["policy.conv.b"] = _fold(model.policy.block[0], model.policy.block[1])
    t["policy.moves.w"], t["policy.moves.b"] = (model.policy.moves.weight.detach().numpy(),
                                                model.policy.moves.bias.detach().numpy())
    value = model.value
    t["value.conv.w"], t["value.conv.b"] = _fold(value.block[0], value.block[1])
    t["value.fc1.w"], t["value.fc1.b"] = value.mlp[0].weight.detach().numpy(), value.mlp[0].bias.detach().numpy()
    t["value.fc2.w"], t["value.fc2.b"] = value.mlp[2].weight.detach().numpy(), value.mlp[2].bias.detach().numpy()
    return {name: np.ascontiguousarray(array, dtype="<f4") for name, array in t.items()}


def test_vectors(predictor: MovePredictor) -> list[dict[str, Any]]:
    """The PyTorch model's answer for :data:`TEST_POSITIONS` (all legal moves' probabilities, win/draw/loss)."""
    out = []
    for fen, rating, opp in TEST_POSITIONS:
        prediction = predictor.predict(chess.Board(fen), rating, opp_ratings=opp)
        out.append({"fen": fen, "rating": rating, "opp_rating": opp,
                    "probs": {p.move.uci(): round(p.prob, 6) for p in prediction.moves},
                    "wdl": [round(x, 6) for x in prediction.wdl]})
    return out


def export_model(model: RatingConditionedNet, ratings: RatingsConfig, out_dir: str | Path,
                 name: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``model.bin`` and ``model.json`` to ``out_dir``; returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = model_tensors(model)
    offset, entries, chunks = 0, {}, []
    for tensor_name, array in tensors.items():
        entries[tensor_name] = {"offset": offset, "shape": list(array.shape)}
        chunks.append(array.tobytes())
        offset += array.size
    c = model.config
    manifest = {
        "format": EXPORT_FORMAT,
        "name": name,
        "channels": c.channels,
        "blocks": c.residual_blocks,
        "bucket_edges": list(ratings.bucket_edges),
        "log_clock_scale": _LOG_CLOCK_SCALE,
        "context_features": NUM_CONTEXT_FEATURES,
        "vocabulary": [m.uci() for m in MOVE_VOCABULARY],
        "gather": [int(i) for i in policy_gather_index()],
        "tensors": entries,
        "floats": offset,
        "tests": test_vectors(MovePredictor(model, ratings)),
        **(extra or {}),
    }
    assert len(manifest["vocabulary"]) == NUM_MOVES
    (out_dir / "model.bin").write_bytes(b"".join(chunks))
    (out_dir / "model.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- styles (step 19)

STYLE_EXPORT_FORMAT = 1


def export_style(base: MovePredictor, adapter: Any, info: Any, weight: float, out_dir: str | Path) -> dict[str, Any]:
    """Write one trained style as ``<name>.json`` for the website: the adapter's numbers (the offset to the
    conditioning vector, each block's modulation shift, a bias per move), how much it counts next to the base model
    (``weight``), and test vectors from the PyTorch styled predictor. Returns what was written."""
    from chess_engine.style.predictor import StylePredictor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in adapter.state_dict().items()}
    c = base.model.config
    if state["offset"].shape != (c.channels,) or state["block_shift"].shape != (c.residual_blocks, 2 * c.channels):
        raise ExportError("the style was trained for a different model")
    data = {
        "format": STYLE_EXPORT_FORMAT,
        "name": info.name, "kind": info.kind, "player": info.player, "games": info.games, "rating": info.rating,
        "until": info.until, "weight": round(float(weight), 6),
        "offset": [float(x) for x in state["offset"]],
        "block_shift": [[float(x) for x in row] for row in state["block_shift"]],
        "policy_bias": [float(x) for x in state["policy_bias"]],
        "tests": test_vectors(StylePredictor(base, adapter, weight, info)),
    }
    (out_dir / f"{info.name}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def write_style_index(out_dir: str | Path, styles: list[dict[str, Any]]) -> Path:
    """``index.json``: the styles the site offers (name, player, kind, games, rating, until), by name."""
    keys = ("name", "kind", "player", "games", "rating", "until")
    index = sorted(({k: s[k] for k in keys} for s in styles), key=lambda s: s["name"])
    path = Path(out_dir) / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path
