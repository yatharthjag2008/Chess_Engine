"""Exporting the model for the website (it runs in the browser there)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from bot_helpers import small_predictor
from chess_engine.web_export import TEST_POSITIONS, ExportError, _fold, export_model, model_tensors


def _trained_bn(predictor):
    """Give every batch norm non-trivial statistics, so folding is really tested."""
    torch.manual_seed(0)
    for module in predictor.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 2.0)
            with torch.no_grad():
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    return predictor


def test_folding_matches_conv_then_batch_norm() -> None:
    predictor = _trained_bn(small_predictor())
    block = predictor.model.blocks[0]
    x = torch.randn(2, block.conv1.in_channels, 8, 8)
    with torch.no_grad():
        expected = block.bn1(block.conv1(x)).numpy()
    w, b = _fold(block.conv1, block.bn1)
    got = torch.nn.functional.conv2d(x, torch.as_tensor(w), torch.as_tensor(b), padding=1).numpy()
    assert np.allclose(got, expected, atol=1e-5)


def test_export_writes_weights_manifest_and_test_vectors(tmp_path) -> None:
    predictor = _trained_bn(small_predictor())
    manifest = export_model(predictor.model, predictor.ratings, tmp_path, name="small")
    data = np.frombuffer((tmp_path / "model.bin").read_bytes(), dtype="<f4")
    assert data.size == manifest["floats"]
    tensors = model_tensors(predictor.model)
    for name, entry in manifest["tensors"].items():
        start = entry["offset"]
        size = int(np.prod(entry["shape"]))
        assert np.array_equal(data[start:start + size].reshape(entry["shape"]), tensors[name]), name
    loaded = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert loaded["format"] == 1 and len(loaded["vocabulary"]) == 1858 and len(loaded["gather"]) == 1858
    assert loaded["vocabulary"][0] == "a1b1" and loaded["bucket_edges"] == predictor.ratings.bucket_edges
    assert len(loaded["tests"]) == len(TEST_POSITIONS)
    first = loaded["tests"][0]
    assert len(first["probs"]) == 20 and abs(sum(first["probs"].values()) - 1) < 1e-4 and len(first["wdl"]) == 3
    promo = loaded["tests"][3]["probs"]
    assert {"a7a8q", "a7a8n", "a7a8b", "a7a8r"} <= set(promo), "promotions are named with their piece"


def test_only_supported_models_export() -> None:
    from chess_engine.config import load_config
    from chess_engine.model.network import build_model

    cfg = load_config()
    fc = build_model(cfg.model_copy(update={"model": cfg.model.model_copy(update={"policy_head": "fc"})}))
    with pytest.raises(ExportError, match="convolutional policy head"):
        model_tensors(fc)
