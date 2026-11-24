"""Tests for the policy and value heads."""

from __future__ import annotations

import chess
import pytest
import torch

from chess_engine.encoding.moves import NUM_MOVES, move_to_index
from chess_engine.model.heads import ConvPolicyHead, FCPolicyHead, ValueHead
from chess_engine.model.policy_map import move_type


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("head_cls", [ConvPolicyHead, FCPolicyHead])
def test_policy_head_shape(head_cls: type) -> None:
    head = head_cls(32)
    assert head(torch.randn(5, 32, 8, 8)).shape == (5, NUM_MOVES)


def test_conv_head_gathers_the_right_plane_and_square() -> None:
    head = ConvPolicyHead(16).eval()
    features = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        planes = head.moves(head.block(features))
        logits = head(features)
    for uci in ("e2e4", "g1f3", "e1g1", "a7a8n", "b7a8b", "h1a8"):
        move = chess.Move.from_uci(uci)
        rank, file = chess.square_rank(move.from_square), chess.square_file(move.from_square)
        assert torch.equal(logits[:, move_to_index(move)], planes[:, move_type(move), rank, file])


def test_conv_head_is_much_smaller_than_fc_head() -> None:
    conv, fc = _params(ConvPolicyHead(64)), _params(FCPolicyHead(64))
    assert conv < 50_000 < 3_000_000 < fc


def test_value_head_shape_and_condition_dependence() -> None:
    head = ValueHead(channels=16, cond_dim=8).eval()
    features = torch.randn(3, 16, 8, 8)
    out = head(features, torch.zeros(3, 8))
    assert out.shape == (3, 3)
    assert not torch.allclose(out, head(features, torch.ones(3, 8)))


def test_gather_index_moves_with_the_module() -> None:
    head = ConvPolicyHead(8)
    assert head.gather_index.dtype == torch.int64 and len(head.gather_index) == NUM_MOVES
    assert "gather_index" not in head.state_dict()  # rebuilt from code, not saved in checkpoints
