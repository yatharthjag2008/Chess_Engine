"""Tests for the conditioning encoder and conditioned residual blocks."""

from __future__ import annotations

import pytest
import torch

from chess_engine.model.features import NUM_CONTEXT_FEATURES, ModelInputs
from chess_engine.model.layers import ConditionedResidualBlock, ConditionEncoder, ConvBlock


def _inputs(batch: int = 4, bucket: int | list[int] = 0, opp: int = 0, clock: float = 0.5) -> ModelInputs:
    buckets = torch.tensor(bucket if isinstance(bucket, list) else [bucket] * batch)
    return ModelInputs(
        planes=torch.zeros(len(buckets), 20, 8, 8),
        bucket=buckets,
        opp_bucket=torch.full((len(buckets),), opp),
        context=torch.full((len(buckets), NUM_CONTEXT_FEATURES), clock),
    )


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_conv_block_shape() -> None:
    assert ConvBlock(20, 32)(torch.randn(2, 20, 8, 8)).shape == (2, 32, 8, 8)


def test_encoder_output_shape_and_bucket_dependence() -> None:
    encoder = ConditionEncoder(num_buckets=8, embedding_dim=16, out_dim=24)
    out = encoder(_inputs(bucket=[0, 3, 7, 3]))
    assert out.shape == (4, 24)
    assert torch.allclose(out[1], out[3]) and not torch.allclose(out[0], out[2])


@pytest.mark.parametrize("use_opponent, use_clock", [(True, True), (False, True), (True, False), (False, False)])
def test_encoder_optional_inputs(use_opponent: bool, use_clock: bool) -> None:
    encoder = ConditionEncoder(8, 16, 24, condition_on_opponent=use_opponent, use_clock=use_clock)
    base = encoder(_inputs())
    assert torch.allclose(base, encoder(_inputs(opp=5))) != use_opponent
    assert torch.allclose(base, encoder(_inputs(clock=0.1))) != use_clock


def test_encoder_rejects_out_of_range_buckets() -> None:
    encoder = ConditionEncoder(8, 16, 24)
    with pytest.raises(IndexError, match="bucket"):
        encoder(_inputs(bucket=8))
    with pytest.raises(IndexError, match="opp_bucket"):
        encoder(_inputs(opp=-1))


@pytest.mark.parametrize("mode", ["add", "film"])
def test_block_starts_unconditioned_then_depends_on_condition(mode: str) -> None:
    block = ConditionedResidualBlock(channels=16, cond_dim=8, conditioning=mode).eval()
    x = torch.randn(3, 16, 8, 8)
    c1, c2 = torch.randn(3, 8), torch.randn(3, 8)
    out = block(x, c1)
    assert out.shape == x.shape and (out >= 0).all()
    assert torch.allclose(out, block(x, c2))  # zero-initialized modulation: condition has no effect yet
    torch.nn.init.normal_(block.modulation.weight)
    assert not torch.allclose(block(x, c1), block(x, c2))


def test_block_modulation_sizes() -> None:
    assert ConditionedResidualBlock(16, 8, "add").modulation.out_features == 16
    assert ConditionedResidualBlock(16, 8, "film").modulation.out_features == 32
    with pytest.raises(ValueError, match="conditioning"):
        ConditionedResidualBlock(16, 8, "concat")  # type: ignore[arg-type]


def test_gradients_reach_the_rating_embedding() -> None:
    encoder = ConditionEncoder(8, 16, 8)
    block = ConditionedResidualBlock(16, 8, "film")
    torch.nn.init.normal_(block.modulation.weight)
    out = block(torch.randn(4, 16, 8, 8), encoder(_inputs(bucket=[1, 2, 3, 4])))
    out.sum().backward()
    grad = encoder.rating.weight.grad
    assert grad is not None
    assert grad[1:5].abs().sum() > 0 and grad[0].abs().sum() == 0 and grad[5:].abs().sum() == 0
