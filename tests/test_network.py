"""Tests for the full rating-conditioned network."""

from __future__ import annotations

import itertools

import pytest
import torch

from chess_engine.config import Config, ModelConfig
from chess_engine.encoding.moves import NUM_MOVES
from chess_engine.model.features import NUM_CONTEXT_FEATURES, ModelInputs
from chess_engine.model.network import RatingConditionedNet, build_model, count_parameters

SMALL = dict(architecture="resnet", channels=16, residual_blocks=2, rating_embedding_dim=8)


def _inputs(batch: int = 6, seed: int = 0, bucket: int | None = None) -> ModelInputs:
    g = torch.Generator().manual_seed(seed)
    buckets = torch.randint(0, 8, (batch,), generator=g) if bucket is None else torch.full((batch,), bucket)
    return ModelInputs(
        planes=(torch.rand(batch, 20, 8, 8, generator=g) > 0.8).float(),
        bucket=buckets,
        opp_bucket=torch.randint(0, 8, (batch,), generator=g),
        context=torch.rand(batch, NUM_CONTEXT_FEATURES, generator=g),
    )


def _perturb_modulation(model: RatingConditionedNet) -> None:
    """Pretend training happened: give the (zero-initialized) modulation layers random weights."""
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation.weight, std=0.5)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize(
    "conditioning, policy_head, value_head, opponent, clock",
    list(itertools.product(["add", "film"], ["conv", "fc"], [True, False], [True, False], [True, False])),
)
def test_every_configuration_runs(conditioning: str, policy_head: str, value_head: bool, opponent: bool,
                                  clock: bool) -> None:
    config = ModelConfig(**SMALL, conditioning=conditioning, policy_head=policy_head, value_head=value_head,
                         condition_on_opponent=opponent, use_clock=clock)
    out = RatingConditionedNet(config, num_buckets=8)(_inputs())
    assert out.policy.shape == (6, NUM_MOVES) and torch.isfinite(out.policy).all()
    if value_head:
        assert out.value is not None and out.value.shape == (6, 3)
    else:
        assert out.value is None


def test_build_model_from_default_config(default_config: Config) -> None:
    model = build_model(default_config)
    assert model.num_buckets == 8 and len(model.blocks) == 6
    counts = count_parameters(model)
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")
    assert set(counts) == {"condition", "stem", "blocks", "policy", "value", "total"}
    assert 400_000 < counts["total"] < 1_500_000  # a laptop-sized network


def test_untrained_policy_ignores_rating_until_modulation_is_learned() -> None:
    model = RatingConditionedNet(ModelConfig(**SMALL), 8).eval()
    with torch.no_grad():
        low, high = model(_inputs(bucket=0)).policy, model(_inputs(bucket=7)).policy
        assert torch.allclose(low, high)  # zero-initialized modulation = a plain ResNet at the start
        _perturb_modulation(model)
        low, high = model(_inputs(bucket=0)).policy, model(_inputs(bucket=7)).policy
    assert not torch.allclose(low, high)


def test_eval_mode_is_deterministic_and_batch_independent() -> None:
    model = RatingConditionedNet(ModelConfig(**SMALL), 8)
    _perturb_modulation(model)
    model.eval()
    inputs = _inputs(batch=5)
    with torch.no_grad():
        full = model(inputs).policy
        again = model(inputs).policy
        single = model(ModelInputs(inputs.planes[2:3], inputs.bucket[2:3], inputs.opp_bucket[2:3],
                                   inputs.context[2:3])).policy
    assert torch.equal(full, again)
    assert torch.allclose(full[2:3], single, atol=1e-5)


def test_every_parameter_gets_a_gradient() -> None:
    model = RatingConditionedNet(ModelConfig(**SMALL), 8)
    out = model(_inputs())
    (out.policy.logsumexp(dim=1).sum() + out.value.logsumexp(dim=1).sum()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing


def test_unknown_bucket_count_mismatch_raises() -> None:
    model = RatingConditionedNet(ModelConfig(**SMALL), num_buckets=4)
    with pytest.raises(IndexError):
        model(_inputs(bucket=5))


def test_conv_policy_head_keeps_the_model_small() -> None:
    conv = count_parameters(RatingConditionedNet(ModelConfig(**{**SMALL, "channels": 64}, policy_head="conv"), 8))
    fc = count_parameters(RatingConditionedNet(ModelConfig(**{**SMALL, "channels": 64}, policy_head="fc"), 8))
    assert fc["policy"] > 50 * conv["policy"]
