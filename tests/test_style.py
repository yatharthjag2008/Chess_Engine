"""Style imitation and ghost mode (step 19)."""

from __future__ import annotations

import torch

from bot_helpers import small_predictor
from chess_engine.style.adapter import StyleAdapter, StyledNet


def _inputs(predictor, n: int = 4):
    import chess
    import numpy as np

    from chess_engine.encoding.board import encode_boards
    from chess_engine.model.features import buckets_from_ratings, context_features, make_inputs

    boards = [chess.Board() for _ in range(n)]
    for i, board in enumerate(boards):
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6"][:i]:
            board.push_uci(uci)
    ratings = np.full(n, 1500)
    return make_inputs(encode_boards(boards), buckets_from_ratings(ratings, predictor.ratings),
                       buckets_from_ratings(ratings, predictor.ratings),
                       context_features(np.full(n, 180.0), np.full(n, 180.0), np.full(n, 180), np.zeros(n)))


def test_a_new_adapter_changes_nothing() -> None:
    predictor = small_predictor()
    base = predictor.model
    inputs = _inputs(predictor)
    with torch.no_grad():
        expected = base(inputs)
        styled = StyledNet(base)(inputs)
    assert torch.allclose(styled.policy, expected.policy) and torch.allclose(styled.value, expected.value)


def test_only_the_adapter_learns_and_the_base_stays_in_eval_mode() -> None:
    predictor = small_predictor()
    net = StyledNet(predictor.model)
    net.train()
    assert not net.base.training and net.adapter.training
    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    assert trainable == ["adapter.offset", "adapter.block_shift", "adapter.policy_bias"]
    out = net(_inputs(predictor))
    out.policy.logsumexp(dim=1).mean().backward()
    assert net.adapter.offset.grad is not None and all(p.grad is None for p in net.base.parameters())
    size = sum(p.numel() for p in net.adapter.parameters())
    cfg = predictor.model.config
    assert size == cfg.channels + cfg.residual_blocks * cfg.channels * 2 + 1858


def test_the_adapter_moves_the_policy() -> None:
    predictor = small_predictor()
    net = StyledNet(predictor.model)
    inputs = _inputs(predictor)
    with torch.no_grad():
        before = net(inputs).policy.clone()
        net.adapter.policy_bias[100] = 5.0
        net.adapter.offset += 0.5
        after = net(inputs).policy
    assert (after[:, 100] - before[:, 100]).min() > 4
    assert not torch.allclose(after, before)
