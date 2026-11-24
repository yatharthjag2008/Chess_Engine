"""Tests for the learning-rate schedules."""

from __future__ import annotations

import math

import pytest
import torch

from chess_engine.training.schedule import build_scheduler, lr_factor


def test_warmup_is_linear_and_reaches_one() -> None:
    factors = [lr_factor(s, 100, "cosine", warmup_steps=4) for s in range(4)]
    assert factors == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert lr_factor(4, 100, "cosine", warmup_steps=4) == pytest.approx(1.0)


def test_constant_schedule() -> None:
    assert [lr_factor(s, 10, "constant") for s in (0, 5, 9, 50)] == [1.0] * 4


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_decay_ends_at_min_fraction_and_is_monotonic(schedule: str) -> None:
    total, warmup = 200, 20
    factors = [lr_factor(s, total, schedule, warmup, min_fraction=0.1) for s in range(warmup, total + 50)]
    assert factors[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-12 for a, b in zip(factors, factors[1:]))
    assert lr_factor(total, total, schedule, warmup, 0.1) == pytest.approx(0.1)
    assert factors[-1] == pytest.approx(0.1)  # steps past the end stay at the minimum


def test_cosine_and_linear_halfway() -> None:
    assert lr_factor(50, 100, "cosine") == pytest.approx(0.5)
    assert lr_factor(25, 100, "cosine") == pytest.approx(0.5 * (1 + math.cos(math.pi / 4)))
    assert lr_factor(25, 100, "linear") == pytest.approx(0.75)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="schedule"):
        lr_factor(0, 10, "step")
    with pytest.raises(ValueError, match="total_steps"):
        lr_factor(0, 0)


def test_scheduler_drives_the_optimizer_and_resumes_from_its_state() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=10, schedule="linear", warmup_steps=2)
    seen = []
    for _ in range(6):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    expected = [0.1 * lr_factor(s, 10, "linear", 2) for s in range(6)]
    assert seen == pytest.approx(expected)

    # A fresh scheduler loaded from the saved state continues where the first one stopped.
    optimizer2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler2 = build_scheduler(optimizer2, total_steps=10, schedule="linear", warmup_steps=2)
    scheduler2.load_state_dict(scheduler.state_dict())
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(0.1 * lr_factor(6, 10, "linear", 2))
