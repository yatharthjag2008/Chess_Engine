"""Learning-rate schedules: linear warmup, then constant / cosine / linear decay.

The schedule is a pure function of the step (:func:`lr_factor`), so resuming a
run only needs the step number, and it is easy to test and plot.

::

    factor
    1.0 |    ________                 cosine: 0.5 * (1 + cos(pi * progress))
        |   /        ''--..
        |  /               ''-.
    min | /                    ''---  min_lr_fraction at total_steps
        +----------------------------> step
          warmup
"""

from __future__ import annotations

import math

import torch

SCHEDULES = ("constant", "cosine", "linear")


def lr_factor(
    step: int, total_steps: int, schedule: str = "cosine", warmup_steps: int = 0, min_fraction: float = 0.0
) -> float:
    """Multiplier of the peak learning rate for optimizer step ``step`` (0-based).

    Warmup rises linearly from ``1 / warmup_steps`` to 1 over the first
    ``warmup_steps`` steps; afterwards the factor decays from 1 to
    ``min_fraction`` at ``total_steps`` (and stays there for any extra steps).
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    shape = 0.5 * (1 + math.cos(math.pi * progress)) if schedule == "cosine" else 1 - progress
    return min_fraction + (1 - min_fraction) * shape


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    schedule: str = "cosine",
    warmup_steps: int = 0,
    min_fraction: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """A ``LambdaLR`` that applies :func:`lr_factor` to every parameter group's peak learning rate."""
    lr_factor(0, total_steps, schedule, warmup_steps, min_fraction)  # validate the arguments now
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, schedule, warmup_steps, min_fraction)
    )
