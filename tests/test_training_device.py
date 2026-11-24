"""Tests for device selection, seeding and mixed-precision settings."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from chess_engine.training import device as device_module
from chess_engine.training.device import (
    DeviceError,
    Precision,
    configure_threads,
    resolve_device,
    resolve_precision,
    seed_everything,
)


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)
    assert resolve_device("auto").type == "mps"
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_unavailable_devices_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    with pytest.raises(DeviceError, match="cuda"):
        resolve_device("cuda")
    with pytest.raises(DeviceError, match="mps"):
        resolve_device("mps")
    with pytest.raises(DeviceError, match="unknown"):
        resolve_device("tpu")
    assert resolve_device("cpu") == torch.device("cpu")


def test_seed_everything_makes_all_generators_repeat() -> None:
    def draw() -> tuple:
        return random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(7)
    first = draw()
    seed_everything(7)
    assert draw() == first
    seed_everything(8)
    assert draw() != first


def test_configure_threads() -> None:
    before = torch.get_num_threads()
    try:
        assert configure_threads(1) == 1
        assert configure_threads(None) == 1  # None keeps the current setting
    finally:
        torch.set_num_threads(before)


def test_precision_on_cpu_is_fp32_unless_asked() -> None:
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) == Precision("cpu")
    assert not resolve_precision("off", cpu).enabled
    bf16 = resolve_precision("bf16", cpu)
    assert bf16.dtype == torch.bfloat16 and bf16.name == "bf16" and not bf16.needs_grad_scaler
    fp16 = resolve_precision("fp16", cpu)
    assert fp16.needs_grad_scaler and fp16.name == "fp16"
    with pytest.raises(ValueError):
        resolve_precision("fp8", cpu)


def test_auto_precision_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_precision("auto", cuda).dtype == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_precision("auto", cuda).dtype == torch.float16


def test_autocast_context() -> None:
    layer = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    with Precision("cpu").autocast():
        assert layer(x).dtype == torch.float32
    with Precision("cpu", torch.bfloat16).autocast():
        assert layer(x).dtype == torch.bfloat16
