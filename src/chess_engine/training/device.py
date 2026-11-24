"""Pick the device, seed every random generator, and decide on mixed precision.

    device = resolve_device("auto")                  # cuda > mps > cpu
    seed_everything(42)
    precision = resolve_precision("auto", device)    # Precision(dtype=None) on CPU, bf16/fp16 on GPU
    with precision.autocast():
        out = model(inputs)
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch


class DeviceError(RuntimeError):
    """The requested device is not available on this machine."""


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` -> the best available device (CUDA, then Apple MPS, then CPU); others are checked.

    Raises:
        DeviceError: ``cuda`` or ``mps`` was asked for but is not available.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise DeviceError("training.device is 'cuda' but no CUDA GPU is available (use 'auto' or 'cpu')")
    if name == "mps" and not _mps_available():
        raise DeviceError("training.device is 'mps' but Apple MPS is not available (use 'auto' or 'cpu')")
    if name not in ("cpu", "cuda", "mps"):
        raise DeviceError(f"unknown device {name!r}")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed Python's, NumPy's and PyTorch's (CPU and every GPU) global random generators."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds every CUDA device


def configure_threads(threads: int | None) -> int:
    """Set the number of torch CPU threads (``None`` keeps torch's default); returns the number in use."""
    if threads is not None:
        torch.set_num_threads(threads)
    return torch.get_num_threads()


@dataclass(frozen=True)
class Precision:
    """Mixed-precision setting for a device: ``dtype`` None means plain float32."""

    device_type: str
    dtype: torch.dtype | None = None

    @property
    def enabled(self) -> bool:
        return self.dtype is not None

    @property
    def needs_grad_scaler(self) -> bool:
        """float16 gradients can underflow, so fp16 training scales the loss (bf16 does not need it)."""
        return self.dtype == torch.float16

    @property
    def name(self) -> str:
        return {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[self.dtype]

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        """Run the enclosed forward pass under autocast (a no-op for fp32)."""
        if self.dtype is None:
            yield
            return
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            yield


def resolve_precision(setting: str, device: torch.device) -> Precision:
    """``auto``: bf16 on GPUs that support it, else fp16 on CUDA; fp32 on CPU and MPS.

    CPU mixed precision is not used automatically: on the development laptop
    bf16 training is not supported by oneDNN and fp16 is slower than fp32.
    """
    kind = device.type
    if setting == "off":
        return Precision(kind)
    if setting == "auto":
        if kind != "cuda":
            return Precision(kind)
        return Precision(kind, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(setting)
    if dtype is None:
        raise ValueError(f"unknown mixed_precision setting {setting!r}")
    return Precision(kind, dtype)
