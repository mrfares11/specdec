"""The single device/dtype resolver.

HARD RULE: no other module may call ``.cuda()``, hardcode ``"cuda"``, or make
device decisions. Everything goes through :func:`resolve_device` /
:func:`resolve_dtype`, and timing code brackets measured regions with
:func:`sync` (a no-op on CPU). Enforced by ``tests/test_device.py``.
"""

from __future__ import annotations

from typing import Optional

import torch

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a config device spec to a concrete :class:`torch.device`.

    ``"auto"`` picks CUDA when available, else CPU. Explicit CUDA specs fail
    fast when CUDA is absent instead of silently falling back.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"config requested device '{spec}' but CUDA is not available; "
            "use runtime.device=auto or cpu"
        )
    return dev


def resolve_dtype(spec: str = "auto", device: Optional[torch.device] = None) -> torch.dtype:
    """Resolve a config dtype spec. ``"auto"``: bfloat16 on CUDA, float32 on CPU."""
    if spec == "auto":
        if device is not None and device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    try:
        return _DTYPES[spec]
    except KeyError:
        raise ValueError(f"unknown dtype '{spec}'; expected one of {sorted(_DTYPES)} or 'auto'")


def sync(device: torch.device) -> None:
    """Barrier for accurate timing: ``cuda.synchronize`` on GPU, no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
