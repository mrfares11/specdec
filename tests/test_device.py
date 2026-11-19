"""Device resolver behavior + repo-wide enforcement of the no-bare-cuda rule."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from src.common.device import resolve_device, resolve_dtype, sync

ROOT = Path(__file__).resolve().parents[1]


def test_auto_resolves_on_this_machine():
    dev = resolve_device("auto")
    assert dev.type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert dev.type == "cpu"


def test_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_explicit_cuda_fails_fast_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the fail-fast branch is untestable here")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_dtype_resolution():
    cpu = torch.device("cpu")
    assert resolve_dtype("auto", cpu) == torch.float32
    assert resolve_dtype("float32", cpu) == torch.float32
    assert resolve_dtype("bfloat16", cpu) == torch.bfloat16
    assert resolve_dtype("float16", cpu) == torch.float16
    with pytest.raises(ValueError):
        resolve_dtype("fp64", cpu)


def test_sync_is_noop_on_cpu():
    sync(torch.device("cpu"))  # must not raise


def test_no_bare_cuda_calls_anywhere():
    """HARD CONSTRAINT: device handling only via src/common/device.py."""
    offenders = []
    pattern = re.compile(r"\.cuda\(|torch\.device\(\s*[\"']cuda|to\(\s*[\"']cuda")
    for folder in ("src", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name == "device.py":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "bare CUDA usage outside the resolver:\n" + "\n".join(offenders)
