"""Shared fixtures: tiny config + tiny random-weight models. NO downloads, CPU only."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.common.config import load_config
from src.common.modeling import load_target
from src.models.drafter import Drafter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """base.yaml + tiny.yaml with paths.root pointed at a pytest tmp dir."""
    run_root = tmp_path_factory.mktemp("runs")
    return load_config(
        [ROOT / "configs" / "base.yaml", ROOT / "configs" / "tiny.yaml"],
        [f"paths.root={run_root.as_posix()}"],
    )


@pytest.fixture(scope="session")
def tiny_target(tiny_cfg):
    """Random-weight 2-layer hidden-128 Llama, frozen, on CPU."""
    torch.manual_seed(0)
    target = load_target(tiny_cfg, torch.device("cpu"), torch.float32)
    assert not any(p.requires_grad for p in target.parameters())
    return target


@pytest.fixture(scope="session")
def tiny_drafter(tiny_cfg, tiny_target):
    """Tiny drafter bound to the tiny target's frozen embed/norm/head."""
    torch.manual_seed(1)
    drafter = Drafter.from_config(tiny_cfg.drafter)
    drafter.bind_target(tiny_target)
    return drafter
