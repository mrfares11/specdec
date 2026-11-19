"""Deterministic seeding for scripts and tests."""

from __future__ import annotations

import random

import torch


def seed_all(seed: int) -> None:
    """Seed torch (all devices), Python's ``random``, and NumPy if present."""
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover
        pass
