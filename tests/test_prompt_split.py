"""Dev/test prompt separation.

The experiment protocol forbids selecting checkpoints, thread caps or
configurations on the prompts that get reported. That is only enforceable if
the two pools are provably disjoint and stable across runs, so these are
regression tests, not smoke tests. No network: the dataset loader is stubbed.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.eval import benchmark as B


def _stub_dataset(monkeypatch, n_rows: int = 500):
    """Stand in for datasets.load_dataset with deterministic rows."""
    rows = [{"question": f"q{i}"} for i in range(n_rows)]

    def _load(*_a, **_k):
        return rows

    mod = types.ModuleType("datasets")
    mod.load_dataset = _load
    monkeypatch.setitem(sys.modules, "datasets", mod)
    monkeypatch.setitem(
        B._BENCH_SOURCES, "stub", ("stub", None, "test", lambda r: r["question"])
    )


def test_dev_and_test_pools_are_disjoint(monkeypatch):
    _stub_dataset(monkeypatch)
    dev = B._benchmark_prompts("stub", 50, split="dev", dev_size=200)
    test = B._benchmark_prompts("stub", 50, split="test", dev_size=200)
    dev_ids = {i for i, _ in dev}
    test_ids = {i for i, _ in test}
    assert dev_ids.isdisjoint(test_ids), "development prompts leaked into the test pool"


def test_split_is_deterministic(monkeypatch):
    _stub_dataset(monkeypatch)
    a = B._benchmark_prompts("stub", 30, split="test", dev_size=200)
    b = B._benchmark_prompts("stub", 30, split="test", dev_size=200)
    assert a == b, "test prompt selection must be frozen across runs"


def test_legacy_split_is_dataset_order(monkeypatch):
    """Pre-plan results used first-n order; keep them reproducible."""
    _stub_dataset(monkeypatch)
    legacy = B._benchmark_prompts("stub", 10, split="legacy")
    assert [i for i, _ in legacy] == list(range(10))


def test_unknown_split_is_rejected(monkeypatch):
    _stub_dataset(monkeypatch)
    with pytest.raises(ValueError, match="legacy|dev|test"):
        B._benchmark_prompts("stub", 5, split="train")


def test_dev_pool_size_is_respected(monkeypatch):
    _stub_dataset(monkeypatch, n_rows=300)
    with pytest.raises(ValueError, match="only"):
        B._benchmark_prompts("stub", 250, split="dev", dev_size=200)


def test_small_dataset_still_leaves_a_test_pool(monkeypatch):
    """Regression: a fixed dev_size once consumed a whole small benchmark.

    HumanEval has 164 rows; with dev_size=200 the dev pool took every row and
    the test pool came back empty, which surfaced only as a run-time failure
    mid-experiment. dev_size is a cap, and at most a third may be reserved.
    """
    _stub_dataset(monkeypatch, n_rows=164)
    dev = B._benchmark_prompts("stub", 40, split="dev", dev_size=200)
    test = B._benchmark_prompts("stub", 100, split="test", dev_size=200)
    assert len(dev) == 40 and len(test) == 100
    assert {i for i, _ in dev}.isdisjoint({i for i, _ in test})


@pytest.mark.parametrize("n_rows", [80, 164, 500, 1319])
def test_test_pool_keeps_the_majority(monkeypatch, n_rows):
    """The reported split must never be the minority of the data."""
    _stub_dataset(monkeypatch, n_rows=n_rows)
    dev = B._benchmark_prompts("stub", 1, split="dev", dev_size=200)
    test = B._benchmark_prompts("stub", 1, split="test", dev_size=200)
    # Ask for everything each pool holds, to measure the actual sizes.
    n_dev = max(1, min(200, n_rows // 3))
    full_dev = B._benchmark_prompts("stub", n_dev, split="dev", dev_size=200)
    full_test = B._benchmark_prompts("stub", n_rows - n_dev, split="test",
                                     dev_size=200)
    assert len(full_test) > len(full_dev), f"test pool not the majority at n={n_rows}"
    assert {i for i, _ in full_dev}.isdisjoint({i for i, _ in full_test})
    assert dev and test
