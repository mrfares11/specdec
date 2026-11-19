"""Runnable-end-to-end smoke tests: profiling, extraction, eval — all tiny, all CPU."""

from __future__ import annotations

import csv
import warnings

import pytest
import torch

from src.common.modeling import clone_prefix_cache, validate_feature_config
from src.data.extract_features import run as extract_run
from src.data.shards import ShardReader
from src.eval.benchmark import run as eval_run
from src.profiling.verifier_latency import run as profile_run


def test_verifier_latency_profile(tiny_cfg):
    """The first thing that runs on the pod must run here too."""
    csv_path = profile_run(tiny_cfg)
    assert csv_path.exists()
    assert csv_path.with_name("latency.png").exists()
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    counts = [int(x) for x in tiny_cfg.profiling.token_counts]
    assert [int(r["num_query_tokens"]) for r in rows] == counts
    for r in rows:
        assert float(r["mean_ms"]) > 0
        assert float(r["ms_per_token"]) > 0


def test_extract_features_synthetic_and_resume(tiny_cfg, tmp_path):
    cfg = tiny_cfg.copy()
    cfg.data.out_dir = (tmp_path / "feats").as_posix()

    # dry run: estimate only, nothing written
    cfg.data.dry_run = True
    assert extract_run(cfg) is None
    assert not (tmp_path / "feats" / "manifest.json").exists()

    cfg.data.dry_run = False
    out = extract_run(cfg)
    r = ShardReader(out)
    assert len(r) == int(cfg.data.max_samples)
    hidden_size = int(cfg.target.random_config.hidden_size)
    for _, hidden, ids in r.samples():
        assert hidden.shape == (ids.shape[0], hidden_size)
        assert hidden.isfinite().all()

    # re-run: everything already present → nothing new, no error
    extract_run(cfg)
    assert len(ShardReader(out)) == int(cfg.data.max_samples)


def test_prefix_cache_isolation(tiny_cfg, tiny_target):
    """The profiler's core assumption: cloned caches never leak appended tokens
    back into the legacy prefix, so every timed iteration starts at prefix_len.
    Also a canary for deprecation of the legacy-cache API (pyproject silences
    warning categories suite-wide, so we check explicitly here)."""
    prefix_len = 16
    vocab = int(tiny_cfg.target.random_config.vocab_size)
    g = torch.Generator().manual_seed(0)
    prefix = torch.randint(0, vocab, (1, prefix_len), generator=g)
    with torch.no_grad():
        out = tiny_target(input_ids=prefix, use_cache=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = out.past_key_values.to_legacy_cache()
        for _ in range(3):
            cache = clone_prefix_cache(legacy)
            assert cache.get_seq_length() == prefix_len
            ids = torch.randint(0, vocab, (1, 4), generator=g)
            pos = torch.arange(prefix_len, prefix_len + 4).unsqueeze(0)
            with torch.no_grad():
                tiny_target(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True)
            # the legacy prefix tensors must be untouched by the append
            assert all(k.shape[2] == prefix_len and v.shape[2] == prefix_len for k, v in legacy)
    deprecations = [w for w in caught if issubclass(w.category, (DeprecationWarning, FutureWarning))
                    and "legacy" in str(w.message).lower()]
    assert not deprecations, f"legacy-cache API deprecated upstream: {deprecations[0].message}"


def test_benchmark_prompt_sources_registered(tiny_cfg):
    """Benchmark loaders exist for the paper's domain table; extraction logic
    is exercised on synthetic rows (no downloads in tests)."""
    from src.eval.benchmark import _BENCH_SOURCES

    assert set(_BENCH_SOURCES) == {"mtbench", "humaneval", "gsm8k", "alpaca", "mbpp"}
    fake = {
        "mtbench": {"prompt": ["Explain X.", "follow-up"]},
        "humaneval": {"prompt": "def add(a, b):\n"},
        "gsm8k": {"question": "If x=2 and y=3, what is x+y?"},
        "alpaca": {"instruction": "Summarize:", "input": "some text"},
        "mbpp": {"text": "Write a function to reverse a list.",
                 "test_list": ["assert rev([1,2]) == [2,1]"]},
    }
    for name, (_, _, _, extract) in _BENCH_SOURCES.items():
        text = extract(fake[name])
        assert isinstance(text, str) and len(text) > 5, name


def test_feature_config_guard(tiny_cfg):
    """layer_index=-1 with apply_final_norm=true double-norms — must be rejected."""
    cfg = tiny_cfg.copy()
    validate_feature_config(cfg)  # default (-2, true) is fine
    cfg.data.layer_index = -1
    with pytest.raises(ValueError, match="apply_final_norm"):
        validate_feature_config(cfg)
    cfg.drafter.apply_final_norm = False
    validate_feature_config(cfg)  # (-1, false) is the exact pairing


def test_eval_benchmark_smoke(tiny_cfg, tiny_drafter):
    csv_path = eval_run(tiny_cfg)
    assert csv_path.exists()
    assert csv_path.with_name("benchmark.png").exists()
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["K"]) for r in rows] == [int(k) for k in tiny_cfg.eval.K_sweep]
    for r in rows:
        assert 0.0 <= float(r["mean_accept_len"]) <= float(r["K"])
        assert float(r["tokens_per_step"]) >= 1.0
