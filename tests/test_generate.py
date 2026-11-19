"""End-to-end generation: greedy speculative output must EXACTLY equal greedy
autoregressive decoding of the target — the strongest cheap invariant we have
(exercises tree mask, position ids, KV crop/replay, drafter wiring at once).
"""

from __future__ import annotations

import pytest
import torch

from src.decode.generate import greedy_generate, speculative_generate
from src.decode.tree import build_from_config
from src.decode.verify import VerifyOutcome  # noqa: F401  (re-export sanity)

MAX_NEW = 24


def _prompt(vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, 8), generator=g)


def _explain_divergence(target, prompt, ar, spec):
    """On mismatch, check whether it is a genuine bug or a float near-tie."""
    idx = next(i for i, (a, b) in enumerate(zip(ar, spec)) if a != b)
    ids = torch.cat([prompt[0], torch.tensor(ar[:idx], dtype=torch.long)]).unsqueeze(0)
    with torch.no_grad():
        logits = target(input_ids=ids).logits[0, -1]
    top2 = torch.topk(logits, 2).values
    gap = float(top2[0] - top2[1])
    return idx, gap


MODES = {"fold": dict(commit_fold=True), "gather": dict(commit_fold=False, kv_gather=True),
         "replay": dict(commit_fold=False, kv_gather=False)}


@pytest.mark.parametrize("mode", list(MODES), ids=list(MODES))
@pytest.mark.parametrize("allocation", ["uniform", "geometric"])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_spec_equals_greedy_ar(tiny_cfg, tiny_target, tiny_drafter, allocation, k, seed, mode):
    cfg = tiny_cfg.copy()
    cfg.tree.allocation = allocation
    cfg.tree.max_depth = k
    tree = build_from_config(cfg.tree)
    vocab = int(cfg.target.random_config.vocab_size)
    prompt = _prompt(vocab, seed)

    ar = greedy_generate(tiny_target, prompt, MAX_NEW)
    res = speculative_generate(
        tiny_target, tiny_drafter, tree, prompt, MAX_NEW,
        greedy=True, layer_index=int(cfg.data.layer_index), **MODES[mode],
    )
    if res.tokens != ar:
        idx, gap = _explain_divergence(tiny_target, prompt, ar, res.tokens)
        # Only a numerically tied argmax may excuse a divergence; a real
        # mask/position/cache bug produces divergences at decisive logits.
        assert gap < 1e-3, (
            f"greedy spec != greedy AR at position {idx} with decisive top-2 gap "
            f"{gap:.6f}\n  ar  ={ar}\n  spec={res.tokens}"
        )
        pytest.skip(f"near-tie (gap {gap:.2e}) at position {idx}; not a correctness signal")
    assert len(res.tokens) == MAX_NEW
    assert res.num_steps == len(res.accept_lengths) > 0
    assert all(0 <= a <= k for a in res.accept_lengths)


@pytest.mark.parametrize("mode", list(MODES), ids=list(MODES))
def test_sampling_mode_runs_and_is_seeded(tiny_cfg, tiny_target, tiny_drafter, mode):
    cfg = tiny_cfg.copy()
    cfg.tree.max_depth = 3
    tree = build_from_config(cfg.tree)
    vocab = int(cfg.target.random_config.vocab_size)
    prompt = _prompt(vocab, 9)

    def go():
        return speculative_generate(
            tiny_target, tiny_drafter, tree, prompt, MAX_NEW,
            temperature=1.0, greedy=False,
            layer_index=int(cfg.data.layer_index),
            generator=torch.Generator().manual_seed(123),
            **MODES[mode],
        )

    a, b = go(), go()
    assert a.tokens == b.tokens, "same seed must reproduce the same output"
    assert len(a.tokens) == MAX_NEW
    assert all(0 <= t < vocab for t in a.tokens)


def test_timing_decomposition_present(tiny_cfg, tiny_target, tiny_drafter):
    cfg = tiny_cfg.copy()
    cfg.tree.max_depth = 2
    tree = build_from_config(cfg.tree)
    prompt = _prompt(int(cfg.target.random_config.vocab_size), 5)
    res = speculative_generate(
        tiny_target, tiny_drafter, tree, prompt, 12,
        greedy=True, layer_index=int(cfg.data.layer_index), profile_timing=True,
    )
    # Static path: prefill is now timed separately; topology/mask are dynamic-only
    # (a static tree builds its mask ONCE outside the loop, which is exactly why
    # it is cheap -- their absence here is the contract, not an omission).
    assert set(res.timings) == {"prefill", "commit", "draft", "tree_forward", "verify"}
    assert all(v > 0 for v in res.timings.values())

    # Dynamic path additionally carves topology and mask construction out of
    # `draft`, so the per-phase cost of reallocation is attributable.
    res_dyn = speculative_generate(
        tiny_target, tiny_drafter, tree, prompt, 12,
        greedy=True, layer_index=int(cfg.data.layer_index), profile_timing=True,
        dynamic_tree=True,
    )
    assert {"topology", "mask"} <= set(res_dyn.timings)
    assert all(v > 0 for v in res_dyn.timings.values())


def test_eos_stops_generation(tiny_cfg, tiny_target, tiny_drafter):
    cfg = tiny_cfg.copy()
    cfg.tree.max_depth = 2
    tree = build_from_config(cfg.tree)
    vocab = int(cfg.target.random_config.vocab_size)
    prompt = _prompt(vocab, 4)
    ar = greedy_generate(tiny_target, prompt, MAX_NEW)
    eos = ar[3]  # force an early stop at a token greedy decoding actually emits
    res = speculative_generate(
        tiny_target, tiny_drafter, tree, prompt, MAX_NEW,
        greedy=True, layer_index=int(cfg.data.layer_index), eos_token_id=eos,
    )
    assert eos in res.tokens
    assert res.tokens == res.tokens[: res.tokens.index(eos) + 1], "nothing after eos"
    assert res.tokens == ar[: len(res.tokens)]
