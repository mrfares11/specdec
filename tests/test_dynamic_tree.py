"""M4 dynamic tree reallocation: structure validity, budget invariance, and —
the hard invariant — greedy losslessness with per-step dynamic topologies.
CPU-only, tiny random models."""

from __future__ import annotations

import pytest
import torch

from src.decode.generate import _draft_dynamic, greedy_generate, speculative_generate
from src.decode.tree import build_from_config, build_from_parents
from src.models.drafter import DrafterCache

MAX_NEW = 24


def test_build_from_parents_chain_and_fanout():
    t = build_from_parents([-1, 0, 1])  # chain
    assert t.num_nodes == 3 and t.max_depth == 3 and t.num_leaves == 1
    assert t.paths() == [[0, 1, 2]]
    t2 = build_from_parents([-1, -1, 0, 0, 1])
    assert t2.max_depth == 2 and t2.num_leaves == 3
    assert t2.children_of(-1) == (0, 1)
    assert bool(t2.attn_mask[4, 1]) and not bool(t2.attn_mask[4, 0])


def test_build_from_parents_rejects_non_bfs():
    with pytest.raises(ValueError):
        build_from_parents([-1, 1])  # forward parent reference
    with pytest.raises(ValueError):
        build_from_parents([0])  # self/forward at node 0
    with pytest.raises(ValueError):
        build_from_parents([-1, 0, -1])  # depth decreases: not BFS order


def test_draft_dynamic_budget_and_bfs(tiny_cfg, tiny_target, tiny_drafter):
    torch.manual_seed(0)
    h = int(tiny_cfg.drafter.hidden_size)
    prompt = torch.randint(0, int(tiny_cfg.target.random_config.vocab_size), (1, 6))
    with torch.no_grad():
        out = tiny_target(input_ids=prompt, output_hidden_states=True)
        feats = out.hidden_states[-2]
        _, dcache = tiny_drafter(prompt[:, 1:], feats[:, :-1],
                                 torch.arange(1, 6).unsqueeze(0), attn_bias=None, cache=None)
        q0 = tiny_drafter.logits(feats[:, -1])[0]
        widths = [5, 4, 3]
        toks, parents, _ = _draft_dynamic(
            tiny_drafter, widths, dcache, dcache.seq_length(), 6, q0, feats[:, -1:],
        )
    assert len(parents) == sum(widths) == toks.shape[0]
    tree = build_from_parents(parents)  # raises if not BFS-consistent
    assert torch.bincount(tree.depth).tolist() == widths
    # every non-root parent must sit exactly one level up
    for i, p in enumerate(parents):
        assert p == -1 or int(tree.depth[p]) == int(tree.depth[i]) - 1


def test_dynamic_rejects_sampling_and_replay(tiny_cfg, tiny_target, tiny_drafter):
    cfg = tiny_cfg.copy()
    cfg.tree.max_depth = 3
    tree = build_from_config(cfg.tree)
    prompt = torch.randint(0, int(cfg.target.random_config.vocab_size), (1, 8))
    with pytest.raises(ValueError, match="greedy"):
        speculative_generate(tiny_target, tiny_drafter, tree, prompt, 8,
                             greedy=False, dynamic_tree=True,
                             generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="commit_fold"):
        speculative_generate(tiny_target, tiny_drafter, tree, prompt, 8,
                             greedy=True, dynamic_tree=True, commit_fold=False)


@pytest.mark.parametrize("allocation", ["uniform", "geometric"])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dynamic_greedy_is_lossless(tiny_cfg, tiny_target, tiny_drafter, allocation, k, seed):
    cfg = tiny_cfg.copy()
    cfg.tree.allocation = allocation
    cfg.tree.max_depth = k
    tree = build_from_config(cfg.tree)
    vocab = int(cfg.target.random_config.vocab_size)
    prompt = torch.randint(0, vocab, (1, 8), generator=torch.Generator().manual_seed(seed))

    ar = greedy_generate(tiny_target, prompt, MAX_NEW)
    res = speculative_generate(
        tiny_target, tiny_drafter, tree, prompt, MAX_NEW,
        greedy=True, layer_index=int(cfg.data.layer_index), dynamic_tree=True,
    )
    if res.tokens != ar:
        idx = next(i for i, (a, b) in enumerate(zip(ar, res.tokens)) if a != b)
        ids = torch.cat([prompt[0], torch.tensor(ar[:idx], dtype=torch.long)]).unsqueeze(0)
        with torch.no_grad():
            logits = tiny_target(input_ids=ids).logits[0, -1]
        gap = float(torch.topk(logits, 2).values.diff().abs())
        assert gap < 1e-3, f"dynamic-tree lossless violation at {idx} (gap {gap:.2e})"
        pytest.skip(f"near-tie at {idx}")
    assert len(res.tokens) == MAX_NEW
    assert all(0 <= a <= k for a in res.accept_lengths)


def test_dynamic_is_deterministic(tiny_cfg, tiny_target, tiny_drafter):
    cfg = tiny_cfg.copy()
    cfg.tree.allocation = "geometric"
    cfg.tree.max_depth = 3
    tree = build_from_config(cfg.tree)
    prompt = torch.randint(0, int(cfg.target.random_config.vocab_size), (1, 8),
                           generator=torch.Generator().manual_seed(7))

    def go():
        return speculative_generate(tiny_target, tiny_drafter, tree, prompt, MAX_NEW,
                                    greedy=True, dynamic_tree=True)
    a, b = go(), go()
    assert a.tokens == b.tokens and a.accept_lengths == b.accept_lengths
