"""Verification acceptance rule: perfect drafter, adversarial drafter, edge cases."""

from __future__ import annotations

import pytest
import torch

from src.decode.tree import build_tree, chain
from src.decode.verify import verify_chain, verify_tree

V = 32


def _rand_probs(rows: int, g: torch.Generator, sharp: float = 1.0) -> torch.Tensor:
    return torch.softmax(torch.randn(rows, V, generator=g) * sharp, dim=-1)


def _draft_tree_tokens(tree, draft_probs, g):
    """Sample node tokens i.i.d. with replacement from each parent slot (theorem hypothesis)."""
    tokens = torch.full((tree.num_nodes,), -1, dtype=torch.long)
    for parent in range(-1, tree.num_nodes):
        kids = tree.children_of(parent)
        if kids:
            picks = torch.multinomial(draft_probs[parent + 1], len(kids), replacement=True, generator=g)
            for rank, node in enumerate(kids):
                tokens[node] = picks[rank]
    return tokens


# ---------- drafter == target accepts everything ----------

def test_chain_identical_dists_accepts_all():
    g = torch.Generator().manual_seed(0)
    k = 6
    probs = _rand_probs(k + 1, g)
    draft = probs[:k]
    tokens = torch.tensor([int(torch.multinomial(draft[i], 1, generator=g)) for i in range(k)])
    out = verify_chain(tokens, draft, probs, generator=g)
    assert out.num_accepted == k
    assert out.tokens[:k] == tokens.tolist()
    assert len(out.tokens) == k + 1  # + bonus
    assert not out.corrected
    assert out.path == list(range(k))


def test_tree_identical_dists_accepts_full_path():
    g = torch.Generator().manual_seed(1)
    tree = build_tree([3, 2, 2])
    probs = _rand_probs(tree.num_nodes + 1, g)
    tokens = _draft_tree_tokens(tree, probs, g)
    out = verify_tree(tree, tokens, probs, probs, generator=g)
    # q == p → the first candidate at every level is accepted with prob 1
    assert out.num_accepted == tree.max_depth
    assert len(out.tokens) == tree.max_depth + 1
    assert not out.corrected


def test_greedy_matching_drafter_accepts_all():
    g = torch.Generator().manual_seed(2)
    k = 5
    probs = _rand_probs(k + 1, g)
    tokens = probs[:k].argmax(dim=-1)
    out = verify_chain(tokens, probs[:k], probs, generator=g, greedy=True)
    assert out.num_accepted == k
    assert out.tokens[-1] == int(probs[k].argmax())


# ---------- deliberately bad drafter still yields valid tokens ----------

@pytest.mark.parametrize("seed", range(5))
def test_chain_adversarial_drafter_valid_output(seed):
    g = torch.Generator().manual_seed(seed)
    k = 6
    target = _rand_probs(k + 1, g)
    # drafter concentrated on the target's LEAST likely token
    worst = target[:k].argmin(dim=-1)
    draft = torch.full((k, V), 1e-6)
    draft[torch.arange(k), worst] = 1.0
    draft = draft / draft.sum(-1, keepdim=True)
    tokens = worst.clone()
    out = verify_chain(tokens, draft, target, generator=g)
    assert 1 <= len(out.tokens) <= k + 1
    assert all(0 <= t < V for t in out.tokens)
    assert out.num_accepted <= k
    assert len(out.tokens) == out.num_accepted + 1


@pytest.mark.parametrize("seed", range(5))
def test_tree_adversarial_drafter_valid_output(seed):
    g = torch.Generator().manual_seed(100 + seed)
    tree = build_tree([3, 2, 2])
    target = _rand_probs(tree.num_nodes + 1, g)
    draft = _rand_probs(tree.num_nodes + 1, g, sharp=4.0)  # confidently wrong
    tokens = _draft_tree_tokens(tree, draft, g)
    out = verify_tree(tree, tokens, draft, target, generator=g)
    assert 1 <= len(out.tokens) <= tree.max_depth + 1
    assert all(0 <= t < V for t in out.tokens)
    assert len(out.tokens) == out.num_accepted + 1
    # accepted path must be a real root-descending path
    for a, b in zip([-1] + out.path, out.path):
        assert b in tree.children_of(a)


def test_greedy_wrong_drafter_commits_target_argmax():
    g = torch.Generator().manual_seed(3)
    k = 4
    target = _rand_probs(k + 1, g)
    t_star = int(target[0].argmax())
    wrong = (t_star + 1) % V
    tokens = torch.full((k,), wrong, dtype=torch.long)
    out = verify_chain(tokens, target[:k], target, generator=g, greedy=True)
    assert out.num_accepted == 0
    assert out.tokens == [t_star]
    assert out.corrected


# ---------- edge cases ----------

def test_zero_draft_prob_token_handled():
    g = torch.Generator().manual_seed(4)
    k = 2
    target = _rand_probs(k + 1, g)
    draft = target[:k].clone()
    tokens = torch.tensor([5, 7])
    draft[0, 5] = 0.0  # drafted token with q=0 (broken proposal input)
    draft = draft / draft.sum(-1, keepdim=True)
    out = verify_chain(tokens, draft, target, generator=g)
    assert all(0 <= t < V for t in out.tokens)


def test_determinism_with_seeded_generator():
    tree = build_tree([2, 3, 2])
    g1 = torch.Generator().manual_seed(42)
    target = _rand_probs(tree.num_nodes + 1, g1)
    draft = _rand_probs(tree.num_nodes + 1, g1)
    tokens = _draft_tree_tokens(tree, draft, g1)
    out_a = verify_tree(tree, tokens, draft, target, generator=torch.Generator().manual_seed(7))
    out_b = verify_tree(tree, tokens, draft, target, generator=torch.Generator().manual_seed(7))
    assert out_a.tokens == out_b.tokens
    assert out_a.path == out_b.path


def test_shape_validation():
    tree = chain(3)
    g = torch.Generator().manual_seed(5)
    probs = _rand_probs(4, g)
    with pytest.raises(ValueError):
        verify_chain(torch.tensor([1, 2, 3]), probs[:2], probs, generator=g)
    with pytest.raises(ValueError):
        verify_tree(tree, torch.tensor([1, 2]), probs, probs, generator=g)
