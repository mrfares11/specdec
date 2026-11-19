"""Statistical losslessness: spec-decoded token distributions match the target.

Over many seeded trials with a genuinely different drafter distribution, the
empirical distribution of committed tokens must match the target distribution
within a tolerance set 2.5x above the expected multinomial noise floor.
Folded-normal per bin: E|p̂_i - p_i| ≈ sqrt(2 p_i (1 - p_i) / (π n)), summed
and halved ⇒ E[TV] ≲ 0.5 · sqrt(2 V / (π n)). Deterministic given the seeds.
(Known-broken residual-update variants measure TV ≈ 0.08–0.30 here, an order
of magnitude above the bound.)
"""

from __future__ import annotations

import torch

from src.decode.tree import build_tree
from src.decode.verify import verify_chain, verify_tree

V = 16


def _tv(counts: torch.Tensor, probs: torch.Tensor) -> float:
    emp = counts / counts.sum()
    return 0.5 * float((emp - probs).abs().sum())


def _noise_floor(n: int) -> float:
    """E[TV] of an n-sample empirical distribution vs its true V-bin multinomial."""
    return 0.5 * (2 * V / (torch.pi * n)) ** 0.5


def test_chain_first_and_second_token_distributions():
    g = torch.Generator().manual_seed(0)
    k = 3
    target = torch.softmax(torch.randn(k + 1, V, generator=g), dim=-1)
    # drafter: noticeably different (temperature-warped + noise), acceptance well below 1
    draft = torch.softmax(torch.randn(k, V, generator=g) + 0.5 * torch.log(target[:k]), dim=-1)

    n = 25_000
    first = torch.zeros(V)
    second = torch.zeros(V)
    n_second = 0
    for _ in range(n):
        tokens = torch.stack(
            [torch.multinomial(draft[i], 1, generator=g)[0] for i in range(k)]
        )
        out = verify_chain(tokens, draft, target, generator=g)
        first[out.tokens[0]] += 1
        if len(out.tokens) >= 2:
            second[out.tokens[1]] += 1
            n_second += 1

    tol1 = 2.5 * _noise_floor(n)
    assert _tv(first, target[0]) < tol1, f"first-token TV {_tv(first, target[0]):.4f} > {tol1:.4f}"
    # Given the first draft was accepted, the second committed token ~ p_1 exactly.
    assert n_second > n // 10, "acceptance collapsed; test setup broken"
    tol2 = 2.5 * _noise_floor(n_second)
    assert _tv(second, target[1]) < tol2, f"second-token TV {_tv(second, target[1]):.4f} > {tol2:.4f}"


def test_tree_first_token_distribution():
    g = torch.Generator().manual_seed(1)
    tree = build_tree([3, 2, 2])
    n_slots = tree.num_nodes + 1
    target = torch.softmax(torch.randn(n_slots, V, generator=g), dim=-1)
    draft = torch.softmax(torch.randn(n_slots, V, generator=g) + 0.5 * torch.log(target), dim=-1)

    n = 20_000
    first = torch.zeros(V)
    for _ in range(n):
        tokens = torch.full((tree.num_nodes,), -1, dtype=torch.long)
        for parent in range(-1, tree.num_nodes):
            kids = tree.children_of(parent)
            if kids:
                picks = torch.multinomial(draft[parent + 1], len(kids), replacement=True, generator=g)
                for rank, node in enumerate(kids):
                    tokens[node] = picks[rank]
        out = verify_tree(tree, tokens, draft, target, generator=g)
        first[out.tokens[0]] += 1

    tol = 2.5 * _noise_floor(n)
    tv = _tv(first, target[0])
    assert tv < tol, f"tree first-token TV {tv:.4f} > {tol:.4f} — tree verification is NOT lossless"


def test_tree_broken_verifier_would_fail_here():
    """Sanity: committing the drafter's top pick (no rejection) must violate the bound.

    Guards the test's own power — if this ever passes, the tolerance is meaningless.
    """
    g = torch.Generator().manual_seed(2)
    target = torch.softmax(torch.randn(V, generator=g), dim=-1)
    draft = torch.softmax(torch.randn(V, generator=g), dim=-1)
    n = 20_000
    first = torch.zeros(V)
    for _ in range(n):
        first[int(torch.multinomial(draft, 1, generator=g))] += 1  # "accept everything" bug
    assert _tv(first, target) > 2.5 * _noise_floor(n)
