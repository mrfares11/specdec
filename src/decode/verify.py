"""Lossless verification: chain and tree acceptance rules.

LOSSLESSNESS IS A HARD INVARIANT. Both verifiers guarantee that committed
tokens are distributed exactly as sampling (or argmax, in greedy mode) from the
target — guarded by ``tests/test_lossless_stats.py`` and the greedy-equivalence
test in ``tests/test_generate.py``.

Notation: ``p`` = target distribution, ``q`` = drafter (proposal) distribution.

Chain rule (Leviathan et al., 2023): accept draft ``x_i`` with probability
``min(1, p_i(x_i) / q_i(x_i))``; on rejection, commit a sample from the residual
``norm(max(p_i - q_i, 0))`` and stop; on full acceptance, append a bonus token
sampled from the final target slot.

Tree rule (SpecInfer-style recursive rejection): at the current node, try its
children in order. Maintain ``r`` (initially ``p``); accept child with token
``x`` w.p. ``min(1, r(x) / q(x))``; after each sibling rejection update
``r ← norm(max(r - q, 0))``. If no child is accepted, commit a sample from
``r``. The theorem requires sibling tokens drawn i.i.d. WITH replacement
from ``q`` — see ``generate._draft``.

RNG CONTRACT: all randomness is drawn on CPU with a (CPU) ``torch.Generator``,
and both verifiers move their inputs to CPU on entry. torch requires a
generator's device to match the tensor device of ``rand``/``multinomial``, and
no single generator device satisfies the mixed usage here on CUDA — CPU-side
verification sidesteps that entirely, keeps seeded runs reproducible across
devices, and costs only [N, V]-sized transfers per step (device-resident RNG
is a pod-phase optimization, not a scaffold concern).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .tree import DraftTree

_EPS = 1e-9


@dataclass
class VerifyOutcome:
    """Result of one verification step.

    Attributes:
        tokens: all tokens committed this step — the accepted draft tokens plus
            exactly one final token (a residual/argmax correction on rejection,
            or a bonus continuation after full acceptance). Never empty.
        num_accepted: number of DRAFT tokens accepted (excludes the final token).
        path: accepted tree-node indices in root→leaf order (for chains this is
            ``[0, 1, ..., num_accepted - 1]``).
        corrected: True if the final token came from a rejection, False if it is
            a bonus token after full acceptance.
    """

    tokens: List[int]
    num_accepted: int
    path: List[int]
    corrected: bool


def _sample(probs: torch.Tensor, generator: Optional[torch.Generator]) -> int:
    """One multinomial draw from a 1-D probability vector."""
    return int(torch.multinomial(probs, 1, generator=generator).item())


def _accept(target_p: float, draft_p: float, generator: Optional[torch.Generator]) -> bool:
    """Bernoulli(min(1, target_p / draft_p)) acceptance test."""
    if draft_p <= 0.0:
        # A correct proposal never emits a q=0 token; if handed one anyway,
        # accept only when the target actually supports it.
        return target_p > 0.0
    if target_p >= draft_p:
        return True
    u = float(torch.rand((), generator=generator).item())
    return u < target_p / draft_p


def _residual(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> Optional[torch.Tensor]:
    """``norm(max(target - draft, 0))``; None if numerically all mass cancels."""
    r = (target_probs - draft_probs).clamp_min(0.0)
    s = float(r.sum().item())
    if s <= _EPS:
        return None
    return r / s


def verify_chain(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    greedy: bool = False,
) -> VerifyOutcome:
    """Sequential accept over a drafted chain.

    Args:
        draft_tokens: long [K] — drafted tokens, ``x_i`` sampled from ``q_i``.
        draft_probs: float [K, V] — ``q_i`` for each draft position.
        target_probs: float [K + 1, V] — ``p_i`` for each draft position plus
            one bonus slot for the continuation after full acceptance.
        generator: RNG for reproducibility (sampling mode).
        greedy: exact-match acceptance against ``argmax p_i`` (lossless w.r.t.
            greedy decoding); sampling mode is lossless w.r.t. temperature
            sampling.
    """
    k = int(draft_tokens.shape[0])
    if target_probs.shape[0] != k + 1:
        raise ValueError(f"target_probs must have K+1={k + 1} rows, got {target_probs.shape[0]}")
    if draft_probs.shape[0] != k:
        raise ValueError(f"draft_probs must have K={k} rows, got {draft_probs.shape[0]}")
    # RNG contract (module docstring): verification happens on CPU.
    draft_tokens = draft_tokens.detach().cpu()
    draft_probs = draft_probs.detach().float().cpu()
    target_probs = target_probs.detach().float().cpu()

    tokens: List[int] = []
    path: List[int] = []
    for i in range(k):
        x = int(draft_tokens[i].item())
        if greedy:
            t_star = int(target_probs[i].argmax().item())
            if x != t_star:
                tokens.append(t_star)
                return VerifyOutcome(tokens, i, path, corrected=True)
        else:
            if not _accept(float(target_probs[i, x].item()), float(draft_probs[i, x].item()), generator):
                r = _residual(target_probs[i], draft_probs[i])
                probs = r if r is not None else target_probs[i]
                tokens.append(_sample(probs, generator))
                return VerifyOutcome(tokens, i, path, corrected=True)
        tokens.append(x)
        path.append(i)

    bonus_probs = target_probs[k]
    bonus = int(bonus_probs.argmax().item()) if greedy else _sample(bonus_probs, generator)
    tokens.append(bonus)
    return VerifyOutcome(tokens, k, path, corrected=False)


def verify_tree_greedy(
    tree: DraftTree,
    node_tokens: torch.Tensor,
    slot_argmax: torch.Tensor,
) -> VerifyOutcome:
    """Greedy tree verification from precomputed per-slot argmaxes.

    Equivalent to ``verify_tree(..., greedy=True)`` but consumes only
    ``slot_argmax`` (long [N + 1]: target argmax per slot, slot layout as in
    :func:`verify_tree`) — so callers can keep full logits on the device and
    transfer N + 1 integers instead of [N + 1, V] probabilities. This is the
    serial-CPU reduction step of the RQ3 overlap work; greedy losslessness is
    still guarded by the exact AR-equivalence test.
    """
    n = tree.num_nodes
    if node_tokens.shape[0] != n or slot_argmax.shape[0] != n + 1:
        raise ValueError("shape mismatch (node_tokens [N], slot_argmax [N+1])")
    node_tokens = node_tokens.detach().cpu()
    slot_argmax = slot_argmax.detach().cpu()

    tokens: List[int] = []
    path: List[int] = []
    cur = -1
    while True:
        slot = cur + 1
        kids = tree.children_of(cur)
        t_star = int(slot_argmax[slot].item())
        if not kids:
            tokens.append(t_star)  # bonus after an accepted leaf
            return VerifyOutcome(tokens, len(path), path, corrected=False)
        accepted: Optional[int] = None
        for c in kids:
            if int(node_tokens[c].item()) == t_star:
                accepted = c
                break
        if accepted is None:
            tokens.append(t_star)
            return VerifyOutcome(tokens, len(path), path, corrected=True)
        tokens.append(t_star)
        path.append(accepted)
        cur = accepted


def verify_tree(
    tree: DraftTree,
    node_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    greedy: bool = False,
) -> VerifyOutcome:
    """Multi-candidate verification over a draft tree.

    Slot layout (both prob tensors are float [N + 1, V]):
        slot 0     — distribution after the root (over depth-0 candidates);
        slot i + 1 — distribution after node ``i`` (over its children).
    ``node_tokens`` is long [N]. In sampling mode, the children of each node
    must have been drawn i.i.d. with replacement from ``draft_probs[slot]`` —
    that is the hypothesis of the losslessness theorem (duplicates are fine
    and are handled by the residual updates).

    Walks from the root: tries children in order with recursive-rejection
    acceptance; descends on accept; on exhausting siblings commits a residual
    sample (or ``argmax p`` in greedy mode); at an accepted leaf commits a
    bonus token from the leaf's slot.
    """
    n = tree.num_nodes
    if node_tokens.shape[0] != n:
        raise ValueError(f"node_tokens must have N={n} entries, got {node_tokens.shape[0]}")
    if draft_probs.shape[0] != n + 1 or target_probs.shape[0] != n + 1:
        raise ValueError(f"prob tensors must have N+1={n + 1} rows (slot layout)")
    # RNG contract (module docstring): verification happens on CPU.
    node_tokens = node_tokens.detach().cpu()
    draft_probs = draft_probs.detach().float().cpu()
    target_probs = target_probs.detach().float().cpu()

    tokens: List[int] = []
    path: List[int] = []
    cur = -1  # root
    while True:
        slot = cur + 1
        kids = tree.children_of(cur)
        p = target_probs[slot]

        if not kids:  # accepted a leaf: bonus continuation
            bonus = int(p.argmax().item()) if greedy else _sample(p, generator)
            tokens.append(bonus)
            return VerifyOutcome(tokens, len(path), path, corrected=False)

        accepted: Optional[int] = None
        if greedy:
            t_star = int(p.argmax().item())
            for c in kids:
                if int(node_tokens[c].item()) == t_star:
                    accepted = c
                    break
            if accepted is None:
                tokens.append(t_star)
                return VerifyOutcome(tokens, len(path), path, corrected=True)
        else:
            q = draft_probs[slot]
            r: Optional[torch.Tensor] = p.clone()
            for c in kids:
                x = int(node_tokens[c].item())
                assert r is not None
                if _accept(float(r[x].item()), float(q[x].item()), generator):
                    accepted = c
                    break
                r = _residual(r, q)
                if r is None:
                    break  # numerically exhausted; fall through to fallback
            if accepted is None:
                probs = r if r is not None else p
                tokens.append(_sample(probs, generator))
                return VerifyOutcome(tokens, len(path), path, corrected=True)

        tokens.append(int(node_tokens[accepted].item()))
        path.append(accepted)
        cur = accepted
