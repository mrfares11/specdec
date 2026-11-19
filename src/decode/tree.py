"""Draft-tree topology for speculative decoding — the RQ1 core object.

A :class:`DraftTree` is a *static topology*: node structure, ancestor-preserving
attention mask, depth-based positions, and root→leaf retrieval paths. Tokens are
filled in at draft time (``src/decode/generate.py``).

Conventions
-----------
- The tree ROOT is the last committed token. It already lives in the KV cache
  and is NOT a node; nodes are the drafted candidates hanging below it.
- ``depth`` is 0-based: depth-0 nodes are candidate next tokens (children of
  the root). A node at depth ``d`` sits at absolute position
  ``context_len + d`` — position offset == depth, by construction.
- Nodes are BFS-ordered (all of depth ``d`` before any of depth ``d+1``), so a
  node's ancestors always have smaller indices and per-level slices of the node
  list are contiguous.

Fan-out allocation across depth is pluggable: an allocation strategy maps
``(num_nodes, max_depth) -> per-depth node counts`` and is registered by name
(``uniform``, ``geometric``, or your own via :func:`register_allocation`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from omegaconf import DictConfig

AllocationFn = Callable[[int, int], List[int]]

_ALLOCATIONS: Dict[str, Callable[..., List[int]]] = {}


def register_allocation(name: str) -> Callable[[Callable[..., List[int]]], Callable[..., List[int]]]:
    """Decorator: register an allocation strategy under ``name``.

    The strategy receives ``(num_nodes, max_depth, **params)`` and returns
    per-depth node counts (length ``max_depth``, each >= 1, summing to
    ``num_nodes``). Select it via ``cfg.tree.allocation`` and put its params in
    a same-named subsection of ``cfg.tree``.
    """

    def deco(fn: Callable[..., List[int]]) -> Callable[..., List[int]]:
        _ALLOCATIONS[name] = fn
        return fn

    return deco


def get_allocation(name: str, **params) -> AllocationFn:
    """Look up a registered strategy and bind its params."""
    if name not in _ALLOCATIONS:
        raise KeyError(f"unknown allocation '{name}'; registered: {sorted(_ALLOCATIONS)}")
    fn = _ALLOCATIONS[name]
    return lambda num_nodes, max_depth: fn(num_nodes, max_depth, **params)


def _apportion(weights: Sequence[float], total: int) -> List[int]:
    """Largest-remainder apportionment with a minimum of 1 per bucket.

    Guarantees ``sum == total`` and every bucket >= 1; remainder ties break
    toward shallower depths (earlier buckets).
    """
    n = len(weights)
    if n == 0:
        raise ValueError("need at least one depth")
    if total < n:
        raise ValueError(f"cannot place {total} nodes across {n} depths with >=1 per depth")
    spare = total - n
    s = float(sum(weights))
    if not math.isfinite(s) or s <= 0:
        raise ValueError(f"weights must have a positive finite sum, got {s}")
    raw = [w / s * spare for w in weights]
    base = [int(x) for x in raw]
    rem = spare - sum(base)
    order = sorted(range(n), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return [1 + b for b in base]


@register_allocation("uniform")
def uniform_allocation(num_nodes: int, max_depth: int) -> List[int]:
    """Even split across depths; remainders go to shallower depths."""
    return _apportion([1.0] * max_depth, num_nodes)


@register_allocation("geometric")
def geometric_allocation(num_nodes: int, max_depth: int, decay: float = 0.5) -> List[int]:
    """Per-depth counts ∝ ``decay**depth`` (shallow-heavy for decay < 1), min 1 per depth."""
    if decay <= 0:
        raise ValueError(f"decay must be > 0, got {decay}")
    return _apportion([decay**d for d in range(max_depth)], num_nodes)


@dataclass(frozen=True)
class DraftTree:
    """Static draft-tree topology (token-free). See module docstring for conventions."""

    parent: torch.Tensor
    """long [N] — parent node index; -1 for depth-0 nodes (children of the root)."""

    depth: torch.Tensor
    """long [N] — 0-based depth; node position offset from context length == depth."""

    attn_mask: torch.Tensor
    """bool [N, N] — ``[i, j]`` True iff node i may attend node j (self + ancestors only)."""

    retrieve_indices: torch.Tensor
    """long [num_leaves, max_depth] — root→leaf node paths, -1-padded at the tail."""

    children: Tuple[Tuple[int, ...], ...]
    """``children[0]`` = depth-0 nodes; ``children[i + 1]`` = children of node i."""

    @property
    def num_nodes(self) -> int:
        return int(self.parent.shape[0])

    @property
    def num_leaves(self) -> int:
        return int(self.retrieve_indices.shape[0])

    @property
    def max_depth(self) -> int:
        """Number of depth levels (== K for a horizon-K tree)."""
        return int(self.depth.max().item()) + 1

    def children_of(self, node: int) -> Tuple[int, ...]:
        """Children of ``node``; pass -1 for the root."""
        return self.children[node + 1]

    def position_offsets(self) -> torch.Tensor:
        """Per-node position offset from the current context length (== depth)."""
        return self.depth

    def position_ids(self, prefix_len: int) -> torch.Tensor:
        """Absolute position ids [1, N] for a context of ``prefix_len`` tokens."""
        return (self.depth + prefix_len).unsqueeze(0)

    def attn_bias(
        self,
        prefix_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Additive attention bias [1, 1, N, prefix_len + N].

        0 where visible, ``finfo(dtype).min`` where blocked. The committed
        prefix is fully visible to every node; within the tree block a node
        sees exactly itself and its ancestors.
        """
        n = self.num_nodes
        neg = torch.finfo(dtype).min
        bias = torch.zeros(1, 1, n, prefix_len + n, dtype=dtype, device=device)
        block = torch.full((n, n), neg, dtype=dtype, device=device)
        block.masked_fill_(self.attn_mask.to(device), 0.0)
        bias[0, 0, :, prefix_len:] = block
        return bias

    def paths(self) -> List[List[int]]:
        """Root→leaf node-index paths (unpadded), one per leaf."""
        out: List[List[int]] = []
        for row in self.retrieve_indices.tolist():
            out.append([i for i in row if i >= 0])
        return out


def build_tree(counts: Sequence[int]) -> DraftTree:
    """Build a BFS-ordered :class:`DraftTree` from per-depth node counts.

    Children at depth ``d+1`` are distributed over depth-``d`` parents as evenly
    as possible, with earlier (higher-ranked) parents receiving the extras; if
    there are fewer children than parents, later parents become leaves.
    """
    counts = [int(c) for c in counts]
    if not counts:
        raise ValueError("counts must be non-empty")
    if any(c < 1 for c in counts):
        raise ValueError(f"every depth needs >= 1 node, got {counts}")

    parent: List[int] = []
    prev: List[int] = [-1]  # conceptual root
    node = 0
    for c in counts:
        q, r = divmod(c, len(prev))
        level: List[int] = []
        for j, p in enumerate(prev):
            for _ in range(q + (1 if j < r else 0)):
                parent.append(p)
                level.append(node)
                node += 1
        prev = level

    return build_from_parents(parent)


def build_from_parents(parent: Sequence[int]) -> DraftTree:
    """Build a :class:`DraftTree` from a BFS-ordered parent list.

    ``parent[i]`` is the parent node index of node ``i`` (-1 = child of the
    root). Requires ``parent[i] < i`` (BFS/topological order) — this is the
    entry point for per-step DYNAMIC topologies, where drafting decides the
    structure and this function materializes masks and retrieval paths.
    """
    parent = [int(p) for p in parent]
    n = len(parent)
    if n == 0:
        raise ValueError("need at least one node")
    depth: List[int] = []
    for i, p in enumerate(parent):
        if p >= i or p < -1:
            raise ValueError(f"parent[{i}]={p} is not BFS-ordered (need -1 <= p < i)")
        depth.append(0 if p == -1 else depth[p] + 1)
    if any(depth[i] > depth[i + 1] for i in range(n - 1)):
        raise ValueError("nodes must be BFS-ordered (non-decreasing depth)")

    kids: List[List[int]] = [[] for _ in range(n + 1)]
    for i, p in enumerate(parent):
        kids[p + 1].append(i)
    children = tuple(tuple(k) for k in kids)

    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j >= 0:
            mask[i, j] = True
            j = parent[j]

    leaves = [i for i in range(n) if not children[i + 1]]
    depth_levels = max(depth) + 1
    ri = torch.full((len(leaves), depth_levels), -1, dtype=torch.long)
    for row, leaf in enumerate(leaves):
        path: List[int] = []
        j = leaf
        while j >= 0:
            path.append(j)
            j = parent[j]
        path.reverse()
        ri[row, : len(path)] = torch.tensor(path, dtype=torch.long)

    return DraftTree(
        parent=torch.tensor(parent, dtype=torch.long),
        depth=torch.tensor(depth, dtype=torch.long),
        attn_mask=mask,
        retrieve_indices=ri,
        children=children,
    )


def build_from_config(tree_cfg: DictConfig) -> DraftTree:
    """``cfg.tree`` → :class:`DraftTree` via the allocation registry.

    Strategy params live in a subsection named after the strategy
    (e.g. ``tree.geometric.decay`` for ``allocation: geometric``).
    """
    parents_file = tree_cfg.get("parents_file", None)
    if parents_file:
        # An explicit static topology (e.g. one fitted offline by
        # scripts/e9_build_tree.py). The file pins its own depth and budget;
        # they must agree with the run's, so a K sweep cannot silently pair a
        # depth-4 tree with K=8.
        import json

        spec = json.loads(open(str(parents_file), encoding="utf-8").read())
        tree = build_from_parents(spec["parents"])
        # A fitted tree may use fewer levels than the run's horizon (it is
        # optimised under "depth at most K"), never more.
        if tree.max_depth > int(tree_cfg.max_depth):
            raise ValueError(
                f"parents_file {parents_file} has depth {tree.max_depth}, run has max_depth={tree_cfg.max_depth}")
        if tree.num_nodes != int(tree_cfg.num_nodes):
            raise ValueError(
                f"parents_file {parents_file} has {tree.num_nodes} nodes, run has num_nodes={tree_cfg.num_nodes}")
        return tree
    name = str(tree_cfg.allocation)
    params: Dict[str, object] = {}
    if name in tree_cfg and tree_cfg[name] is not None:
        params = {str(k): v for k, v in dict(tree_cfg[name]).items()}
    alloc = get_allocation(name, **params)
    counts = alloc(int(tree_cfg.num_nodes), int(tree_cfg.max_depth))
    return build_tree(counts)


def chain(k: int) -> DraftTree:
    """Degenerate tree: a single path of length ``k`` (classic chain drafting)."""
    return build_tree([1] * k)
