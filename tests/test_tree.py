"""DraftTree correctness: allocation, ancestor mask, positions, retrieval paths."""

from __future__ import annotations

import pytest
import torch

from src.decode.tree import (
    DraftTree,
    build_from_config,
    build_tree,
    chain,
    geometric_allocation,
    get_allocation,
    uniform_allocation,
)

CASES = [
    ([1, 1, 1], "chain-3"),
    ([3, 3, 3, 3], "uniform-12"),
    ([5, 3, 2, 2], "geometric-ish"),
    ([2, 4, 4], "widening"),
    ([4, 2, 1], "narrowing"),
    ([1, 6, 2], "bulge"),
]


def _ancestors(tree: DraftTree, i: int) -> set:
    """Independent ancestor computation via parent pointers."""
    out = set()
    j = int(tree.parent[i].item())
    while j >= 0:
        out.add(j)
        j = int(tree.parent[j].item())
    return out


@pytest.mark.parametrize("counts,label", CASES)
def test_mask_is_exactly_ancestors_plus_self(counts, label):
    tree = build_tree(counts)
    n = tree.num_nodes
    assert n == sum(counts)
    for i in range(n):
        allowed = _ancestors(tree, i) | {i}
        for j in range(n):
            assert bool(tree.attn_mask[i, j]) == (j in allowed), (
                f"{label}: node {i} vs {j} — attends outside its ancestor path"
            )


@pytest.mark.parametrize("counts,label", CASES)
def test_mask_never_crosses_branches(counts, label):
    """No node sees a sibling/cousin: mask ∧ different-subtree ⇒ False."""
    tree = build_tree(counts)
    for i in range(tree.num_nodes):
        for j in range(tree.num_nodes):
            if i == j:
                continue
            if bool(tree.attn_mask[i, j]):
                assert j in _ancestors(tree, i)
                assert j < i, "BFS order: ancestors must have smaller indices"


@pytest.mark.parametrize("counts,label", CASES)
def test_position_ids_equal_depth(counts, label):
    tree = build_tree(counts)
    # position offset == depth (0-based), by definition
    assert torch.equal(tree.position_offsets(), tree.depth)
    prefix = 17
    pos = tree.position_ids(prefix)
    assert pos.shape == (1, tree.num_nodes)
    assert torch.equal(pos[0], tree.depth + prefix)
    # depth is consistent with parents
    for i in range(tree.num_nodes):
        p = int(tree.parent[i].item())
        if p == -1:
            assert int(tree.depth[i]) == 0
        else:
            assert int(tree.depth[i]) == int(tree.depth[p]) + 1


@pytest.mark.parametrize("counts,label", CASES)
def test_retrieve_indices_reconstruct_paths(counts, label):
    tree = build_tree(counts)
    leaves = [i for i in range(tree.num_nodes) if not tree.children_of(i)]
    assert tree.num_leaves == len(leaves)
    seen_leaves = set()
    for row in tree.retrieve_indices:
        path = [int(v) for v in row if v >= 0]
        assert path, "empty path"
        # padding only at the tail
        vals = row.tolist()
        assert vals[len(path):] == [-1] * (len(vals) - len(path))
        # path starts at a depth-0 node and follows parent pointers
        assert int(tree.parent[path[0]].item()) == -1
        for a, b in zip(path, path[1:]):
            assert int(tree.parent[b].item()) == a
        seen_leaves.add(path[-1])
    assert seen_leaves == set(leaves)
    # every node lies on at least one root→leaf path
    on_path = set()
    for p in tree.paths():
        on_path.update(p)
    assert on_path == set(range(tree.num_nodes))


def test_attn_bias_layout():
    tree = build_tree([2, 2])
    prefix = 5
    bias = tree.attn_bias(prefix, dtype=torch.float32)
    n = tree.num_nodes
    assert bias.shape == (1, 1, n, prefix + n)
    assert (bias[0, 0, :, :prefix] == 0).all(), "prefix must be fully visible"
    neg = torch.finfo(torch.float32).min
    block = bias[0, 0, :, prefix:]
    assert ((block == 0) == tree.attn_mask).all()
    assert (block[~tree.attn_mask] == neg).all()


def test_uniform_allocation_counts():
    assert uniform_allocation(12, 4) == [3, 3, 3, 3]
    assert uniform_allocation(10, 4) == [3, 3, 2, 2]
    assert uniform_allocation(4, 4) == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        uniform_allocation(3, 4)  # cannot give every depth a node


def test_geometric_allocation_counts():
    counts = geometric_allocation(12, 4, decay=0.5)
    assert sum(counts) == 12
    assert all(c >= 1 for c in counts)
    assert counts == sorted(counts, reverse=True), "decay<1 must be shallow-heavy"
    # decay > 1 flips the shape deep-heavy
    deep = geometric_allocation(12, 4, decay=2.0)
    assert deep == sorted(deep), f"decay>1 must be deep-heavy, got {deep}"


def test_allocation_registry_pluggable(tiny_cfg):
    with pytest.raises(KeyError):
        get_allocation("nope")
    for name in ("uniform", "geometric"):
        cfg = tiny_cfg.copy()
        cfg.tree.allocation = name
        tree = build_from_config(cfg.tree)
        assert tree.num_nodes == int(cfg.tree.num_nodes)
        assert tree.max_depth == int(cfg.tree.max_depth)


def test_chain_is_single_path():
    tree = chain(5)
    assert tree.num_nodes == 5
    assert tree.num_leaves == 1
    assert tree.paths() == [[0, 1, 2, 3, 4]]
    assert tree.max_depth == 5


def test_build_tree_rejects_bad_counts():
    with pytest.raises(ValueError):
        build_tree([])
    with pytest.raises(ValueError):
        build_tree([2, 0, 1])


def test_build_from_config_parents_file(tmp_path):
    """An explicit topology file overrides the allocation and must match K and N."""
    import json

    import pytest
    from omegaconf import OmegaConf

    from src.decode.tree import build_from_config

    parents = [-1, -1, -1, 0, 0, 1, 3, 3]  # widths (3, 3, 2), depth 3, 8 nodes
    f = tmp_path / "tree.json"
    f.write_text(json.dumps({"parents": parents}))
    cfg = OmegaConf.create({"num_nodes": 8, "max_depth": 3, "allocation": "uniform",
                            "parents_file": str(f)})
    t = build_from_config(cfg)
    assert t.parent.tolist() == parents and t.max_depth == 3
    assert t.children_of(0) == (3, 4) and t.children_of(3) == (6, 7)
    # shallower than the horizon is allowed (depth at most K); deeper is not
    assert build_from_config(OmegaConf.merge(cfg, {"max_depth": 4})).max_depth == 3
    with pytest.raises(ValueError, match="depth"):
        build_from_config(OmegaConf.merge(cfg, {"max_depth": 2}))
    with pytest.raises(ValueError, match="nodes"):
        build_from_config(OmegaConf.merge(cfg, {"num_nodes": 12}))
    # null / absent keeps the registry path
    t2 = build_from_config(OmegaConf.create({"num_nodes": 8, "max_depth": 3, "allocation": "uniform",
                                             "parents_file": None}))
    assert t2.num_nodes == 8
