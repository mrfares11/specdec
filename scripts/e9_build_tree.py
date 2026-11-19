"""E9 --- offline-optimised static draft trees from development-prompt statistics.

The standard construction (Medusa / EAGLE-1 / Sequoia-style): estimate, on the
DEVELOPMENT prompts, the probability p_d(r) that the target's greedy token has
rank r in the drafter's distribution at depth d given that the path was
correct so far (marginals of the rollout ranks from e9_rollout_stats.py,
made non-increasing in r); score every candidate node (r_1, ..., r_d) by
V = prod_j p_j(r_j), the probability the greedy path passes through it; and
pick nodes greedily by V. Because V never increases along a path, the picked
set is prefix-closed and each parent's children are its top-ranked tokens, so
the tree is a valid static topology for the repo's drafting code.

Two variants:
  same-profile  -- keep the paper's per-depth widths b_d and choose, at each
                   depth, the b_d highest-V extensions of the selected parents
                   (data-driven quotas under the identical depth budget);
  free-profile  -- the N highest-V nodes with depth <= K (the classic
                   offline-optimal static tree at the same node budget).

    python scripts/e9_build_tree.py --stats <dev rollout.pt> [--stats ...] \
        --K 4 --nodes 256 --variant same-profile --out tree.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decode.tree import build_from_parents, build_tree, geometric_allocation  # noqa: E402


def rank_marginals(stats_files: List[str], K: int, max_rank: int = 256) -> torch.Tensor:
    """p[d, r] = P(rank at depth d+1 == r), non-increasing in r, from pooled dev roots."""
    counts = torch.zeros(K, max_rank + 1)
    n = 0
    for f in stats_files:
        per = torch.load(f)
        for v in per.values():
            rk = v["ranks"][:, :K].long().clamp(max=max_rank)
            for d in range(K):
                counts[d] += torch.bincount(rk[:, d], minlength=max_rank + 1).float()
            n += rk.shape[0]
    # additive prior of 0.1 pseudo-counts per rank: negligible where counts are
    # large, and it keeps every rank strictly positive so depth quotas always
    # fill (tail ranks then order by rank, which is what a practitioner would do).
    p = (counts[:, :max_rank] + 0.1) / (max(n, 1) + 0.1 * max_rank)
    # ranks are an ordering, so the true probability is non-increasing in r;
    # enforce it on the noisy empirical estimate (pool-adjacent-violators).
    for d in range(K):
        row = p[d].tolist()
        blocks: List[Tuple[float, int]] = []  # (sum, size)
        for x in row:
            blocks.append((x, 1))
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
                s2, n2 = blocks.pop()
                s1, n1 = blocks.pop()
                blocks.append((s1 + s2, n1 + n2))
        out = []
        for s, m in blocks:
            out += [s / m] * m
        p[d] = torch.tensor(out)
    return p, n


def build(p: torch.Tensor, K: int, nodes: int, variant: str, profile: List[int]):
    """Return (parents BFS list, per-node value list, expected accepted tokens)."""
    # node = (depth d (1-based), rank tuple); value V
    selected: List[Tuple[Tuple[int, ...], float]] = []
    if variant == "free-profile":
        heap = [(-float(p[0, r]), (r,)) for r in range(p.shape[1])]
        heapq.heapify(heap)
        while heap and len(selected) < nodes:
            negv, tup = heapq.heappop(heap)
            selected.append((tup, -negv))
            d = len(tup)
            if d < K:
                for r in range(p.shape[1]):
                    v = -negv * float(p[d, r])
                    if v > 0:
                        heapq.heappush(heap, (-v, tup + (r,)))
                        # children are pushed lazily in rank order? No: all children
                        # pushed; heap order handles it. Bounded by nodes*max_rank.
    elif variant == "same-profile":
        parents_lvl: List[Tuple[Tuple[int, ...], float]] = [((), 1.0)]
        for d in range(K):
            cands = []
            for tup, v in parents_lvl:
                for r in range(p.shape[1]):
                    vv = v * float(p[d, r])
                    if vv > 0:
                        cands.append((vv, tup + (r,)))
            cands.sort(key=lambda x: -x[0])
            chosen = [(tup, vv) for vv, tup in cands[: profile[d]]]
            selected += chosen
            parents_lvl = chosen
    else:
        raise ValueError(variant)
    # BFS order: by depth, then by parent's index, then rank
    by_depth: Dict[int, List[Tuple[Tuple[int, ...], float]]] = {}
    for tup, v in selected:
        by_depth.setdefault(len(tup), []).append((tup, v))
    index: Dict[Tuple[int, ...], int] = {}
    parents: List[int] = []
    values: List[float] = []
    for d in sorted(by_depth):
        lvl = by_depth[d]
        lvl.sort(key=lambda x: (index.get(x[0][:-1], -1), x[0][-1]))
        for tup, v in lvl:
            par = index.get(tup[:-1], -1) if d > 1 else -1
            if d > 1 and tup[:-1] not in index:
                raise RuntimeError(f"not prefix-closed: {tup}")
            index[tup] = len(parents)
            parents.append(par)
            values.append(v)
    # children of each parent must be a rank prefix 0..k-1
    kids: Dict[int, List[int]] = {}
    for tup, i in index.items():
        kids.setdefault(index.get(tup[:-1], -1) if len(tup) > 1 else -1, []).append(tup[-1])
    for par, rks in kids.items():
        assert sorted(rks) == list(range(len(rks))), (par, sorted(rks))
    tree = build_from_parents(parents)  # validates BFS consistency
    return parents, values, sum(values), torch.bincount(tree.depth).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="append", required=True, help="dev rollout.pt (repeatable, pooled)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=256)
    ap.add_argument("--variant", choices=["same-profile", "free-profile"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, n_roots = rank_marginals(a.stats, a.K)
    profile = geometric_allocation(a.nodes, a.K, 0.5)
    parents, values, expected, widths = build(p, a.K, a.nodes, a.variant, profile)
    # the paper's chain-static tree under the same model, for reference
    chain = build_tree(profile)
    chain_vals = []
    # value of chain nodes: each parent extended by rank 0 only; depth-1 ranks 0..b1-1
    stack = {-1: ((), 1.0)}
    for i in range(chain.num_nodes):
        par = int(chain.parent[i])
        ptup, pv = stack[par]
        r = list(chain.children_of(par)).index(i)
        d = len(ptup)
        v = pv * float(p[d, r])
        stack[i] = (ptup + (r,), v)
        chain_vals.append(v)
    spec = {"K": a.K, "nodes": a.nodes, "variant": a.variant, "parents": parents,
            "widths": widths, "n_dev_roots": n_roots, "model_expected_accept": expected,
            "chain_model_expected_accept": sum(chain_vals),
            "p_top20_by_depth": [[round(float(x), 4) for x in p[d, :20]] for d in range(a.K)],
            "stats": a.stats}
    Path(a.out).write_text(json.dumps(spec))
    print(f"[E9] {a.variant} K={a.K} N={a.nodes}: widths={widths}, model-expected accept "
          f"{expected:.3f} (chain-static under the same model: {sum(chain_vals):.3f}); "
          f"from {n_roots} dev roots -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
