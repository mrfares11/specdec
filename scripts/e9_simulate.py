"""E9 --- greedy-acceptance simulator that needs only the drafter.

Under greedy verification the committed sequence IS the target's greedy
continuation (exactly, in exact arithmetic), and the accepted path at every
step is the longest tree path that matches that continuation. So once a
prompt's greedy continuation S and the target's features along it are known,
every step of the commit-fold decode loop (src/decode/generate.py,
``_generate_folded``) can be replayed with the target replaced by lookups:

  * the root slot's argmax is S[c+1]; an on-path node at depth d has argmax
    S[c+2+d]; off-path nodes get an impossible token, so verification with
    the real ``verify_tree_greedy`` yields the same accepted path;
  * the "true target features" the drafter consumes are the precomputed
    features at the corresponding positions of S.

The drafting code (``_draft_levels`` / ``_draft_dynamic``) is the repo's own,
called unchanged, so the tree contents are what the GPU runs drafted (modulo
float32-CPU vs bfloat16-GPU rounding, which the paper quantifies). Only
acceptance is measured; no wall-clock claim can come out of this script.

``--self-test`` replays a tiny random target/drafter and checks that the
accept-length list equals ``speculative_generate``'s, static and dynamic.

    python scripts/e9_simulate.py --self-test
    python scripts/e9_simulate.py --model <dir> --ckpt <pt> --data <pt> \
        --benchmark mtbench --split test --K 4 --tree dynamic --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decode.generate import _build_level_plans, _draft_dynamic, _draft_levels  # noqa: E402
from src.decode.tree import DraftTree, build_from_parents, build_tree, geometric_allocation  # noqa: E402
from src.decode.verify import verify_tree_greedy  # noqa: E402
from src.models.drafter import Drafter, RMSNorm  # noqa: E402


# --------------------------------------------------------------------------- #
# drafter construction without a resident target
# --------------------------------------------------------------------------- #

def bind_from_safetensors(drafter: Drafter, model_dir: Path, *, head_dtype=torch.float32) -> None:
    """Bind the frozen embed / final norm / lm_head straight from the shards."""
    from safetensors import safe_open

    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def get(name):
        f = idx[name]
        if f not in files:
            files[f] = safe_open(str(model_dir / f), "pt")
        return files[f].get_tensor(name)

    cfg = json.loads((model_dir / "config.json").read_text())
    emb_w = get("model.embed_tokens.weight")  # keep bf16: rows are cast on lookup
    embed = nn.Embedding(emb_w.shape[0], emb_w.shape[1], _weight=emb_w)
    norm = RMSNorm(emb_w.shape[1], eps=float(cfg.get("rms_norm_eps", 1e-5)))
    norm.weight.data = get("model.norm.weight").to(torch.float32)
    head_w = get("lm_head.weight").to(head_dtype)
    head = nn.Linear(head_w.shape[1], head_w.shape[0], bias=False)
    head.weight.data = head_w
    for m in (embed, norm, head):
        m.requires_grad_(False)
    drafter._bound = [embed, norm, head]


def load_drafter(model_dir: Path, ckpt: Path, drafter_cfg: dict, device: str = "cpu") -> Drafter:
    from omegaconf import OmegaConf

    drafter = Drafter.from_config(OmegaConf.create(drafter_cfg))
    state = torch.load(str(ckpt), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    drafter.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    bind_from_safetensors(drafter, model_dir)
    drafter.eval()
    if device != "cpu":
        drafter.to(device)
        drafter._bound = [m.to(device) for m in drafter._bound]
    return drafter


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #

def _slot_argmax(tree: DraftTree, node_tokens: torch.Tensor, S: torch.Tensor, c: int) -> torch.Tensor:
    """Target argmax per slot from the greedy continuation S (see module docstring)."""
    n = tree.num_nodes
    out = torch.full((n + 1,), -1, dtype=torch.long)
    L = int(S.shape[0])
    if c + 1 < L:
        out[0] = S[c + 1]
    on_path = [False] * n
    parent = tree.parent.tolist()
    depth = tree.depth.tolist()
    toks = node_tokens.tolist()
    for i in range(n):
        p = parent[i]
        d = depth[i]
        if (p == -1 or on_path[p]) and c + 1 + d < L and toks[i] == int(S[c + 1 + d]):
            on_path[i] = True
            if c + 2 + d < L:
                out[i + 1] = S[c + 2 + d]
    return out


@torch.no_grad()
def replay(
    drafter: Drafter,
    S: torch.Tensor,
    F: torch.Tensor,
    prompt_len: int,
    max_new: int,
    *,
    K: int,
    tree: Optional[DraftTree],
    dynamic: bool,
    eos_token_id: Optional[int],
    skip_leaf_logits: bool = True,
) -> Tuple[List[int], int]:
    """Replay ``_generate_folded`` for one prompt. Returns (accept_lengths, committed)."""
    S = S.long()
    F = F.to(torch.float32)
    c = prompt_len
    feats = F[:c].unsqueeze(0)
    pending = int(S[c])
    feat_last = feats[:, -1:, :]
    if c >= 2:
        d_tokens, d_prev = S[1:c].unsqueeze(0), feats[:, : c - 1]
        d_pos = torch.arange(1, c).unsqueeze(0)
    else:
        d_tokens, d_prev = S[:c].unsqueeze(0), feats.new_zeros(1, 1, feats.shape[-1])
        d_pos = torch.arange(0, 1).unsqueeze(0)
    _h, dcache = drafter(d_tokens, d_prev, d_pos, attn_bias=None, cache=None)
    ctx_rows = dcache.seq_length()

    if dynamic:
        level_widths = torch.bincount(tree.depth).tolist()
        plans = None
    else:
        plans = _build_level_plans(tree, torch.float32, torch.device("cpu"))
        if skip_leaf_logits:
            # logits of leaves are never read by _draft_levels; drop them from
            # the head matmul (the expensive part) -- identical outputs.
            for plan in plans:
                keep = [r for r, i in enumerate(plan["level"]) if tree.children_of(i)]
                plan["logit_rows"] = keep
    committed: List[int] = []
    accept_lengths: List[int] = []
    hit_eos = False
    while len(committed) < max_new and not hit_eos:
        p_tok = torch.tensor([[pending]], dtype=torch.long)
        p_pos = torch.tensor([[c]], dtype=torch.long)
        h_p, dcache = drafter(p_tok, feat_last, p_pos, attn_bias=None, cache=dcache)
        ctx_rows = dcache.seq_length()
        q0 = drafter.logits(h_p[:, -1])[0]
        if dynamic:
            node_tokens, parents, dcache = _draft_dynamic(
                drafter, level_widths, dcache, ctx_rows, c + 1, q0, h_p[:, -1:, :])
            tree_s = build_from_parents(parents)
        else:
            node_tokens, _dp, dcache = _draft_levels_fast(
                drafter, tree, plans, dcache, ctx_rows, c + 1, q0, h_p[:, -1:, :])
            tree_s = tree
        outcome = verify_tree_greedy(tree_s, node_tokens, _slot_argmax(tree_s, node_tokens, S, c))
        accept_lengths.append(outcome.num_accepted)
        a = outcome.num_accepted
        dcache.crop(ctx_rows)
        new_tokens = [pending] + outcome.tokens[:a]
        f_pending = F[c].view(1, 1, -1)
        if a > 0:
            acc_feats = F[c + 1 : c + 1 + a].unsqueeze(0)
            acc_t = S[c + 1 : c + 1 + a].unsqueeze(0)
            acc_pos = torch.arange(c + 1, c + 1 + a).unsqueeze(0)
            _h2, dcache = drafter(acc_t, torch.cat([f_pending, acc_feats[:, :-1]], dim=1),
                                  acc_pos, attn_bias=None, cache=dcache)
            ctx_rows = dcache.seq_length()
            feat_last = acc_feats[:, -1:, :]
        else:
            feat_last = f_pending
        committed.extend(new_tokens)
        if eos_token_id is not None and eos_token_id in new_tokens:
            hit_eos = True
        c += 1 + a
        pending = outcome.tokens[a]
        if c >= int(S.shape[0]) - 1:
            break  # continuation exhausted (only possible if S is too short)
    return accept_lengths, len(committed)


def _draft_levels_fast(drafter, tree, plans, dcache, ctx_rows, context_len, q0, feat_root_prev):
    """``_draft_levels`` (greedy) with the head evaluated only on rows that
    have children when a plan carries ``logit_rows``. Same outputs."""
    if not any("logit_rows" in p for p in plans):
        return _draft_levels(drafter, tree, plans, dcache, ctx_rows, context_len, q0,
                             feat_root_prev, temperature=1.0, greedy=True, generator=None)
    n = tree.num_nodes
    node_tokens = torch.full((n,), -1, dtype=torch.long)
    slot: List[Optional[torch.Tensor]] = [None] * (n + 1)
    slot[0] = q0
    node_feat: List[Optional[torch.Tensor]] = [None] * n
    parent = tree.parent.tolist()
    for d, plan in enumerate(plans):
        level = plan["level"]
        for p, kid_idx in plan["parents"]:
            picks = torch.topk(slot[p + 1], k=kid_idx.shape[0]).indices
            node_tokens[kid_idx] = picks
        toks = node_tokens[plan["level_t"]].unsqueeze(0)
        prev = torch.cat([(feat_root_prev if parent[i] == -1 else node_feat[parent[i]]) for i in level], dim=1)
        pos_lvl = torch.full((1, len(level)), context_len + d, dtype=torch.long)
        block = plan["block"]
        bias = torch.cat([torch.zeros(1, 1, len(level), ctx_rows, dtype=block.dtype), block], dim=-1)
        h_lvl, dcache = drafter(toks, prev, pos_lvl, attn_bias=bias, cache=dcache)
        rows = plan["logit_rows"]
        if rows:
            logits = drafter.logits(h_lvl[0, rows])
            for j, r in enumerate(rows):
                slot[level[r] + 1] = logits[j]
        for r, i in enumerate(level):
            node_feat[i] = h_lvl[:, r : r + 1]
    return node_tokens, None, dcache


def make_static_tree(K: int, num_nodes: int, parents_file: Optional[str]) -> DraftTree:
    if parents_file:
        spec = json.loads(Path(parents_file).read_text())
        return build_from_parents(spec["parents"])
    return build_tree(geometric_allocation(num_nodes, K, 0.5))


# --------------------------------------------------------------------------- #
# self-test against the real decode loop on a tiny random model
# --------------------------------------------------------------------------- #

def self_test() -> int:
    from src.common.config import load_config
    from src.common.modeling import load_target
    from src.decode.generate import greedy_generate, speculative_generate
    from src.decode.tree import build_from_config

    cfg = load_config([ROOT / "configs/base.yaml", ROOT / "configs/tiny.yaml"], [])
    torch.manual_seed(0)
    target = load_target(cfg, torch.device("cpu"), torch.float32)
    torch.manual_seed(1)
    drafter = Drafter.from_config(cfg.drafter)
    drafter.bind_target(target)
    drafter.eval()
    vocab = int(cfg.target.random_config.vocab_size)
    ok = True
    n_checked = 0
    for seed in range(4):
        for K in (2, 4):
            for alloc in ("uniform", "geometric"):
                for dynamic in (False, True):
                    c2 = cfg.copy()
                    c2.tree.allocation = alloc
                    c2.tree.max_depth = K
                    tree = build_from_config(c2.tree)
                    torch.manual_seed(100 + seed)
                    prompt = torch.randint(0, vocab, (1, 7 + seed))
                    max_new = 20
                    ref = speculative_generate(target, drafter, tree, prompt, max_new,
                                               greedy=True, dynamic_tree=dynamic)
                    cont = greedy_generate(target, prompt, max_new + K + 2)
                    S = torch.cat([prompt[0], torch.tensor(cont, dtype=torch.long)])
                    with torch.no_grad():
                        out = target(input_ids=S.unsqueeze(0), output_hidden_states=True)
                    F = out.hidden_states[-2][0]
                    acc, n_committed = replay(drafter, S, F, int(prompt.shape[1]), max_new,
                                              K=K, tree=tree, dynamic=dynamic, eos_token_id=None)
                    same = acc == list(ref.accept_lengths)
                    n_checked += 1
                    if not same:
                        ok = False
                        print(f"MISMATCH seed={seed} K={K} {alloc} dynamic={dynamic}: "
                              f"ref={list(ref.accept_lengths)} sim={acc}")
    print(f"[E9] self-test: {n_checked} configurations, {'ALL IDENTICAL' if ok else 'FAILED'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", help="target dir (safetensors) for embed/norm/head")
    ap.add_argument("--ckpt", help="drafter checkpoint .pt")
    ap.add_argument("--drafter-cfg", default="base", help="base | eagle (configs/*.yaml drafter section)")
    ap.add_argument("--data", help="e9 data file with S / features per prompt")
    ap.add_argument("--benchmark")
    ap.add_argument("--split", default="test")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=256)
    ap.add_argument("--tree", default="static", help="static | dynamic | <parents.json>")
    ap.add_argument("--feature", default="feat", help="feat (hidden_states[-2]) | final (post-norm, EAGLE)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=0, help="limit (0 = all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=False)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.threads:
        torch.set_num_threads(a.threads)
    from omegaconf import OmegaConf

    base = OmegaConf.load(ROOT / "configs/base.yaml").drafter
    if a.drafter_cfg == "eagle":
        base = OmegaConf.merge(base, OmegaConf.load(ROOT / "configs/eagle.yaml").drafter)
    drafter = load_drafter(Path(a.model), Path(a.ckpt), OmegaConf.to_container(base, resolve=True))
    data = torch.load(a.data)
    rows = [r for r in data["rows"] if r["benchmark"] == a.benchmark and r["split"] == a.split]
    if a.prompts:
        rows = rows[: a.prompts]
    eos = int(data["eos_token_id"])
    dynamic = a.tree == "dynamic"
    tree = make_static_tree(a.K, a.nodes, None if a.tree in ("static", "dynamic") else a.tree)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_prompt_results.csv"
    done = set()
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path, newline="", encoding="utf-8")):
            done.add(int(r["prompt_id"]))
    fields = ["benchmark", "split", "prompt_id", "K", "tree", "tag", "ckpt", "steps",
              "committed_tokens", "mean_accept_len", "accept_lengths", "seconds"]
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    if not done:
        w.writeheader()
    accs = []
    t_all = time.time()
    for r in rows:
        if r["prompt_id"] in done:
            continue
        t0 = time.time()
        S = r["S"]
        F = r[a.feature]
        acc, n_c = replay(drafter, S, F, int(r["prompt_len"]), a.max_new, K=a.K, tree=tree,
                          dynamic=dynamic, eos_token_id=eos)
        m = statistics.mean(acc) if acc else 0.0
        accs.append(m)
        w.writerow({"benchmark": a.benchmark, "split": a.split, "prompt_id": r["prompt_id"], "K": a.K,
                    "tree": a.tree, "tag": a.tag, "ckpt": Path(a.ckpt).stem, "steps": len(acc),
                    "committed_tokens": n_c, "mean_accept_len": f"{m:.6f}",
                    "accept_lengths": " ".join(map(str, acc)), "seconds": f"{time.time() - t0:.1f}"})
        f.flush()
        print(f"  {a.benchmark}/{a.split} pid={r['prompt_id']} steps={len(acc)} "
              f"acc={m:.3f} ({time.time() - t0:.0f}s, elapsed {time.time() - t_all:.0f}s)", flush=True)
    f.close()
    if accs:
        print(f"[E9] {a.benchmark}/{a.split} K={a.K} tree={a.tree} {Path(a.ckpt).stem}: "
              f"mean accept over {len(accs)} new prompts = {statistics.mean(accs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
