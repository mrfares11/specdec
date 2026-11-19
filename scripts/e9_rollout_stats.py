"""E9 --- per-depth diagnostics of a drafter along the target's greedy path.

For every decode-step root c on a prompt's greedy continuation S, the drafter
is rolled out exactly as the on-path chain of a draft tree would be at decode
time: step 1 consumes (S[c], true feature F[c-1]) on top of the true-feature
context; step s >= 2 consumes (S[c+s-1], its own predicted feature) and attends
to the true-feature context plus its own chain rows. At every step it records

  rank      -- rank of the true next token S[c+s] in the drafter's logits
               (0 = the drafter's top-1 agrees with the target's greedy choice);
  logq      -- drafter log-probability of that token;
  ce_full   -- cross-entropy between the FULL target's next-token distribution
               (recomputed from the stored post-norm state) and the drafter's;
  ce_proxy  -- cross-entropy between the proxy distribution H(F[c+s-1]) of the
               intermediate feature and the drafter's -- the training loss.

Marginals of ``rank`` per depth are what an offline tree optimiser needs
(scripts/e9_build_tree.py); the per-depth top-1 agreement decomposes chain
acceptance; the two cross-entropies separate "proxy vs target" from
"averaging over depths" as explanations of the loss/acceptance misranking.

    python scripts/e9_rollout_stats.py --model <dir> --ckpt <pt> --data <pt> \
        --benchmark mtbench --split test --K 8 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e9_simulate import load_drafter  # noqa: E402
from src.models.drafter import DrafterCache  # noqa: E402


@torch.no_grad()
def rollout_prompt(drafter, S: torch.Tensor, F: torch.Tensor, final: torch.Tensor,
                   prompt_len: int, K: int, root_cap: int = 128):
    dev = next(drafter.parameters()).device
    S = S.long().to(dev)
    F = F.float().to(dev)
    final = final.to(dev)
    L = int(S.shape[0])
    H = F.shape[-1]
    roots = list(range(prompt_len, min(L - K - 1, prompt_len + root_cap)))
    R = len(roots)
    if R == 0:
        return None
    # target distributions at every position (identical for every checkpoint)
    head = drafter._lm_head
    logp_full = torch.log_softmax(head(final.float()), dim=-1)       # [L, V]
    logp_proxy = torch.log_softmax(drafter.logits(F), dim=-1)          # [L, V]
    # causal pass over the whole sequence: rows are positions 1..L-1
    tok_all = S[1:].unsqueeze(0)
    prev_all = F[:-1].unsqueeze(0)
    pos_all = torch.arange(1, L, device=dev).unsqueeze(0)
    h_all, cache = drafter(tok_all, prev_all, pos_all, attn_bias=None, cache=None)
    ctx_k, ctx_v = cache.layers[0]                                     # [1, kv, L-1, D]
    n_ctx = ctx_k.shape[2]
    kv, D = ctx_k.shape[1], ctx_k.shape[3]
    root_t = torch.tensor(roots, device=dev)
    ranks = torch.zeros(R, K, dtype=torch.int32, device=dev)
    logq = torch.zeros(R, K, device=dev)
    ce_full = torch.zeros(R, K, device=dev)
    ce_proxy = torch.zeros(R, K, device=dev)

    def record(s: int, logits: torch.Tensor):
        # logits [R, V] predict S[c+s]; the target distributions live at position c+s-1
        pos = root_t + s - 1
        true = S[root_t + s]
        lq = torch.log_softmax(logits.float(), dim=-1)
        lt = lq.gather(1, true.unsqueeze(1)).squeeze(1)
        ranks[:, s - 1] = (lq > lt.unsqueeze(1)).sum(dim=1).to(torch.int32)
        logq[:, s - 1] = lt
        ce_full[:, s - 1] = -(logp_full[pos].exp() * lq).sum(dim=1)
        ce_proxy[:, s - 1] = -(logp_proxy[pos].exp() * lq).sum(dim=1)

    # step 1: row for position c is h_all[c-1]
    h_prev = h_all[0, root_t - 1]                                      # [R, H]
    record(1, drafter.logits(h_prev))
    chain_k = ctx_k.new_zeros(R, kv, 0, D)
    chain_v = ctx_v.new_zeros(R, kv, 0, D)
    col_pos = torch.arange(1, L, device=dev).unsqueeze(0)              # position of ctx column j
    ctx_visible = (col_pos <= root_t.unsqueeze(1))                      # [R, L-1]
    neg = torch.finfo(torch.float32).min
    for s in range(2, K + 1):
        tok = S[root_t + s - 1].unsqueeze(1)                           # [R, 1]
        pos = (root_t + s - 1).unsqueeze(1)
        past_k = torch.cat([ctx_k.expand(R, -1, -1, -1), chain_k], dim=2)
        past_v = torch.cat([ctx_v.expand(R, -1, -1, -1), chain_v], dim=2)
        n_past = past_k.shape[2]
        bias = torch.full((R, 1, 1, n_past + 1), neg, device=dev, dtype=torch.float32)
        bias[:, 0, 0, :n_ctx][ctx_visible] = 0.0
        bias[:, 0, 0, n_ctx:] = 0.0                                    # own chain rows + self
        h, new = drafter(tok, h_prev.unsqueeze(1), pos, attn_bias=bias,
                         cache=DrafterCache([(past_k, past_v)]))
        k_new, v_new = new.layers[0]
        chain_k = torch.cat([chain_k, k_new[:, :, n_past:, :]], dim=2)
        chain_v = torch.cat([chain_v, v_new[:, :, n_past:, :]], dim=2)
        h_prev = h[:, 0]
        record(s, drafter.logits(h_prev))
    return {"roots": root_t.cpu(), "ranks": ranks.cpu(), "logq": logq.cpu(),
            "ce_full": ce_full.cpu(), "ce_proxy": ce_proxy.cpu()}


@torch.no_grad()
def rollout_prompt_training(drafter, S: torch.Tensor, F: torch.Tensor, final: torch.Tensor,
                            prompt_len: int, K: int, root_cap: int = 128):
    """Same metrics as ``rollout_prompt`` but in the TRAINING convention of
    scripts/heldout_ce.py: at step s the whole sequence is re-run with the
    previous step's predicted features as inputs (positions restart at 1 each
    step, as there), so context rows are predicted features, not true ones.
    Rows are re-indexed to (root, step) so the two conventions are comparable."""
    dev = next(drafter.parameters()).device
    S = S.long().to(dev)
    F = F.float().to(dev)
    final = final.to(dev)
    L = int(S.shape[0])
    roots = list(range(prompt_len, min(L - K - 1, prompt_len + root_cap)))
    R = len(roots)
    if R == 0:
        return None
    head = drafter._lm_head
    logp_full = torch.log_softmax(head(final.float()), dim=-1)
    logp_proxy = torch.log_softmax(drafter.logits(F), dim=-1)
    root_t = torch.tensor(roots, device=dev)
    ranks = torch.zeros(R, K, dtype=torch.int32, device=dev)
    logq = torch.zeros(R, K, device=dev)
    ce_full = torch.zeros(R, K, device=dev)
    ce_proxy = torch.zeros(R, K, device=dev)
    tok = S[1:].unsqueeze(0)
    prev = F[:-1].unsqueeze(0)
    for s in range(1, K + 1):
        pos = torch.arange(1, 1 + tok.shape[1], device=dev).unsqueeze(0)
        pred, _ = drafter(tok, prev, pos, attn_bias=None, cache=None)   # rows: positions s..L-1
        i0 = root_t + s - 1                                             # position of root's step-s row
        rows = i0 - s                                                   # row index within pred
        logits = drafter.logits(pred[0, rows])
        true = S[i0 + 1]
        lq = torch.log_softmax(logits.float(), dim=-1)
        lt = lq.gather(1, true.unsqueeze(1)).squeeze(1)
        ranks[:, s - 1] = (lq > lt.unsqueeze(1)).sum(dim=1).to(torch.int32)
        logq[:, s - 1] = lt
        ce_full[:, s - 1] = -(logp_full[i0].exp() * lq).sum(dim=1)
        ce_proxy[:, s - 1] = -(logp_proxy[i0].exp() * lq).sum(dim=1)
        prev = pred[:, :-1]
        tok = tok[:, 1:]
    return {"roots": root_t.cpu(), "ranks": ranks.cpu(), "logq": logq.cpu(),
            "ce_full": ce_full.cpu(), "ce_proxy": ce_proxy.cpu()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--drafter-cfg", default="base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--feature", default="feat")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--convention", default="decode", choices=["decode", "training"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from omegaconf import OmegaConf

    base = OmegaConf.load(ROOT / "configs/base.yaml").drafter
    if a.drafter_cfg == "eagle":
        base = OmegaConf.merge(base, OmegaConf.load(ROOT / "configs/eagle.yaml").drafter)
    drafter = load_drafter(Path(a.model), Path(a.ckpt), OmegaConf.to_container(base, resolve=True),
                           device=a.device)
    data = torch.load(a.data)
    rows = [r for r in data["rows"] if r["benchmark"] == a.benchmark and r["split"] == a.split]
    if a.prompts:
        rows = rows[: a.prompts]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_prompt = {}
    pt_path = out / "rollout.pt"
    if pt_path.exists():
        per_prompt = torch.load(pt_path)
    t_all = time.time()
    for r in rows:
        pid = int(r["prompt_id"])
        if pid in per_prompt:
            continue
        t0 = time.time()
        fn = rollout_prompt if a.convention == "decode" else rollout_prompt_training
        res = fn(drafter, r["S"], r[a.feature], r["final"], int(r["prompt_len"]), a.K)
        if res is None:
            continue
        per_prompt[pid] = res
        torch.save(per_prompt, pt_path)
        agree = (res["ranks"] == 0).float().mean(dim=0).tolist()
        print(f"  {a.benchmark}/{a.split} pid={pid} roots={len(res['roots'])} "
              f"top1@depth={[round(x, 2) for x in agree]} ({time.time() - t0:.0f}s, {time.time() - t_all:.0f}s)",
              flush=True)
    # summary: per-depth means over prompts (each prompt weighted equally)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth", "top1_agreement", "mean_rank", "ce_full", "ce_proxy", "mean_logq", "n_prompts"])
        for d in range(a.K):
            ag = [float((v["ranks"][:, d] == 0).float().mean()) for v in per_prompt.values()]
            mr = [float(v["ranks"][:, d].float().mean()) for v in per_prompt.values()]
            cf = [float(v["ce_full"][:, d].mean()) for v in per_prompt.values()]
            cp = [float(v["ce_proxy"][:, d].mean()) for v in per_prompt.values()]
            lq = [float(v["logq"][:, d].mean()) for v in per_prompt.values()]
            w.writerow([d + 1, f"{statistics.mean(ag):.4f}", f"{statistics.mean(mr):.3f}",
                        f"{statistics.mean(cf):.4f}", f"{statistics.mean(cp):.4f}",
                        f"{statistics.mean(lq):.4f}", len(per_prompt)])
    print(f"[E9] wrote {out}/summary.csv ({len(per_prompt)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
