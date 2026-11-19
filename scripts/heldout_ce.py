"""ARM-INDEPENDENT held-out rollout CE (mock-review remedy for the RQ2
train-CE confound: each arm's logged CE used its own step weights).

``--full-target`` additionally scores every rollout step against the FULL
target's next-token distribution (one target forward per sample) and reports
per-step top-1 agreement with the target's argmax, so proxy-vs-target
mismatch can be separated from depth averaging.

Evaluates a drafter checkpoint on held-out feature shards with a K-step
self-rollout mirroring training alignment, reporting the PER-STEP soft-CE
curve and its UNIFORM mean — identical batching, masks, and order for every
checkpoint (sequential shard order, no shuffle, fixed sample budget).

    python scripts/heldout_ce.py --config configs/base.yaml \
        --set data.out_dir=<dev_shards> eval.drafter_ckpt=<ckpt> \
        [--samples 400] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config  # noqa: E402
from src.common.device import resolve_device, resolve_dtype  # noqa: E402
from src.common.modeling import load_target, validate_feature_config  # noqa: E402
from src.data.shards import ShardReader  # noqa: E402
from src.models.drafter import Drafter  # noqa: E402


@torch.no_grad()
def _soft_ce(head, pred, tgt, m_f, denom, chunk=256):
    total = 0.0
    for c0 in range(0, pred.shape[1], chunk):
        sl = slice(c0, c0 + chunk)
        pl = torch.log_softmax(head(pred[:, sl]).float(), dim=-1)
        tp = torch.softmax(head(tgt[:, sl]).float(), dim=-1)
        total += float((-(tp * pl).sum(-1) * m_f[:, sl]).sum().item())
    return total / float(denom)


@torch.no_grad()
def _full_target_terms(head, pred, lp_full, am_full, m_f, denom, chunk=256):
    """Cross-entropy against the FULL target distribution and top-1 agreement
    with the target's argmax, for pred rows aligned with lp_full/am_full."""
    ce = 0.0
    agree = 0.0
    for c0 in range(0, pred.shape[1], chunk):
        sl = slice(c0, c0 + chunk)
        lg = head(pred[:, sl]).float()
        pl = torch.log_softmax(lg, dim=-1)
        pf = lp_full[sl].exp().unsqueeze(0)
        ce += float((-(pf * pl).sum(-1) * m_f[:, sl]).sum().item())
        agree += float(((lg.argmax(-1) == am_full[sl].unsqueeze(0)).float() * m_f[:, sl]).sum().item())
    return ce / float(denom), agree / float(denom)


@torch.no_grad()
def evaluate(cfg, ckpt: str, n_samples: int, full_target: bool = False) -> dict:
    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)
    drafter = Drafter.from_config(cfg.drafter).to(device=device, dtype=dtype)
    drafter.bind_target(target)
    state = torch.load(ckpt, map_location=device)
    drafter.load_state_dict(state["model"])
    drafter.eval()

    reader = ShardReader(str(cfg.data.out_dir))
    K = int(cfg.train.rollout_K)
    ce_sum = [0.0] * K
    feat_sum = [0.0] * K
    tok_cnt = [0.0] * K
    ce_full_sum = [0.0] * K
    top1_full_sum = [0.0] * K
    used = 0
    for sid in reader.sample_ids():  # sequential: identical for every ckpt
        if used >= n_samples:
            break
        hidden, ids = reader.read_sample(sid)
        if ids.shape[0] < 3:
            continue
        used += 1
        feats = hidden.to(device=device, dtype=dtype).unsqueeze(0)
        tokens = ids.to(device).unsqueeze(0)
        if full_target:
            # the target's own next-token distribution at every position of the
            # recorded sequence; rows at rollout step s align with positions 1+s..T-1
            tlog = target(input_ids=tokens).logits[0]
            lp_full_all = torch.log_softmax(tlog.float(), dim=-1)
            am_full_all = tlog.argmax(dim=-1)
            del tlog
        tok = tokens[:, 1:]
        prev = feats[:, :-1]
        tgt = feats[:, 1:]
        m = torch.ones_like(tok, dtype=torch.bool)
        n_steps = min(K, int(tok.shape[1]))
        offset = 1
        for s in range(n_steps):
            pos = torch.arange(offset, offset + tok.shape[1], device=device).unsqueeze(0)
            pred, _ = drafter(tok, prev, pos, attn_bias=None, cache=None)
            m_f = m.to(pred.dtype)
            denom = float(m_f.sum().clamp_min(1.0).item())
            l_feat = float(((F.smooth_l1_loss(pred, tgt, reduction="none").mean(-1) * m_f).sum() / denom).item())
            ce = _soft_ce(drafter.logits, pred, tgt, m_f, denom)
            ce_sum[s] += ce * denom
            feat_sum[s] += l_feat * denom
            tok_cnt[s] += denom
            if full_target:
                n = int(pred.shape[1])
                cf, ag = _full_target_terms(drafter.logits, pred, lp_full_all[1 + s : 1 + s + n],
                                            am_full_all[1 + s : 1 + s + n], m_f, denom)
                ce_full_sum[s] += cf * denom
                top1_full_sum[s] += ag * denom
            if s == n_steps - 1 or tok.shape[1] <= 1:
                break
            prev = pred[:, :-1]
            tok = tok[:, 1:]
            tgt = tgt[:, 1:]
            m = m[:, 1:]
    per_ce = [ce_sum[s] / tok_cnt[s] if tok_cnt[s] else None for s in range(K)]
    per_feat = [feat_sum[s] / tok_cnt[s] if tok_cnt[s] else None for s in range(K)]
    valid = [c for c in per_ce if c is not None]
    out = {
        "ckpt": ckpt, "samples": used, "K": K,
        "per_step_ce": per_ce, "per_step_feat": per_feat,
        "uniform_ce": sum(valid) / len(valid) if valid else None,
    }
    if full_target:
        per_full = [ce_full_sum[s] / tok_cnt[s] if tok_cnt[s] else None for s in range(K)]
        per_top1 = [top1_full_sum[s] / tok_cnt[s] if tok_cnt[s] else None for s in range(K)]
        vf = [c for c in per_full if c is not None]
        out.update({"per_step_ce_full": per_full, "uniform_ce_full": sum(vf) / len(vf) if vf else None,
                    "per_step_top1_full": per_top1})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", dest="sets", nargs="*", default=[])
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--full-target", action="store_true",
                    help="also report CE against the full target distribution and top-1 agreement")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.sets)
    validate_feature_config(cfg)
    res = evaluate(cfg, str(cfg.eval.drafter_ckpt), args.samples, full_target=args.full_target)
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
