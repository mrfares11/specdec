"""E1 --- numerical correctness and bf16 practical fidelity.

Three procedures from the experiment protocol, emitting raw per-position rows
so the analysis can be redone without re-running the GPU work.

  A  free-running equivalence   : generate speculatively and autoregressively
                                  from the same prompt, compare token by token.
                                  At float32 ANY divergence blocks the
                                  exact-arithmetic correctness claim.
  B  same-prefix logit compare  : walk the AR-generated sequence and, at every
                                  position, score the identical prefix through
                                  the sequential and tree-batched paths. This
                                  isolates numerical difference BEFORE
                                  free-running divergence changes the context,
                                  and yields the top-1/top-2 margin that
                                  explains which positions are at risk.
  C  bf16 free-running          : procedure A at bf16, joined to B's margins.

    python scripts/e1_correctness.py --config configs/base.yaml \
        --set eval.drafter_ckpt=... --precision float32 --procedure A \
        --prompts 25 --out results/e1/<cell>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.device import resolve_device, resolve_dtype, sync  # noqa: E402
from src.common.modeling import (  # noqa: E402
    FeatureTap, load_target, validate_feature_config,
)
from src.decode.generate import greedy_generate, speculative_generate  # noqa: E402
from src.decode.tree import build_from_config  # noqa: E402
from src.eval.benchmark import _prompt_ids  # noqa: E402
from src.models.drafter import Drafter  # noqa: E402


def _build(cfg, device, dtype):
    target = load_target(cfg, device, dtype)
    drafter = Drafter.from_config(cfg.drafter).to(device=device, dtype=dtype)
    drafter.bind_target(target)
    if cfg.eval.drafter_ckpt is not None:
        state = torch.load(str(cfg.eval.drafter_ckpt), map_location=device)
        drafter.load_state_dict(state["model"])
    drafter.eval()
    return target, drafter


def procedure_a_c(cfg, target, drafter, prompts, K, tap, device, precision):
    """Free-running equivalence at the configured precision."""
    tree = build_from_config(OmegaConf.merge(cfg.tree, {"max_depth": K}))
    max_new = int(cfg.eval.max_new_tokens)
    rows = []
    for pid, ids in prompts:
        gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
        res = speculative_generate(
            target, drafter, tree, ids, max_new,
            temperature=float(cfg.sampling.temperature), greedy=True,
            layer_index=int(cfg.data.layer_index), tap=tap, generator=gen,
            kv_gather=bool(cfg.decode.kv_gather),
            dynamic_tree=bool(cfg.decode.get("dynamic_tree", False)),
            commit_fold=bool(cfg.decode.get("commit_fold", True)),
        )
        ar = greedy_generate(target, ids, max_new)
        spec, aref = list(res.tokens), list(ar)
        n = min(len(spec), len(aref))
        first = next((i for i in range(n) if spec[i] != aref[i]), -1)
        agree = n if first < 0 else first
        rows.append({
            "precision": precision, "K": K, "prompt_id": pid,
            "spec_len": len(spec), "ar_len": len(aref), "compared": n,
            "first_divergence": first, "agreeing_prefix": agree,
            "positions_matched": agree, "positions_compared": n,
            "sequence_exact_match": int(first < 0),
        })
        print(f"[E1-{precision}] prompt {pid}: {agree}/{n} matched"
              + ("" if first < 0 else f", first divergence at {first}"), flush=True)
    return rows


@torch.no_grad()
def procedure_b(cfg, target, prompts, tap, device, precision, tree_nodes=256):
    """Same-prefix comparison: sequential vs tree-batched logits, with margins.

    The autoregressive continuation supplies a common reference prefix. At each
    position we score that identical prefix two ways -- one token at a time, and
    as a batched row inside a tree-shaped forward -- so any disagreement is
    purely numerical rather than a consequence of the two runs having drifted
    onto different contexts.
    """
    max_new = int(cfg.eval.max_new_tokens)
    rows = []
    for pid, ids in prompts:
        ar = greedy_generate(target, ids, max_new)
        seq = torch.cat([ids, torch.tensor([ar], device=ids.device)], dim=1)
        # Sequential reference pass over the whole sequence.
        out_seq = target(input_ids=seq, use_cache=False)
        lg_seq = out_seq.logits[0].float()
        # Batched pass: same tokens, but presented as a batch of rows, which is
        # the reduction order the tree forward actually uses.
        out_bat = target(input_ids=seq.repeat(2, 1), use_cache=False)
        lg_bat = out_bat.logits[0].float()

        start = int(ids.shape[1]) - 1
        for i in range(start, lg_seq.shape[0] - 1):
            a = lg_seq[i]
            b = lg_bat[i]
            top2 = torch.topk(a, 2)
            margin = float(top2.values[0] - top2.values[1])
            t1_seq = int(top2.indices[0])
            t1_bat = int(torch.argmax(b))
            # Record the top-1 logit magnitude so the margin can be expressed in
            # UNITS IN THE LAST PLACE of the execution dtype. An absolute margin
            # threshold is meaningless across magnitudes: bf16 has a 7-bit
            # mantissa, so near a logit of 16-32 the representable spacing is
            # already 0.125, and a "wide" absolute margin can still be a single
            # ULP -- i.e. two tokens the format cannot tell apart.
            top1_val = float(top2.values[0])
            exp = math.floor(math.log2(abs(top1_val))) if top1_val != 0 else 0
            mant_bits = 7 if precision == "bfloat16" else 23
            ulp = 2.0 ** (exp - mant_bits)
            rows.append({
                "precision": precision, "prompt_id": pid, "position": i - start,
                "top1_sequential": t1_seq, "top1_batched": t1_bat,
                "argmax_agrees": int(t1_seq == t1_bat),
                "top1_top2_margin": margin,
                "top1_logit": top1_val,
                "ulp_at_top1": ulp,
                "margin_in_ulps": (margin / ulp) if ulp > 0 else float("inf"),
                "max_abs_logit_diff": float((a - b).abs().max()),
                "mean_abs_logit_diff": float((a - b).abs().mean()),
                "rank_of_sequential_winner_in_batched":
                    int((b > b[t1_seq]).sum()),
            })
        print(f"[E1-B-{precision}] prompt {pid}: {lg_seq.shape[0]-1-start} positions",
              flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--precision", choices=["float32", "bfloat16"], required=True)
    ap.add_argument("--procedure", choices=["A", "B"], default="A",
                    help="A = free-running (also serves procedure C at bf16); "
                         "B = same-prefix logit comparison")
    ap.add_argument("--prompts", type=int, default=25)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    overrides = list(a.set) + [f"runtime.dtype={a.precision}",
                              f"eval.num_prompts={a.prompts}"]
    cfg = load_config(a.config, overrides)
    validate_feature_config(cfg)

    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    print(f"[E1] precision={dtype} procedure={a.procedure} K={a.K} "
          f"prompts={a.prompts} source={cfg.eval.prompt_source}", flush=True)

    target, drafter = _build(cfg, device, dtype)
    tap = FeatureTap.from_config(cfg, target=target)
    prompts = _prompt_ids(cfg, device)

    if a.procedure == "A":
        rows = procedure_a_c(cfg, target, drafter, prompts, a.K, tap, device,
                             a.precision)
    else:
        rows = procedure_b(cfg, target, prompts, tap, device, a.precision)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"procedure_{a.procedure}_{a.precision}.csv"
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if a.procedure == "A":
        matched = sum(r["positions_matched"] for r in rows)
        compared = sum(r["positions_compared"] for r in rows)
        exact = sum(r["sequence_exact_match"] for r in rows)
        summary = {"procedure": "A", "precision": a.precision, "K": a.K,
                   "positions_matched": matched, "positions_compared": compared,
                   "token_agreement": matched / max(compared, 1),
                   "sequences_exact": exact, "sequences": len(rows)}
    else:
        agree = sum(r["argmax_agrees"] for r in rows)
        summary = {"procedure": "B", "precision": a.precision,
                   "positions": len(rows), "argmax_agree": agree,
                   "argmax_agreement": agree / max(len(rows), 1)}
    (out / f"summary_{a.procedure}_{a.precision}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("[E1] " + json.dumps(summary), flush=True)

    if a.procedure == "A" and a.precision == "float32":
        if summary["token_agreement"] < 1.0:
            print("[E1] *** FLOAT32 DIVERGENCE -- exact-arithmetic correctness "
                  "claim is BLOCKED. Preserve the trace and debug before "
                  "continuing. ***", flush=True)
            return 2
        print("[E1] float32 free-running equivalence: EXACT", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
