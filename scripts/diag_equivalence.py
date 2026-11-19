"""Diagnose speculative-vs-AR divergence: is it an off-by-one in the comparison,
or genuine numerical divergence on the real model?

Reports, per prompt: both lengths, the first index where they differ, and how
many leading positions agree. An off-by-one shows as first_diff=0; genuine
bf16 argmax divergence shows as a large matching prefix then a single flip.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config                      # noqa: E402
from src.common.device import resolve_device, resolve_dtype     # noqa: E402
from src.common.modeling import load_target, validate_feature_config  # noqa: E402
from src.decode.generate import greedy_generate, speculative_generate  # noqa: E402
from src.decode.tree import build_from_config                   # noqa: E402
from omegaconf import OmegaConf                                  # noqa: E402
from src.models.drafter import Drafter                          # noqa: E402
from src.eval.benchmark import _prompt_ids                      # noqa: E402
from src.common.modeling import FeatureTap                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    validate_feature_config(cfg)

    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)
    drafter = Drafter.from_config(cfg.drafter).to(device=device, dtype=dtype)
    drafter.bind_target(target)
    state = torch.load(str(cfg.eval.drafter_ckpt), map_location=device)
    drafter.load_state_dict(state["model"])
    drafter.eval()

    tap = FeatureTap.from_config(cfg, target=target)
    K = int(cfg.eval.K_sweep[0])
    tree = build_from_config(OmegaConf.merge(cfg.tree, {"max_depth": K}))
    cfg.eval.num_prompts = a.prompts
    prompts = _prompt_ids(cfg, device)

    print(f"K={K} nodes={tree.num_nodes} dtype={dtype} max_new={a.max_new}")
    tot_match = tot_pos = n_exact = 0
    for i, ids in enumerate(prompts):
        gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
        res = speculative_generate(
            target, drafter, tree, ids, a.max_new,
            temperature=float(cfg.sampling.temperature), greedy=True,
            layer_index=int(cfg.data.layer_index), tap=tap,
            generator=gen, kv_gather=bool(cfg.decode.kv_gather),
            dynamic_tree=bool(cfg.decode.get("dynamic_tree", False)),
            commit_fold=bool(cfg.decode.get("commit_fold", True)),
        )
        ar = greedy_generate(target, ids, a.max_new)
        s, r = list(res.tokens), list(ar)
        n = min(len(s), len(r))
        first = next((j for j in range(n) if s[j] != r[j]), -1)
        match = first if first >= 0 else n
        tot_match += match; tot_pos += n
        n_exact += int(first < 0)
        print(f"  prompt {i}: spec_len={len(s):4d} ar_len={len(r):4d} common={n:4d} "
              f"first_diff={first:4d} matching_prefix={match:4d}"
              + ("  EXACT" if first < 0 else f"  spec={s[first]} ar={r[first]}"))
    print(f"\n{n_exact}/{len(prompts)} prompts exactly equal on the common prefix")
    print(f"aggregate matching positions: {tot_match}/{tot_pos} = {100*tot_match/max(tot_pos,1):.3f}%")


if __name__ == "__main__":
    main()
