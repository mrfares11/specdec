"""E9 --- greedy continuations and target features for a prompt set (GPU).

Same prompt selection and chat template as the evaluation harness
(``_prompt_ids`` from src/eval/benchmark.py), then for each prompt: the
target's greedy continuation of ``--n-cont`` tokens (EOS ignored on purpose:
tree nodes past an EOS are verified against real target argmaxes, so those
positions are needed) and one teacher-forced forward over prompt+continuation
that records hidden_states[-2] (the drafter's feature space) and the post-norm
state feeding lm_head. Output rows are consumed by e9_rollout_stats.py and
e9_simulate.py.

    python scripts/e9_gpu_data.py --config configs/base.yaml --set \
        target.model_id=NousResearch/Meta-Llama-3-8B-Instruct \
        eval.prompt_source=mtbench eval.prompt_split=test eval.num_prompts=50 \
        --out /workspace/specdec-runs/e9/data_mtbench_test.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.device import resolve_device, resolve_dtype  # noqa: E402
from src.common.modeling import load_target, load_tokenizer  # noqa: E402
from src.decode.generate import greedy_generate  # noqa: E402
from src.eval.benchmark import _prompt_ids  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", dest="sets", nargs="*", default=[])
    ap.add_argument("--n-cont", type=int, default=140)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config, a.sets)
    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)
    tok = load_tokenizer(cfg)
    eos = int(tok.eos_token_id)
    prompts = _prompt_ids(cfg, device)
    bench, split = str(cfg.eval.prompt_source), str(cfg.eval.get("prompt_split", "legacy"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for pid, ids in prompts:
        cont = greedy_generate(target, ids, a.n_cont, eos_token_id=None)
        S = torch.cat([ids[0].cpu(), torch.tensor(cont, dtype=torch.long)])
        o = target(input_ids=S.unsqueeze(0).to(device), output_hidden_states=True)
        feat = o.hidden_states[-2][0].to(torch.float16).cpu()
        final = o.hidden_states[-1][0].to(torch.float16).cpu()
        eos_off = next((j for j, t in enumerate(cont) if t == eos), -1)
        rows.append({"benchmark": bench, "split": split, "prompt_id": int(pid), "prompt_len": int(ids.shape[1]),
                     "S": S, "feat": feat, "final": final, "eos_offset": eos_off})
        print(f"  {bench}/{split} pid={pid} len={S.shape[0]} eos@{eos_off} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"rows": rows, "eos_token_id": eos}, out)
    n_eos = sum(1 for r in rows if 0 <= r["eos_offset"] < 128)
    print(f"[E9] wrote {out}: {len(rows)} prompts, {n_eos} reach EOS within 128 tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
