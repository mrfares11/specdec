"""E1 procedure D --- downstream task quality for the bf16 headline system.

Gate A needs more than "the tokens differ 1% of the time": it needs evidence
that the bf16 differences do not cost task quality. This generates the SAME
prompts autoregressively and speculatively, scores both with one deterministic
harness, and reports the PAIRED difference with a bootstrap interval.

Two benchmarks are scored without any judge model, so the result is
reproducible:

  HumanEval  pass@1   -- the generated function is executed against the
                        problem's own test suite, in a subprocess with a
                        timeout.
  GSM8K      exact match on the final numeric answer.

MT-Bench needs an LLM judge, which would make the number depend on a third
model's version. It is deliberately out of scope here and reported as such.

A non-significant difference is NOT reported as "no quality loss": the protocol
requires the interval and the largest degradation still compatible with it.

    python scripts/e1_quality.py --config configs/base.yaml --set ... \
        --benchmark humaneval --precision bfloat16 --out results/e1q/<cell>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.device import resolve_device, resolve_dtype  # noqa: E402
from src.common.modeling import (  # noqa: E402
    FeatureTap, load_target, load_tokenizer, validate_feature_config,
)
from src.decode.generate import greedy_generate, speculative_generate  # noqa: E402
from src.decode.tree import build_from_config  # noqa: E402
from src.models.drafter import Drafter  # noqa: E402

EXEC_TIMEOUT = 10


def run_humaneval_case(completion: str, test: str, entry_point: str) -> bool:
    """Execute one HumanEval solution against its tests in a fresh subprocess."""
    prog = f"{completion}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=EXEC_TIMEOUT)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_answer(text: str):
    """Last number in the text; GSM8K gold is after '#### '."""
    nums = _NUM.findall(text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def boot_ci(diffs, n=10000, seed=20260904):
    rng = random.Random(seed)
    if not diffs:
        return None, None
    out = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        out.append(sum(s) / len(s))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--benchmark", choices=["humaneval", "gsm8k"], required=True)
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--prompts", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset  # noqa: PLC0415

    cfg = load_config(a.config, list(a.set) + [f"runtime.dtype={a.precision}"])
    validate_feature_config(cfg)
    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)
    tok = load_tokenizer(cfg)
    drafter = Drafter.from_config(cfg.drafter).to(device=device, dtype=dtype)
    drafter.bind_target(target)
    state = torch.load(str(cfg.eval.drafter_ckpt), map_location=device)
    drafter.load_state_dict(state["model"])
    drafter.eval()
    tap = FeatureTap.from_config(cfg, target=target)
    tree = build_from_config(OmegaConf.merge(cfg.tree, {"max_depth": a.K}))

    if a.benchmark == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")

    # Use the SAME frozen test split the timing runs use, so quality and speed
    # are measured on the same prompts.
    import random as _r
    order = list(range(len(ds)))
    _r.Random(20260902).shuffle(order)
    n_dev = max(1, min(200, len(ds) // 3))
    test_ids = order[n_dev:][:a.prompts]

    rows = []
    for i, idx in enumerate(test_ids):
        ex = ds[idx]
        if a.benchmark == "humaneval":
            prompt = ex["prompt"]
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        else:
            prompt = ex["question"] + "\nAnswer step by step, then give the final number.\n"
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True,
                                          return_tensors="pt").to(device)
        gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
        spec = speculative_generate(
            target, drafter, tree, ids, a.max_new, temperature=1.0, greedy=True,
            layer_index=int(cfg.data.layer_index), tap=tap, generator=gen,
            kv_gather=bool(cfg.decode.kv_gather),
            dynamic_tree=bool(cfg.decode.get("dynamic_tree", False)),
            commit_fold=bool(cfg.decode.get("commit_fold", True)))
        ar = greedy_generate(target, ids, a.max_new)
        s_txt = tok.decode(spec.tokens, skip_special_tokens=True)
        a_txt = tok.decode(ar, skip_special_tokens=True)

        if a.benchmark == "humaneval":
            s_ok = run_humaneval_case(ex["prompt"] + s_txt, ex["test"], ex["entry_point"])
            a_ok = run_humaneval_case(ex["prompt"] + a_txt, ex["test"], ex["entry_point"])
        else:
            gold = gsm8k_answer(ex["answer"].split("####")[-1])
            s_ok = gsm8k_answer(s_txt) == gold
            a_ok = gsm8k_answer(a_txt) == gold
        rows.append({"idx": idx, "spec_correct": int(s_ok), "ar_correct": int(a_ok),
                     "identical_output": int(list(spec.tokens) == list(ar)),
                     "spec_len": len(spec.tokens), "ar_len": len(ar)})
        print(f"[E1D] {a.benchmark} {i+1}/{len(test_ids)} idx={idx} "
              f"spec={'PASS' if s_ok else 'fail'} ar={'PASS' if a_ok else 'fail'}",
              flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"quality_{a.benchmark}_{a.precision}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    spec_acc = sum(r["spec_correct"] for r in rows) / n
    ar_acc = sum(r["ar_correct"] for r in rows) / n
    diffs = [r["spec_correct"] - r["ar_correct"] for r in rows]
    lo, hi = boot_ci(diffs)
    summary = {
        "benchmark": a.benchmark, "precision": a.precision, "n": n,
        "speculative_score": spec_acc, "ar_score": ar_acc,
        "paired_difference": spec_acc - ar_acc,
        "difference_ci95": [lo, hi],
        "identical_outputs": sum(r["identical_output"] for r in rows),
        "largest_degradation_compatible_with_ci": (-lo if lo is not None else None),
    }
    (out / f"summary_quality_{a.benchmark}_{a.precision}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("[E1D] " + json.dumps(summary), flush=True)
    print(f"[E1D] NOTE: a non-significant difference is not evidence of no loss. "
          f"The largest degradation still compatible with this interval is "
          f"{summary['largest_degradation_compatible_with_ci']}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
