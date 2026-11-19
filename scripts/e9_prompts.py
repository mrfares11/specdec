"""E9 --- materialise the frozen dev/test prompt sets as token ids.

Uses the SAME selection code as every reported run (``_benchmark_prompts`` in
src/eval/benchmark.py, frozen permutation seed) and the target tokenizer's chat
template, then cross-checks the per-prompt token counts against the committed
E2 raw rows so the prompt sets are provably identical to the paper's.

    python scripts/e9_prompts.py --tokenizer <dir> --out results/e9/prompts.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.benchmark import _benchmark_prompts  # noqa: E402

SETS = [("mtbench", "test", 50), ("humaneval", "test", 50),
        ("mtbench", "dev", 26), ("humaneval", "dev", 54)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/e9/prompts.pt"))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = []
    for bench, split, n in SETS:
        for idx, text in _benchmark_prompts(bench, n, split=split, dev_size=200):
            ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                          add_generation_prompt=True, return_tensors="pt")[0]
            rows.append({"benchmark": bench, "split": split, "prompt_id": int(idx),
                         "input_ids": ids.long(), "n_tokens": int(ids.shape[0])})
    # cross-check against E2 (test) and E4 sweep (dev) per-prompt rows
    ref = {}
    for f in [ROOT / "results/e2/scaled-static-mtbench-k4-s1-r1/raw_prompt_results.csv",
              ROOT / "results/e2/scaled-static-humaneval-k4-s1-r1/raw_prompt_results.csv",
              ROOT / "results/e4/sweep-humaneval-static-k4-t8/raw_prompt_results.csv"]:
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            ref[(r["benchmark"], r["prompt_split"], int(r["prompt_id"]))] = int(r["input_tokens"])
    n_ok = n_chk = 0
    for r in rows:
        key = (r["benchmark"], r["split"], r["prompt_id"])
        if key in ref:
            n_chk += 1
            if ref[key] == r["n_tokens"]:
                n_ok += 1
            else:
                print(f"MISMATCH {key}: committed input_tokens={ref[key]} here={r['n_tokens']}")
    print(f"[E9] {len(rows)} prompts; token-count cross-check {n_ok}/{n_chk} match")
    print("  eos ids:", tok.eos_token_id, "| pad:", tok.pad_token_id)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "eos_token_id": int(tok.eos_token_id)}, a.out)
    print(f"[E9] wrote {a.out}")
    return 0 if n_ok == n_chk else 1


if __name__ == "__main__":
    raise SystemExit(main())
