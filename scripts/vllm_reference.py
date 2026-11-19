"""Optimized-engine AR reference row (context for our same-harness baseline).

Measures vLLM greedy decode ms/token on the SAME prompts/lengths as the
benchmark harness. Reported as external context only — our speedups remain
same-harness paired measurements.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--prompts", type=int, default=80)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = []
    for row in ds:
        p = (row.get("prompt") or [None])[0]
        if p:
            texts.append(tok.apply_chat_template([{"role": "user", "content": str(p)[:2000]}],
                                                 add_generation_prompt=True, tokenize=False))
        if len(texts) >= args.prompts:
            break

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=False, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    t0 = time.perf_counter()
    outs = llm.generate(texts, sp)          # batched: engine's best case
    batched = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)

    t0 = time.perf_counter()               # bs=1 sequential: latency-comparable
    seq_tok = 0
    for t in texts[:20]:
        o = llm.generate([t], sp)
        seq_tok += len(o[0].outputs[0].token_ids)
    seq = time.perf_counter() - t0

    res = {"model": args.model, "prompts": len(texts), "max_new": args.max_new,
           "batched_ms_per_token": 1000 * batched / max(ntok, 1),
           "batched_tokens": ntok,
           "sequential_bs1_ms_per_token": 1000 * seq / max(seq_tok, 1),
           "sequential_prompts": 20,
           "note": "external context only; paper speedups are same-harness paired"}
    print(json.dumps(res, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
