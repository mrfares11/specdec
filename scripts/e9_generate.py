"""E9 --- exact greedy continuations + target features, streamed on a small CPU.

Batched greedy decoding of the float32 target with its weights re-streamed
from the safetensors shards at every step (one decoder layer resident at a
time, ~0.9 GB) and the KV cache held in float16. Each step costs one pass over
the 16 GB of weights, so prompts are decoded in batches of up to ``--batch``
and the per-step cost is shared by the whole batch. The loop records, for
every position of prompt + continuation, the drafter's feature space
(hidden_states[-2]) and the post-norm state feeding lm_head, exactly as
scripts/e9_target_pass.py does for a fixed sequence -- the two are checked
against each other and against HF's generate() in ``--self-test``.

Continuations are ``--n-cont`` tokens long and ignore EOS on purpose: the
speculative loop verifies tree nodes past an EOS against the target's real
argmaxes, so those positions are needed; the EOS offset is stored.

    python scripts/e9_generate.py --model <dir> --prompts results/e9/prompts.pt \
        --benchmark mtbench --split test --out <data.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e9_target_pass import StreamingLlama  # noqa: E402
from src.models.drafter import RMSNorm, _rope_cos_sin  # noqa: E402


class BatchedStreamingDecoder:
    """Greedy decode a batch of prompts with layer-streamed float32 weights."""

    def __init__(self, sl: StreamingLlama, kv_dtype=torch.float16):
        self.sl = sl
        self.kv_dtype = kv_dtype

    @torch.no_grad()
    def decode(self, prompts: List[torch.Tensor], n_cont: int, log=print) -> List[Dict[str, torch.Tensor]]:
        sl = self.sl
        B = len(prompts)
        lens = [int(p.shape[0]) for p in prompts]
        Lmax = max(lens)
        H = sl.hidden
        neg = torch.finfo(torch.float32).min
        # left-pad prompts; row b occupies columns [Lmax - lens[b], Lmax)
        ids = torch.zeros(B, Lmax, dtype=torch.long)
        pos = torch.zeros(B, Lmax, dtype=torch.long)
        valid = torch.zeros(B, Lmax, dtype=torch.bool)
        for b, p in enumerate(prompts):
            off = Lmax - lens[b]
            ids[b, off:] = p
            pos[b, off:] = torch.arange(lens[b])
            valid[b, off:] = True
        n_layers = sl.n_layers
        feat_after = n_layers - 2  # output of layer n-2 == hidden_states[-2]
        S = Lmax + n_cont
        K_cache = [torch.zeros(B, sl.n_kv, S, sl.head_dim, dtype=self.kv_dtype) for _ in range(n_layers)]
        V_cache = [torch.zeros(B, sl.n_kv, S, sl.head_dim, dtype=self.kv_dtype) for _ in range(n_layers)]
        feats = torch.zeros(B, S, H, dtype=torch.float16)
        finals = torch.zeros(B, S, H, dtype=torch.float16)
        out_ids = torch.zeros(B, S, dtype=torch.long)
        out_ids[:, :Lmax] = ids
        embed = sl._get("model.embed_tokens.weight")
        norm = RMSNorm(H, eps=sl.eps)
        norm.weight.data = sl._get("model.norm.weight")
        head = sl._get("lm_head.weight") if "lm_head.weight" in sl.weight_map else embed
        t0 = time.time()

        def run_step(x: torch.Tensor, cur_pos: torch.Tensor, col0: int, col1: int, bias: torch.Tensor,
                     record_from: int):
            """x [B, T, H] for columns [col0, col1); bias [B,1,T,col1]. Returns final-norm state."""
            cos, sin = _rope_cos_sin(cur_pos, sl.head_dim, sl.theta, torch.float32)
            for i in range(n_layers):
                lyr = sl.layer(i)
                past = (K_cache[i][:, :, :col0].float(), V_cache[i][:, :, :col0].float()) if col0 > 0 else None
                x, (k, v) = lyr(x, cos, sin, bias, past)
                K_cache[i][:, :, col0:col1] = k[:, :, col0:].to(self.kv_dtype)
                V_cache[i][:, :, col0:col1] = v[:, :, col0:].to(self.kv_dtype)
                if i == feat_after:
                    feats[:, col0:col1] = x.to(torch.float16)
                del lyr
            h = norm(x)
            finals[:, col0:col1] = h.to(torch.float16)
            return h

        # ---- prefill: all prompt columns at once
        x = embed[ids]
        bias = torch.full((B, 1, Lmax, Lmax), neg)
        causal = torch.tril(torch.ones(Lmax, Lmax, dtype=torch.bool))
        for b in range(B):
            bias[b, 0][causal & valid[b].unsqueeze(0)] = 0.0
        h = run_step(x, pos, 0, Lmax, bias, 0)
        logits = h[:, -1] @ head.T
        nxt = logits.argmax(dim=-1)
        log(f"  prefill of {B} prompts (Lmax={Lmax}) done, {time.time() - t0:.0f}s")
        # ---- decode n_cont tokens
        cur = torch.tensor(lens)  # next absolute position per row
        for j in range(n_cont):
            col = Lmax + j
            out_ids[:, col] = nxt
            x = embed[nxt].unsqueeze(1)
            bias = torch.full((B, 1, 1, col + 1), neg)
            bias[:, 0, 0, :Lmax][valid] = 0.0
            bias[:, 0, 0, Lmax:] = 0.0
            h = run_step(x, cur.unsqueeze(1), col, col + 1, bias, col)
            logits = h[:, -1] @ head.T
            nxt = logits.argmax(dim=-1)
            cur += 1
            if (j + 1) % 10 == 0 or j == n_cont - 1:
                log(f"  step {j + 1}/{n_cont}, {time.time() - t0:.0f}s")
        # argmax after the last continuation token is `nxt` (not stored: beyond S)
        res = []
        for b in range(B):
            off = Lmax - lens[b]
            res.append({"S": out_ids[b, off:].clone(), "feat": feats[b, off:].clone(),
                        "final": finals[b, off:].clone(), "prompt_len": lens[b]})
        return res


def self_test() -> int:
    import tempfile

    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=211, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=2, rms_norm_eps=1e-5,
                      rope_theta=500000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    ok = True
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)
        sl = StreamingLlama(Path(d))
        dec = BatchedStreamingDecoder(sl, kv_dtype=torch.float32)
        torch.manual_seed(3)
        prompts = [torch.randint(0, 211, (n,)) for n in (5, 12, 9, 12, 3)]
        res = dec.decode(prompts, 15, log=lambda *_: None)
        for p, r in zip(prompts, res):
            with torch.no_grad():
                ref = model.generate(p.unsqueeze(0), max_new_tokens=15, do_sample=False,
                                     eos_token_id=None, pad_token_id=0)[0]
                full = model(input_ids=r["S"].unsqueeze(0), output_hidden_states=True)
            same_tokens = bool((ref == r["S"]).all())
            fe = (full.hidden_states[-2][0] - r["feat"].float()).abs().max().item()
            fi = (full.hidden_states[-1][0] - r["final"].float()).abs().max().item()
            print(f"T={p.shape[0]}: tokens equal={same_tokens} feat err={fe:.1e} final err={fi:.1e}")
            ok &= same_tokens and fe < 1e-2 and fi < 1e-2
        # fp16 KV cache must not change the greedy tokens on this tiny model either
        dec16 = BatchedStreamingDecoder(sl, kv_dtype=torch.float16)
        res16 = dec16.decode(prompts, 15, log=lambda *_: None)
        same16 = all(bool((a["S"] == b["S"]).all()) for a, b in zip(res, res16))
        print("fp16 KV cache reproduces fp32 tokens:", same16)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--benchmark")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out")
    ap.add_argument("--n-cont", type=int, default=140)
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.threads:
        torch.set_num_threads(a.threads)
    P = torch.load(a.prompts)
    rows = [r for r in P["rows"] if r["benchmark"] == a.benchmark and r["split"] == a.split]
    if a.limit:
        rows = rows[: a.limit]
    eos = int(P["eos_token_id"])
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[int, dict] = {}
    if out_path.exists():
        for r in torch.load(out_path)["rows"]:
            done[int(r["prompt_id"])] = r
    todo = [r for r in rows if int(r["prompt_id"]) not in done]
    todo.sort(key=lambda r: r["n_tokens"])  # similar lengths together: less padding
    sl = StreamingLlama(Path(a.model))
    dec = BatchedStreamingDecoder(sl)
    print(f"[E9] {a.benchmark}/{a.split}: {len(rows)} prompts, {len(done)} done, "
          f"{len(todo)} to decode in batches of {a.batch}; threads={torch.get_num_threads()}", flush=True)
    for b0 in range(0, len(todo), a.batch):
        batch = todo[b0 : b0 + a.batch]
        t0 = time.time()
        res = dec.decode([r["input_ids"] for r in batch], a.n_cont, log=lambda m: print(m, flush=True))
        for r, o in zip(batch, res):
            pid = int(r["prompt_id"])
            plen = int(o["prompt_len"])
            cont = o["S"][plen:].tolist()
            eos_off = next((j for j, t in enumerate(cont) if t == eos), -1)
            done[pid] = {"benchmark": a.benchmark, "split": a.split, "prompt_id": pid, "prompt_len": plen,
                         "S": o["S"], "feat": o["feat"], "final": o["final"], "eos_offset": eos_off}
        torch.save({"rows": list(done.values()), "eos_token_id": eos}, out_path)
        print(f"[E9] batch {b0 // a.batch + 1}: {len(batch)} prompts in {time.time() - t0:.0f}s -> {out_path}",
              flush=True)
    n_eos = sum(1 for r in done.values() if 0 <= r["eos_offset"] < 128)
    print(f"[E9] done: {len(done)} prompts, {n_eos} reach EOS within 128 tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
