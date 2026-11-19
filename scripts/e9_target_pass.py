"""E9 --- exact teacher-forced target pass on a CPU with less RAM than the model.

The target (Llama-3-8B-Instruct, 16 GB in bf16) does not fit this machine, so
its weights are streamed from the safetensors shards ONE DECODER LAYER AT A
TIME in float32 (~0.9 GB resident), and every sequence's hidden state is
carried through all 32 layers. One pass over the weights processes every
sequence, so the cost is dominated by matmuls, not disk.

For each input sequence (prompt + candidate continuation) it records

  * ``argmax``  -- the target's greedy next token at every position, which is
                   all that greedy verification needs;
  * ``feat``    -- hidden_states[-2] (the last decoder layer's INPUT), the
                   drafter's feature space, stored in float16;
  * ``final``   -- the post-norm hidden state feeding lm_head, stored in
                   float16, so the full target distribution can be recomputed
                   later for cross-entropy diagnostics.

Correctness is checked by ``--self-test``, which builds a tiny random Llama with
transformers, saves it as safetensors, and compares against HF's own forward.

    python scripts/e9_target_pass.py --model <dir> --seqs in.pt --out out.pt
    python scripts/e9_target_pass.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.drafter import RMSNorm, _DecoderLayer, _rope_cos_sin  # noqa: E402


class StreamingLlama:
    """Layer-streaming float32 forward for a Llama checkpoint in safetensors."""

    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        cfg = json.loads((self.dir / "config.json").read_text())
        self.hidden = int(cfg["hidden_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.inter = int(cfg["intermediate_size"])
        self.eps = float(cfg.get("rms_norm_eps", 1e-5))
        self.theta = float(cfg.get("rope_theta", 10000.0))
        if cfg.get("rope_scaling"):
            raise ValueError("rope_scaling is not implemented (Llama-3-8B needs none)")
        self.head_dim = self.hidden // self.n_heads
        idx_file = self.dir / "model.safetensors.index.json"
        if idx_file.exists():
            self.weight_map: Dict[str, str] = json.loads(idx_file.read_text())["weight_map"]
        else:
            self.weight_map = {}
            with safe_open(str(self.dir / "model.safetensors"), "pt") as f:
                for k in f.keys():
                    self.weight_map[k] = "model.safetensors"
        self._open: Dict[str, object] = {}

    def _get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.dir / shard), "pt")
        return self._open[shard].get_tensor(name).to(torch.float32)

    def layer(self, i: int) -> _DecoderLayer:
        lyr = _DecoderLayer(self.hidden, self.n_heads, self.n_kv, self.inter)
        lyr.input_norm = RMSNorm(self.hidden, eps=self.eps)
        lyr.post_attn_norm = RMSNorm(self.hidden, eps=self.eps)
        p = f"model.layers.{i}."
        sd = {
            "input_norm.weight": self._get(p + "input_layernorm.weight"),
            "post_attn_norm.weight": self._get(p + "post_attention_layernorm.weight"),
            "attn.q_proj.weight": self._get(p + "self_attn.q_proj.weight"),
            "attn.k_proj.weight": self._get(p + "self_attn.k_proj.weight"),
            "attn.v_proj.weight": self._get(p + "self_attn.v_proj.weight"),
            "attn.o_proj.weight": self._get(p + "self_attn.o_proj.weight"),
            "mlp.gate_proj.weight": self._get(p + "mlp.gate_proj.weight"),
            "mlp.up_proj.weight": self._get(p + "mlp.up_proj.weight"),
            "mlp.down_proj.weight": self._get(p + "mlp.down_proj.weight"),
        }
        lyr.load_state_dict(sd)
        lyr.eval()
        return lyr

    @torch.no_grad()
    def run(self, seqs: List[torch.Tensor], *, feat_layer: int = -2,
            log=print) -> List[Dict[str, torch.Tensor]]:
        """Teacher-forced forward over every sequence; see module docstring."""
        t0 = time.time()
        embed = self._get("model.embed_tokens.weight")
        xs = [embed[s.long()] for s in seqs]  # float32 [T, H] each
        del embed
        # hidden_states[k] in HF numbering: k=0 embeddings, k=i+1 output of layer i,
        # the last one post-norm. feat_layer=-2 -> output of layer n_layers-2.
        feat_after = (self.n_layers + 1 + feat_layer) - 1 if feat_layer < 0 else feat_layer - 1
        feats: List[torch.Tensor] = [None] * len(seqs)  # type: ignore[list-item]
        for i in range(self.n_layers):
            lyr = self.layer(i)
            for j, x in enumerate(xs):
                t = x.shape[0]
                pos = torch.arange(t).unsqueeze(0)
                cos, sin = _rope_cos_sin(pos, self.head_dim, self.theta, x.dtype)
                y, _ = lyr(x.unsqueeze(0), cos, sin, None, None)
                xs[j] = y[0]
                if i == feat_after:
                    feats[j] = xs[j].to(torch.float16)
            del lyr
            log(f"  layer {i + 1}/{self.n_layers} done, {time.time() - t0:.0f}s")
        norm = RMSNorm(self.hidden, eps=self.eps)
        norm.weight.data = self._get("model.norm.weight")
        head = self._get("lm_head.weight") if "lm_head.weight" in self.weight_map \
            else self._get("model.embed_tokens.weight")
        out = []
        for j, x in enumerate(xs):
            h = norm(x)
            am = torch.empty(h.shape[0], dtype=torch.long)
            for c0 in range(0, h.shape[0], 64):
                am[c0:c0 + 64] = (h[c0:c0 + 64] @ head.T).argmax(dim=-1)
            out.append({"argmax": am, "feat": feats[j], "final": h.to(torch.float16)})
        log(f"  head done, {time.time() - t0:.0f}s total")
        return out


def self_test() -> int:
    import tempfile

    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                      num_attention_heads=8, num_key_value_heads=2, rms_norm_eps=1e-5,
                      rope_theta=500000.0, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)
        sl = StreamingLlama(Path(d))
        seqs = [torch.randint(0, 257, (n,)) for n in (5, 17, 33)]
        got = sl.run(seqs, log=lambda *_: None)
        ok = True
        for s, g in zip(seqs, got):
            with torch.no_grad():
                ref = model(input_ids=s.unsqueeze(0), output_hidden_states=True)
            feat_err = (ref.hidden_states[-2][0] - g["feat"].float()).abs().max().item()
            fin_err = (ref.hidden_states[-1][0] - g["final"].float()).abs().max().item()
            am_ok = bool((ref.logits[0].argmax(-1) == g["argmax"]).all())
            print(f"T={s.shape[0]}: feat max|err|={feat_err:.2e} final max|err|={fin_err:.2e} argmax equal={am_ok}")
            ok &= feat_err < 1e-2 and fin_err < 1e-2 and am_ok
        print("SELF-TEST", "PASSED" if ok else "FAILED")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="directory with config.json + safetensors")
    ap.add_argument("--seqs", help="torch file: list of 1-D long tensors")
    ap.add_argument("--out", help="torch file for the per-sequence results")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.threads:
        torch.set_num_threads(a.threads)
    seqs = torch.load(a.seqs)
    sl = StreamingLlama(Path(a.model))
    n_tok = sum(int(s.shape[0]) for s in seqs)
    print(f"[E9] {len(seqs)} sequences, {n_tok} tokens, threads={torch.get_num_threads()}", flush=True)
    res = sl.run(seqs, log=lambda m: print(m, flush=True))
    torch.save(res, a.out)
    print(f"[E9] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
