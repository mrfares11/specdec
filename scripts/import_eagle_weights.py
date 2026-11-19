"""Convert an official EAGLE drafter checkpoint into our Drafter format.

The official EAGLE checkpoints (e.g. yuhuili/EAGLE-LLaMA3-Instruct-8B) are a
single Llama-style decoder layer plus an FC over [embedding; feature] — the
same family as our drafter, with three deltas: the FC has a bias, layer 0 has
no input RMSNorm, and features are the target's last-layer output pre-final-
norm. The first two are weight-mapping concerns handled here; the third is a
runtime concern handled by ``data.feature_tap=prenorm_last`` (configs/eagle.yaml).

Key mapping (theirs -> ours):
    fc.weight / fc.bias                         -> fc.weight / fc.bias
    layers.N.self_attn.{q,k,v,o}_proj.weight    -> layers.N.attn.{q,k,v,o}_proj.weight
    layers.N.mlp.{gate,up,down}_proj.weight     -> layers.N.mlp.{gate,up,down}_proj.weight
    layers.N.post_attention_layernorm.weight    -> layers.N.post_attn_norm.weight
    layers.N.input_layernorm.weight             -> layers.N.input_norm.weight (absent for N=0)
    embed_tokens.weight                         -> dropped (we bind the frozen target's)

Output is our standard checkpoint dict {"step": 0, "model": state_dict}, loadable
via ``eval.drafter_ckpt`` with drafter.fc_bias=true + drafter.layer0_no_input_norm=true.

Usage:
    python scripts/import_eagle_weights.py --src <hf_repo_id_or_local_path> --out eagle.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_RULES = [
    (re.compile(r"^fc\.(weight|bias)$"), r"fc.\1"),
    (
        re.compile(r"^layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$"),
        r"layers.\1.attn.\2_proj.weight",
    ),
    (
        re.compile(r"^layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$"),
        r"layers.\1.mlp.\2_proj.weight",
    ),
    (
        re.compile(r"^layers\.(\d+)\.post_attention_layernorm\.weight$"),
        r"layers.\1.post_attn_norm.weight",
    ),
    (
        re.compile(r"^layers\.(\d+)\.input_layernorm\.weight$"),
        r"layers.\1.input_norm.weight",
    ),
]
_DROP = re.compile(r"^embed_tokens\.")


def remap_eagle_state_dict(src: dict) -> dict:
    """Map official-EAGLE key names onto our Drafter's; raise on unknown keys."""
    out, dropped, unknown = {}, [], []
    for key, tensor in src.items():
        if _DROP.match(key):
            dropped.append(key)
            continue
        for pat, repl in _RULES:
            if pat.match(key):
                out[pat.sub(repl, key)] = tensor
                break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"unmapped EAGLE checkpoint keys: {unknown}")
    if dropped:
        print(f"[import] dropped (bound from target at runtime): {dropped}")
    return out


def _load_source(src: str) -> dict:
    """Load a state dict from a local file/dir or a HF repo id."""
    p = Path(src)
    if not p.exists():  # HF repo id -> download the checkpoint file
        from huggingface_hub import hf_hub_download

        for fname in ("pytorch_model.bin", "model.safetensors"):
            try:
                p = Path(hf_hub_download(repo_id=src, filename=fname))
                break
            except Exception:
                continue
        else:
            raise FileNotFoundError(f"no checkpoint file found in HF repo '{src}'")
    if p.is_dir():
        for fname in ("pytorch_model.bin", "model.safetensors"):
            if (p / fname).exists():
                p = p / fname
                break
        else:
            raise FileNotFoundError(f"no checkpoint file in {p}")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(p))
    return torch.load(str(p), map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="HF repo id or local ckpt path")
    ap.add_argument("--out", required=True, help="output .pt path (our format)")
    args = ap.parse_args()

    state = _load_source(args.src)
    print(f"[import] source keys: {len(state)}")
    mapped = remap_eagle_state_dict(state)
    print(f"[import] mapped keys: {len(mapped)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "model": mapped}, out_path)
    print(f"[import] wrote {out_path}")


if __name__ == "__main__":
    main()
