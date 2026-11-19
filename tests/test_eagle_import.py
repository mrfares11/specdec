"""EAGLE-checkpoint import path: key remapping, architecture flags, and the
pre-final-norm feature tap. CPU-only, tiny random weights, no downloads.

The lossless invariant must hold for an EAGLE-shaped drafter driven by the
prenorm tap exactly as it does for ours — verification never trusts the
drafter, so greedy speculative output must still equal greedy AR decoding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.common.modeling import FeatureTap, validate_feature_config
from src.decode.generate import greedy_generate, speculative_generate
from src.decode.tree import build_from_config
from src.models.drafter import Drafter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_eagle_weights import remap_eagle_state_dict  # noqa: E402

MAX_NEW = 24


def _eagle_format_state(
    hidden: int, inter: int, heads: int, kv_heads: int, vocab: int, with_bias: bool
):
    """A state dict with the OFFICIAL EAGLE key names/shapes (layer 0 only).

    ``with_bias`` covers both released variants: the LLaMA3-Instruct-8B
    checkpoint ships fc.weight only; some other EAGLE releases add fc.bias.
    """
    head_dim = hidden // heads
    g = torch.Generator().manual_seed(7)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.02

    return {
        "embed_tokens.weight": r(vocab, hidden),  # must be dropped by the remap
        "fc.weight": r(hidden, 2 * hidden),
        **({"fc.bias": r(hidden)} if with_bias else {}),
        "layers.0.self_attn.q_proj.weight": r(heads * head_dim, hidden),
        "layers.0.self_attn.k_proj.weight": r(kv_heads * head_dim, hidden),
        "layers.0.self_attn.v_proj.weight": r(kv_heads * head_dim, hidden),
        "layers.0.self_attn.o_proj.weight": r(hidden, heads * head_dim),
        "layers.0.mlp.gate_proj.weight": r(inter, hidden),
        "layers.0.mlp.up_proj.weight": r(inter, hidden),
        "layers.0.mlp.down_proj.weight": r(hidden, inter),
        "layers.0.post_attention_layernorm.weight": torch.ones(hidden),
        # NOTE: no layers.0.input_layernorm.weight — EAGLE layer 0 has none.
    }


def _eagle_drafter(tiny_cfg, tiny_target, fc_bias: bool = False) -> Drafter:
    cfg = OmegaConf.merge(
        tiny_cfg.drafter, {"fc_bias": fc_bias, "layer0_no_input_norm": True}
    )
    drafter = Drafter.from_config(cfg)
    drafter.bind_target(tiny_target)
    return drafter


@pytest.mark.parametrize("with_bias", [False, True], ids=["llama3-nobias", "with-bias"])
def test_remap_loads_strict(tiny_cfg, tiny_target, with_bias):
    d = tiny_cfg.drafter
    state = _eagle_format_state(
        int(d.hidden_size), int(d.intermediate_size), int(d.num_heads),
        int(d.num_kv_heads), int(tiny_cfg.target.random_config.vocab_size),
        with_bias=with_bias,
    )
    mapped = remap_eagle_state_dict(state)
    assert "embed_tokens.weight" not in mapped
    assert ("fc.bias" in mapped) == with_bias
    drafter = _eagle_drafter(tiny_cfg, tiny_target, fc_bias=with_bias)
    drafter.load_state_dict(mapped, strict=True)  # raises on any mismatch
    # the imported weights actually run
    h, _ = drafter(
        torch.tensor([[1, 2, 3]]),
        torch.zeros(1, 3, int(d.hidden_size)),
        torch.arange(3).unsqueeze(0),
    )
    assert h.shape == (1, 3, int(d.hidden_size))


def test_remap_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unmapped"):
        remap_eagle_state_dict({"totally.unknown.weight": torch.zeros(1)})


def test_default_drafter_has_no_eagle_keys(tiny_cfg):
    """Default construction must be unchanged: no fc.bias, input_norm present."""
    drafter = Drafter.from_config(tiny_cfg.drafter)
    keys = set(drafter.state_dict().keys())
    assert "fc.bias" not in keys
    assert "layers.0.input_norm.weight" in keys


def test_eagle_flags_shape_state_dict(tiny_cfg, tiny_target):
    keys = set(_eagle_drafter(tiny_cfg, tiny_target, fc_bias=True).state_dict().keys())
    assert "fc.bias" in keys
    assert "layers.0.input_norm.weight" not in keys  # Identity has no params
    keys_nb = set(_eagle_drafter(tiny_cfg, tiny_target).state_dict().keys())
    assert "fc.bias" not in keys_nb  # LLaMA3 checkpoint layout


def test_prenorm_tap_is_the_norm_input(tiny_cfg, tiny_target):
    """norm(tap output) == hidden_states[-1], and the tap is NOT hidden_states[-2]."""
    tap = FeatureTap("prenorm_last", -2, target=tiny_target)
    try:
        ids = torch.randint(0, int(tiny_cfg.target.random_config.vocab_size), (1, 12))
        with torch.no_grad():
            out = tiny_target(input_ids=ids, output_hidden_states=True)
            feats = tap.take(out)
            renormed = tiny_target.model.norm(feats)
        assert torch.allclose(renormed, out.hidden_states[-1], atol=1e-5)
        assert not torch.allclose(feats, out.hidden_states[-2], atol=1e-3)
    finally:
        tap.remove()


def test_validate_rejects_prenorm_without_final_norm(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.data.feature_tap = "prenorm_last"
    cfg.drafter.apply_final_norm = False
    with pytest.raises(ValueError, match="prenorm_last"):
        validate_feature_config(cfg)
    cfg.drafter.apply_final_norm = True
    validate_feature_config(cfg)  # valid pairing passes


@pytest.mark.parametrize("mode", ["fold", "gather", "replay"])
@pytest.mark.parametrize("seed", [0, 1])
def test_eagle_shaped_drafter_is_still_lossless(tiny_cfg, tiny_target, mode, seed):
    """Greedy equivalence with EAGLE flags + prenorm tap, in every decode mode."""
    modes = {
        "fold": dict(commit_fold=True),
        "gather": dict(commit_fold=False, kv_gather=True),
        "replay": dict(commit_fold=False, kv_gather=False),
    }
    cfg = tiny_cfg.copy()
    cfg.tree.max_depth = 3
    tree = build_from_config(cfg.tree)
    vocab = int(cfg.target.random_config.vocab_size)
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, vocab, (1, 8), generator=g)

    torch.manual_seed(3)
    drafter = _eagle_drafter(tiny_cfg, tiny_target)  # random weights: quality-free test
    tap = FeatureTap("prenorm_last", -2, target=tiny_target)
    try:
        ar = greedy_generate(tiny_target, prompt, MAX_NEW)
        res = speculative_generate(
            tiny_target, drafter, tree, prompt, MAX_NEW,
            greedy=True, tap=tap, **modes[mode],
        )
    finally:
        tap.remove()
    if res.tokens != ar:
        ids = torch.cat(
            [prompt[0], torch.tensor(ar, dtype=torch.long)]
        ).unsqueeze(0)
        idx = next(i for i, (a, b) in enumerate(zip(ar, res.tokens)) if a != b)
        with torch.no_grad():
            logits = tiny_target(input_ids=ids[:, : prompt.shape[1] + idx]).logits[0, -1]
        gap = float(torch.topk(logits, 2).values.diff().abs())
        assert gap < 1e-3, f"lossless violation at {idx} (gap {gap:.2e})"
        pytest.skip(f"near-tie at {idx}")
    assert len(res.tokens) == MAX_NEW
