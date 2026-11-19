"""Incremental drafter-cache extension must equal a single causal prefill.

Regression for the review finding: multi-token cached forwards previously ran
UNMASKED (extension tokens attended later tokens in their own block) — latent
for 1-layer drafters, corrupting for >= 2 layers. We compare features and
cache contents between incremental extension and one full causal prefill, for
1-layer AND 2-layer drafters.
"""

from __future__ import annotations

import pytest
import torch

from src.models.drafter import Drafter


def _make(tiny_cfg, tiny_target, num_layers: int) -> Drafter:
    torch.manual_seed(3)
    cfg = tiny_cfg.copy()
    cfg.drafter.num_layers = num_layers
    d = Drafter.from_config(cfg.drafter)
    d.bind_target(tiny_target)
    d.eval()
    return d


@pytest.mark.parametrize("num_layers", [1, 2])
def test_incremental_extension_equals_causal_prefill(tiny_cfg, tiny_target, num_layers):
    drafter = _make(tiny_cfg, tiny_target, num_layers)
    vocab = int(tiny_cfg.target.random_config.vocab_size)
    hidden = int(tiny_cfg.target.random_config.hidden_size)
    g = torch.Generator().manual_seed(11)
    t_full = 9
    tokens = torch.randint(0, vocab, (1, t_full), generator=g)
    feats = torch.randn(1, t_full, hidden, generator=g)
    pos = torch.arange(t_full).unsqueeze(0)

    with torch.no_grad():
        h_ref, cache_ref = drafter(tokens, feats, pos, attn_bias=None, cache=None)
        # split 4 + 3 + 2 with cached multi-token extensions
        h_a, cache = drafter(tokens[:, :4], feats[:, :4], pos[:, :4], attn_bias=None, cache=None)
        h_b, cache = drafter(tokens[:, 4:7], feats[:, 4:7], pos[:, 4:7], attn_bias=None, cache=cache)
        h_c, cache = drafter(tokens[:, 7:], feats[:, 7:], pos[:, 7:], attn_bias=None, cache=cache)

    h_inc = torch.cat([h_a, h_b, h_c], dim=1)
    assert torch.allclose(h_ref, h_inc, atol=1e-5), (
        f"features diverge (num_layers={num_layers}): "
        f"max {(h_ref - h_inc).abs().max():.2e}"
    )
    for layer_idx, ((k_r, v_r), (k_i, v_i)) in enumerate(zip(cache_ref.layers, cache.layers)):
        assert torch.allclose(k_r, k_i, atol=1e-5), f"K cache layer {layer_idx}"
        assert torch.allclose(v_r, v_i, atol=1e-5), f"V cache layer {layer_idx}"


def test_single_token_extension_unaffected(tiny_cfg, tiny_target):
    """t=1 cached extension (full visibility, no mask needed) still matches."""
    drafter = _make(tiny_cfg, tiny_target, 2)
    vocab = int(tiny_cfg.target.random_config.vocab_size)
    hidden = int(tiny_cfg.target.random_config.hidden_size)
    g = torch.Generator().manual_seed(12)
    tokens = torch.randint(0, vocab, (1, 5), generator=g)
    feats = torch.randn(1, 5, hidden, generator=g)
    pos = torch.arange(5).unsqueeze(0)
    with torch.no_grad():
        h_ref, _ = drafter(tokens, feats, pos, attn_bias=None, cache=None)
        h4, cache = drafter(tokens[:, :4], feats[:, :4], pos[:, :4], attn_bias=None, cache=None)
        h1, _ = drafter(tokens[:, 4:], feats[:, 4:], pos[:, 4:], attn_bias=None, cache=cache)
    assert torch.allclose(h_ref[:, -1:], h1, atol=1e-5)
