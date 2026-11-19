"""Chunked soft-CE must be gradient-exact vs the unchunked path.

The chunked path (losses.chunked_soft_ce_backward + leaf-gradient injection)
exists purely to cap memory; any numerical divergence from the full-tensor
path would silently change RQ2's training. We compare LOSS VALUES and EVERY
PARAMETER GRADIENT between the two paths on identical batches, for both
detached and backprop-through rollouts.
"""

from __future__ import annotations

import pytest
import torch

from src.train.train_drafter import _rollout_backward


def _grads(drafter):
    return {n: p.grad.clone() for n, p in drafter.named_parameters() if p.grad is not None}


def _run_path(tiny_cfg, tiny_drafter, ce_chunk: int, detach: bool, seed: int = 0):
    cfg = tiny_cfg.copy()
    cfg.train.ce_chunk = ce_chunk
    cfg.train.rollout_K = 3
    cfg.train.rollout_detach = detach
    g = torch.Generator().manual_seed(seed)
    vocab = int(cfg.target.random_config.vocab_size)
    hidden = int(cfg.target.random_config.hidden_size)
    b, t = 2, 17  # deliberately not a multiple of the chunk size
    tokens = torch.randint(0, vocab, (b, t), generator=g)
    feats = torch.randn(b, t, hidden, generator=g)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[1, 12:] = False  # ragged batch

    tiny_drafter.zero_grad(set_to_none=True)
    torch.manual_seed(7)  # dropout-free but keep any RNG identical
    vals = _rollout_backward(tiny_drafter, tokens, feats, mask, cfg, loss_scale=0.5)
    grads = _grads(tiny_drafter)
    tiny_drafter.zero_grad(set_to_none=True)
    return vals, grads


@pytest.mark.parametrize("detach", [True, False])
def test_chunked_matches_full(tiny_cfg, tiny_drafter, detach):
    full_vals, full_grads = _run_path(tiny_cfg, tiny_drafter, ce_chunk=0, detach=detach)
    chunk_vals, chunk_grads = _run_path(tiny_cfg, tiny_drafter, ce_chunk=5, detach=detach)

    for key in ("loss", "l_feat", "l_logit"):
        assert chunk_vals[key] == pytest.approx(full_vals[key], rel=1e-5, abs=1e-6), key

    assert full_grads.keys() == chunk_grads.keys()
    assert full_grads, "no gradients produced"
    for name in full_grads:
        assert torch.allclose(full_grads[name], chunk_grads[name], rtol=1e-4, atol=1e-6), (
            f"gradient mismatch in {name}: max abs diff "
            f"{(full_grads[name] - chunk_grads[name]).abs().max():.2e}"
        )


def test_chunk_size_irrelevant(tiny_cfg, tiny_drafter):
    """Any chunk size gives the same gradients (chunking is purely mechanical)."""
    _, g5 = _run_path(tiny_cfg, tiny_drafter, ce_chunk=5, detach=False)
    _, g64 = _run_path(tiny_cfg, tiny_drafter, ce_chunk=64, detach=False)
    for name in g5:
        assert torch.allclose(g5[name], g64[name], rtol=1e-4, atol=1e-6), name


def test_step_decay_zero_equals_single_step_rollout(tiny_cfg, tiny_drafter):
    """decay=0 puts all weight on step 1 → gradients must equal rollout_K=1."""
    cfg_a = tiny_cfg.copy()
    cfg_a.train.rollout_step_decay = 0.0
    _, g_decay0 = _run_path(cfg_a, tiny_drafter, ce_chunk=5, detach=False)

    cfg_b = tiny_cfg.copy()
    cfg_b.train.rollout_K = 1  # overridden inside _run_path? no — _run_path sets rollout_K=3
    _, g_k1 = _run_path_k(cfg_b, tiny_drafter, rollout_k=1)
    for name in g_decay0:
        assert torch.allclose(g_decay0[name], g_k1[name], rtol=1e-4, atol=1e-6), name


def _run_path_k(cfg_in, tiny_drafter, rollout_k: int, seed: int = 0):
    cfg = cfg_in.copy()
    cfg.train.ce_chunk = 5
    cfg.train.rollout_K = rollout_k
    cfg.train.rollout_detach = False
    g = torch.Generator().manual_seed(seed)
    vocab = int(cfg.target.random_config.vocab_size)
    hidden = int(cfg.target.random_config.hidden_size)
    tokens = torch.randint(0, vocab, (2, 17), generator=g)
    feats = torch.randn(2, 17, hidden, generator=g)
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[1, 12:] = False
    tiny_drafter.zero_grad(set_to_none=True)
    torch.manual_seed(7)
    vals = _rollout_backward(tiny_drafter, tokens, feats, mask, cfg, loss_scale=0.5)
    grads = _grads(tiny_drafter)
    tiny_drafter.zero_grad(set_to_none=True)
    return vals, grads


def test_bp_through_rollout_differs_from_detached(tiny_cfg, tiny_drafter):
    """Sanity: detach true/false must NOT be equivalent (else the RQ2 comparison is vacuous)."""
    _, g_bp = _run_path(tiny_cfg, tiny_drafter, ce_chunk=5, detach=False)
    _, g_det = _run_path(tiny_cfg, tiny_drafter, ce_chunk=5, detach=True)
    diffs = [not torch.allclose(g_bp[n], g_det[n], rtol=1e-3) for n in g_bp]
    assert any(diffs), "backprop-through-rollout produced identical grads to detached"
