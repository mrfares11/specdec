"""Drafter training losses.

EAGLE-style pairing: a feature-regression loss (smooth L1 between predicted
and true target features) plus a classification loss (soft cross-entropy of
drafter logits against the target's distribution). NOTE: the target logits
are recomputed from stored features via the drafter's head path
(``Drafter.logits``). That reproduces the target's true logits EXACTLY only
for the pairing ``layer_index=-1`` (post-final-norm features in HF) with
``drafter.apply_final_norm=false``; any other layer choice makes the CE
target an EAGLE-style approximation. The invalid pairing (-1 with the norm
re-applied) is rejected by ``validate_feature_config`` — see docs/working-agreements.md.

MEMORY (measured on H100 80GB, 2026-08-06): at pod scale (B=4, T=2048,
V=128k, rollout_K=8) the unchunked path's per-step full-vocab tensors plus
retained cross-step graphs exceed 80 GB. RESOLVED by
:func:`chunked_soft_ce_backward` (``train.ce_chunk``, default 256): CE runs
through the frozen head in T-chunks with immediate per-chunk backward, so
backprop-through-rollout fits at B=4 (~60-65 GB peak with the bf16 target,
whose unused decoder stack is additionally freed after binding). The
unchunked path (``ce_chunk=0``) remains for CPU-exact equivalence tests.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


def chunked_soft_ce_backward(
    head_fn: Callable[[torch.Tensor], torch.Tensor],
    pred_feats_leaf: torch.Tensor,
    true_feats: torch.Tensor,
    mask: torch.Tensor,
    denom: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
) -> float:
    """Soft-CE over the frozen head, chunked over T, with per-chunk backward.

    ``pred_feats_leaf`` must be a detached leaf (``pred.detach().requires_grad_()``).
    Each chunk projects [B, c, H] features through the FROZEN head, computes its
    share of the masked soft-CE, and backwards immediately — so only one
    [B, c, V] logits pair is ever alive, and gradients accumulate into
    ``pred_feats_leaf.grad``. Exact because the head/norm hold no trainable
    parameters: the caller re-injects the leaf gradient into the main graph via
    ``torch.autograd.backward([...,), pred], [None, leaf.grad])``. This is the
    fix for the measured 80 GB rollout blowup (see module docstring).

    Returns the (detached) total CE value for logging.
    """
    if pred_feats_leaf.grad_fn is not None or not pred_feats_leaf.requires_grad:
        raise ValueError("pred_feats_leaf must be a detached requires_grad leaf")
    t_total = pred_feats_leaf.shape[1]
    total = 0.0
    for t0 in range(0, t_total, chunk_size):
        sl = slice(t0, min(t0 + chunk_size, t_total))
        pl = head_fn(pred_feats_leaf[:, sl])
        with torch.no_grad():
            tl = head_fn(true_feats[:, sl])
        logp = F.log_softmax(pl.float(), dim=-1)
        p_tgt = F.softmax(tl.float(), dim=-1)
        ce = (-(p_tgt * logp).sum(-1) * mask[:, sl].to(logp.dtype)).sum() / denom
        total += float(ce.detach().item())
        (ce * scale).backward()
    return total


def drafter_losses(
    pred_feats: torch.Tensor,
    true_feats: torch.Tensor,
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    w_feat: float = 1.0,
    w_logit: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Masked feature-regression + soft-CE losses.

    Args:
        pred_feats/true_feats: float [B, T, H].
        pred_logits/target_logits: float [B, T, V] (target_logits should be
            detached / computed under no_grad).
        mask: bool [B, T] — True where the position counts toward the loss.
        w_feat/w_logit: loss weights (``cfg.train.loss``).

    Returns:
        dict with ``loss`` (weighted total), ``l_feat``, ``l_logit``.
    """
    m = mask.to(pred_feats.dtype)
    denom = m.sum().clamp_min(1.0)

    l_feat = (F.smooth_l1_loss(pred_feats, true_feats, reduction="none").mean(-1) * m).sum() / denom

    logp = F.log_softmax(pred_logits.float(), dim=-1)
    p_tgt = F.softmax(target_logits.float(), dim=-1)
    l_logit = (-(p_tgt * logp).sum(-1) * m).sum() / denom

    total = w_feat * l_feat + w_logit * l_logit
    return {"loss": total, "l_feat": l_feat, "l_logit": l_logit}
