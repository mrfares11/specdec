"""Drafter training loop over extracted feature shards.

Teacher-forced step plus HASS-style multi-step rollout: rollout step ``s``
re-feeds the previous step's PREDICTED features as inputs (shifted by one), so
the drafter learns to stay accurate over the horizons it will actually draft
(``train.rollout_K``, which interpolates from the global K). This is a
simplified rollout scheme — the exact scheme (per-step weights, detach policy,
tree-structured rollouts) is RQ2 territory; the knobs live in ``cfg.train``.

The target is loaded ONLY to bind its frozen embedding/norm/head to the
drafter. It receives no gradients; checkpoints contain drafter weights only.

Checkpoints (``train.ckpt_dir``, on /workspace): ``latest.pt`` after every
save plus ``step_<n>.pt`` snapshots; auto-resume from ``latest.pt`` when
``train.resume`` is true. Long runs go under tmux.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from ..common.device import resolve_device, resolve_dtype
from ..common.modeling import load_target, validate_feature_config
from ..common.seeding import seed_all
from ..data.shards import ShardReader
from ..models.drafter import Drafter
from .losses import chunked_soft_ce_backward, drafter_losses


def _batches(
    reader: ShardReader,
    batch_size: int,
    generator: torch.Generator,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Endless shuffled batches: (tokens [B, T], feats [B, T, H], mask [B, T]).

    Right-pads to the batch max length. Samples shorter than 2 tokens are
    skipped. Loads one shard at a time (reader caches the last shard file).
    """
    sids = [s for s in reader.sample_ids()]
    if not sids:
        raise ValueError("no samples in shard dir — run extraction first")
    # Locality-preserving shuffle. A global randperm over sharded storage
    # makes nearly every read a shard-cache miss: measured on the M3 pod as
    # ~16 GB of shard re-reads PER STEP (4 TB total by step 1150) and
    # 25-38 s/step with the GPU idle. Shuffling SHARD order per epoch and
    # sample order WITHIN each shard keeps one shard resident at a time.
    # Same seeded-generator determinism contract; batches are shard-local
    # (mild intra-batch correlation — recorded in RESULTS.md).
    by_shard = [
        [sid for sid in shard["samples"]] for shard in reader.manifest["shards"]
    ]
    by_shard = [ids_in for ids_in in by_shard if ids_in]
    while True:
        buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for si in torch.randperm(len(by_shard), generator=generator).tolist():
            ids_in = by_shard[si]
            for j in torch.randperm(len(ids_in), generator=generator).tolist():
                hidden, ids = reader.read_sample(ids_in[j])
                if ids.shape[0] < 2:
                    continue
                buf.append((ids, hidden))
                if len(buf) == batch_size:
                    yield _collate(buf)
                    buf = []
        if buf:
            yield _collate(buf)


def _collate(
    items: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(ids.shape[0] for ids, _ in items)
    b = len(items)
    hidden_size = items[0][1].shape[1]
    tokens = torch.zeros(b, max_len, dtype=torch.long)
    feats = torch.zeros(b, max_len, hidden_size, dtype=torch.float32)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    for i, (ids, h) in enumerate(items):
        t = ids.shape[0]
        tokens[i, :t] = ids
        feats[i, :t] = h.float()
        mask[i, :t] = True
    return tokens, feats, mask


def _atomic_save(state: Dict, path: Path) -> None:
    """torch.save via tmp + os.replace — a mid-write crash never corrupts a checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _lr_lambda(step: int, warmup: int, max_steps: int) -> float:
    """Linear warmup then cosine decay to 10%."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    frac = (step - warmup) / max(max_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))


def _rollout_backward(
    drafter: Drafter,
    tokens: torch.Tensor,
    feats: torch.Tensor,
    mask: torch.Tensor,
    cfg: DictConfig,
    loss_scale: float,
) -> Dict[str, float]:
    """Teacher-forced + rollout loss for one micro-batch, with PER-STEP backward.

    Every rollout step materializes two full-vocab [B, T, V] float32 tensors
    (drafter and target logits) whose softmax activations autograd would
    otherwise retain until the end of the loop — tens of GB at pod config
    (B=4, T=2048, V=128k, K=8). Backwarding each step's share of the loss
    immediately (``retain_graph`` only for the feature chain later steps
    reuse) frees those activations step by step. The accumulated gradients
    are mathematically identical to one backward over the summed loss.

    Returns detached per-step-mean loss components for logging.
    """
    device = tokens.device
    # Alignment: input pair (t_i, f_{i-1}) predicts f_i, for i = 1..T-1.
    tok = tokens[:, 1:]
    prev = feats[:, :-1]
    tgt = feats[:, 1:]
    m = mask[:, 1:] & mask[:, :-1]
    offset = 1

    n_steps = min(max(int(cfg.train.rollout_K), 1), int(tok.shape[1]))
    detach = bool(cfg.train.rollout_detach)
    # Per-step loss weights ∝ decay^s, normalized to sum 1. decay=1 recovers
    # the uniform average; decay<1 anchors early steps (RQ2 finding: accept
    # length is dominated by step-1 token fidelity).
    decay = float(cfg.train.rollout_step_decay)
    raw_w = [decay**s for s in range(n_steps)]
    w_sum = sum(raw_w)
    step_w = [w / w_sum for w in raw_w]
    agg = {"loss": 0.0, "l_feat": 0.0, "l_logit": 0.0}
    for s in range(n_steps):
        pos = torch.arange(offset, offset + tok.shape[1], device=device).unsqueeze(0).expand(tok.shape[0], -1)
        pred, _ = drafter(tok, prev, pos, attn_bias=None, cache=None)
        # Later steps backprop through pred's feature chain iff not detaching.
        retain = (not detach) and s < n_steps - 1
        ce_chunk = int(cfg.train.ce_chunk)
        w_feat = float(cfg.train.loss.w_feat)
        w_logit = float(cfg.train.loss.w_logit)
        if ce_chunk > 0:
            # Memory-lean path: CE chunked through the frozen head with
            # immediate per-chunk backward into a feature leaf, then ONE
            # traversal of the drafter body carrying feature-loss + injected
            # CE gradients. Enables backprop-through-rollout at pod batch.
            m_f = m.to(pred.dtype)
            denom = m_f.sum().clamp_min(1.0)
            l_feat = (F.smooth_l1_loss(pred, tgt, reduction="none").mean(-1) * m_f).sum() / denom
            leaf = pred.detach().requires_grad_(True)
            l_logit_val = chunked_soft_ce_backward(
                drafter.logits, leaf, tgt, m_f, denom,
                scale=loss_scale * w_logit * step_w[s], chunk_size=ce_chunk,
            )
            leaf_grad = leaf.grad if leaf.grad is not None else torch.zeros_like(pred)
            torch.autograd.backward(
                [l_feat * (loss_scale * w_feat * step_w[s]), pred],
                [None, leaf_grad],
                retain_graph=retain,
            )
            step_vals = {
                "loss": w_feat * float(l_feat.item()) + w_logit * l_logit_val,
                "l_feat": float(l_feat.item()),
                "l_logit": l_logit_val,
            }
        else:
            pred_logits = drafter.logits(pred)
            with torch.no_grad():
                target_logits = drafter.logits(tgt)
            losses = drafter_losses(
                pred, tgt, pred_logits, target_logits, m,
                w_feat=w_feat, w_logit=w_logit,
            )
            (losses["loss"] * (loss_scale * step_w[s])).backward(retain_graph=retain)
            step_vals = {k: float(losses[k].item()) for k in agg}
        for key in agg:
            agg[key] += step_vals[key] * step_w[s]
        if s == n_steps - 1:
            break
        # Next rollout step: predict f_{i+1} from (t_{i+1}, f̂_i).
        nxt = pred.detach() if detach else pred
        prev = nxt[:, :-1]
        tok = tok[:, 1:]
        tgt = tgt[:, 1:]
        m = m[:, 1:]
        offset += 1
    return agg


def _check_shard_meta(meta: Dict, cfg: DictConfig) -> None:
    """Refuse to train on shards extracted under a different feature config.

    Keys absent from the manifest (e.g. hand-built test shards) are skipped —
    only a recorded-and-different value is a provenance error.
    """
    wanted = {"layer_index": int(cfg.data.layer_index), "model_id": str(cfg.target.model_id)}
    for key, want in wanted.items():
        have = meta.get(key)
        if have is not None and have != want:
            raise ValueError(
                f"shard meta mismatch in {cfg.data.out_dir}: manifest has {key}={have!r} "
                f"but the config wants {want!r} — re-extract features or fix the config"
            )


def run(cfg: DictConfig) -> int:
    """Train the drafter per ``cfg.train``; returns the final optimizer step."""
    seed_all(int(cfg.runtime.seed))
    validate_feature_config(cfg)
    device = resolve_device(str(cfg.runtime.device))
    if bool(cfg.train.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Drafter math is always fp32; the frozen target may be loaded compact
    # (train.target_dtype=bfloat16 on the pod) — its three bound modules are
    # then re-bound as fp32 copies, which is lossless for bf16 checkpoints.
    tdtype = resolve_dtype(str(cfg.train.target_dtype), device)
    target = load_target(cfg, device, tdtype)
    drafter = Drafter.from_config(cfg.drafter).to(device)
    cast = None if tdtype == torch.float32 else torch.float32
    drafter.bind_target(target, cast_dtype=cast)
    if cast is not None:
        # Bound modules are fp32 COPIES; the frozen decoder stack is unused in
        # training — free it (~16 GB on the pod) instead of keeping it resident.
        del target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[train] trainable params: {drafter.num_trainable_parameters():,}")

    reader = ShardReader(str(cfg.data.out_dir))
    _check_shard_meta(reader.meta, cfg)

    opt = torch.optim.AdamW(
        drafter.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, int(cfg.train.warmup_steps), int(cfg.train.max_steps))
    )

    ckpt_dir = Path(str(cfg.train.ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    latest = ckpt_dir / "latest.pt"
    if bool(cfg.train.resume) and latest.exists():
        state = torch.load(latest, map_location=device)
        drafter.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step = int(state["step"])
        print(f"[train] resumed from {latest} at step {step}")

    # Seed the data stream with (seed + resume step) so repeated crash/resume
    # cycles do not replay the same head-of-permutation samples over and over.
    g = torch.Generator().manual_seed(int(cfg.runtime.seed) + step)
    batches = _batches(reader, int(cfg.train.batch_size), g)

    log_path = ckpt_dir / "train_log.csv"
    log_new = not log_path.exists()
    log_f = log_path.open("a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if log_new:
        log.writerow(["step", "loss", "l_feat", "l_logit", "lr", "time"])

    accum = max(int(cfg.train.grad_accum), 1)
    max_steps = int(cfg.train.max_steps)
    drafter.train()
    t_last = time.perf_counter()

    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "l_feat": 0.0, "l_logit": 0.0}
        for _ in range(accum):
            tokens, feats, mask = next(batches)
            micro = _rollout_backward(
                drafter, tokens.to(device), feats.to(device), mask.to(device),
                cfg, loss_scale=1.0 / accum,
            )
            for k in agg:
                agg[k] += micro[k] / accum
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), float(cfg.train.max_grad_norm))
        opt.step()
        sched.step()
        step += 1

        if step % int(cfg.train.log_every) == 0 or step == max_steps:
            now = time.perf_counter()
            print(f"[train] step {step}/{max_steps} loss={agg['loss']:.4f} "
                  f"feat={agg['l_feat']:.4f} logit={agg['l_logit']:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({now - t_last:.1f}s)")
            t_last = now
            log.writerow([step, agg["loss"], agg["l_feat"], agg["l_logit"],
                          sched.get_last_lr()[0], time.time()])
            log_f.flush()

        if step % int(cfg.train.save_every) == 0 or step == max_steps:
            state = {
                "step": step,
                "model": drafter.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "train_cfg": OmegaConf.to_container(cfg.train, resolve=True),
            }
            _atomic_save(state, ckpt_dir / f"step_{step}.pt")
            _atomic_save(state, latest)

    log_f.close()
    print(f"[train] done at step {step}; checkpoints in {ckpt_dir}")
    return step
