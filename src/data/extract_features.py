"""Extract target hidden states over a dataset into resumable shards.

Per token we store ``hidden_states[cfg.data.layer_index]`` (see the docs/working-agreements.md
note on -2 vs -1) plus the token ids. Before writing ANYTHING the script
prints a total-disk estimate; ``data.dry_run=true`` stops there.

Dataset modes:
  - ``data.dataset_name: _random`` — synthetic random token ids (offline sanity
    check; works with the random tiny target, no tokenizer needed);
  - anything else — a HuggingFace dataset; rows are either ShareGPT-style
    conversation lists (rendered via the tokenizer's chat template) or plain
    text under ``data.text_field``.

TODO(research): loss masking (train only on assistant tokens) — store a mask
alongside ids when that decision is made.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ..common.device import resolve_device, resolve_dtype
from ..common.modeling import load_target, load_tokenizer
from ..common.seeding import seed_all
from .shards import ShardWriter, estimate_bytes, format_bytes

_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}


def _conversation_to_ids(conv: List[dict], tokenizer, max_seq_len: int) -> Optional[torch.Tensor]:
    """ShareGPT-style [{from, value}, ...] → token ids via the chat template."""
    messages = []
    for turn in conv:
        role = _ROLE_MAP.get(str(turn.get("from", turn.get("role", ""))).lower())
        content = turn.get("value", turn.get("content"))
        if role is None or content is None:
            return None
        messages.append({"role": role, "content": str(content)})
    if not messages:
        return None
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")[0]
    return ids[:max_seq_len]


def _row_to_ids(row: dict, cfg: DictConfig, tokenizer) -> Optional[torch.Tensor]:
    """One dataset row → token ids [T] (None = skip the row)."""
    value = row.get(str(cfg.data.text_field))
    if value is None:
        return None
    if isinstance(value, list):
        return _conversation_to_ids(value, tokenizer, int(cfg.data.max_seq_len))
    ids = tokenizer(str(value), return_tensors="pt", truncation=True,
                    max_length=int(cfg.data.max_seq_len)).input_ids[0]
    return ids


def _iter_token_ids(cfg: DictConfig) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield (sample_id, token_ids) per sample, honoring ``data.max_samples``."""
    d = cfg.data
    if str(d.dataset_name) == "_random":
        # Synthetic mode: random ids over the target vocab (offline sanity).
        vocab = int(cfg.target.random_config.vocab_size) if cfg.target.source == "random" else 128_256
        g = torch.Generator().manual_seed(int(cfg.runtime.seed))
        n = int(d.max_samples or 8)
        for i in range(n):
            t = int(torch.randint(4, int(d.max_seq_len) + 1, (1,), generator=g).item())
            yield f"{d.split}-{i}", torch.randint(0, vocab, (t,), generator=g)
        return

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'datasets>=2.19' (extra: specdec[data]) for extraction") from e

    tokenizer = load_tokenizer(cfg)
    ds = load_dataset(str(d.dataset_name), name=d.dataset_config, split=str(d.split))
    skip = int(d.get("skip_samples", 0) or 0)  # held-out slices: start past the train rows
    limit = skip + (int(d.max_samples) if d.max_samples is not None else len(ds))
    for i, row in enumerate(ds):
        if i < skip:
            continue
        if i >= limit:
            break
        ids = _row_to_ids(row, cfg, tokenizer)
        if ids is None or ids.shape[0] < 2:
            continue
        yield f"{d.split}-{i}", ids


def _corpus_size(cfg: DictConfig) -> int:
    """Number of rows extraction will visit (before per-row filtering)."""
    d = cfg.data
    if str(d.dataset_name) == "_random":
        return int(d.max_samples or 8)
    from datasets import load_dataset  # noqa: PLC0415  (cached by HF across calls)

    ds = load_dataset(str(d.dataset_name), name=d.dataset_config, split=str(d.split))
    n = len(ds)
    return min(int(d.max_samples), n) if d.max_samples is not None else n


def _estimate(cfg: DictConfig, hidden_size: int) -> int:
    """Upfront disk estimate: mean tokens over a BOUNDED probe × corpus size.

    Only ``data.estimate_samples`` rows are tokenized (the loop breaks — no
    full-corpus tokenization pass); the total row count comes from dataset
    metadata. Rows the probe filtered out still count toward the total, so
    the estimate errs slightly high — the safe direction for a disk check.
    """
    d = cfg.data
    probe = int(d.estimate_samples)
    seen_tokens = 0
    seen_samples = 0
    for _, ids in _iter_token_ids(cfg):
        seen_tokens += int(ids.shape[0])
        seen_samples += 1
        if seen_samples >= probe:
            break
    if seen_samples == 0:
        return 0
    est_tokens = int(seen_tokens / seen_samples * _corpus_size(cfg))
    return estimate_bytes(est_tokens, hidden_size, str(d.storage_dtype))


def _hidden_size(cfg: DictConfig) -> int:
    """Target hidden size WITHOUT loading weights (dry runs stay light)."""
    if cfg.target.source == "random":
        return int(cfg.target.random_config.hidden_size)
    from transformers import AutoConfig  # noqa: PLC0415

    return int(
        AutoConfig.from_pretrained(str(cfg.target.model_id), revision=cfg.target.revision).hidden_size
    )


def run(cfg: DictConfig) -> Optional[str]:
    """Run extraction per ``cfg.data``. Returns the shard dir (None on dry run)."""
    seed_all(int(cfg.runtime.seed))
    hidden_size = _hidden_size(cfg)

    est = _estimate(cfg, hidden_size)
    per_shard = estimate_bytes(int(cfg.data.shard_tokens), hidden_size, str(cfg.data.storage_dtype))
    # NOTE: console prints stay ASCII — Windows terminals default to cp1252.
    print(f"[extract] estimated total size: {format_bytes(est)} "
          f"({format_bytes(per_shard)}/shard nominal) -> {cfg.data.out_dir}")
    if bool(cfg.data.dry_run):
        print("[extract] dry_run=true -- stopping before any write (no weights were loaded)")
        return None

    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)

    meta = {
        "model_id": str(cfg.target.model_id),
        "target_source": str(cfg.target.source),
        "layer_index": int(cfg.data.layer_index),
        "max_seq_len": int(cfg.data.max_seq_len),
        "dataset_name": str(cfg.data.dataset_name),
        "split": str(cfg.data.split),
    }
    with ShardWriter(
        cfg.data.out_dir,
        shard_tokens=int(cfg.data.shard_tokens),
        storage_dtype=str(cfg.data.storage_dtype),
        meta=meta,
        resume=bool(cfg.data.resume),
    ) as writer:
        done = writer.completed_ids
        skipped = 0
        written = 0
        for sid, ids in tqdm(_iter_token_ids(cfg), desc="extract", unit="sample"):
            if sid in done:
                skipped += 1
                continue
            with torch.no_grad():
                out = target(
                    input_ids=ids.unsqueeze(0).to(device),
                    output_hidden_states=True,
                )
            hidden = out.hidden_states[int(cfg.data.layer_index)][0]
            writer.add(sid, hidden, ids)
            written += 1
        print(f"[extract] wrote {written} samples ({skipped} already present) -> {cfg.data.out_dir}")
    return str(cfg.data.out_dir)
