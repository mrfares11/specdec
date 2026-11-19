"""Phase 0: verifier forward latency vs number of query tokens.

The question this answers: how many draft tokens can ride a single
verification forward before the latency curve leaves the flat, memory-bound
regime? That knee is the budget that makes long-horizon drafting (large K)
free-ish, so it anchors all three RQs.

Methodology (the details that make the numbers trustworthy):
  - a realistic KV-cache prefix of ``profiling.prefix_len`` tokens is built
    ONCE; every iteration gets a FRESH cache view over it (the forward
    torch.cat's new keys onto the cache — reusing one cache across iterations
    would silently grow the prefix and corrupt the curve);
  - cache re-wrapping happens OUTSIDE the timed region;
  - the timed region is bracketed by ``sync(device)`` — a CUDA synchronize on
    GPU, a no-op on CPU — after ``warmup_iters`` untimed warmups per size.

Outputs: ``latency.csv`` + ``latency.png`` under ``profiling.out_dir``, and a
console table.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # file output only; no display on pods
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from ..common.device import resolve_device, resolve_dtype, sync
from ..common.modeling import clone_prefix_cache, load_target
from ..common.seeding import seed_all

# dataviz reference palette (light surface — PNGs are static)
_BLUE = "#2a78d6"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_SURFACE = "#fcfcfb"
_GRID = "#e5e4e0"


def run(cfg: DictConfig) -> Path:
    """Profile the verifier per ``cfg.profiling``; returns the CSV path."""
    seed_all(int(cfg.runtime.seed))
    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)
    vocab = int(target.config.vocab_size)

    p = cfg.profiling
    prefix_len = int(p.prefix_len)
    g = torch.Generator().manual_seed(int(cfg.runtime.seed))
    prefix = torch.randint(0, vocab, (1, prefix_len), generator=g).to(device)

    with torch.no_grad():
        out = target(input_ids=prefix, use_cache=True)
    legacy = out.past_key_values.to_legacy_cache()

    rows: List[Dict[str, float]] = []
    print(f"[profile] model={cfg.target.model_id if cfg.target.source == 'hub' else 'random-tiny'} "
          f"device={device.type} dtype={dtype} prefix={prefix_len}")
    print(f"{'tokens':>7} {'mean ms':>9} {'std':>7} {'min':>8} {'p50':>8} {'ms/tok':>8} {'tok/s':>9}")

    for n in [int(x) for x in p.token_counts]:
        ids = torch.randint(0, vocab, (1, n), generator=g).to(device)
        pos = torch.arange(prefix_len, prefix_len + n, device=device).unsqueeze(0)

        def once() -> float:
            cache = clone_prefix_cache(legacy)  # outside the timed region
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                target(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True)
            sync(device)
            return (time.perf_counter() - t0) * 1e3

        for _ in range(int(p.warmup_iters)):
            once()
        times = [once() for _ in range(int(p.timed_iters))]

        mean = statistics.fmean(times)
        row = {
            "num_query_tokens": n,
            "mean_ms": mean,
            "std_ms": statistics.pstdev(times),
            "min_ms": min(times),
            "p50_ms": statistics.median(times),
            "ms_per_token": mean / n,
            "tokens_per_s": n / (mean / 1e3),
        }
        rows.append(row)
        print(f"{n:>7} {row['mean_ms']:>9.2f} {row['std_ms']:>7.2f} {row['min_ms']:>8.2f} "
              f"{row['p50_ms']:>8.2f} {row['ms_per_token']:>8.3f} {row['tokens_per_s']:>9.0f}")

    out_dir = Path(str(p.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, out_dir / "latency.png", device.type, prefix_len)
    print(f"[profile] wrote {csv_path} and {out_dir / 'latency.png'}")
    return csv_path


def _plot(rows: List[Dict[str, float]], path: Path, device_type: str, prefix_len: int) -> None:
    """Two panels (different scales → separate axes, never dual-axis):
    total latency and per-token latency vs query size."""
    x = [r["num_query_tokens"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=_SURFACE)
    specs = [
        ("mean_ms", "Forward latency (ms)", "Total latency per forward"),
        ("ms_per_token", "Latency per token (ms)", "Amortized cost per query token"),
    ]
    for ax, (key, ylabel, title) in zip(axes, specs):
        y = [r[key] for r in rows]
        ax.set_facecolor(_SURFACE)
        ax.plot(x, y, color=_BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_xlabel("query tokens per forward", color=_INK_2)
        ax.set_ylabel(ylabel, color=_INK_2)
        ax.set_title(title, color=_INK, fontsize=11)
        ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=_INK_2)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.set_ylim(bottom=0)
    fig.suptitle(
        f"Verifier latency vs query-token count  ·  {device_type}, prefix {prefix_len}",
        color=_INK, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
