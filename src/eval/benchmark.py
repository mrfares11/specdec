"""Benchmark harness: acceptance length and throughput across the K sweep.

For each K in ``eval.K_sweep`` the draft tree is rebuilt with ``max_depth=K``
(same node budget and allocation strategy), then ``eval.num_prompts`` prompts
are decoded speculatively and once autoregressively (the baseline). Reported
per K: mean accepted draft tokens per step, committed tokens per step, wall
ms/token for both paths, and the speedup. CSV + plot + console table.

Wall-clock numbers on CPU with tiny models are for plumbing verification only;
real numbers come from the pod.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from ..common.device import resolve_device, resolve_dtype, sync
from ..common.modeling import FeatureTap, load_target, load_tokenizer, validate_feature_config
from ..common.seeding import seed_all
from ..decode.generate import greedy_generate, speculative_generate
from ..decode.tree import build_from_config
from ..models.drafter import Drafter

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_SURFACE = "#fcfcfb"
_GRID = "#e5e4e0"

_PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about distributed systems.",
    "What is the capital of Lebanon, and what is it known for?",
    "Summarize the plot of Hamlet in one paragraph.",
    "Give three tips for writing fast Python code.",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


def _file_sha256(path: Optional[str]) -> str:
    """SHA-256 of a checkpoint. Chunked, so a multi-GB file costs seconds, not RAM."""
    if not path:
        return ""
    f = Path(str(path))
    if not f.is_file():
        return ""
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_uuid(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip().splitlines()
        idx = device.index or 0
        return out[idx].strip() if idx < len(out) else out[0].strip()
    except Exception:
        try:
            return torch.cuda.get_device_name(device)
        except Exception:
            return ""


def _run_metadata(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> Dict[str, str]:
    """Run-level provenance replicated onto every per-prompt record.

    Per-prompt rows are the unit the statistics need (hierarchical paired
    bootstrap over session / repeat / prompt), and they are only interpretable
    if each row carries the identity of the code, checkpoint and host that
    produced it.
    """
    ckpt = cfg.eval.drafter_ckpt
    return {
        "experiment_id": str(cfg.eval.get("experiment_id", "") or Path(str(cfg.eval.out_dir)).name),
        "experiment_version": str(cfg.eval.get("experiment_version", "1")),
        "git_commit": _git_commit(),
        "session_id": str(os.environ.get("SPECDEC_SESSION_ID", "") or ""),
        "run_id": uuid.uuid4().hex[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_id": platform.node(),
        "gpu_uuid": _gpu_uuid(device),
        "verifier_revision": "{}@{}".format(
            cfg.target.get("model_id", "") or cfg.target.get("source", ""),
            cfg.target.get("revision", None) or "default"),
        "drafter_name": "" if ckpt is None else Path(str(ckpt)).parent.name,
        "drafter_checkpoint_sha": _file_sha256(None if ckpt is None else str(ckpt)),
        "precision": str(dtype).replace("torch.", ""),
        "thread_cap": str(os.environ.get("OMP_NUM_THREADS", "")) or str(torch.get_num_threads()),
        "seed": str(int(cfg.runtime.seed)),
    }


def _heldout_prompts(cfg: DictConfig, n: int) -> List[str]:
    """First user turns of held-out ShareGPT rows (beyond the training slice)."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(str(cfg.data.dataset_name), name=cfg.data.dataset_config,
                      split=str(cfg.data.split))
    offset = int(cfg.eval.heldout_offset)
    out: List[str] = []
    i = offset
    while len(out) < n and i < len(ds):
        conv = ds[i].get(str(cfg.data.text_field)) or []
        first = next((t.get("value") for t in conv
                      if str(t.get("from", "")).lower() == "human" and t.get("value")), None)
        if first:
            out.append(str(first)[:2000])
        i += 1
    if len(out) < n:
        raise ValueError(f"only {len(out)} held-out prompts past offset {offset}")
    return out


# Standard-benchmark prompt sources. For LOSSLESS speculative decoding these
# are prompt sets only — output quality is the target model's by construction,
# so per-benchmark we measure accept length / speedup, never task scores.
_BENCH_SOURCES = {
    # eval.prompt_source -> (hf dataset, config, split, extractor)
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", None, "train",
                lambda row: (row.get("prompt") or [None])[0]),
    "humaneval": ("openai/openai_humaneval", None, "test",
                  lambda row: "Complete the following Python function:\n\n" + str(row.get("prompt", ""))),
    "gsm8k": ("openai/gsm8k", "main", "test",
              lambda row: str(row.get("question", ""))),
    "alpaca": ("tatsu-lab/alpaca", None, "train",
               lambda row: str(row.get("instruction", "")) +
               ("\n\n" + str(row["input"]) if row.get("input") else "")),
    "mbpp": ("google-research-datasets/mbpp", "full", "test",
             lambda row: str(row.get("text", "")) +
             "\n\nWrite the Python function. Test cases:\n" +
             "\n".join(str(t) for t in (row.get("test_list") or [])[:1])),
}


# Frozen. Changing it silently redefines every dev/test prompt set, so it is a
# constant in code rather than a config knob.
_SPLIT_SEED = 20260902


def _benchmark_prompts(source: str, n: int, split: str = "legacy",
                       dev_size: int = 200) -> List[tuple]:
    """``n`` prompts as ``(dataset_index, text)`` pairs.

    ``split``:
      ``legacy`` -- the first ``n`` usable rows in dataset order. This is the
        selection every pre-plan result used; kept so those results stay
        reproducible, but it does NOT separate development from test.
      ``dev`` / ``test`` -- disjoint pools carved from one frozen permutation.
        Development prompts are for debugging and for choosing checkpoints,
        thread caps and configurations. Test prompts are reported and must
        never drive a choice.
    """
    from datasets import load_dataset  # noqa: PLC0415

    name, config, hf_split, extract = _BENCH_SOURCES[source]
    ds = load_dataset(name, config, split=hf_split)

    usable: List[tuple] = []
    for idx, row in enumerate(ds):
        text = extract(row)
        if text:
            usable.append((idx, str(text)[:2000]))

    if split == "legacy":
        pool = usable
    elif split in ("dev", "test"):
        # dev_size is a CAP, not a fixed count. These benchmarks differ in size
        # by three orders of magnitude (MT-Bench 80 rows, Alpaca ~52k), so a
        # fixed count silently consumed small datasets entirely -- HumanEval's
        # 164 rows all landed in dev and the test pool came back empty. Reserve
        # at most a third, so the reported split always keeps the majority.
        n_dev = max(1, min(dev_size, len(usable) // 3))
        order = list(range(len(usable)))
        random.Random(_SPLIT_SEED).shuffle(order)
        chosen = order[:n_dev] if split == "dev" else order[n_dev:]
        # Iterate in permutation order, so "the first n" is a stable frozen choice.
        pool = [usable[i] for i in chosen]
    else:
        raise ValueError(f"eval.prompt_split must be legacy|dev|test, got {split!r}")

    if len(pool) < n:
        raise ValueError(
            f"{source}/{split}: only {len(pool)} prompts available, wanted {n}"
            + (" (reduce eval.prompt_dev_size?)" if split != "legacy" else "")
        )
    return pool[:n]


def _prompt_ids(cfg: DictConfig, device: torch.device) -> List[tuple]:
    """Prompt token batches per ``eval.prompt_source``:
    builtin canned prompts, held-out ShareGPT turns, or random ids (tiny targets)."""
    n = int(cfg.eval.num_prompts)
    split = str(cfg.eval.get("prompt_split", "legacy"))
    dev_size = int(cfg.eval.get("prompt_dev_size", 200))
    if cfg.target.source == "hub":
        tok = load_tokenizer(cfg)
        source = str(cfg.eval.prompt_source)
        if source == "sharegpt":
            pairs = list(enumerate(_heldout_prompts(cfg, n)))
        elif source in _BENCH_SOURCES:
            pairs = _benchmark_prompts(source, n, split=split, dev_size=dev_size)
        else:
            pairs = [(i, _PROMPTS[i % len(_PROMPTS)]) for i in range(n)]
        return [
            (
                idx,
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device),
            )
            for idx, p in pairs
        ]
    g = torch.Generator().manual_seed(int(cfg.runtime.seed))
    vocab = int(cfg.target.random_config.vocab_size)
    return [(i, torch.randint(0, vocab, (1, 8), generator=g).to(device)) for i in range(n)]


def run(cfg: DictConfig) -> Path:
    """Run the benchmark per ``cfg.eval``; returns the CSV path."""
    seed_all(int(cfg.runtime.seed))
    validate_feature_config(cfg)
    device = resolve_device(str(cfg.runtime.device))
    dtype = resolve_dtype(str(cfg.runtime.dtype), device)
    target = load_target(cfg, device, dtype)
    drafter = Drafter.from_config(cfg.drafter).to(device=device, dtype=dtype)
    drafter.bind_target(target)
    if cfg.eval.drafter_ckpt is not None:
        state = torch.load(str(cfg.eval.drafter_ckpt), map_location=device)
        drafter.load_state_dict(state["model"])
        print(f"[eval] loaded drafter from {cfg.eval.drafter_ckpt} (step {state.get('step')})")
    else:
        print("[eval] WARNING: random-init drafter (no eval.drafter_ckpt) -- smoke run only")
    drafter.eval()

    prompts = _prompt_ids(cfg, device)
    greedy = bool(cfg.eval.greedy)
    temperature = float(cfg.sampling.temperature)
    max_new = int(cfg.eval.max_new_tokens)
    layer_index = int(cfg.data.layer_index)
    tap = FeatureTap.from_config(cfg, target=target)

    split_name = str(cfg.eval.get("prompt_split", "legacy"))
    meta = _run_metadata(cfg, device, dtype)
    per_prompt: List[Dict[str, object]] = []

    rows: List[Dict[str, float]] = []
    for k in [int(x) for x in cfg.eval.K_sweep]:
        tree_cfg = OmegaConf.merge(cfg.tree, {"max_depth": k})
        tree = build_from_config(tree_cfg)

        spec_s = 0.0
        spec_tokens = 0
        ar_s = 0.0
        ar_tokens = 0
        accepts: List[int] = []
        steps = 0
        phase_totals: Dict[str, float] = {}
        # Roadmap M5: record output-equivalence status per cell. Both outputs are
        # already produced here; comparing them costs nothing and is done OUTSIDE
        # the timed brackets so it cannot affect any reported latency.
        equivalent = True if greedy else None
        for prompt_idx, ids in prompts:
            gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
            sync(device)
            t0 = time.perf_counter()
            res = speculative_generate(
                target, drafter, tree, ids, max_new,
                temperature=temperature, greedy=greedy,
                layer_index=layer_index, tap=tap, generator=gen,
                kv_gather=bool(cfg.decode.kv_gather),
                dynamic_tree=bool(cfg.decode.get("dynamic_tree", False)),
                commit_fold=bool(cfg.decode.get("commit_fold", True)),
                profile_timing=bool(cfg.decode.profile_timing),
            )
            for phase, secs in res.timings.items():
                phase_totals[phase] = phase_totals.get(phase, 0.0) + secs
            sync(device)
            spec_dt = time.perf_counter() - t0
            spec_s += spec_dt
            spec_tokens += len(res.tokens)
            accepts.extend(res.accept_lengths)
            steps += res.num_steps

            sync(device)
            t0 = time.perf_counter()
            ar = greedy_generate(target, ids, max_new)
            sync(device)
            ar_dt = time.perf_counter() - t0
            ar_s += ar_dt
            ar_tokens += len(ar)

            # --- per-prompt record: the unit the statistics actually need ---
            n_cmp = min(len(res.tokens), len(ar))
            prompt_match = (list(res.tokens[:n_cmp]) == list(ar[:n_cmp])) if greedy else None
            ph = res.timings
            steps_p = max(res.num_steps, 1)
            rec: Dict[str, object] = dict(meta)
            rec.update({
                "tree_type": "dynamic" if bool(cfg.decode.get("dynamic_tree", False)) else
                             (f"static:{Path(str(cfg.tree.parents_file)).stem}"
                              if cfg.tree.get("parents_file", None) else str(cfg.tree.allocation)),
                "tree_budget": int(tree.num_nodes),
                "K": k,
                "benchmark": str(cfg.eval.prompt_source),
                "prompt_split": split_name,
                "prompt_id": prompt_idx,
                "input_tokens": int(ids.shape[1]),
                "output_tokens": len(res.tokens),
                "total_decode_ms": spec_dt * 1e3,
                "ms_per_token": spec_dt * 1e3 / max(len(res.tokens), 1),
                "ar_total_ms": ar_dt * 1e3,
                "ar_ms_per_token": ar_dt * 1e3 / max(len(ar), 1),
                "prefill_ms": ph.get("prefill", float("nan")) * 1e3,
                "topology_ms": ph.get("topology", float("nan")) * 1e3 / steps_p,
                "mask_ms": ph.get("mask", float("nan")) * 1e3 / steps_p,
                "draft_ms": ph.get("draft", float("nan")) * 1e3 / steps_p,
                "target_tree_forward_ms": ph.get("tree_forward", float("nan")) * 1e3 / steps_p,
                "verify_ms": ph.get("verify", float("nan")) * 1e3 / steps_p,
                "kv_commit_ms": ph.get("commit", float("nan")) * 1e3 / steps_p,
                "committed_tokens": len(res.tokens),
                "verifier_forwards": res.num_steps,
                "tokens_per_verifier_forward": len(res.tokens) / steps_p,
                "mean_accept_len": res.mean_accept_length,
                "output_token_hash": hashlib.sha256(
                    ",".join(str(t) for t in res.tokens).encode()).hexdigest()[:16],
                "ar_output_token_hash": hashlib.sha256(
                    ",".join(str(t) for t in ar).encode()).hexdigest()[:16],
                "exact_match_to_ar": "" if prompt_match is None else
                                     ("yes" if prompt_match else "NO"),
                "rejected": "",
                "rejection_reason": "",
            })
            per_prompt.append(rec)

            # Speculative decoding COMMITS whole accepted spans, so it breaks
            # only after exceeding max_new_tokens and can overshoot; greedy_generate
            # stops exactly at max_new_tokens. Equivalence is therefore agreement on
            # the common prefix -- comparing full lists fails on length alone.
            if greedy:
                n = min(len(res.tokens), len(ar))
                if list(res.tokens[:n]) != list(ar[:n]):
                    equivalent = False

        mean_accept = sum(accepts) / max(len(accepts), 1)
        row = {
            "K": k,
            "num_nodes": tree.num_nodes,
            "mean_accept_len": mean_accept,
            "tokens_per_step": spec_tokens / max(steps, 1),
            "spec_ms_per_token": spec_s * 1e3 / max(spec_tokens, 1),
            "ar_ms_per_token": ar_s * 1e3 / max(ar_tokens, 1),
        }
        row["speedup"] = row["ar_ms_per_token"] / max(row["spec_ms_per_token"], 1e-9)
        # Column deliberately renamed: an earlier revision wrote `output_equivalent`
        # using a length-sensitive comparison that reported NO for every cell. The
        # new name guarantees those stale values can never be read as if correct.
        row["lossless_vs_ar"] = ("n/a(sampling)" if equivalent is None
                                 else ("yes" if equivalent else "NO"))
        if phase_totals:
            for phase, secs in phase_totals.items():
                row[f"t_{phase}_ms_per_step"] = secs * 1e3 / max(steps, 1)
        rows.append(row)
        print(f"[eval] K={k:>3} accept={mean_accept:5.2f} tok/step={row['tokens_per_step']:5.2f} "
              f"spec={row['spec_ms_per_token']:7.2f} ms/tok  ar={row['ar_ms_per_token']:7.2f} ms/tok  "
              f"speedup={row['speedup']:4.2f}x")

    out_dir = Path(str(cfg.eval.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # --- result bundle: raw per-prompt rows + provenance -------------------
    if per_prompt:
        raw_path = out_dir / "raw_prompt_results.csv"
        with raw_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_prompt[0].keys()))
            w.writeheader()
            w.writerows(per_prompt)
        print(f"[eval] wrote {raw_path} ({len(per_prompt)} per-prompt rows)")

    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    env = {
        **{k: v for k, v in meta.items()
           if k in ("git_commit", "host_id", "gpu_uuid", "precision", "thread_cap",
                    "timestamp_utc", "verifier_revision", "drafter_checkpoint_sha", "seed")},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "cpu_count": os.cpu_count(),
    }
    try:
        import transformers  # noqa: PLC0415
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    manifest = []
    for f_ in sorted(out_dir.iterdir()):
        if f_.is_file() and f_.name != "SHA256SUMS":
            h = hashlib.sha256(f_.read_bytes()).hexdigest()
            manifest.append(f"{h}  {f_.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    _plot(rows, out_dir / "benchmark.png")
    print(f"[eval] wrote {csv_path} and {out_dir / 'benchmark.png'}")
    return csv_path


def _plot(rows: List[Dict[str, float]], path: Path) -> None:
    """Accept length vs K, and ms/token (speculative vs autoregressive) vs K."""
    x = [r["K"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=_SURFACE)

    ax = axes[0]
    ax.set_facecolor(_SURFACE)
    ax.plot(x, [r["mean_accept_len"] for r in rows], color=_BLUE, linewidth=2,
            marker="o", markersize=5, zorder=3)
    ax.set_title("Mean accepted draft tokens per step", color=_INK, fontsize=11)
    ax.set_ylabel("accepted tokens", color=_INK_2)

    ax = axes[1]
    ax.set_facecolor(_SURFACE)
    ax.plot(x, [r["spec_ms_per_token"] for r in rows], color=_BLUE, linewidth=2,
            marker="o", markersize=5, zorder=3, label="speculative")
    ax.plot(x, [r["ar_ms_per_token"] for r in rows], color=_ORANGE, linewidth=2,
            marker="o", markersize=5, zorder=3, label="autoregressive")
    ax.set_title("Wall-clock cost per committed token", color=_INK, fontsize=11)
    ax.set_ylabel("ms / token", color=_INK_2)
    ax.legend(frameon=False, labelcolor=_INK_2)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_xlabel("draft horizon K", color=_INK_2)
        ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=_INK_2)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.set_ylim(bottom=0)

    fig.suptitle("Speculative decoding benchmark across the K sweep", color=_INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
