"""Capture the full host/hardware identity required by the experiment protocol.

Run at the START of every timing session and store the output beside the
results. The per-run bundle records a subset; this captures the things that
silently invalidate a comparison if they change underneath you -- driver, CPU
topology, NUMA layout, thread caps, GPU clocks, persistence/MIG/power state.

    python scripts/capture_environment.py --out results/env/session-<id>.json

Degrades gracefully: anything unavailable is recorded as null rather than
omitted, so a missing field is visible instead of being mistaken for absent.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True, timeout=30
        ).strip()
    except Exception:
        return None


def _nvidia(fields: str) -> str | None:
    return _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def collect() -> dict:
    env = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_id": platform.node(),
        "os": platform.platform(),
        "kernel": _run(["uname", "-r"]),
        "container_image": os.environ.get("CONTAINER_IMAGE")
        or os.environ.get("DOCKER_IMAGE"),
    }
    env.update({
        # --- GPU identity and state -------------------------------------
        "gpu": _nvidia("name,uuid,driver_version,memory.total"),
        "gpu_clocks": _nvidia("clocks.applications.graphics,clocks.max.graphics"),
        "gpu_current_clocks": _nvidia("clocks.current.graphics,clocks.current.sm"),
        "gpu_temperature_c": _nvidia("temperature.gpu"),
        "gpu_power_limit": _nvidia("power.limit,power.max_limit"),
        "gpu_persistence_mode": _nvidia("persistence_mode"),
        "gpu_mig_mode": _nvidia("mig.mode.current"),
        "gpu_compute_mode": _nvidia("compute_mode"),
        # --- CPU / memory / NUMA ----------------------------------------
        "cpu_model": (_run(["lscpu"]) or "").split("Model name:")[-1].split("\n")[0].strip()
        or platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "lscpu": _run(["lscpu"]),
        "numa": _run(["numactl", "--hardware"]),
        "meminfo_total": (_read("/proc/meminfo") or "").split("\n")[0] or None,
        # --- thread configuration (the E4 variable) ---------------------
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else None,
        # --- software ----------------------------------------------------
        "python": sys.version.split()[0],
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    })

    try:
        import torch
        env.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })
    except Exception as e:
        env["torch"] = f"import failed: {e}"

    for mod in ("transformers", "datasets", "safetensors", "omegaconf", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    args = ap.parse_args()
    env = collect()
    text = json.dumps(env, indent=2, default=str)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        missing = [k for k, v in env.items() if v is None]
        print(f"wrote {out} ({len(env)} fields, {len(missing)} unavailable)")
        if missing:
            print("  unavailable here: " + ", ".join(missing))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
