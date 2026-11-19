"""Verify every artifact the M3/M5 campaign was supposed to produce is on THIS
machine before the pod is released.

The pod's volume dies with the rental, so "the chain finished" is not the same
claim as "we have the results". This exits non-zero while anything is missing.

    python scripts/verify_campaign.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

M5_FACTORIAL = [f"{d}-{t}-{b}"
                for d in ("base", "scaled", "eagle")
                for t in ("static", "dyn")
                for b in ("mtbench", "humaneval")]
M5_EXTRAS = ["long-base-static", "long-scaled-static",
             "temp1-base-static", "temp1-scaled-static"]
M5_TAIL = ["base-static-mbpp", "base-dyn-mbpp"]          # queued after M5
M3_ARMS = ["m3-sw80k", "m3-tf80k", "m3-det80k", "m3-bp80k", "m3-sw40k", "m3-sw20k"]

# Roadmap Milestone 5 compliance: every factorial config repeated to n=3 with
# variance, plus a timers-on latency decomposition per config.
M5_CONFIGS = [f"{d}-{t}-{b}"
              for b in ("mtbench", "humaneval")
              for d in ("base", "scaled", "eagle")
              for t in ("static", "dyn")]
M5_REPEATS = [f"{c}-r{r}" for r in (2, 3) for c in M5_CONFIGS]


def check(label: str, paths: list[tuple[str, Path]]) -> list[str]:
    missing = [n for n, p in paths if not (p.exists() and p.stat().st_size > 0)]
    have = len(paths) - len(missing)
    mark = "OK " if not missing else "MISS"
    print(f"[{mark}] {label}: {have}/{len(paths)}")
    for n in missing:
        print(f"         missing: {n}")
    return missing


def main() -> int:
    missing: list[str] = []

    missing += check("M5 factorial cells", [
        (c, ROOT / f"results/m5/{c}/benchmark.csv") for c in M5_FACTORIAL])
    missing += check("M5 protocol extras", [
        (c, ROOT / f"results/m5/{c}/benchmark.csv") for c in M5_EXTRAS])
    missing += check("M5 tail (MBPP dynamic re-measure)", [
        (c, ROOT / f"results/m5/{c}/benchmark.csv") for c in M5_TAIL])
    missing += check("M3 held-out CE (incl. step-matched cell)", [
        (a, ROOT / f"results/m3/heldout-ce/{a}.json") for a in M3_ARMS + ["m3-sw80k-s10"]])
    missing += check("M3 step-matched HumanEval", [
        ("m3-sw80k-s10-humaneval",
         ROOT / "results/m3/m3-sw80k-s10-humaneval/benchmark.csv")])
    missing += check("M5 repeats (roadmap: variance on headline timings)", [
        (c, ROOT / f"results/m5-rep/{c}/benchmark.csv") for c in M5_REPEATS])
    missing += check("M5 latency decomposition (roadmap: draft/verify latency)", [
        (c, ROOT / f"results/m5-timing/{c}/benchmark.csv") for c in M5_CONFIGS])
    missing += check("Environment manifest", [
        ("h100-manifest.txt", ROOT / "results/env/h100-manifest.txt")])

    # Protocol section 0: every result directory must carry a classification, so
    # a new directory cannot quietly end up in a table without one.
    status_f = ROOT / "results/STATUS.json"
    if not status_f.exists():
        print("[MISS] results/STATUS.json absent - directories unclassified")
        missing.append("results/STATUS.json")
    else:
        classified = json.loads(status_f.read_text())["directories"]
        on_disk = {d.name for d in (ROOT / "results").iterdir() if d.is_dir()}
        unclassified = sorted(on_disk - set(classified))
        if unclassified:
            print(f"[MISS] result dirs with no active/superseded/diagnostic status: "
                  f"{', '.join(unclassified)}")
            missing.extend(unclassified)
        else:
            counts: dict = {}
            for v in classified.values():
                counts[v["status"]] = counts.get(v["status"], 0) + 1
            print(f"[OK ] result-dir classification: {len(on_disk)} dirs, "
                  + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # vLLM row is explicitly failure-tolerant: report, never fail on it.
    vllm = ROOT / "results/m5/vllm-reference.json"
    print(f"[{'OK ' if vllm.exists() else 'note'}] vLLM reference row: "
          f"{'present' if vllm.exists() else 'absent (failure-tolerant, not required)'}")

    print()
    if missing:
        print(f"INCOMPLETE - {len(missing)} artifact(s) missing. Do NOT release the pod;")
        print("copy the missing results into results/ (then re-run this check)")
        return 1
    print("COMPLETE - every required artifact is on this machine.")
    print("Safe to stop the pod. Bulk features/checkpoints on the volume are NOT")
    print("covered by this check - see scripts/pod_closeout.sh before TERMINATING.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
