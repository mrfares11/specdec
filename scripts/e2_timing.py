"""E2 --- final timing replication and factorial robustness.

Drives repeated benchmark cells under the protocol's session design and then
audits the session rather than trusting it:

  * condition order is randomized independently for every repeat, so a
    systematic drift (thermal, neighbour noise) cannot align with one condition;
  * every cell already measures its own autoregressive baseline paired per
    prompt, so AR drift across the session is measurable after the fact;
  * a session whose AR baseline drifts beyond a predeclared threshold is marked
    REJECTED with its reason and kept, never silently dropped.

    python scripts/e2_timing.py --session s1 --repeats 3 \
        --benchmarks humaneval mtbench --out-root /workspace/specdec-runs

Resume-safe: a cell whose benchmark.csv already exists is skipped, so an
interrupted session continues rather than restarting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, drafter-override, dynamic_tree)
SYSTEMS = [
    ("base",   "eval.drafter_ckpt={R}/checkpoints/drafter-20k.pt",        None),
    ("scaled", "eval.drafter_ckpt={R}/checkpoints/m3-sw80k/step_20000.pt", None),
    ("eagle",  None,                                                      None),
]
TREES = [("static", "false"), ("dyn", "true")]


def cells(benchmarks, ks):
    for b in benchmarks:
        for sysname, ckpt, _ in SYSTEMS:
            for tree, dyn in TREES:
                for k in ks:
                    yield {"bench": b, "system": sysname, "ckpt": ckpt,
                           "tree": tree, "dyn": dyn, "K": k}


def run_cell(c, rep, session, out_root, model_id, prompts, tokens, dry=False):
    name = f"{c['system']}-{c['tree']}-{c['bench']}-k{c['K']}-{session}-r{rep}"
    out = Path(out_root) / "results" / "e2" / name
    if (out / "benchmark.csv").exists() and (out / "benchmark.csv").stat().st_size > 0:
        print(f"[E2] SKIP {name} (already present)", flush=True)
        return name, True, 0.0

    cfg = ["configs/base.yaml"]
    sets = [f"target.model_id={model_id}",
            f"eval.prompt_source={c['bench']}",
            "eval.prompt_split=test",
            f"eval.num_prompts={prompts}",
            f"eval.max_new_tokens={tokens}",
            f"eval.K_sweep=[{c['K']}]",
            "tree.allocation=geometric", "tree.num_nodes=256",
            f"decode.dynamic_tree={c['dyn']}",
            f"eval.out_dir={out}",
            f"eval.experiment_id={name}"]
    if c["system"] == "eagle":
        cfg.append("configs/eagle.yaml")
    else:
        sets.append(c["ckpt"].format(R=out_root))

    cmd = [sys.executable, "scripts/run_eval.py", "--config", *cfg, "--set", *sets]
    if dry:
        print("[E2] DRY", " ".join(cmd), flush=True)
        return name, True, 0.0
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0 and (out / "benchmark.csv").exists()
    tail = (r.stdout or r.stderr or "").strip().splitlines()[-2:]
    print(f"[E2] {'OK  ' if ok else 'FAIL'} {name} ({dt/60:.1f} min)"
          + ("" if ok else "  :: " + " | ".join(tail)), flush=True)
    return name, ok, dt


def ar_drift(out_root, session):
    """AR ms/token across the session, in completion order."""
    rows = []
    base = Path(out_root) / "results" / "e2"
    for d in sorted(base.glob(f"*-{session}-r*"), key=lambda p: p.stat().st_mtime):
        f = d / "raw_prompt_results.csv"
        if not f.exists():
            continue
        vals = [float(r["ar_ms_per_token"]) for r in csv.DictReader(f.open())
                if r.get("ar_ms_per_token")]
        if vals:
            rows.append((d.name, statistics.mean(vals)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--benchmarks", nargs="+", default=["humaneval", "mtbench"])
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out-root", default="/workspace/specdec-runs")
    ap.add_argument("--model-id", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--drift-threshold", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("SPECDEC_SESSION_ID", a.session)
    todo = list(cells(a.benchmarks, a.ks))
    print(f"[E2] session={a.session} cells={len(todo)} repeats={a.repeats} "
          f"total_runs={len(todo)*a.repeats} prompts={a.prompts}", flush=True)

    results = []
    for rep in range(1, a.repeats + 1):
        order = list(todo)
        # Independent shuffle per repeat: drift cannot align with a condition.
        random.Random(hash((a.session, rep)) & 0xffffffff).shuffle(order)
        print(f"[E2] --- repeat {rep}/{a.repeats} ---", flush=True)
        for c in order:
            results.append(run_cell(c, rep, a.session, a.out_root, a.model_id,
                                    a.prompts, a.tokens, a.dry_run))

    drift = ar_drift(a.out_root, a.session)
    verdict = {"session": a.session, "runs": len(results),
               "failed": [n for n, ok, _ in results if not ok]}
    if len(drift) >= 6:
        vals = [v for _, v in drift]           # already in completion order
        mean = statistics.mean(vals)
        # DRIFT is a temporal trend, not a range. (max-min)/mean grows with the
        # number of runs by construction -- with 72 runs it just finds the two
        # extremes of ordinary run-to-run noise and calls it drift. Compare the
        # first and last thirds of the session instead, which is what the
        # threshold is actually about: is the baseline moving under us?
        third = max(len(vals) // 3, 2)
        early, late = statistics.mean(vals[:third]), statistics.mean(vals[-third:])
        drift_frac = abs(late - early) / early
        cv = statistics.stdev(vals) / mean if len(vals) > 1 else 0.0
        verdict.update({
            "ar_ms_per_token_min": min(vals), "ar_ms_per_token_max": max(vals),
            "ar_ms_per_token_mean": mean,
            "ar_range_frac": (max(vals) - min(vals)) / mean,
            "ar_cv": cv,
            "ar_early_mean": early, "ar_late_mean": late,
            "ar_temporal_drift": drift_frac,
            "drift_threshold": a.drift_threshold,
            "rejected": drift_frac > a.drift_threshold,
            "rejection_reason": (
                f"AR baseline drifted {drift_frac:.1%} from session start to end, "
                f"exceeding {a.drift_threshold:.0%}"
                if drift_frac > a.drift_threshold else "")})
        print(f"[E2] AR baseline: mean {mean:.2f} ms/token, CV {cv:.2%}, "
              f"range {(max(vals)-min(vals))/mean:.2%}", flush=True)
        print(f"[E2] AR temporal drift (first third {early:.2f} -> last third "
              f"{late:.2f}): {drift_frac:.2%} vs threshold "
              f"{a.drift_threshold:.0%}", flush=True)
        if drift_frac > a.drift_threshold:
            print("[E2] *** SESSION FLAGGED: AR baseline drifted across the "
                  "session. Runs are preserved and marked, not deleted. ***",
                  flush=True)
        else:
            print("[E2] session accepted: no systematic baseline drift. Run-to-run "
                  "noise is controlled by per-prompt AR pairing within each cell.",
                  flush=True)

    out = Path(a.out_root) / "results" / "e2" / f"session_{a.session}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("[E2] " + json.dumps(verdict), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
