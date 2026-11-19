"""E10 --- pod-D timing campaign for the revision (same protocol as E2).

Three cell plans, all measured with the E2 session design: condition order
randomized independently per repeat, every cell paired with its own
autoregressive baseline per prompt, and an AR-drift audit over the session.
A cell whose benchmark.csv exists is skipped, so an interrupted session resumes.

  w3          the replicated RQ1 checkpoints (step-decayed vs detached, seeds
              1-2; teacher forcing optional) x {static, dynamic} x benchmark,
              K in {4, 8} in one run each: does the training intervention change
              what dynamic allocation buys, in acceptance AND latency?
  q3-scaled   the scaled drafter with the paper's chain-static tree, dynamic
              allocation, and the two offline-optimised static trees (same
              depth profile / free profile) at K=4 and K=8, all in-session.
  q3-transfer the same four trees for the baseline and official EAGLE drafters.

    python scripts/e10_campaign.py --plan w3 --session d1 --repeats 3 \
        --out-root /workspace/specdec-runs
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

CK = "{R}/checkpoints"
E5 = {"sw-s1": f"{CK}/e5-sw-s1-step_10000.pt", "sw-s2": f"{CK}/e5-sw-s2-step_10000.pt",
      "det-s1": f"{CK}/e5-det-s1-step_10000.pt", "det-s2": f"{CK}/e5-det-s2-step_10000.pt",
      "tf-s1": f"{CK}/e5-tf-s1-step_10000.pt", "tf-s2": f"{CK}/e5-tf-s2-step_10000.pt"}
DRAFTERS = {"scaled": f"{CK}/m3-sw80k-step_20000.pt", "base": f"{CK}/drafter-20k.pt", "eagle": None}
TREE_DIR = "{R}/e10/trees"


def plan_cells(plan: str, benchmarks, with_tf: bool):
    cells = []
    if plan == "w3":
        names = ["sw-s1", "sw-s2", "det-s1", "det-s2"] + (["tf-s1", "tf-s2"] if with_tf else [])
        for b in benchmarks:
            for n in names:
                for tree, dyn in (("static", "false"), ("dyn", "true")):
                    cells.append({"bench": b, "system": n, "ckpt": E5[n], "tree": tree, "dyn": dyn,
                                  "ks": [4, 8], "parents": None})
    elif plan in ("q3-scaled", "q3-transfer"):
        systems = ["scaled"] if plan == "q3-scaled" else ["base", "eagle"]
        for b in benchmarks:
            for s in systems:
                for tree, dyn in (("static", "false"), ("dyn", "true")):
                    cells.append({"bench": b, "system": s, "ckpt": DRAFTERS[s], "tree": tree, "dyn": dyn,
                                  "ks": [4, 8], "parents": None})
                for variant in ("optsame", "optfree"):
                    for k in (4, 8):
                        cells.append({"bench": b, "system": s, "ckpt": DRAFTERS[s], "tree": f"{variant}{k}",
                                      "dyn": "false", "ks": [k],
                                      "parents": f"{TREE_DIR}/{s}-{variant}-k{k}.json"})
    else:
        raise ValueError(plan)
    return cells


def run_cell(c, rep, session, out_root, model_id, prompts, tokens, dry=False):
    name = f"{c['system']}-{c['tree']}-{c['bench']}-{session}-r{rep}"
    out = Path(out_root) / "results" / "e10" / name
    if (out / "benchmark.csv").exists() and (out / "benchmark.csv").stat().st_size > 0:
        print(f"[E10] SKIP {name} (already present)", flush=True)
        return name, True, 0.0
    cfg = ["configs/base.yaml"]
    sets = [f"target.model_id={model_id}",
            f"eval.prompt_source={c['bench']}", "eval.prompt_split=test",
            f"eval.num_prompts={prompts}", f"eval.max_new_tokens={tokens}",
            "eval.K_sweep=[" + ",".join(str(k) for k in c["ks"]) + "]",
            "tree.allocation=geometric", "tree.num_nodes=256",
            f"decode.dynamic_tree={c['dyn']}",
            f"eval.out_dir={out}", f"eval.experiment_id={name}"]
    if c["parents"]:
        sets.append("tree.parents_file=" + c["parents"].format(R=out_root))
    if c["system"] == "eagle":
        cfg.append("configs/eagle.yaml")
    else:
        sets.append("eval.drafter_ckpt=" + c["ckpt"].format(R=out_root))
    cmd = [sys.executable, "scripts/run_eval.py", "--config", *cfg, "--set", *sets]
    if dry:
        print("[E10] DRY", " ".join(cmd), flush=True)
        return name, True, 0.0
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0 and (out / "benchmark.csv").exists()
    tail = (r.stdout or r.stderr or "").strip().splitlines()[-2:]
    print(f"[E10] {'OK  ' if ok else 'FAIL'} {name} ({dt / 60:.1f} min)"
          + ("" if ok else "  :: " + " | ".join(tail)), flush=True)
    return name, ok, dt


def ar_drift(out_root, session):
    rows = []
    base = Path(out_root) / "results" / "e10"
    for d in sorted(base.glob(f"*-{session}-r*"), key=lambda p: p.stat().st_mtime):
        f = d / "raw_prompt_results.csv"
        if not f.exists():
            continue
        vals = [float(r["ar_ms_per_token"]) for r in csv.DictReader(f.open()) if r.get("ar_ms_per_token")]
        if vals:
            rows.append((d.name, statistics.mean(vals)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["w3", "q3-scaled", "q3-transfer"])
    ap.add_argument("--session", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--benchmarks", nargs="+", default=["humaneval", "mtbench"])
    ap.add_argument("--with-tf", action="store_true")
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out-root", default="/workspace/specdec-runs")
    ap.add_argument("--model-id", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--drift-threshold", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("SPECDEC_SESSION_ID", a.session)
    todo = plan_cells(a.plan, a.benchmarks, a.with_tf)
    print(f"[E10] plan={a.plan} session={a.session} cells={len(todo)} repeats={a.repeats} "
          f"runs={len(todo) * a.repeats} threads={os.environ.get('OMP_NUM_THREADS', 'UNCAPPED')}", flush=True)
    results = []
    for rep in range(1, a.repeats + 1):
        order = list(todo)
        random.Random(hash((a.plan, a.session, rep)) & 0xffffffff).shuffle(order)
        print(f"[E10] --- repeat {rep}/{a.repeats} ---", flush=True)
        for c in order:
            results.append(run_cell(c, rep, a.session, a.out_root, a.model_id, a.prompts, a.tokens, a.dry_run))
    drift = ar_drift(a.out_root, a.session)
    verdict = {"plan": a.plan, "session": a.session, "runs": len(results),
               "failed": [n for n, ok, _ in results if not ok]}
    if len(drift) >= 6:
        vals = [v for _, v in drift]
        mean = statistics.mean(vals)
        third = max(1, len(vals) // 3)
        early, late = statistics.mean(vals[:third]), statistics.mean(vals[-third:])
        trend = abs(late - early) / mean
        verdict.update({"ar_ms_per_token_mean": mean, "ar_early_mean": early, "ar_late_mean": late,
                        "ar_temporal_drift": trend, "drift_threshold": a.drift_threshold,
                        "rejected": trend > a.drift_threshold})
    Path(a.out_root, "results", "e10").mkdir(parents=True, exist_ok=True)
    (Path(a.out_root) / "results" / "e10" / f"session_{a.session}_{a.plan}.json").write_text(json.dumps(verdict, indent=1))
    print("[E10]", json.dumps(verdict), flush=True)
    return 0 if not verdict["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
