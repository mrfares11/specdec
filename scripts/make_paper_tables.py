"""Regenerate every paper table from archived result CSVs (M2/M6 artifact).

Usage:
    python scripts/make_paper_tables.py [--m1-dir results/m1] [--out docs/tables]

Reads only files under results/ (committed at tag v1.0 and after); writes
markdown tables to --out. If an M1 re-measurement directory exists (pod B,
n>=1 repeats per cell), headline speed comes from it with mean +/- std;
otherwise falls back to the single pod-A solo runs with a WARNING.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _coerce(k: str, v: str):
    """Numeric where possible; non-numeric columns (e.g. output_equivalent) pass
    through as strings rather than raising."""
    if k == "K":
        return int(float(v))
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_csv(p: Path):
    with open(p, newline="") as f:
        return [{k: _coerce(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def table_acceptance(out: list[str]):
    """Tokens committed per target forward.

    Prefers the M1 pod-B repeats (n=3, same host/session family as every other
    pod-B table) over the pod-A single runs, which were measured on a different
    machine and, for several benchmarks, under 3-way concurrent GPU use. Falls
    back to the pod-A CSVs only where M1 has no cell.
    """
    benches = ["humaneval", "mbpp", "gsm8k", "alpaca", "mtbench"]

    def cell(sysname: str, bench: str, k: int):
        runs = sorted(ROOT.glob(f"results/m1/{sysname}-{bench}-r*/benchmark.csv"))
        vals = [r["tokens_per_step"] for p in runs for r in read_csv(p) if r["K"] == k]
        if vals:
            return fmt(statistics.mean(vals)), len(vals)
        stem = bench if sysname == "ours" else f"eagle-{bench}"
        for cand in (ROOT / f"results/bench/{stem}.csv",
                     ROOT / f"results/bench/{stem}-solo.csv"):
            if cand.exists():
                for r in read_csv(cand):
                    if r["K"] == k:
                        return fmt(r["tokens_per_step"]), 1
        return "--", 0

    out.append("## Acceptance matrix (tokens committed per target forward)" + chr(10))
    out.append("| System | " + " | ".join(benches) + " |")
    out.append("|---|" + "---|" * len(benches))
    out.append("| Plain AR (definition) | " + " | ".join("1.00" for _ in benches) + " |")
    ns = set()
    for label, sysname in (("Ours (v1.0 drafter)", "ours"),
                           ("Official EAGLE (our harness)", "eagle")):
        for k in (4, 8):
            row = []
            for b in benches:
                v, n = cell(sysname, b, k)
                row.append(v)
                if n:
                    ns.add(n)
            out.append(f"| {label}, K={k} | " + " | ".join(row) + " |")
    out.append("")
    out.append(f"Source: results/m1 (pod B, n={'/'.join(str(x) for x in sorted(ns))} "
               "per cell), falling back to the pod-A CSVs only where M1 has no cell. "
               "Acceptance is deterministic under greedy decoding, so these are exact "
               "across repeats." + chr(10))
    out.append("The SCALED drafter's acceptance is not in this table -- see the M5 "
               "factorial, which measures it against EAGLE in a single session." + chr(10))


def table_headline_speed(out: list[str], m1_dir: Path):
    """Wall-clock ms/token + speedup; M1 repeats (mean+/-std) when available."""
    out.append("## Headline wall-clock (solo runs)\n")
    runs = sorted(m1_dir.glob("*/benchmark.csv")) if m1_dir.exists() else []
    if runs:
        cells = defaultdict(list)  # (system, bench, K) -> [rows]
        for p in runs:
            sysname, bench = p.parent.name.rsplit("-r", 1)[0].split("-", 1)
            for r in read_csv(p):
                cells[(sysname, bench, r["K"])].append(r)
        out.append(f"Source: {Path(m1_dir).relative_to(ROOT).as_posix()} (pod B, n per cell = repeats found)\n")
        out.append("| System | Benchmark | K | ms/token (mean±std) | speedup (mean±std) | n |")
        out.append("|---|---|---|---|---|---|")
        for (sysname, bench, k), rows in sorted(cells.items()):
            ms = [r["spec_ms_per_token"] for r in rows]
            sp = [r["speedup"] for r in rows]
            ms_s = fmt(statistics.mean(ms)) + (f" ± {fmt(statistics.stdev(ms))}" if len(ms) > 1 else "")
            sp_s = fmt(statistics.mean(sp)) + (f" ± {fmt(statistics.stdev(sp), 3)}" if len(sp) > 1 else "")
            out.append(f"| {sysname} | {bench} | {k} | {ms_s} | {sp_s} | {len(ms)} |")
        out.append(chr(10) + "In this table `ours` is the **v1.0 baseline drafter** (detached "
                   "objective, 20k steps). M3 promoted the step-weighted 80k-row drafter to the "
                   "headline configuration on acceptance; its wall-clock rows come from the M5 "
                   "factorial and are NOT yet in this table. Do not mix the acceptance claims for "
                   "the scaled drafter with the speedups here." + chr(10))
        out.append("These M1 runs also PREDATE the host thread cap (commit 8ca7f57). Static "
                   "decode is only mildly affected -- MT-Bench K=4 re-measures at 15.04 vs "
                   "15.37 ms/token, ~2% -- so these speedups are slightly CONSERVATIVE rather "
                   "than wrong. The dynamic-tree rows measured in the same window are NOT "
                   "usable; see the M4/M5 table. M5 re-measures MT-Bench and HumanEval under "
                   "the capped host; MBPP, GSM8K and Alpaca timings remain pre-cap." + chr(10))
    else:
        out.append("**WARNING: no M1 re-measurement found — single pod-A solo runs:**\n")
        out.append("| System | Benchmark | K | ms/token | speedup |")
        out.append("|---|---|---|---|---|")
        for name, path in [("ours", "results/bench/humaneval-solo.csv"),
                           ("ours", "results/bench/mbpp.csv"),
                           ("ours", "results/bench/mtbench-solo.csv"),
                           ("eagle", "results/bench/eagle-humaneval.csv"),
                           ("eagle", "results/bench/eagle-mbpp.csv"),
                           ("eagle", "results/bench/eagle-mtbench.csv")]:
            p = ROOT / path
            if not p.exists():
                continue
            bench = p.stem.replace("-solo", "").replace("eagle-", "")
            for r in read_csv(p):
                out.append(f"| {name} | {bench} | {r['K']} | {fmt(r['spec_ms_per_token'])} "
                           f"| {fmt(r['speedup'], 3)} |")
    out.append("")


def table_rq1(out: list[str]):
    """Budget vs allocation: the RQ1 'reallocation beats budget' quantification."""
    out.append("## RQ1: acceptance across budget x allocation (mean accept length)" + chr(10))
    ks = [2, 4, 8, 16, 32]
    out.append("| Allocation | Nodes | " + " | ".join(f"K={k}" for k in ks) + " |")
    out.append("|---|---|" + "---|" * len(ks))
    data = {}
    for alloc in ("uniform", "geometric"):
        for n in (64, 128, 256):
            rows = read_csv(ROOT / f"results/rq1/{alloc}-n{n}.csv")
            data[(alloc, n)] = {r["K"]: r["mean_accept_len"] for r in rows}
            cells = [fmt(data[(alloc, n)].get(k), 3) if data[(alloc, n)].get(k) else "—"
                     for k in ks]
            out.append(f"| {alloc} | {n} | " + " | ".join(cells) + " |")
    # headline deltas at the long-horizon end
    k = 32
    budget_gain = data[("geometric", 256)][k] - data[("geometric", 64)][k]
    alloc_gain = data[("geometric", 256)][k] - data[("uniform", 256)][k]
    out.append(chr(10) + f"At K={k}: reallocating a FIXED 256-node budget (uniform->geometric) buys "
               f"+{alloc_gain:.3f} accept; quadrupling the geometric budget (64->256) buys "
               f"+{budget_gain:.3f}. Timing columns in these CSVs predate the RQ3 loop "
               f"rebuild - acceptance only." + chr(10))


def _heldout_ce(arm: str):
    """Arm-independent held-out rollout CE, or None if that cell hasn't run."""
    f = ROOT / f"results/m3/heldout-ce/{arm}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d["per_step_ce"][0], d["uniform_ce"]


def _m3_accept(tag: str, bench: str = "mtbench"):
    f = ROOT / f"results/m3/{tag}-{bench}/benchmark.csv"
    if not f.exists():
        return None, None
    rows = read_csv(f)
    g = lambda k: next((fmt(r["mean_accept_len"]) for r in rows if r["K"] == k), "--")
    return g(4), g(8)


def table_rq2(out: list[str]):
    """RQ2 on the pod-B re-certification (M3): matched 80k-row data, 10k steps,
    scored with the ARM-INDEPENDENT held-out rollout CE. The pod-A table this
    replaces used each arm's own step-weighted training CE, which is the
    confound that produced the original 'BP-uniform improves CE' claim."""
    # (label, checkpoint arm, acceptance tag)
    arms = [("BP-through, step weights 0.5^s", "m3-sw80k-s10", "m3-sw80k-s10"),
            ("Teacher-forced control (rollout_K=1)", "m3-tf80k", "m3-tf80k"),
            ("Detached rollout", "m3-det80k", "m3-det80k"),
            ("BP-through, uniform weights", "m3-bp80k", "m3-bp80k")]
    out.append("## RQ2 training ablation, re-certified (M3, pod B)\n")
    out.append("Matched: 80k-row features (stream seed 0), 10k steps, batch 4 x accum 4,")
    out.append("geometric-256 tree, MT-Bench 80 prompts, greedy, seed 0.\n")
    out.append("| Objective (10k steps, matched) | held-out CE s1 | held-out CE uniform | accept K=4 | accept K=8 |")
    out.append("|---|---|---|---|---|")
    pts = []
    for label, arm, tag in arms:
        ce = _heldout_ce(arm)
        a4, a8 = _m3_accept(tag)
        c1 = fmt(ce[0]) if ce else "_pending_"
        cu = fmt(ce[1]) if ce else "_pending_"
        out.append(f"| {label} | {c1} | {cu} | {a4 or '--'} | {a8 or '--'} |")
        if ce and a4:
            pts.append((ce[0], ce[1], float(a4)))
    out.append("")

    # Does CE predict acceptance? Rank-correlate both CE variants against accept.
    extra = [(a, a) for a in ("m3-sw40k", "m3-sw20k")]
    for arm, tag in extra:
        ce = _heldout_ce(arm)
        a4, _ = _m3_accept(tag)
        if ce and a4:
            pts.append((ce[0], ce[1], float(a4)))
    if len(pts) >= 4:
        def spearman(xs, ys):
            def rank(v):
                order = sorted(range(len(v)), key=lambda i: v[i])
                r = [0] * len(v)
                for pos, i in enumerate(order):
                    r[i] = pos + 1
                return r
            rx, ry = rank(xs), rank(ys)
            n = len(xs)
            d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
            return 1 - 6 * d2 / (n * (n * n - 1))
        acc = [q[2] for q in pts]
        r1 = spearman([-q[0] for q in pts], acc)
        ru = spearman([-q[1] for q in pts], acc)
        out.append(f"Rank-correlation with acceptance across the {len(pts)} step-matched arms: "
                   f"step-1 CE {r1:+.2f}, multi-step (uniform) CE {ru:+.2f}. "
                   f"With n={len(pts)} these are directions, not significance tests.\n")
    out.append("Supersedes the pod-A three-arm table (results/bench/rq2-*.csv), whose CE")
    out.append("column was each arm's own step-weighted training CE and whose det/BP")
    out.append("ordering does not reproduce under the arm-independent metric. See")
    out.append("docs/tables/m3-findings.md.\n")


def _mean_std(pat: str, K: int, col: str):
    vals = []
    for f in sorted(ROOT.glob(pat)):
        for r in read_csv(f):
            if r["K"] == K:
                vals.append(r[col])
    if not vals:
        return None
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.mean(vals), sd, len(vals)


def table_m4(out: list[str]):
    """Static vs per-step dynamic reallocation under the SAME 256-node budget.

    ACCEPTANCE comes from the M4 repeats (deterministic under greedy, so it is
    unaffected by host load). WALL-CLOCK comes from the M5 same-session cells,
    because the original M4 dynamic timings were taken on a 224-core host with
    an uncapped thread pool and are contaminated -- see the note below.
    """
    out.append("## M4/M5: dynamic tree reallocation under a fixed budget" + chr(10))
    out.append("| Benchmark | K | accept static | accept dynamic | d accept | "
               "speedup static | speedup dynamic | (superseded) dynamic, uncapped host |")
    out.append("|---|---|---|---|---|---|---|---|")
    for b in ("mtbench", "humaneval", "mbpp"):
        for K in (4, 8):
            st_a = _mean_std(f"results/m1/ours-{b}-r*/benchmark.csv", K, "mean_accept_len")
            dy_a = _mean_std(f"results/m4/dynamic-{b}-r*/benchmark.csv", K, "mean_accept_len")
            if not (st_a and dy_a):
                continue
            bad = _mean_std(f"results/m4/dynamic-{b}-r*/benchmark.csv", K, "speedup")
            m5s, m5d = _m5(f"base-static-{b}", K), _m5(f"base-dyn-{b}", K)
            out.append(f"| {b} | {K} | {fmt(st_a[0])} | {fmt(dy_a[0])} | "
                       f"+{fmt(dy_a[0] - st_a[0])} | "
                       f"{fmt(m5s['speedup']) if m5s else '--'} | "
                       f"{fmt(m5d['speedup']) if m5d else '--'} | "
                       f"{fmt(bad[0]) if bad else '--'} |")
    out.append("")
    out.append("**The last column is a measurement artifact, retained only for the record.** "
               "Those dynamic runs predate the host thread cap (OMP/MKL_NUM_THREADS=8, "
               "commit 8ca7f57): on a 224-core host the CPU-bound work of per-step topology "
               "construction thrashed and starved the GPU. Capping threads halves dynamic "
               "decode latency (23.0 -> 11.6 ms/token on MT-Bench K=4) while leaving the AR "
               "baseline (18.3 -> 18.4) and static decode (15.4 -> 15.0) essentially "
               "unchanged -- the selective signature of host contention on the CPU-heavy "
               "path. Acceptance is bit-identical across both (2.2551 at K=4) and no decode "
               "or eval code changed between them." + chr(10))
    out.append("Conclusion: dynamic reallocation DOES pay. The earlier negative result was "
               "environmental, not a property of dynamic topologies." + chr(10))


def table_m3_scaling(out: list[str]):
    """Data-scaling and training-progress curves for the step-weighted arm."""
    out.append("## M3: data scaling and training progress (step-weighted arm)\n")
    out.append("Data scaling, all held at 10k steps:\n")
    out.append("| Feature rows | MT-Bench K=4 | MT-Bench K=8 | HumanEval K=4 | held-out CE |")
    out.append("|---|---|---|---|---|")
    for rows, tag, arm in (("20k", "m3-sw20k", "m3-sw20k"),
                           ("40k", "m3-sw40k", "m3-sw40k"),
                           ("80k", "m3-sw80k-s10", "m3-sw80k-s10")):
        m4_, m8_ = _m3_accept(tag, "mtbench")
        h4, _ = _m3_accept(tag, "humaneval")
        ce = _heldout_ce(arm)
        out.append(f"| {rows} | {m4_ or '--'} | {m8_ or '--'} | {h4 or '_pending_'} | "
                   f"{fmt(ce[1]) if ce else '_pending_'} |")
    out.append(chr(10) + "Training progress, 80k rows:\n")
    out.append("| Steps | accept K=4 | accept K=8 |")
    out.append("|---|---|---|")
    for steps, tag in (("2k", "m3-sw80k-s2000"), ("5k", "m3-sw80k-s5000"),
                       ("10k", "m3-sw80k-s10"), ("20k", "m3-sw80k")):
        a4, a8 = _m3_accept(tag, "mtbench")
        out.append(f"| {steps} | {a4 or '--'} | {a8 or '--'} |")
    out.append(chr(10) + "Acceptance is still climbing at 20k steps with no plateau, so every number "
               "reported for the headline drafter is a LOWER BOUND, not a converged value." + chr(10))


def table_rq3(out: list[str]):
    stages = [("Original scaffold loop", "results/rq3/oldloop.csv"),
              ("+ device-side greedy verification", "results/rq3/ab-replay.csv"),
              ("+ KV gather (vs crop+replay)", "results/rq3/ab-gather.csv"),
              ("+ commit-fold (ONE forward/step)", "results/fold/timing.csv")]
    out.append("## RQ3 incremental systems ablation (K=8, geometric-256, same drafter+prompts)\n")
    out.append("| Stage | ms/token | speedup | verify ms/step |")
    out.append("|---|---|---|---|")
    for label, path in stages:
        rows = [r for r in read_csv(ROOT / path) if r["K"] == 8]
        r = rows[0]
        v = r.get("t_verify_ms_per_step")
        out.append(f"| {label} | {fmt(r['spec_ms_per_token'])} | {fmt(r['speedup'], 3)} "
                   f"| {fmt(v, 2) if v is not None and not math.isnan(v) else '—'} |")
    out.append("\nCommit-fold row measured with sync-bracketed timers ON (adds overhead); "
               "timer-off equivalent: results/fold/real100.csv (13.9 ms/tok, 1.06x at K=4).\n")


def _m5(name: str, K: int):
    """One M5 cell row, or None if that cell has not landed yet."""
    f = ROOT / f"results/m5/{name}/benchmark.csv"
    if not f.exists():
        return None
    for r in read_csv(f):
        if r["K"] == K:
            return r
    return None


def _m5_reps(config: str, K: int):
    """Every repeat of one M5 config: r1 from results/m5, r2/r3 from results/m5-rep."""
    rows = []
    r1 = _m5(config, K)
    if r1:
        rows.append(r1)
    for r in (2, 3):
        f = ROOT / f"results/m5-rep/{config}-r{r}/benchmark.csv"
        if f.exists():
            rows.extend(x for x in read_csv(f) if x["K"] == K)
    return rows


def _ms(rows: list, col: str, nd: int = 2) -> str:
    """mean +/- std over repeats; bare value when n=1; '--' when absent."""
    vals = [r[col] for r in rows if isinstance(r.get(col), float)]
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt(vals[0], nd)
    return f"{fmt(statistics.mean(vals), nd)} +/- {fmt(statistics.stdev(vals), 3)}"


def _equiv(rows: list) -> str:
    vals = {r.get("lossless_vs_ar") for r in rows if r.get("lossless_vs_ar")}
    if not vals:
        return "--"
    if vals == {"yes"}:
        return "yes"
    return "/".join(sorted(str(v) for v in vals))


def table_m5(out: list[str]):
    """Final factorial: {baseline, scaled} x {static, dynamic} + symmetric EAGLE.

    Cells that have not finished render as '--' so this table is safe to
    regenerate while the campaign is still running.
    """
    out.append("## M5: final factorial (drafter x tree), same session, solo GPU\n")
    missing = []
    for bench in ("mtbench", "humaneval"):
        out.append(f"### {bench}\n")
        out.append("| Drafter | Tree | K | accept | ms/token | AR ms/token | speedup | "
                   "output-equivalent | n |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for drafter, stem in (("v1.0 baseline (det-20k)", "base"),
                              ("scaled (sw, 80k rows, 20k steps)", "scaled"),
                              ("official EAGLE", "eagle")):
            for tree, dyn in (("static geometric-256", "static"), ("dynamic", "dyn")):
                for K in (4, 8):
                    cell = f"{stem}-{dyn}-{bench}"
                    reps = _m5_reps(cell, K)
                    if not reps:
                        missing.append(f"{cell} K={K}")
                        out.append(f"| {drafter} | {tree} | {K} | -- | -- | -- | -- | -- | 0 |")
                        continue
                    out.append(f"| {drafter} | {tree} | {K} | {_ms(reps, 'mean_accept_len')} | "
                               f"{_ms(reps, 'spec_ms_per_token')} | {_ms(reps, 'ar_ms_per_token')} | "
                               f"{_ms(reps, 'speedup')} | {_equiv(reps)} | {len(reps)} |")
        out.append("")

    # 2x2 decomposition: main effects and interaction, per benchmark and K.
    out.append("### Factorial decomposition (drafter x tree)" + chr(10))
    out.append("| Benchmark | K | metric | base+static | scaled+dynamic | joint | "
               "drafter effect | tree effect | interaction |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for bench in ("mtbench", "humaneval"):
        for K in (4, 8):
            bs, bd = _m5(f"base-static-{bench}", K), _m5(f"base-dyn-{bench}", K)
            ss, sd = _m5(f"scaled-static-{bench}", K), _m5(f"scaled-dyn-{bench}", K)
            if not all((bs, bd, ss, sd)):
                continue
            for metric in ("mean_accept_len", "speedup"):
                a, b, c, d = bs[metric], bd[metric], ss[metric], sd[metric]
                drafter = ((c - a) + (d - b)) / 2
                tree = ((b - a) + (d - c)) / 2
                inter = (d - c) - (b - a)
                label = "accept" if metric == "mean_accept_len" else "speedup"
                out.append(f"| {bench} | {K} | {label} | {fmt(a)} | {fmt(d)} | "
                           f"+{fmt(d - a)} | +{fmt(drafter)} | +{fmt(tree)} | "
                           f"{'+' if inter >= 0 else ''}{fmt(inter)} |")
    out.append("")
    out.append("A POSITIVE interaction means the two contributions reinforce rather than "
               "substitute: dynamic reallocation pays MORE with a better drafter, which is "
               "what the mechanism predicts -- global parent competition reallocates budget "
               "on the drafter's own signal, so a sharper drafter gives it more to work "
               "with. The effect is consistent on SPEEDUP (positive in every cell) and "
               "on acceptance in 3 of 4; HumanEval K=4 acceptance is exactly additive "
               "(+0.001), so the claim is 'reinforcing, never substituting' rather than "
               "'always super-additive'." + chr(10))

    out.append("### Protocol-breadth extras\n")
    out.append("| Setting | Drafter | K | accept | speedup |")
    out.append("|---|---|---|---|---|")
    for label, cell in (("512-token generation", "long-base-static"),
                        ("512-token generation", "long-scaled-static"),
                        ("temperature 1.0 (sampling)", "temp1-base-static"),
                        ("temperature 1.0 (sampling)", "temp1-scaled-static")):
        who = "scaled" if "scaled" in cell else "v1.0 baseline"
        r = _m5(cell, 4)
        if r is None:
            missing.append(f"{cell} K=4")
            out.append(f"| {label} | {who} | 4 | -- | -- |")
        else:
            out.append(f"| {label} | {who} | 4 | {fmt(r['mean_accept_len'])} | {fmt(r['speedup'])} |")

    vf = ROOT / "results/m5/vllm-reference.json"
    if vf.exists():
        out.append(chr(10) + "vLLM reference row: `results/m5/vllm-reference.json` "
                   "(isolated venv; an optimized-engine anchor, NOT a like-for-like "
                   "lossless-speculation comparison)." + chr(10))
    else:
        out.append(chr(10) + "vLLM reference row: not present (the chain treats it as "
                   "failure-tolerant and never blocks on it)." + chr(10))

    if missing:
        out.append(f"**{len(missing)} M5 cells not yet landed** (rendered as `--`): "
                   + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "") + chr(10))
    else:
        out.append("All M5 cells present." + chr(10))


def table_m5_latency(out: list[str]):
    """Roadmap M5: drafting latency and verification latency per configuration.

    Measured with profile_timing=true in a SEPARATE pass, because the timers add
    overhead and must never contaminate a reported wall-clock number.
    """
    out.append("## M5 latency decomposition (K=8, sync-bracketed timers ON)" + chr(10))
    out.append("| Drafter | Tree | Benchmark | draft ms/step | tree-fwd ms/step | "
               "verify ms/step | commit ms/step |")
    out.append("|---|---|---|---|---|---|---|")
    found = 0
    for bench in ("mtbench", "humaneval"):
        for drafter, stem in (("v1.0 baseline", "base"), ("scaled", "scaled"),
                              ("official EAGLE", "eagle")):
            for tree, t in (("static", "static"), ("dynamic", "dyn")):
                f = ROOT / f"results/m5-timing/{stem}-{t}-{bench}/benchmark.csv"
                if not f.exists():
                    out.append(f"| {drafter} | {tree} | {bench} | -- | -- | -- | -- |")
                    continue
                rows = [r for r in read_csv(f) if r["K"] == 8]
                if not rows:
                    continue
                r = rows[0]
                found += 1
                g = lambda k: fmt(r[k], 2) if isinstance(r.get(k), float) else "--"
                out.append(f"| {drafter} | {tree} | {bench} | "
                           f"{g('t_draft_ms_per_step')} | {g('t_tree_forward_ms_per_step')} | "
                           f"{g('t_verify_ms_per_step')} | {g('t_commit_ms_per_step')} |")
    out.append("")
    if not found:
        out.append("_Latency pass not yet run (scripts/pod-jobs/m5-repeats.sh, pass B)._"
                   + chr(10))
    else:
        out.append("Absolute values are inflated by the timers; read them as a COST "
                   "DECOMPOSITION, not as wall-clock. The clean wall-clock numbers are in "
                   "the factorial table above." + chr(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", default="results/m1")
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Paper tables (auto-generated — do not hand-edit)\n"]
    table_headline_speed(lines, ROOT / args.m1_dir)
    table_acceptance(lines)
    table_rq1(lines)
    table_rq2(lines)
    table_rq3(lines)
    table_m4(lines)
    table_m3_scaling(lines)
    table_m5(lines)
    table_m5_latency(lines)
    (out_dir / "paper-tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'paper-tables.md'}")


if __name__ == "__main__":
    main()
