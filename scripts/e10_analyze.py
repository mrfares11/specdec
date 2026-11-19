"""E10 --- tables for the revision: Q3 (optimised static trees), W3 (RQ1
checkpoints x allocation) and W2 (per-depth diagnostics), from the pod-D outputs.

All timing quantities are paired by prompt (repeats averaged within prompt
first) with 95% percentile bootstrap intervals over prompts (10,000 resamples,
seed 20260904, as in scripts/e3_interaction.py). Robust to partial campaigns:
cells that do not exist yet are simply absent from the tables.

    python scripts/e10_analyze.py --root results/e10 --rollout results/e10-rollout \
        --heldout results/e10-heldout --out paper/tables-fact --md docs/tables/e10-summary.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, SEED = 10000, 20260904
BENCH = {"mtbench": "MT-Bench", "humaneval": "HumanEval"}
TREE_NAME = {"static": "Chain-static (paper)", "optsame": "Optimised static, same profile",
             "optfree": "Optimised static, free profile", "dyn": "Dynamic"}
SYS_NAME = {"scaled": "Scaled", "base": "Baseline", "eagle": "Official EAGLE",
            "sw-s1": "Step-decayed, seed 1", "sw-s2": "Step-decayed, seed 2",
            "det-s1": "Detached, seed 1", "det-s2": "Detached, seed 2",
            "tf-s1": "Teacher forcing, seed 1", "tf-s2": "Teacher forcing, seed 2"}


def boot(fn, pids, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pids = np.array(pids)
    vals = np.array([fn(rng.choice(pids, len(pids), replace=True)) for _ in range(n)])
    vals.sort()
    return float(vals[int(0.025 * n)]), float(vals[int(0.975 * n)])


def pm(x, d=2):
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"


def ci(lo, hi, d=2):
    return f"{{\\scriptsize [{pm(lo, d)},{pm(hi, d)}]}}"


def load_cells(root: Path):
    """cells[(system, tree, bench, K)][pid] -> list of (ms, ar_ms, acc, run)"""
    cells = defaultdict(lambda: defaultdict(list))
    for f in sorted(root.glob("*/raw_prompt_results.csv")):
        name = f.parent.name  # <system>-<tree>-<bench>-<session>-r<rep>
        parts = name.split("-")
        # system may contain '-' (sw-s1): tree is the token before bench
        try:
            bench_i = max(i for i, p in enumerate(parts) if p in BENCH)
        except ValueError:
            continue
        tree = parts[bench_i - 1]
        system = "-".join(parts[: bench_i - 1])
        if tree.startswith("opt"):
            tree = tree[:-1]  # optsame4 -> optsame (K comes from the rows)
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            try:
                cells[(system, tree, r["benchmark"], int(float(r["K"])))][int(r["prompt_id"])].append(
                    (float(r["ms_per_token"]), float(r["ar_ms_per_token"]), float(r["mean_accept_len"]), name,
                     float(r["total_decode_ms"]) / max(float(r["verifier_forwards"]), 1.0)))
            except (KeyError, ValueError):
                continue
    return cells


def per_prompt(cell):
    return {p: (float(np.mean([x[0] for x in l])), float(np.mean([x[1] for x in l])),
                float(np.mean([x[2] for x in l])), float(np.mean([x[1] / x[0] for x in l])),
                float(np.mean([x[4] for x in l]))) for p, l in cell.items()}


def summary(pp):
    return dict(n=len(pp), ms=float(np.mean([v[0] for v in pp.values()])),
                acc=float(np.mean([v[2] for v in pp.values()])), sp=float(np.mean([v[3] for v in pp.values()])),
                step=float(np.mean([v[4] for v in pp.values()])))


# --------------------------------------------------------------------------- #
# Q3: optimised static trees
# --------------------------------------------------------------------------- #

def q3_tables(cells, out: Path, md: list):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Static trees fitted offline against per-step dynamic allocation, on the 50 test prompts per workload "
             r"(pod session d1, same protocol as the main results; three repeats for the scaled drafter, one for the baseline "
             r"and official EAGLE drafters). Both optimised trees are fitted on the development prompts "
             r"of both workloads pooled; \emph{same profile} keeps the paper's per-depth widths, \emph{free profile} only the "
             r"256-node budget. $\Delta$ is the paired latency saving relative to the chain-static tree with a 95\% bootstrap "
             r"interval over prompts; the last column is the share of the dynamic tree's saving that the static tree captures.}",
             r"\label{tab:q3}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lllrrrll}", r"\toprule",
             r"Drafter & Workload, $K$ & Tree & $\tau$ & ms/token & Speedup & $\Delta$ vs.\ chain-static & Share of dynamic \\",
             r"\midrule"]
    md += ["## Q3 — optimised static trees vs dynamic", "",
           "| drafter | bench | K | tree | acc | ms/token | speedup | ms/step | Δ vs chain-static [95% CI] | share of dynamic saving |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for system in ("scaled", "base", "eagle"):
        for bench in ("mtbench", "humaneval"):
            for K in (4, 8):
                ref = cells.get((system, "static", bench, K))
                if not ref:
                    continue
                dyn = cells.get((system, "dyn", bench, K))
                ppr = per_prompt(ref)
                ppd = per_prompt(dyn) if dyn else None
                first = True
                for tree in ("static", "optsame", "optfree", "dyn"):
                    c = cells.get((system, tree, bench, K))
                    if not c:
                        continue
                    pp = per_prompt(c)
                    s = summary(pp)
                    pids = sorted(set(pp) & set(ppr))
                    if tree == "static":
                        dcol, share = "--", "--"
                        mdd, mds = "--", "--"
                    else:
                        g = lambda smp, pp=pp: float(np.mean([ppr[p][0] - pp[p][0] for p in smp]))
                        lo, hi = boot(g, pids)
                        dcol = f"{pm(g(pids))} {ci(lo, hi)}"
                        mdd = f"{g(pids):+.2f} [{lo:+.2f}, {hi:+.2f}]"
                        if ppd and tree != "dyn":
                            pids2 = sorted(set(pids) & set(ppd))
                            sh = lambda smp: float(np.mean([ppr[p][0] - pp[p][0] for p in smp]) /
                                                   np.mean([ppr[p][0] - ppd[p][0] for p in smp]))
                            lo2, hi2 = boot(sh, pids2)
                            share = f"{100 * sh(pids2):.0f}\\% {ci(100 * lo2, 100 * hi2, 0)}"
                            mds = f"{100 * sh(pids2):.0f}% [{100 * lo2:.0f}, {100 * hi2:.0f}]"
                        else:
                            share, mds = ("100\\%", "100%") if tree == "dyn" else ("--", "--")
                    label = f"{SYS_NAME[system]}" if first else ""
                    wl = f"{BENCH[bench]}, {K}" if first else ""
                    first = False
                    lines.append(f"{label} & {wl} & {TREE_NAME[tree]} & {s['acc']:.2f} & {s['ms']:.2f} & "
                                 f"{s['sp']:.2f}$\\times$ & {dcol} & {share} \\\\")
                    md.append(f"| {system} | {bench} | {K} | {tree} | {s['acc']:.3f} | {s['ms']:.2f} | {s['sp']:.3f} | {s['step']:.1f} | {mdd} | {mds} |")
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_q3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md.append("")


# --------------------------------------------------------------------------- #
# W3: RQ1 checkpoints x allocation
# --------------------------------------------------------------------------- #

def w3_tables(cells, out: Path, md: list):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Static and dynamic allocation on the replicated RQ1 checkpoints (10,000 steps, seeds 1--2), "
             r"50 test prompts per workload, pod session d1. $G$ is the paired latency saving of dynamic over static allocation. "
             r"$I$ compares $G$ between the step-decayed and detached objectives at the same seed on the absolute (ms/token) "
             r"and proportional (log-ratio, \%) scales; brackets are 95\% bootstrap intervals over prompts.}",
             r"\label{tab:w3}", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{llrrrrrrl}", r"\toprule",
             r"Workload, $K$ & Checkpoint & $\tau_{\mathrm{st}}$ & $\tau_{\mathrm{dyn}}$ & ms$_{\mathrm{st}}$ & ms$_{\mathrm{dyn}}$ & "
             r"Speedup$_{\mathrm{st}}$ & Speedup$_{\mathrm{dyn}}$ & $G$ (ms/token) \\", r"\midrule"]
    md += ["## W3 — RQ1 checkpoints × allocation", "",
           "| bench | K | ckpt | acc static | acc dyn | ms static | ms dyn | speedup static | speedup dyn | G [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    inter_lines = [r"\begin{table}[htbp]", r"\centering",
                   r"\caption{Objective--allocation interaction on the replicated checkpoints: "
                   r"$I_{\mathrm{abs}} = G(\text{step-decayed}) - G(\text{detached})$ in ms/token and $I_{\mathrm{rel}}$ as the "
                   r"difference in mean $\log(L_{\mathrm{static}}/L_{\mathrm{dynamic}})$ (in \%), per seed; "
                   r"$\Delta\tau$ is the same contrast on the acceptance gain of dynamic allocation.}",
                   r"\label{tab:w3-interaction}", r"\small", r"\setlength{\tabcolsep}{3pt}",
                   r"\begin{tabular}{llllll}", r"\toprule",
                   r"Workload & $K$ & Seed & $I_{\mathrm{abs}}$ (ms/token) & $I_{\mathrm{rel}}$ (\%) & $\Delta\tau$ (dyn $-$ static) \\",
                   r"\midrule"]
    md2 = ["", "### interaction (sw − det), per seed", "",
           "| bench | K | seed | I_abs ms [CI] | I_rel % [CI] | Δτ contrast [CI] |", "|---|---|---|---|---|---|"]
    G = {}
    for bench in ("mtbench", "humaneval"):
        for K in (4, 8):
            first = True
            for sysname in ("sw-s1", "sw-s2", "det-s1", "det-s2", "tf-s1", "tf-s2"):
                st, dy = cells.get((sysname, "static", bench, K)), cells.get((sysname, "dyn", bench, K))
                if not st or not dy:
                    continue
                pps, ppd = per_prompt(st), per_prompt(dy)
                ss, sd = summary(pps), summary(ppd)
                pids = sorted(set(pps) & set(ppd))
                g = lambda smp: float(np.mean([pps[p][0] - ppd[p][0] for p in smp]))
                lo, hi = boot(g, pids)
                G[(sysname, bench, K)] = (pps, ppd, pids)
                wl = f"{BENCH[bench]}, {K}" if first else ""
                first = False
                lines.append(f"{wl} & {SYS_NAME[sysname]} & {ss['acc']:.2f} & {sd['acc']:.2f} & {ss['ms']:.2f} & {sd['ms']:.2f} & "
                             f"{ss['sp']:.2f}$\\times$ & {sd['sp']:.2f}$\\times$ & {g(pids):.2f} {ci(lo, hi)} \\\\")
                md.append(f"| {bench} | {K} | {sysname} | {ss['acc']:.3f} | {sd['acc']:.3f} | {ss['ms']:.2f} | {sd['ms']:.2f} | "
                          f"{ss['sp']:.3f} | {sd['sp']:.3f} | {g(pids):.3f} [{lo:.3f}, {hi:.3f}] |")
            for seed in ("s1", "s2"):
                a, b = G.get((f"sw-{seed}", bench, K)), G.get((f"det-{seed}", bench, K))
                if not a or not b:
                    continue
                (ppa_s, ppa_d, _), (ppb_s, ppb_d, _) = a, b
                pids = sorted(set(ppa_s) & set(ppa_d) & set(ppb_s) & set(ppb_d))
                iabs = lambda smp: float(np.mean([(ppa_s[p][0] - ppa_d[p][0]) - (ppb_s[p][0] - ppb_d[p][0]) for p in smp]))
                irel = lambda smp: float(np.mean([np.log(ppa_s[p][0] / ppa_d[p][0]) - np.log(ppb_s[p][0] / ppb_d[p][0]) for p in smp]))
                dacc = lambda smp: float(np.mean([(ppa_d[p][2] - ppa_s[p][2]) - (ppb_d[p][2] - ppb_s[p][2]) for p in smp]))
                lo1, hi1 = boot(iabs, pids)
                lo2, hi2 = boot(irel, pids)
                lo3, hi3 = boot(dacc, pids)
                pct = lambda x: 100 * (np.exp(x) - 1)
                inter_lines.append(f"{BENCH[bench]} & {K} & {seed[1]} & {pm(iabs(pids))} {ci(lo1, hi1)} & "
                                   f"{pm(pct(irel(pids)), 1)} {ci(pct(lo2), pct(hi2), 1)} & {pm(dacc(pids), 3)} {ci(lo3, hi3, 3)} \\\\")
                md2.append(f"| {bench} | {K} | {seed} | {iabs(pids):+.3f} [{lo1:+.3f}, {hi1:+.3f}] | "
                           f"{pct(irel(pids)):+.2f} [{pct(lo2):+.2f}, {pct(hi2):+.2f}] | {dacc(pids):+.3f} [{lo3:+.3f}, {hi3:+.3f}] |")
            if not first:
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    inter_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_w3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "tab_w3_interaction.tex").write_text("\n".join(inter_lines) + "\n", encoding="utf-8")
    md += md2 + [""]


# --------------------------------------------------------------------------- #
# W2: per-depth diagnostics
# --------------------------------------------------------------------------- #

def w2_tables(rollout_root: Path, heldout_root: Path, out: Path, md: list):
    # (a) decode-convention rollouts on test prompts: seeds averaged
    groups = {"Step-decayed (10k, seeds 1--2)": ["e5-sw-s1", "e5-sw-s2"],
              "Teacher forcing (10k, seeds 1--2)": ["e5-tf-s1", "e5-tf-s2"],
              "Detached (10k, seeds 1--2)": ["e5-det-s1", "e5-det-s2"],
              "Scaled (20k)": ["scaled"], "Baseline (detached, 20k)": ["base"], "Official EAGLE": ["eagle"]}
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Per-depth diagnostics along the target's greedy continuations of the 50 test prompts, in the decode-time "
             r"convention (true-feature context, the drafter's own predicted features along the chain). Top-1: fraction of roots "
             r"at which the drafter's most likely token at that depth is the target's greedy token. CE$_{\mathrm{full}}$: "
             r"cross-entropy against the full target distribution; CE$_{\mathrm{proxy}}$: against the intermediate-feature proxy. "
             r"Seeds averaged where two exist.}",
             r"\label{tab:w2-depth}", r"\small", r"\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Workload & Objective & Metric & $d{=}1$ & $d{=}2$ & $d{=}3$ & $d{=}4$ & $d{=}6$ & $d{=}8$ & Mean$_{1:8}$ \\", r"\midrule"]
    md += ["## W2 — per-depth diagnostics on test prompts (decode convention)", "",
           "| bench | objective | metric | d1 | d2 | d3 | d4 | d6 | d8 | mean |", "|---|---|---|---|---|---|---|---|---|---|"]
    for bench in ("mtbench", "humaneval"):
        first = True
        for label, names in groups.items():
            rows = []
            for n in names:
                f = rollout_root / f"{n}-{bench}-test" / "summary.csv"
                if f.exists():
                    rows.append({int(r["depth"]): r for r in csv.DictReader(open(f, newline="", encoding="utf-8"))})
            if not rows:
                continue
            def col(metric, d):
                return float(np.mean([float(r[d][metric]) for r in rows if d in r]))
            for metric, mname, fmt in (("top1_agreement", "Top-1", "{:.3f}"), ("ce_full", "CE$_{\\mathrm{full}}$", "{:.2f}"),
                                       ("ce_proxy", "CE$_{\\mathrm{proxy}}$", "{:.2f}")):
                vals = [col(metric, d) for d in (1, 2, 3, 4, 6, 8)]
                mean = float(np.mean([col(metric, d) for d in range(1, 9)]))
                wl = BENCH[bench] if first else ""
                lab = label if metric == "top1_agreement" else ""
                first = False
                lines.append(f"{wl} & {lab} & {mname} & " + " & ".join(fmt.format(v) for v in vals) + f" & {fmt.format(mean)} \\\\")
                md.append(f"| {bench} | {label} | {metric} | " + " | ".join(fmt.format(v) for v in vals) + f" | {fmt.format(mean)} |")
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_w2_depth.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md.append("")
    # (a') the same continuations in the training rollout convention (context on predicted features)
    tl = [r"\begin{table}[htbp]", r"\centering",
          r"\caption{As Table~\ref{tab:w2-depth}, but in the training-time rollout convention of the held-out evaluator: "
          r"at step $s$ the whole sequence is re-run with the previous step's predicted features as inputs, so the context "
          r"rows are predicted features rather than true ones. Same test continuations, same roots and steps.}",
          r"\label{tab:w2-depth-train}", r"\small", r"\setlength{\tabcolsep}{2.6pt}",
          r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
          r"Workload & Objective & Metric & $d{=}1$ & $d{=}2$ & $d{=}3$ & $d{=}4$ & $d{=}6$ & $d{=}8$ & Mean$_{1:8}$ \\", r"\midrule"]
    md += ["## W2 — same continuations, training rollout convention", "",
           "| bench | objective | metric | d1 | d2 | d3 | d4 | d6 | d8 | mean |", "|---|---|---|---|---|---|---|---|---|---|"]
    any_train = False
    for bench in ("mtbench", "humaneval"):
        first = True
        for label, names in groups.items():
            rows = []
            for n in names:
                f = rollout_root / f"{n}-{bench}-test-train" / "summary.csv"
                if f.exists():
                    rows.append({int(r["depth"]): r for r in csv.DictReader(open(f, newline="", encoding="utf-8"))})
            if not rows:
                continue
            any_train = True

            def colt(metric, d, rows=rows):
                return float(np.mean([float(r[d][metric]) for r in rows if d in r]))
            for metric, mname, fmt in (("top1_agreement", "Top-1", "{:.3f}"), ("ce_full", "CE$_{\\mathrm{full}}$", "{:.2f}"),
                                       ("ce_proxy", "CE$_{\\mathrm{proxy}}$", "{:.2f}")):
                vals = [colt(metric, d) for d in (1, 2, 3, 4, 6, 8)]
                mean = float(np.mean([colt(metric, d) for d in range(1, 9)]))
                wl = BENCH[bench] if first else ""
                lab = label if metric == "top1_agreement" else ""
                first = False
                tl.append(f"{wl} & {lab} & {mname} & " + " & ".join(fmt.format(v) for v in vals) + f" & {fmt.format(mean)} \\\\")
                md.append(f"| {bench} | {label} | {metric} | " + " | ".join(fmt.format(v) for v in vals) + f" | {fmt.format(mean)} |")
            tl.append(r"\addlinespace[1pt]")
    tl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    if any_train:
        (out / "tab_w2_depth_train.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")
    md.append("")
    # (b) held-out ShareGPT evaluator with --full-target (training convention)
    hl = [r"\begin{table}[htbp]", r"\centering",
          r"\caption{The paper's held-out ShareGPT evaluator (400 conversations, eight-step rollout, training convention) "
          r"re-run with the full target distribution. Uniform: equal-weight mean over the eight steps. Top-1: agreement of the "
          r"drafter's argmax with the target's argmax at each step. Seeds averaged where two exist.}",
          r"\label{tab:w2-heldout}", r"\small", r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular}{llrrrrrrr}", r"\toprule",
          r"Objective & Metric & $s{=}1$ & $s{=}2$ & $s{=}3$ & $s{=}4$ & $s{=}6$ & $s{=}8$ & Uniform \\", r"\midrule"]
    md += ["## W2 — held-out ShareGPT evaluator with full target (training convention)", "",
           "| objective | metric | s1 | s2 | s3 | s4 | s6 | s8 | uniform |", "|---|---|---|---|---|---|---|---|---|"]
    hgroups = {"Step-decayed (10k, seeds 1--2)": ["e5-sw-s1", "e5-sw-s2"],
               "Teacher forcing (10k, seeds 1--2)": ["e5-tf-s1", "e5-tf-s2"],
               "Detached (10k, seeds 1--2)": ["e5-det-s1", "e5-det-s2"],
               "Step-decayed (10k, seed 0)": ["m3-sw10k"], "Teacher forcing (10k, seed 0)": ["m3-tf80k"],
               "Detached (10k, seed 0)": ["m3-det80k"], "Uniform full-gradient (10k, seed 0)": ["m3-bp80k"],
               "Scaled (20k, seed 0)": ["m3-sw20k"]}
    for label, names in hgroups.items():
        js = [json.load(open(heldout_root / f"{n}.json")) for n in names if (heldout_root / f"{n}.json").exists()]
        if not js:
            continue
        for key, mname, fmt in (("per_step_ce", "CE$_{\\mathrm{proxy}}$", "{:.2f}"), ("per_step_ce_full", "CE$_{\\mathrm{full}}$", "{:.2f}"),
                                ("per_step_top1_full", "Top-1", "{:.3f}")):
            if key not in js[0]:
                continue
            steps = np.mean([j[key] for j in js], axis=0)
            uni = float(np.mean(steps))
            lab = label if key == "per_step_ce" else ""
            hl.append(f"{lab} & {mname} & " + " & ".join(fmt.format(steps[i]) for i in (0, 1, 2, 3, 5, 7)) + f" & {fmt.format(uni)} \\\\")
            md.append(f"| {label} | {key} | " + " | ".join(fmt.format(steps[i]) for i in (0, 1, 2, 3, 5, 7)) + f" | {fmt.format(uni)} |")
        hl.append(r"\addlinespace[1pt]")
    hl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_w2_heldout.tex").write_text("\n".join(hl) + "\n", encoding="utf-8")
    md.append("")


def tree_summary(tree_root: Path, md: list):
    md += ["## fitted trees (model-expected acceptance under the independence model)", "",
           "| tree | widths | expected accept (fitted tree) | expected accept (chain-static) | dev roots |", "|---|---|---|---|---|"]
    for f in sorted(tree_root.glob("*.json")):
        j = json.load(open(f))
        md.append(f"| {f.stem} | {j['widths']} | {j['model_expected_accept']:.3f} | {j['chain_model_expected_accept']:.3f} | {j['n_dev_roots']} |")
    md.append("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e10")
    ap.add_argument("--rollout", default="results/e10-rollout")
    ap.add_argument("--heldout", default="results/e10-heldout")
    ap.add_argument("--trees", default="results/e10-trees")
    ap.add_argument("--out", default="paper/tables-fact")
    ap.add_argument("--md", default="docs/tables/e10-summary.md")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# E10 — revision campaign (pod D, session d1)", "",
          f"Generated by `scripts/e10_analyze.py`; paired by prompt, {N_BOOT:,} bootstrap resamples (seed {SEED}).", ""]
    cells = load_cells(Path(a.root))
    md.append(f"cells present: {len(cells)}")
    md.append("")
    if Path(a.trees).exists():
        tree_summary(Path(a.trees), md)
    q3_tables(cells, out, md)
    w3_tables(cells, out, md)
    if Path(a.rollout).exists() or Path(a.heldout).exists():
        w2_tables(Path(a.rollout), Path(a.heldout), out, md)
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[E10] {len(cells)} cells -> {out}, {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
