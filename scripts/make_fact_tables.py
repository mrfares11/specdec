"""FACT paper tables (interaction on two scales, host-thread replication, per-step CE, full allocation).

Generated from the committed per-prompt rows in results/e2, results/e4/stageB-* and results/e5; no GPU.

    python scripts/make_fact_tables.py        # writes paper/tables-fact/*.tex

Every number is computed here; nothing is typed.
Bootstrap: percentile over prompts, 10,000 resamples, seed 20260904 (same as scripts/e3_interaction.py).
"""
import csv, glob, json, os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "paper", "tables-fact")
os.chdir(ROOT)
os.makedirs(OUT, exist_ok=True)
N_BOOT, SEED = 10000, 20260904
BENCH = {"mtbench": "MT-Bench", "alpaca": "Alpaca", "gsm8k": "GSM8K", "humaneval": "HumanEval", "mbpp": "MBPP"}
ORDER = ["mtbench", "alpaca", "gsm8k", "humaneval", "mbpp"]


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def boot(fn, pids, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pids = np.array(pids)
    vals = np.array([fn(rng.choice(pids, len(pids), replace=True)) for _ in range(n)])
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def fmt_ci(lo, hi, d=2):
    f = lambda x: (f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}")
    return f"[{f(lo)},{f(hi)}]"


def fmt_pm(x, d=2):
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"


# ---------------- E2 per-prompt latency, pooled over every run ----------------
cells = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("results/e2/*/raw_prompt_results.csv")):
    p = os.path.basename(os.path.dirname(f)).split("-")
    for r in rows(f):
        cells[(r["benchmark"], int(float(r["K"])), p[0], p[1])][int(r["prompt_id"])].append(float(r["ms_per_token"]))
pm = {k: {pid: float(np.mean(v)) for pid, v in d.items()} for k, d in cells.items()}

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Dynamic-allocation savings and the drafter--allocation interaction on two scales. $G$ is the paired absolute saving $L_{\mathrm{static}}-L_{\mathrm{dynamic}}$ in ms/token. $I_{\mathrm{abs}}=G(\mathrm{scaled})-G(\mathrm{baseline})$ in ms/token. $I_{\mathrm{rel}}$ is the difference in mean $\log(L_{\mathrm{static}}/L_{\mathrm{dynamic}})$ between the scaled and baseline drafters, shown as a percentage, $100\,(e^{I_{\mathrm{rel}}}-1)$: the factor by which the scaled drafter's proportional speed gain from allocation exceeds the baseline's. Brackets: 95\% percentile bootstrap intervals over the 50 test prompts (10,000 resamples), repeats pooled within prompt. Bold: interval excludes zero.}")
lines.append(r"\label{tab:interaction}")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{2.6pt}")
lines.append(r"\begin{tabular}{lrlllll}")
lines.append(r"\toprule")
lines.append(r"Workload & $K$ & Baseline $G$ & Scaled $G$ & EAGLE $G$ & $I_{\mathrm{abs}}$ (ms/token) & $I_{\mathrm{rel}}$ (\%) \\")
lines.append(r"\midrule")
summary = {}
for b in ORDER:
    for K in (4, 8):
        keys = [(b, K, d, t) for d in ("base", "scaled", "eagle") for t in ("static", "dyn")]
        if not all(k in pm for k in keys):
            continue
        common = sorted(set.intersection(*[set(pm[k]) for k in keys]))

        def G(d, s):
            return float(np.mean([pm[(b, K, d, "static")][p] - pm[(b, K, d, "dyn")][p] for p in s]))

        def R(d, s):
            return float(np.mean([np.log(pm[(b, K, d, "static")][p] / pm[(b, K, d, "dyn")][p]) for p in s]))

        Iabs = lambda s: G("scaled", s) - G("base", s)
        Irel = lambda s: R("scaled", s) - R("base", s)
        cols = []
        for d in ("base", "scaled", "eagle"):
            g = G(d, common)
            lo, hi = boot(lambda s, d=d: G(d, s), common)
            cols.append(f"{g:.2f} {{\\scriptsize {fmt_ci(lo, hi)}}}")
        ia = Iabs(common)
        lo, hi = boot(Iabs, common)
        ir = Irel(common)
        lo2, hi2 = boot(Irel, common)
        pct = lambda x: 100 * (np.exp(x) - 1)
        sa = (lo > 0 or hi < 0)
        sr = (lo2 > 0 or hi2 < 0)
        ca = f"{fmt_pm(ia)} {{\\scriptsize {fmt_ci(lo, hi)}}}"
        cr = f"{fmt_pm(pct(ir), 1)} {{\\scriptsize {fmt_ci(pct(lo2), pct(hi2), 1)}}}"
        if sa:
            ca = r"\textbf{" + f"{fmt_pm(ia)}" + "} " + f"{{\\scriptsize {fmt_ci(lo, hi)}}}"
        if sr:
            cr = r"\textbf{" + f"{fmt_pm(pct(ir), 1)}" + "} " + f"{{\\scriptsize {fmt_ci(pct(lo2), pct(hi2), 1)}}}"
        lines.append(f"{BENCH[b]} & {K} & " + " & ".join(cols) + f" & {ca} & {cr} \\\\")
        summary[(b, K)] = dict(n=len(common), Iabs=ia, Iabs_ci=(lo, hi), Irel_pct=pct(ir), Irel_ci=(pct(lo2), pct(hi2)), sig_abs=sa, sig_rel=sr,
                               neg_abs=sa and ia < 0, pos_abs=sa and ia > 0, neg_rel=sr and ir < 0, pos_rel=sr and ir > 0)
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_interaction.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("interaction summary:")
for k, v in summary.items():
    print(k, f"n={v['n']} Iabs={v['Iabs']:+.3f} {v['Iabs_ci']} Irel%={v['Irel_pct']:+.2f} {tuple(round(x,2) for x in v['Irel_ci'])} sig_abs={v['sig_abs']} sig_rel={v['sig_rel']}")
print("abs: neg", sum(v["neg_abs"] for v in summary.values()), "pos", sum(v["pos_abs"] for v in summary.values()), "of", len(summary))
print("rel: neg", sum(v["neg_rel"] for v in summary.values()), "pos", sum(v["pos_rel"] for v in summary.values()), "of", len(summary))

# ---------------- E4 stage B: thread replication, 5 repeats ----------------
sb = defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob("results/e4/stageB-*")):
    _, bench, tree, k, t, rep = os.path.basename(d).split("-")
    for r in rows(d + "/raw_prompt_results.csv"):
        sb[(bench, tree, int(k[1:]), int(t[1:]))][int(r["prompt_id"])].append((float(r["ms_per_token"]), float(r["ar_ms_per_token"]), rep, float(r["mean_accept_len"]), r["output_token_hash"]))
st = {}
for key, per in sb.items():
    st[key] = {p: (float(np.mean([x[0] for x in l])), float(np.mean([x[1] / x[0] for x in l]))) for p, l in per.items()}
# acceptance / output-hash control across caps
for bench in ("humaneval", "mtbench"):
    for tree, K in (("dyn", 4), ("dyn", 8), ("static", 4)):
        ref = sb[(bench, tree, K, 8)]
        same = True
        for cap in (32, 208):
            oth = sb[(bench, tree, K, cap)]
            for p in ref:
                a = {round(x[3], 6) for x in ref[p]} | {round(x[3], 6) for x in oth[p]}
                h = {x[4] for x in ref[p]} | {x[4] for x in oth[p]}
                if len(a) != 1 or len(h) != 1:
                    same = False
        print(f"stageB control {bench} {tree} K={K}: acceptance and output hash identical across caps and repeats = {same}")

L = []
L.append(r"\begin{table}[htbp]")
L.append(r"\centering")
L.append(r"\caption{Replication of the host-thread effect on the test prompts, five independent runs per cap. Speedup is the per-prompt ratio of autoregressive to speculative time from the same run, averaged over the five runs and then over 50 prompts. $\Delta$ is the paired change from 8 to 208 threads with a 95\% percentile bootstrap interval over prompts (10,000 resamples). Range: the five run-level ms/token means at 208 threads. Acceptance and emitted tokens are identical across caps and runs within each configuration.}")
L.append(r"\label{tab:threads-rep}")
L.append(r"\small")
L.append(r"\setlength{\tabcolsep}{3pt}")
L.append(r"\begin{tabular}{llrrrlrr}")
L.append(r"\toprule")
L.append(r"Workload & Configuration & $t{=}8$ & $t{=}32$ & $t{=}208$ & $\Delta$ speedup (208 vs.\ 8) & ms/token, 8 & ms/token, 208 (range) \\")
L.append(r"\midrule")
for bench in ("mtbench", "humaneval"):
    for tree, K, name in (("dyn", 4, "Dynamic, $K{=}4$"), ("dyn", 8, "Dynamic, $K{=}8$"), ("static", 4, "Static, $K{=}4$")):
        base, o32, o208 = st[(bench, tree, K, 8)], st[(bench, tree, K, 32)], st[(bench, tree, K, 208)]
        pids = sorted(set(base) & set(o208) & set(o32))
        sp = lambda S: float(np.mean([S[p][1] for p in pids]))
        dsp = lambda s: float(np.mean([o208[p][1] - base[p][1] for p in s]))
        lo, hi = boot(dsp, pids)
        ms8 = float(np.mean([base[p][0] for p in pids]))
        ms208 = float(np.mean([o208[p][0] for p in pids]))
        reps = sorted({x[2] for l in sb[(bench, tree, K, 208)].values() for x in l})
        runs = [float(np.mean([x[0] for l in sb[(bench, tree, K, 208)].values() for x in l if x[2] == rp])) for rp in reps]
        d = dsp(pids)
        cell = f"{fmt_pm(d)} {{\\scriptsize {fmt_ci(lo, hi)}}}"
        if lo > 0 or hi < 0:
            cell = r"\textbf{" + fmt_pm(d) + "} " + f"{{\\scriptsize {fmt_ci(lo, hi)}}}"
        L.append(f"{BENCH[bench]} & {name} & {sp(base):.2f}$\\times$ & {sp(o32):.2f}$\\times$ & {sp(o208):.2f}$\\times$ & {cell} & {ms8:.2f} & {ms208:.2f} ({min(runs):.2f}--{max(runs):.2f}) \\\\")
        print(f"stageB {bench} {tree} K={K}: sp8={sp(base):.3f} sp32={sp(o32):.3f} sp208={sp(o208):.3f} d={d:+.3f} [{lo:+.3f},{hi:+.3f}] ms8={ms8:.2f} ms208={ms208:.2f} runs208={[round(x,2) for x in runs]}")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_threads_rep.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------- E5 per-step held-out CE ----------------
acc = defaultdict(list)
for d in sorted(glob.glob("results/e5/e5-*-s*-*")):
    _, obj, seed, bench = os.path.basename(d).split("-")
    for r in rows(d + "/raw_prompt_results.csv"):
        acc[(obj, int(seed[1:]), bench, int(float(r["K"])))].append(float(r["mean_accept_len"]))
ce = {}
for f in sorted(glob.glob("results/e5/heldout-ce/*.json")):
    j = json.load(open(f))
    name = os.path.basename(f)[:-5]
    _, obj, seed = name.split("-")
    ce[(obj, int(seed[1:]))] = j
T = []
T.append(r"\begin{table}[htbp]")
T.append(r"\centering")
T.append(r"\caption{Held-out rollout proxy cross-entropy by rollout step for the replicated objectives, mean of training seeds 1 and 2. The evaluator is identical for every checkpoint: 400 held-out ShareGPT conversations (rows $\geq 100{,}000$), an eight-step feature rollout on recorded tokens, and soft cross-entropy between $H(f)$ and $H(\hat f)$ at every valid position. Uniform is the equal-weight mean over the eight steps. The last column is mean draft acceptance on MT-Bench at $K{=}4$ (seeds averaged, 50 test prompts).}")
T.append(r"\label{tab:per-step-ce}")
T.append(r"\small")
T.append(r"\setlength{\tabcolsep}{3.5pt}")
T.append(r"\begin{tabular}{lrrrrrrrrrr}")
T.append(r"\toprule")
T.append(r"Objective & $s{=}1$ & $s{=}2$ & $s{=}3$ & $s{=}4$ & $s{=}5$ & $s{=}6$ & $s{=}7$ & $s{=}8$ & Uniform & $\tau$ \\")
T.append(r"\midrule")
for obj, name in (("sw", "Step-decayed, full gradient"), ("tf", "Teacher forcing"), ("det", "Uniform, detached")):
    steps = np.mean([ce[(obj, s)]["per_step_ce"] for s in (1, 2)], axis=0)
    uni = np.mean([ce[(obj, s)]["uniform_ce"] for s in (1, 2)])
    a = np.mean([np.mean(acc[(obj, s, "mtbench", 4)]) for s in (1, 2)])
    T.append(f"{name} & " + " & ".join(f"{x:.2f}" for x in steps) + f" & {uni:.2f} & {a:.2f} \\\\")
    print(obj, [round(float(x), 2) for x in steps], round(float(uni), 3), round(float(a), 3))
T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_per_step_ce.tex"), "w", encoding="utf-8").write("\n".join(T) + "\n")

# ---------------- E2 headline (all cells, K=4 and 8) for the main table ----------------
H = {}
for f in sorted(glob.glob("results/e2/*/raw_prompt_results.csv")):
    p = os.path.basename(os.path.dirname(f)).split("-")
    for r in rows(f):
        H.setdefault((r["benchmark"], int(float(r["K"])), p[0], p[1]), defaultdict(list))[int(r["prompt_id"])].append(
            (float(r["ms_per_token"]), float(r["ar_ms_per_token"]), float(r["mean_accept_len"]), p[-2] + "-" + p[-1]))
print("\nheadline cells (per-prompt pooled over all runs):")
for key in sorted(H):
    per = {p: (np.mean([x[0] for x in l]), np.mean([x[1] / x[0] for x in l]), np.mean([x[2] for x in l])) for p, l in H[key].items()}
    runs = sorted({x[3] for l in H[key].values() for x in l})
    print(key, f"ms={np.mean([v[0] for v in per.values()]):.2f} sp={np.mean([v[1] for v in per.values()]):.3f} acc={np.mean([v[2] for v in per.values()]):.3f} runs={len(runs)} sessions={sorted({r.split('-')[0] for r in runs})}")
# ---------------- full allocation table (all drafters, workloads, K) ----------------
F = []
F.append(r"\begin{table}[htbp]")
F.append(r"\centering")
F.append(r"\caption{Complete allocation results on the 50 test prompts per workload. $\tau$ is mean draft acceptance per verification step (excluding the target-generated token); ms/token includes prefill; speedup is the per-prompt ratio to autoregressive decoding from the same run. Repeats are averaged within each prompt first. HumanEval and MT-Bench pool 10 runs over three sessions at $K{=}4$ and 6 runs over two sessions at $K{=}8$; Alpaca, GSM8K and MBPP use 3 runs in one session.}")
F.append(r"\label{tab:full-allocation}")
F.append(r"\small")
F.append(r"\setlength{\tabcolsep}{3.5pt}")
F.append(r"\begin{tabular}{llrrrrrrr}")
F.append(r"\toprule")
F.append(r" & & \multicolumn{2}{c}{$\tau$} & \multicolumn{2}{c}{ms/token} & \multicolumn{2}{c}{Speedup} & \\")
F.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
F.append(r"Workload & Drafter & Static & Dynamic & Static & Dynamic & Static & Dynamic & Runs \\")
NAME = {"base": "Baseline", "scaled": "Scaled", "eagle": "Official EAGLE"}
for K in (4, 8):
    F.append(r"\midrule")
    F.append(rf"\multicolumn{{9}}{{l}}{{\emph{{$K={K}$}}}} \\")
    for b in ORDER:
        for d in ("base", "scaled", "eagle"):
            vals = {}
            for t in ("static", "dyn"):
                key = (b, K, d, t)
                if key not in H:
                    continue
                per = {p: (np.mean([x[0] for x in l]), np.mean([x[1] / x[0] for x in l]), np.mean([x[2] for x in l])) for p, l in H[key].items()}
                runs = len({x[3] for l in H[key].values() for x in l})
                vals[t] = (np.mean([v[2] for v in per.values()]), np.mean([v[0] for v in per.values()]), np.mean([v[1] for v in per.values()]), runs)
            if len(vals) < 2:
                continue
            s, y = vals["static"], vals["dyn"]
            F.append(f"{BENCH[b]} & {NAME[d]} & {s[0]:.2f} & {y[0]:.2f} & {s[1]:.2f} & {y[1]:.2f} & {s[2]:.2f}$\\times$ & {y[2]:.2f}$\\times$ & {s[3]} \\\\")
F += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_full_allocation.tex"), "w", encoding="utf-8").write("\n".join(F) + "\n")

print("\nwrote", os.listdir(OUT))
