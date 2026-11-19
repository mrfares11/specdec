#!/usr/bin/env python
"""Data figures and tables for the paper, generated from committed results.

No GPU and no hand-typed numbers: every value is read from the repo's result
files. Each figure is written as vector PDF (for LaTeX), a 300-dpi PNG preview,
and a CSV of exactly what was plotted -- the figure's table view.

    python make_figures.py --repo "<repo root>" --out "<output dir>" [--boot 10000]

Style: the validated categorical order blue / orange / aqua / yellow on a white
page. Text uses ink tokens, never series colours; gridlines are solid hairlines;
bars have a rounded data end and a square baseline; markers carry a surface ring.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Identity is fixed per entity across every figure, never assigned by rank.
OBJ = {"sw": ("Step-decayed", BLUE),
       "tf": ("Teacher-forced", ORANGE),
       "det": ("Detached", AQUA)}
DRAFTER = {"scaled": ("Scaled (ours)", BLUE),
           "eagle": ("Official EAGLE", ORANGE),
           "base": ("v1.0 baseline", AQUA)}
BENCH = {"mtbench": "MT-Bench", "alpaca": "Alpaca", "gsm8k": "GSM8K",
         "humaneval": "HumanEval", "mbpp": "MBPP"}
TEXTWIDTH = 5.5  # ICLR \textwidth, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": BASE, "ytick.color": BASE,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def save(fig, out: Path, name: str, rows: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    if rows:
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"  {name}")


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, pad=4)


def _ppd(ax):
    """Pixels per data unit on each axis, measured once limits are final."""
    if not hasattr(ax, "_ppd_cache"):
        ax.figure.canvas.draw()
        (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
        ax._ppd_cache = (abs(x1 - x0), abs(y1 - y0))
    return ax._ppd_cache


def bar(ax, pos, length, width, color, start=0.0, horizontal=False,
        r_pt=1.6, zorder=2):
    """One bar with a rounded data end and a square baseline end.

    The radius is in points, so the corner stays circular whatever the axis
    aspect. Axis limits must be final before the first call on an axes.
    """
    if length <= 0:
        return
    ppx, ppy = _ppd(ax)
    r_px = r_pt * ax.figure.dpi / 72.0
    rx, ry = r_px / ppx, r_px / ppy
    style = f"round,pad=0,rounding_size={rx}"
    if horizontal:
        xy, w, h = (start, pos - width / 2), length, width
        square = Rectangle(xy, min(w, 2 * rx), h, fc=color, ec="none", lw=0, zorder=zorder)
    else:
        xy, w, h = (pos - width / 2, start), width, length
        square = Rectangle(xy, w, min(h, 2 * ry), fc=color, ec="none", lw=0, zorder=zorder)
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=style, mutation_aspect=ry / rx,
                                fc=color, ec="none", lw=0, zorder=zorder))
    ax.add_patch(square)


def _lum(hex_: str) -> float:
    h = hex_.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ink_on(fill: str) -> str:
    """White or ink for a label set inside a fill, whichever contrasts more."""
    L = _lum(fill)
    return "#ffffff" if 1.05 / (L + 0.05) >= (L + 0.05) / (_lum(INK) + 0.05) else INK


def _e5_acc(R: Path, obj: str, seed: int, bench: str) -> dict:
    import make_latex_tables as mlt
    return {r["K"]: r["mean_accept_len"]
            for r in mlt._rows(R / f"results/e5/e5-{obj}-s{seed}-{bench}/benchmark.csv")}


def _e5_ce(R: Path, obj: str, seed: int) -> dict:
    return json.loads((R / f"results/e5/heldout-ce/e5-{obj}-s{seed}.json").read_text())


_E3 = None


def e3_cells(R: Path, n_boot: int) -> dict:
    """Dynamic-tree gain per drafter and the interaction, with bootstrap CIs.

    Reuses scripts/e3_interaction.py so the numbers are the ones the paper's
    reproduction command produces: paired by prompt, resampling prompts.
    """
    global _E3
    if _E3 is not None:
        return _E3
    import e3_interaction as e3
    rows, _ = e3.load(R / "results/e2")
    out = {}
    for b in BENCH:
        for K in (4, 8):
            pids = e3.common_prompts(rows, b, K, ["base", "scaled", "eagle"])
            if len(pids) < 5:
                continue
            cell = {"n_prompts": len(pids)}
            for d in ("base", "scaled", "eagle"):
                g = e3.paired_gain(rows, b, K, d, pids)
                lo, hi = e3.boot_ci(lambda s, d=d: e3.paired_gain(rows, b, K, d, s),
                                    pids, n_boot)
                st = statistics.mean(statistics.mean(rows[(b, K, d, "static")][p])
                                     for p in pids)
                cell[d] = {"G": g, "lo": lo, "hi": hi, "static_ms": st}

            def inter(s, b=b, K=K):
                a_ = e3.paired_gain(rows, b, K, "scaled", s)
                c_ = e3.paired_gain(rows, b, K, "base", s)
                return None if a_ is None or c_ is None else a_ - c_

            ilo, ihi = e3.boot_ci(inter, pids, n_boot)
            cell["I"] = {"I": cell["scaled"]["G"] - cell["base"]["G"], "lo": ilo, "hi": ihi}
            out[(b, K)] = cell
    _E3 = out
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def fig_headline(R: Path, out: Path) -> None:
    import make_latex_tables as mlt
    order = ["base", "scaled", "eagle"]
    rows, stats = [], {}
    for b in ("mtbench", "humaneval"):
        for d in order:
            for t in ("static", "dyn"):
                rr = mlt._cell(f"{d}-{t}-{b}", 4)
                sp = [r["speedup"] for r in rr]
                m = statistics.mean(sp)
                sd = statistics.stdev(sp) if len(sp) > 1 else 0.0
                acc = statistics.mean(r["mean_accept_len"] for r in rr)
                stats[(b, d, t)] = (m, sd)
                rows.append(dict(benchmark=b, drafter=d, tree=t, K=4, n=len(sp),
                                 speedup_mean=round(m, 4), speedup_sd=round(sd, 4),
                                 accept_len=round(acc, 4)))
    top = max(m + sd for m, sd in stats.values()) * 1.16
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.83, bottom=0.17, wspace=0.06)
    w, off = 0.34, 0.19
    for ax, b in zip(axes, ("mtbench", "humaneval")):
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, top)
        ax.grid(axis="y")
        ax.axhline(1.0, color=INK2, lw=0.6, zorder=1)
        for i, d in enumerate(order):
            for t, dx, col in (("static", -off, MUTED), ("dyn", off, BLUE)):
                m, sd = stats[(b, d, t)]
                bar(ax, i + dx, m, w, col)
                ax.errorbar(i + dx, m, yerr=sd, fmt="none", ecolor=INK2,
                            elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
                if t == "dyn":
                    ax.text(i + dx, m + sd + top * 0.015, f"{m:.2f}×", ha="center",
                            va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["v1.0\nbaseline", "Scaled\n(ours)", "Official\nEAGLE"])
        ax.tick_params(axis="x", length=0)
        title(ax, f"{BENCH[b]}, $K=4$")
    axes[0].set_ylabel("Speedup vs. autoregressive")
    fig.legend(handles=[Patch(color=MUTED, label="static tree"),
                        Patch(color=BLUE, label="dynamic tree")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0),
               handlelength=0.9, handleheight=0.7, columnspacing=1.2)
    save(fig, out, "fig_headline_speedup", rows)


def fig_host_threads(R: Path, out: Path) -> None:
    import make_latex_tables as mlt
    caps = [1, 4, 8, 32, 208]
    series = [("dyn", 4, BLUE, "dynamic tree, $K=4$"),
              ("dyn", 8, ORANGE, "dynamic tree, $K=8$"),
              ("static", 4, AQUA, "static tree, $K=4$")]
    rows, curves = [], {}
    for b in ("humaneval", "gsm8k"):
        for tree, K, _, _ in series:
            xs, ys = [], []
            for t in caps:
                rr = [r for r in mlt._rows(R / f"results/e4/sweep-{b}-{tree}-k{K}-t{t}/benchmark.csv")
                      if r.get("K") == K]
                if not rr:
                    continue
                r = rr[0]
                xs.append(t)
                ys.append(r["speedup"])
                rows.append(dict(benchmark=b, tree=tree, K=K, thread_cap=t,
                                 speedup=round(r["speedup"], 4),
                                 accept_len=round(r["mean_accept_len"], 4),
                                 spec_ms_per_token=round(r["spec_ms_per_token"], 4),
                                 ar_ms_per_token=round(r["ar_ms_per_token"], 4)))
            curves[(b, tree, K)] = (xs, ys)
    allv = [v for _, ys in curves.values() for v in ys] + [1.0]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.12
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.05), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.80, bottom=0.2, wspace=0.06)
    for ax, b in zip(axes, ("humaneval", "gsm8k")):
        ax.set_xscale("log")
        ax.set_xlim(0.75, 290)
        ax.set_ylim(lo - pad, hi + pad * 1.8)
        ax.grid(axis="y")
        ax.axhline(1.0, color=INK2, lw=0.6, zorder=1)
        for tree, K, col, lab in series:
            xs, ys = curves[(b, tree, K)]
            ax.plot(xs, ys, "-o", color=col, ms=4.2, mec=SURFACE, mew=0.8,
                    label=lab, zorder=3)
        xs, ys = curves[(b, "dyn", 4)]
        for t, off, ha, va in ((8, (0, 5), "center", "bottom"), (208, (-6, -4), "right", "top")):
            if t in xs:
                v = ys[xs.index(t)]
                ax.annotate(f"{v:.2f}×", (t, v), xytext=off, textcoords="offset points",
                            ha=ha, va=va, fontsize=6.8, color=INK)
        ax.set_xticks(caps)
        ax.set_xticklabels([str(c) for c in caps])
        ax.minorticks_off()
        ax.set_xlabel("Host thread cap (log scale)")
        title(ax, BENCH[b])
    axes[0].set_ylabel("Speedup vs. autoregressive")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.0),
               columnspacing=1.2)
    save(fig, out, "fig_host_threads", rows)


def fig_objective_replication(R: Path, out: Path) -> None:
    objs = ["det", "tf", "sw"]
    panels = [("mtbench", 4), ("humaneval", 4), ("mtbench", 8), ("humaneval", 8)]
    data, rows = {}, []
    for b, K in panels:
        for o in objs:
            for s in (1, 2):
                v = _e5_acc(R, o, s, b)[K]
                data[(b, K, o, s)] = v
                rows.append(dict(benchmark=b, K=K, objective=o, seed=s,
                                 accept_len=round(v, 4)))
    vals = list(data.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 2.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.93, bottom=0.15,
                        wspace=0.07, hspace=0.45)
    for ax, (b, K) in zip(axes.flat, panels):
        ax.set_xlim(lo - span * 0.08, hi + span * 0.2)
        ax.set_ylim(-0.6, 2.6)
        ax.grid(axis="x")
        for j, o in enumerate(objs):
            xs = [data[(b, K, o, s)] for s in (1, 2)]
            ax.scatter(xs, [j - 0.1, j + 0.1], s=30, color=OBJ[o][1],
                       edgecolors=SURFACE, linewidths=0.8, zorder=3)
        sw = statistics.mean(data[(b, K, "sw", s)] for s in (1, 2))
        de = statistics.mean(data[(b, K, "det", s)] for s in (1, 2))
        ax.text(max(data[(b, K, "sw", s)] for s in (1, 2)) + span * 0.03, 2,
                f"+{100 * (sw / de - 1):.0f}%", va="center", ha="left",
                fontsize=7, color=INK)
        ax.set_yticks(range(3))
        ax.set_yticklabels([OBJ[o][0] for o in objs])
        ax.tick_params(axis="y", length=0)
        title(ax, f"{BENCH[b]}, $K={K}$")
    for ax in axes[1]:
        ax.set_xlabel("Mean acceptance length")
    save(fig, out, "fig_objective_replication", rows)


def fig_ce_vs_acceptance(R: Path, out: Path) -> None:
    objs = ["sw", "tf", "det"]
    pts, rows = {}, []
    for o in objs:
        for s in (1, 2):
            j = _e5_ce(R, o, s)
            a = _e5_acc(R, o, s, "mtbench")[4]
            pts[(o, s)] = (j["per_step_ce"][0], j["uniform_ce"], a)
            rows.append(dict(objective=o, seed=s, step1_ce=round(j["per_step_ce"][0], 4),
                             uniform_ce=round(j["uniform_ce"], 4),
                             mtbench_accept_k4=round(a, 4)))
    ys_all = [p[2] for p in pts.values()]
    ypad = (max(ys_all) - min(ys_all)) * 0.2
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.2, wspace=0.08)
    for ax, idx, xl, ttl in (
            (axes[0], 0, "Held-out CE at rollout step 1", "(a) Step-1 cross-entropy"),
            (axes[1], 1, "Held-out CE, mean over 8 rollout steps", "(b) Multi-step cross-entropy")):
        xs_all = [p[idx] for p in pts.values()]
        xpad = (max(xs_all) - min(xs_all)) * 0.14
        ax.set_xlim(min(xs_all) - xpad, max(xs_all) + xpad)
        ax.set_ylim(min(ys_all) - ypad, max(ys_all) + ypad * 1.5)
        ax.grid(True)
        for o in objs:
            xs = [pts[(o, s)][idx] for s in (1, 2)]
            ys = [pts[(o, s)][2] for s in (1, 2)]
            ax.scatter(xs, ys, s=30, color=OBJ[o][1], edgecolors=SURFACE,
                       linewidths=0.8, zorder=3)
            ax.annotate(OBJ[o][0], (statistics.mean(xs), max(ys)), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.8, color=INK2)
        ax.set_xlabel(xl)
        title(ax, ttl)
    axes[0].set_ylabel("Acceptance length\n(MT-Bench, $K=4$)")
    fig.legend(handles=[Line2D([0], [0], ls="none", marker="o", ms=5, color=OBJ[o][1],
                               mec=SURFACE, mew=0.8, label=OBJ[o][0]) for o in objs],
               loc="upper right", ncol=3, bbox_to_anchor=(0.98, 1.0), columnspacing=1.2)
    save(fig, out, "fig_ce_vs_acceptance", rows)


def fig_per_step_ce(R: Path, out: Path) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(2.75, 2.15))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.84, bottom=0.2)
    ax.grid(axis="y")
    for o in ("tf", "sw", "det"):
        curves = [_e5_ce(R, o, s)["per_step_ce"] for s in (1, 2)]
        n = min(len(c) for c in curves)
        steps = list(range(1, n + 1))
        mean = [statistics.mean(c[i] for c in curves) for i in range(n)]
        lo = [min(c[i] for c in curves) for i in range(n)]
        hi = [max(c[i] for c in curves) for i in range(n)]
        ax.fill_between(steps, lo, hi, color=OBJ[o][1], alpha=0.12, lw=0, zorder=1)
        ax.plot(steps, mean, "-o", color=OBJ[o][1], ms=3.6, mec=SURFACE, mew=0.7,
                zorder=3, label=OBJ[o][0])
        for i in range(n):
            rows.append(dict(objective=o, step=i + 1, ce_mean=round(mean[i], 4),
                             ce_seed1=round(curves[0][i], 4),
                             ce_seed2=round(curves[1][i], 4)))
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Rollout step $s$")
    ax.set_ylabel("Held-out cross-entropy")
    ax.legend(loc="upper left", handlelength=1.4)
    title(ax, "Held-out CE by rollout step")
    save(fig, out, "fig_per_step_ce", rows)


def fig_cost_decomposition(R: Path, out: Path) -> None:
    import make_latex_tables as mlt
    configs = [("base", "static"), ("base", "dyn"), ("scaled", "static"),
               ("scaled", "dyn"), ("eagle", "static"), ("eagle", "dyn")]
    phases = [("t_draft_ms_per_step", "draft", BLUE),
              ("t_tree_forward_ms_per_step", "target tree forward", ORANGE),
              ("t_verify_ms_per_step", "verify", AQUA),
              ("t_commit_ms_per_step", "commit", YELLOW)]
    rows, seg = [], {}
    for d, t in configs:
        rr = [r for r in mlt._rows(R / f"results/m5-timing/{d}-{t}-mtbench/benchmark.csv")
              if r.get("K") == 8]
        vals = [statistics.mean(r[k] for r in rr) for k, _, _ in phases]
        seg[(d, t)] = vals
        rec = dict(drafter=d, tree=t, K=8, benchmark="mtbench", n=len(rr))
        rec.update({lab.replace(" ", "_") + "_ms": round(v, 3)
                    for (_, lab, _), v in zip(phases, vals)})
        rec["total_ms"] = round(sum(vals), 3)
        rows.append(rec)
    totmax = max(sum(v) for v in seg.values())
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.05))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.84, bottom=0.19)
    ax.set_xlim(0, totmax * 1.1)
    ax.set_ylim(len(configs) - 0.4, -0.6)
    ax.grid(axis="x")
    h = 0.56
    for y, (d, t) in enumerate(configs):
        x = 0.0
        vals = seg[(d, t)]
        for k, ((_, _, col), v) in enumerate(zip(phases, vals)):
            last = k == len(phases) - 1
            if last:
                bar(ax, y, v, h, col, start=x, horizontal=True)
            else:
                ax.add_patch(Rectangle((x, y - h / 2), v, h, fc=col, ec="none", lw=0, zorder=2))
            if v > totmax * 0.08:
                ax.text(x + v / 2, y, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.8, color=ink_on(col), zorder=4)
            x += v
            if not last:  # 2-px surface gap between touching segments
                ax.plot([x, x], [y - h / 2, y + h / 2], color=SURFACE, lw=1.0,
                        zorder=3, solid_capstyle="butt")
        ax.text(x + totmax * 0.012, y, f"{x:.1f}", ha="left", va="center",
                fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([f"{DRAFTER[d][0]}, {'dynamic' if t == 'dyn' else 'static'}"
                        for d, t in configs])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Per-step cost (ms/step, timers on; MT-Bench, $K=8$)")
    vmin = min(v[2] for v in seg.values())
    vmax = max(v[2] for v in seg.values())
    labels = ["draft", "target tree forward", f"verify ({vmin:.2f}–{vmax:.2f} ms)", "commit"]
    fig.legend(handles=[Patch(color=c, label=l) for (_, _, c), l in zip(phases, labels)],
               loc="upper right", ncol=4, bbox_to_anchor=(0.99, 1.0),
               handlelength=0.9, handleheight=0.7, columnspacing=1.0)
    save(fig, out, "fig_cost_decomposition", rows)


def fig_learning_curve(R: Path, out: Path) -> None:
    import make_latex_tables as mlt
    steps = [2000, 4000, 6000, 8000, 10000]
    acc, spd, ce, rows = [], [], [], []
    for n in steps:
        r = next(r for r in mlt._rows(R / f"results/e6/dev-mtbench-step{n}/benchmark.csv")
                 if r.get("K") == 4)
        c = json.loads((R / f"results/e6/heldout-ce/step_{n}.json").read_text())["uniform_ce"]
        acc.append(r["mean_accept_len"])
        spd.append(r["speedup"])
        ce.append(c)
        rows.append(dict(step=n, accept_len_k4=round(r["mean_accept_len"], 4),
                         speedup_k4=round(r["speedup"], 4), heldout_uniform_ce=round(c, 4)))
    k = [n / 1000 for n in steps]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.45)
    panels = ((acc, "Acceptance length ($K=4$)", "{:.2f}"),
              (spd, "Speedup vs. autoregressive", "{:.2f}×"),
              (ce, "Held-out CE (8-step mean)", "{:.2f}"))
    for ax, (ys, ttl, fmt) in zip(axes, panels):
        span = max(ys) - min(ys)
        ax.set_xlim(1.2, 10.8)
        ax.set_ylim(min(ys) - span * 0.32, max(ys) + span * 0.32)
        ax.grid(axis="y")
        ax.plot(k, ys, "-o", color=BLUE, ms=4, mec=SURFACE, mew=0.8, zorder=3)
        for i, nb in ((0, 1), (len(ys) - 1, len(ys) - 2)):
            below = ys[nb] > ys[i]
            ax.annotate(fmt.format(ys[i]), (k[i], ys[i]), xytext=(0, -6 if below else 5),
                        textcoords="offset points", ha="center",
                        va="top" if below else "bottom", fontsize=6.8, color=INK)
        ax.set_xticks(k)
        ax.set_xticklabels([f"{int(v)}k" for v in k])
        ax.set_xlabel("Training steps")
        title(ax, ttl)
    save(fig, out, "fig_learning_curve", rows)


def fig_bf16_margins(R: Path, out: Path):
    edges = [("0", 0.5), ("1", 1.5), ("2", 2.5), ("3–7", 7.5), ("8–31", 31.5),
             ("≥32", float("inf"))]
    agrees = lambda r: r["argmax_agrees"].strip().lower() in ("true", "1", "yes")
    agree, dis = Counter(), Counter()
    for f in sorted(glob.glob(str(R / "results/e1/margins-*/procedure_B_bfloat16.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            lab = next(l for l, ub in edges if float(r["margin_in_ulps"]) < ub)
            (agree if agrees(r) else dis)[lab] += 1
    f32_n = f32_dis = 0
    for f in glob.glob(str(R / "results/e1/margins-*/procedure_B_float32.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            f32_n += 1
            f32_dis += not agrees(r)
    na, nd = sum(agree.values()), sum(dis.values())
    rows = [dict(margin_ulps=l, agree_count=agree[l], agree_share=round(agree[l] / na, 5),
                 disagree_count=dis[l], disagree_share=round(dis[l] / nd, 5) if nd else 0.0)
            for l, _ in edges]
    fig, ax = plt.subplots(figsize=(2.75, 2.15))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.74, bottom=0.24)
    ax.set_xlim(-0.6, len(edges) - 0.4)
    ax.set_ylim(0, 108)
    ax.grid(axis="y")
    w = 0.34
    for i, (l, _) in enumerate(edges):
        dv = 100 * dis[l] / nd if nd else 0.0
        av = 100 * agree[l] / na
        bar(ax, i - 0.19, dv, w, BLUE)
        bar(ax, i + 0.19, av, w, MUTED)
        if dv > 0:
            ax.text(i - 0.19, dv + 1.5, f"{dv:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
    ax.set_xticks(range(len(edges)))
    ax.set_xticklabels([l for l, _ in edges])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Top-1 / top-2 logit margin (bf16 ULPs)")
    ax.set_ylabel("Share of positions (%)")
    fig.legend(handles=[Patch(color=BLUE, label=f"argmax differs (n={nd})"),
                        Patch(color=MUTED, label=f"argmax agrees (n={na:,})")],
               loc="lower left", bbox_to_anchor=(0.15, 0.76), ncol=1,
               handlelength=0.9, handleheight=0.7)
    save(fig, out, "fig_bf16_margins", rows)
    print(f"    float32: {f32_dis} of {f32_n} same-prefix positions disagree")


def fig_transfer(R: Path, out: Path, n_boot: int) -> None:
    cells = e3_cells(R, n_boot)
    order = [b for b in BENCH if any((b, K) in cells for K in (4, 8))]
    rows = []
    xmax = max(c[d]["hi"] for c in cells.values() for d in DRAFTER) * 1.06
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.35), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.82, bottom=0.18, wspace=0.06)
    for ax, K in zip(axes, (4, 8)):
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.grid(axis="x")
        for j, b in enumerate(order):
            c = cells.get((b, K))
            if not c:
                continue
            for k, d in enumerate(("scaled", "eagle", "base")):
                g, lo, hi = c[d]["G"], c[d]["lo"], c[d]["hi"]
                y = j + (k - 1) * 0.24
                col = DRAFTER[d][1]
                ax.plot([lo, hi], [y, y], color=col, lw=1.3, zorder=2)
                ax.plot([g], [y], "o", color=col, ms=4.2, mec=SURFACE, mew=0.8, zorder=3)
                rows.append(dict(benchmark=b, K=K, drafter=d, n_prompts=c["n_prompts"],
                                 gain_ms_per_token=round(g, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4),
                                 static_ms_per_token=round(c[d]["static_ms"], 4),
                                 gain_pct_of_static=round(100 * g / c[d]["static_ms"], 2),
                                 n_boot=n_boot))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([BENCH[b] for b in order])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Latency saved by dynamic allocation (ms/token)")
        title(ax, f"$K={K}$")
    handles = [Line2D([0], [0], color=DRAFTER[d][1], lw=1.3, marker="o", ms=4.2,
                      mec=SURFACE, mew=0.8, label=DRAFTER[d][0])
               for d in ("scaled", "eagle", "base")]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.0),
               columnspacing=1.2)
    save(fig, out, "fig_transfer", rows)


def fig_interaction(R: Path, out: Path, n_boot: int) -> None:
    cells = e3_cells(R, n_boot)
    order = [b for b in BENCH if any((b, K) in cells for K in (4, 8))]
    rows = []
    lo_all = min(c["I"]["lo"] for c in cells.values())
    hi_all = max(c["I"]["hi"] for c in cells.values())
    pad = (hi_all - lo_all) * 0.08
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.82, bottom=0.2, wspace=0.06)
    for ax, K in zip(axes, (4, 8)):
        ax.set_xlim(lo_all - pad, hi_all + pad)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.grid(axis="x")
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for j, b in enumerate(order):
            c = cells.get((b, K))
            if not c:
                continue
            i_, lo, hi = c["I"]["I"], c["I"]["lo"], c["I"]["hi"]
            sig = lo > 0 or hi < 0
            col = BLUE if sig else MUTED
            ax.plot([lo, hi], [j, j], color=col, lw=1.3, zorder=2)
            ax.plot([i_], [j], "o", color=col, ms=4.2, mec=SURFACE, mew=0.8, zorder=3)
            rows.append(dict(benchmark=b, K=K, interaction_ms_per_token=round(i_, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             excludes_zero=sig, n_prompts=c["n_prompts"], n_boot=n_boot))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([BENCH[b] for b in order])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Interaction $I$ (ms/token)")
        title(ax, f"$K={K}$")
    handles = [Line2D([0], [0], color=BLUE, lw=1.3, marker="o", ms=4.2, mec=SURFACE,
                      mew=0.8, label="95% interval excludes 0"),
               Line2D([0], [0], color=MUTED, lw=1.3, marker="o", ms=4.2, mec=SURFACE,
                      mew=0.8, label="interval includes 0")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0),
               columnspacing=1.2)
    save(fig, out, "fig_interaction", rows)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def tab_main_results(R: Path, out: Path) -> None:
    import make_latex_tables as mlt
    cols = [("mtbench", 4), ("humaneval", 4), ("mtbench", 8), ("humaneval", 8)]
    order = ["base", "scaled", "eagle"]
    names = {"base": "v1.0 baseline", "scaled": "Scaled (ours)", "eagle": "Official EAGLE"}
    data = {}
    for d in order:
        for t in ("static", "dyn"):
            for b, K in cols:
                rr = mlt._cell(f"{d}-{t}-{b}", K)
                data[(d, t, b, K)] = (statistics.mean(r["mean_accept_len"] for r in rr),
                                      statistics.mean(r["speedup"] for r in rr))
    best = {(b, K, i): max(v[i] for key, v in data.items() if key[2:] == (b, K))
            for b, K in cols for i in (0, 1)}

    def cell(v, b, K, i):
        s = f"{v:.2f}"
        return rf"\textbf{{{s}}}" if abs(v - best[(b, K, i)]) < 1e-12 else s

    L = [
        r"% Generated by make_figures.py from results/m5 + results/m5-rep (no hand-typed numbers).",
        r"% Needs booktabs, and row shading via \usepackage[table]{xcolor} (or colortbl).",
        r"\definecolor{rowtint}{HTML}{F2F6FC}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Main results under identical greedy decoding and a 256-node budget. "
        r"$\tau$ is the mean acceptance length (tokens accepted per target forward); "
        r"speedup is wall-clock against autoregressive decoding, mean of $n{=}3$ repeats. "
        r"Best per column in bold.}",
        r"\label{tab:main-results}",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{l rr rr rr rr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{$K{=}4$} & \multicolumn{4}{c}{$K{=}8$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r" & \multicolumn{2}{c}{MT-Bench} & \multicolumn{2}{c}{HumanEval}"
        r" & \multicolumn{2}{c}{MT-Bench} & \multicolumn{2}{c}{HumanEval} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Drafter & $\tau$ & Speedup & $\tau$ & Speedup & $\tau$ & Speedup & $\tau$ & Speedup \\",
        r"\midrule",
    ]
    for t, head in (("static", "Static tree (geometric profile)"),
                    ("dyn", "Dynamic tree (same 256-node budget)")):
        L.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{head}}}}} \\")
        for i, d in enumerate(order):
            cs = []
            for b, K in cols:
                a, s = data[(d, t, b, K)]
                cs += [cell(a, b, K, 0), cell(s, b, K, 1)]
            shade = r"\rowcolor{rowtint} " if i % 2 == 1 else ""
            L.append(shade + names[d] + " & " + " & ".join(cs) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_main_results.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("  tab_main_results")


def tab_transfer(R: Path, out: Path, n_boot: int) -> None:
    cells = e3_cells(R, n_boot)
    ci = lambda lo, hi: rf"{{\scriptsize$[{lo:+.2f}, {hi:+.2f}]$}}"
    L = [
        r"% Generated by make_figures.py from results/e2 via scripts/e3_interaction.py.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Latency saved by dynamic allocation, $G = L_{\mathrm{static}} - "
        r"L_{\mathrm{dynamic}}$ (ms/token; positive means dynamic is faster), paired by "
        r"prompt and pooled over every committed session. Brackets: 95\% percentile "
        rf"bootstrap intervals over prompts ({n_boot:,} resamples). "
        r"$I = G(\mathrm{Scaled}) - G(\mathrm{v1.0})$ is the drafter $\times$ tree "
        r"interaction.}",
        r"\label{tab:transfer}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lr rrr r}",
        r"\toprule",
        r"Benchmark & $K$ & v1.0 baseline & Scaled (ours) & Official EAGLE & Interaction $I$ \\",
        r"\midrule",
    ]
    for b in BENCH:
        for K in (4, 8):
            c = cells.get((b, K))
            if not c:
                continue
            row = [BENCH[b], str(K)]
            row += [f"{c[d]['G']:.2f} " + ci(c[d]["lo"], c[d]["hi"])
                    for d in ("base", "scaled", "eagle")]
            row += [f"${c['I']['I']:+.2f}$ " + ci(c["I"]["lo"], c["I"]["hi"])]
            L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_transfer.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("  tab_transfer")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="repo root (holds results/ and scripts/)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <repo>/paper; plots go to paper/figures)")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--only", default="",
                    help="comma-separated names to rebuild, e.g. fig_headline,tab_transfer")
    a = ap.parse_args()
    R = Path(a.repo).resolve()
    sys.path.insert(0, str(R / "scripts"))
    if a.out is None:
        plots, tables = R / "paper" / "figures", R / "paper" / "tables"
    else:
        out = Path(a.out)
        plots, tables = out / "plots", out / "tables"
    jobs = {
        "fig_headline": lambda: fig_headline(R, plots),
        "fig_host_threads": lambda: fig_host_threads(R, plots),
        "fig_objective_replication": lambda: fig_objective_replication(R, plots),
        "fig_ce_vs_acceptance": lambda: fig_ce_vs_acceptance(R, plots),
        "fig_per_step_ce": lambda: fig_per_step_ce(R, plots),
        "fig_cost_decomposition": lambda: fig_cost_decomposition(R, plots),
        "fig_learning_curve": lambda: fig_learning_curve(R, plots),
        "fig_bf16_margins": lambda: fig_bf16_margins(R, plots),
        "fig_transfer": lambda: fig_transfer(R, plots, a.boot),
        "fig_interaction": lambda: fig_interaction(R, plots, a.boot),
        "tab_main_results": lambda: tab_main_results(R, tables),
        "tab_transfer": lambda: tab_transfer(R, tables, a.boot),
    }
    sel = [s for s in a.only.split(",") if s]
    unknown = [s for s in sel if s not in jobs]
    if unknown:
        ap.error(f"unknown names {unknown}; choose from {sorted(jobs)}")
    print("building:")
    for name, fn in jobs.items():
        if not sel or name in sel:
            fn()
    if _E3 is not None:
        c = _E3
        t_ok = sum(v["eagle"]["lo"] > 0 for v in c.values())
        neg = sum(v["I"]["hi"] < 0 for v in c.values())
        pos = sum(v["I"]["lo"] > 0 for v in c.values())
        print(f"summary: EAGLE transfer interval excludes 0 in {t_ok}/{len(c)} cells; "
              f"interaction significantly negative {neg}, positive {pos}, "
              f"not significant {len(c) - neg - pos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
