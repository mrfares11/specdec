"""Figures 4, 6, 8, 9 and 11 of arXiv:2611.23463, from committed results only (no GPU).

Run from the repository root:  python scripts/make_arxiv_figures.py <out_dir>
Style matches figures 3-5 of the paper (DejaVu Sans, teal/amber/slate, light grid, 5.5 in wide).
Prints every number the new text quotes, and asserts the qualitative claims.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
TEAL, AMBER, SLATE, GRAY, LIGHT, GRID = "#287C78", "#B57B27", "#525962", "#90969C", "#B5B9BE", "#E9EBED"
INK = "#2B2F33"
W = 5.5

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.6, "xtick.color": GRAY, "ytick.color": GRAY,
    "xtick.labelcolor": SLATE, "ytick.labelcolor": SLATE, "axes.labelcolor": SLATE, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


BENCH = {"mtbench": "MT-Bench", "humaneval": "HumanEval", "alpaca": "Alpaca", "gsm8k": "GSM8K", "mbpp": "MBPP"}


# ------------------------------------------------ 1. per-depth top-1 agreement (RQ1)
def depth_agreement():
    arms = [("sw", "Step-weighted", TEAL, "o"), ("tf", "Teacher forced", AMBER, "^"), ("det", "Detached", SLATE, "s")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for ax, b in zip(axes, ["mtbench", "humaneval"]):
        means = {}
        for key, lab, col, mk in arms:
            runs = [pd.read_csv(f"results/e10-rollout/e5-{key}-s{s}-{b}-test/summary.csv") for s in (1, 2)]
            assert all(len(r) == 8 and (r.n_prompts == 50).all() for r in runs)
            for r in runs:
                ax.plot(r.depth, r.top1_agreement, color=col, lw=0.6, alpha=0.35)
            m = sum(r.top1_agreement.values for r in runs) / 2
            means[key] = m
            ax.plot(runs[0].depth, m, color=col, marker=mk, ms=4, lw=1.4, mec="white", mew=0.6, label=lab)
        # claims used in the text
        assert means["tf"][0] > means["sw"][0] > means["det"][0], b
        assert all(means["sw"][d] > max(means["tf"][d], means["det"][d]) for d in range(1, 8)), b
        assert all(means["det"][d] < min(means["sw"][d], means["tf"][d]) for d in range(8)), b
        print("depth", b, {k: np.round(v, 3).tolist() for k, v in means.items()})
        ax.set_title(BENCH[b])
        ax.set_xlabel("Draft depth")
        ax.set_xticks(range(1, 9))
        ygrid(ax)
    axes[0].set_ylabel("Top-1 agreement with target")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "figure-depth-agreement")


# ------------------------------------------------ 2. per-prompt speedups, five workloads (RQ2)
def prompt_speedups():
    order = ["mtbench", "alpaca", "gsm8k", "humaneval", "mbpp"]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    rng = np.random.default_rng(0)
    total = faster = 0
    for i, b in enumerate(order):
        per = {}
        for alloc in ("static", "dyn"):
            files = sorted(Path("results/e2").glob(f"scaled-{alloc}-{b}-k4-*-r*/raw_prompt_results.csv"))
            df = pd.concat([pd.read_csv(f) for f in files])
            df = df[df.rejected != True] if "rejected" in df else df
            df["speedup"] = df.ar_ms_per_token / df.ms_per_token
            per[alloc] = df.groupby("prompt_id").speedup.mean()
        s, d = per["static"].align(per["dyn"], join="inner")
        assert len(s) == 50, (b, len(s))
        n_faster = int((d > s).sum())
        total += len(s)
        faster += n_faster
        print("prompts", b, "runs", len(files), "static mean", round(s.mean(), 3), "dyn mean", round(d.mean(), 3),
              "dyn faster", n_faster, "/", len(s))
        for x0, ser, col in ((i - 0.18, s, SLATE), (i + 0.18, d, TEAL)):
            bp = ax.boxplot(ser.values, positions=[x0], widths=0.28, showfliers=False, patch_artist=True,
                            medianprops=dict(color="white", lw=1.2), whiskerprops=dict(color=col, lw=0.8),
                            capprops=dict(color=col, lw=0.8), boxprops=dict(facecolor=col, edgecolor=col, alpha=0.35))
            ax.scatter(x0 + rng.uniform(-0.09, 0.09, len(ser)), ser.values, s=5, color=col, alpha=0.8, lw=0, zorder=3)
    print("prompts total dyn faster", faster, "/", total)
    ax.axhline(1.0, color=GRAY, lw=0.6, ls=":")
    ax.set_xticks(range(len(order)), [BENCH[b] for b in order])
    ax.set_ylabel("Speedup over autoregressive")
    ax.tick_params(axis="x", length=0)
    ygrid(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SLATE, alpha=0.6, label="Static tree"), Patch(color=TEAL, alpha=0.6, label="Dynamic tree")],
              ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, "figure-prompt-speedups")
    return faster, total


# ------------------------------------------------ 3. per-step cost breakdown (RQ3)
def cost_breakdown():
    rows = [("Static, 8 threads", "attrib-static-t8"), ("Dynamic, 8 threads", "attrib-dyn-t8"),
            ("Static, 208 threads", "attrib-static-t208"), ("Dynamic, 208 threads", "attrib-dyn-t208")]
    parts = [("Drafting", ["draft_ms"], SLATE), ("Tree construction", ["topology_ms"], AMBER),
             ("Target tree forward", ["target_tree_forward_ms"], TEAL),
             ("Mask, verify and commit", ["mask_ms", "verify_ms", "kv_commit_ms"], LIGHT)]
    data, tok = [], []
    for _, d in rows:
        df = pd.read_csv(f"results/e4/{d}/raw_prompt_results.csv").fillna(0.0)
        w = df.verifier_forwards  # weight per-prompt step means by step count (matches the paper text)
        data.append([sum((df[c] * w).sum() for c in cols) / w.sum() for _, cols, _ in parts])
        tok.append(df.committed_tokens.sum() / w.sum())
        print("cost", d, "prompts", len(df), "split", df.prompt_split.unique(), np.round(data[-1], 2).tolist(),
              "total", round(sum(data[-1]), 2), "tokens/step", round(tok[-1], 2))
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(W, 2.0))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for j, (name, _, col) in enumerate(parts):
        ax.barh(y, data[:, j], left=left, height=0.58, color=col, label=name, edgecolor="white", linewidth=0.6)
        left += data[:, j]
    for yi, tot, t in zip(y, left, tok):
        ax.text(tot + 1.0, yi, f"{tot:.1f} ms, {t:.2f} tokens", va="center", fontsize=7, color=SLATE)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Mean time per decoding step (ms)")
    ax.set_xlim(0, 95)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.3, 1.02), frameon=False, handlelength=1.1,
              columnspacing=1.0, fontsize=7)
    fig.tight_layout()
    save(fig, "figure-cost-breakdown")
    return data


# ------------------------------------------------ 4. acceptance vs speed, per prompt (RQ3)
def accept_vs_speed():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for ax, wl in zip(axes, ["humaneval", "mtbench"]):
        g = {}
        for alloc, thr, col, mk, lab in [("static", 8, SLATE, "s", "Static, 8 threads"),
                                         ("dyn", 8, TEAL, "o", "Dynamic, 8 threads"),
                                         ("dyn", 208, AMBER, "^", "Dynamic, 208 threads"),
                                         ("static", 208, None, None, None)]:
            df = pd.concat([pd.read_csv(f"results/e4/stageB-{wl}-{alloc}-k4-t{thr}-r{r}/raw_prompt_results.csv")
                            for r in range(1, 6)])
            df = df[df.rejected != True] if "rejected" in df else df
            assert set(df.prompt_split) == {"test"}
            df["speedup"] = df.ar_ms_per_token / df.ms_per_token
            g[(alloc, thr)] = df.groupby("prompt_id").agg(x=("tokens_per_verifier_forward", "mean"),
                                                          xsd=("tokens_per_verifier_forward", "std"),
                                                          y=("speedup", "mean"))
            if col:
                ax.scatter(g[(alloc, thr)].x, g[(alloc, thr)].y, s=11, color=col, marker=mk, alpha=0.75,
                           linewidths=0, label=lab)
        for k, v in g.items():
            print("accept", wl, k, "prompts", len(v), "tokens/call", round(v.x.mean(), 2),
                  "max within-prompt sd", round(v.xsd.max(), 4), "speedup", round(v.y.mean(), 2))
        dx = (g[("dyn", 8)].x - g[("dyn", 208)].x).abs().max()
        print("accept", wl, "max |tokens/call 8 vs 208| per prompt", round(dx, 6))
        assert dx < 1e-9
        ax.axhline(1.0, color=GRAY, lw=0.6, ls=":")
        ax.set_title(BENCH[wl] + ", K = 4")
        ax.set_xlabel("Tokens produced per target call")
        ygrid(ax)
    axes[0].set_ylabel("Speedup over autoregressive")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "figure-accept-vs-speed")


# ------------------------------------------------ 5. acceptance vs training time and data (Appendix D)
def training_cost():
    def acc(name):
        return float(pd.read_csv(f"results/m3/{name}/benchmark.csv").mean_accept_len.iloc[0])

    # Recorded 10k-step training times from Appendix D (one H100); step-weighted checkpoints scaled at that rate.
    sw_steps = [2000, 5000, 10000, 20000]
    sw_runs = ["m3-sw80k-s2000-mtbench", "m3-sw80k-s5000-mtbench", "m3-sw80k-s10-mtbench", "m3-sw80k-mtbench"]
    sw_h = [27.0 * s / 10000 for s in sw_steps]
    sw_a = [acc(r) for r in sw_runs]
    tf_a, det_a, bp_a = acc("m3-tf80k-mtbench"), acc("m3-det80k-mtbench"), acc("m3-bp80k-mtbench")
    bp_log = pd.read_csv("results/m3-logs/m3-bp80k-train_log.csv")
    bp_h = (bp_log.time.iloc[-1] - bp_log.time.iloc[0] + (bp_log.time.iloc[1] - bp_log.time.iloc[0])) / 3600
    print("training sw", list(zip(sw_steps, np.round(sw_h, 1), np.round(sw_a, 3))), "tf", round(tf_a, 3),
          "det", round(det_a, 3), "bp", round(bp_a, 3), "bp hours", round(bp_h, 1))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a1.plot(sw_h, sw_a, "-o", color=TEAL, ms=4, mec="white", mew=0.6, label="Step-weighted")
    for h, a, s in zip(sw_h, sw_a, sw_steps):
        a1.annotate(f"{s // 1000}k steps", (h, a), textcoords="offset points", xytext=(6, -3), fontsize=6.5,
                    color=TEAL, ha="left", va="center")
    a1.scatter([2.4], [tf_a], marker="^", s=26, color=AMBER, zorder=3, label="Teacher forced, 10k")
    a1.scatter([14.1], [det_a], marker="s", s=22, color=SLATE, zorder=3, label="Detached, 10k")
    a1.scatter([bp_h], [bp_a], marker="D", s=18, color=GRAY, zorder=3, label="Uniform full-gradient, 10k")
    a1.set_xscale("log")
    a1.set_xlim(1.8, 110)
    a1.set_ylim(1.2, 1.92)
    a1.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    a1.set_xlabel("Training time on one H100 (h, log scale)")
    a1.set_ylabel("Acceptance, MT-Bench, K = 4")
    a1.set_title("Training time")
    a1.legend(loc="upper left", frameon=False, fontsize=6.5, handletextpad=0.3)
    ygrid(a1)

    rows = [20, 40, 80]
    for b, col, mk in (("mtbench", TEAL, "o"), ("humaneval", AMBER, "^")):
        names = [f"m3-sw20k-{b}", f"m3-sw40k-{b}", f"m3-sw80k-s10-{b}"]
        vals = [acc(n) for n in names]
        print("rows", b, list(zip(rows, np.round(vals, 3))))
        a2.plot(rows, vals, "-" + mk, color=col, ms=4, mec="white", mew=0.6, label=BENCH[b])
    a2.set_xscale("log", base=2)
    a2.set_xticks(rows, ["20k", "40k", "80k"])
    a2.set_xlabel("Training feature rows (10k steps)")
    a2.set_title("Training data")
    a2.legend(loc="upper left", frameon=False, fontsize=6.5)
    ygrid(a2)
    fig.tight_layout()
    save(fig, "figure-training-cost")


depth_agreement()
prompt_speedups()
cost_breakdown()
accept_vs_speed()
training_cost()
