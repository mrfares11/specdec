#!/usr/bin/env python
"""CLI: overlay accept-length-vs-K curves from multiple benchmark.csv files.

    python scripts/compare_runs.py --out results/rq1.png \
        --panel uniform uniform-n64=path.csv uniform-n128=... \
        --panel geometric geometric-n64=... ...

Each --panel gets its own axes (same y-scale everywhere); entries are
label=csv pairs. Colors follow the entry order within a panel (max 3 series
per panel keeps the palette all-pairs safe).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz slots 1-3 (all-pairs safe)
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_SURFACE = "#fcfcfb"
_GRID = "#e5e4e0"


def _read(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [int(r["K"]) for r in rows], [float(r["mean_accept_len"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Mean accepted draft tokens per step")
    ap.add_argument("--panel", action="append", nargs="+", metavar="NAME label=csv...",
                    required=True)
    args = ap.parse_args()

    panels = []
    for spec in args.panel:
        name, entries = spec[0], [e.split("=", 1) for e in spec[1:]]
        panels.append((name, entries))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.4),
                             facecolor=_SURFACE, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    y_max = 0.0
    for ax, (name, entries) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
        for idx, (label, path) in enumerate(entries):
            ks, accepts = _read(path)
            y_max = max(y_max, max(accepts))
            color = _SERIES[idx % len(_SERIES)]
            ax.plot(ks, accepts, color=color, linewidth=2, marker="o",
                    markersize=5, zorder=3, label=label)
            ax.annotate(label, (ks[-1], accepts[-1]), textcoords="offset points",
                        xytext=(6, 0), fontsize=8, color=_INK_2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("draft horizon K", color=_INK_2)
        ax.set_title(name, color=_INK, fontsize=11)
        ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=_INK_2)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.legend(frameon=False, labelcolor=_INK_2, fontsize=9)
    axes[0].set_ylabel("mean accept length", color=_INK_2)
    for ax in axes:
        ax.set_ylim(0, y_max * 1.15)
    fig.suptitle(args.title, color=_INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=_SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
