"""GN iterations-to-tolerance vs mesh size, trained ROMs, both packages.

Panel (a) HEAT (fixed rollout, k=64 and rank=256 FIXED across N — controlled):
  median GN iterations per warm-started time step at three relative
  gradient-norm tolerances. Flat lines = cost at fixed accuracy is n-free.
Panel (b) POISSON (shipped configs: k and m GROW with N; analytic-data
  inconsistency — see memory/README): median iterations per cold-start solve.
  Growth here tracks the shipped configuration family, not N alone.

Iteration counts are censored at the 30-iteration probe; cells with heavy
censoring are annotated with the capped fraction.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"
TOLS = [(0.01, BLUE, "tol 1e-2"), (0.001, ORANGE, "tol 1e-3"),
        (0.0001, AQUA, "tol 1e-4")]
NS = [32, 64, 128, 256]

data = {}
for f in glob.glob(os.path.join(R, "gn_*.json")):
    d = json.load(open(f))
    data[(d["package"], d["N"])] = d


def stats(pkg, N, tol):
    a = np.asarray(data[(pkg, N)]["iters_per_tol"][str(tol)], dtype=float)
    if pkg == "heat":
        warm = a[:, 1:].ravel()
        return np.median(warm), (a >= 30).mean()
    return np.median(a), (a >= 30).mean()


plt.rcParams.update({
    "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=150)
fig.patch.set_facecolor(SURFACE)

for ax, pkg, title, note in (
    (ax1, "heat",
     "(a)  Heat (fixed rollout): iters per warm step   (k=64, rank=256 fixed)",
     "controlled sweep: k, rank fixed across N\n(m = 100/100/128/256)"),
    (ax2, "poisson",
     "(b)  Poisson: iters per cold-start solve   (shipped configs)",
     "NOT controlled: k = 8/8/12/16 and m = 640/640/960/1280\ngrow with N; analytic-data inconsistency (see README)"),
):
    ax.set_facecolor(SURFACE)
    used_offsets = {}
    for si, (tol, color, label) in enumerate(TOLS):
        med = [stats(pkg, N, tol)[0] for N in NS]
        cap = [stats(pkg, N, tol)[1] for N in NS]
        ax.plot(NS, med, "-o", color=color, lw=2, ms=6, mec=SURFACE, mew=1)
        # stagger end labels that land on the same y (both capped at 30)
        key = round(med[-1])
        bump = used_offsets.get(key, 0)
        used_offsets[key] = bump + 1
        end_cap = f"  ({cap[-1]:.0%} capped)" if cap[-1] >= 0.25 else ""
        ax.annotate(label + end_cap, (NS[-1], med[-1]), (8, -3 - 12 * bump),
                    textcoords="offset points", color=INK, fontsize=9.5)
        for N, m, c in zip(NS[:-1], med[:-1], cap[:-1]):
            if c >= 0.25:
                ax.annotate(f"{c:.0%} capped", (N, m), (-2, -14),
                            textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlim(26, 560)
    ax.set_ylim(0, 37)
    ax.axhline(30, color=GRID, lw=1, ls="--")
    ax.text(29, 30.8, "probe cap (30)", fontsize=8, color=INK2, ha="left")
    ax.set_xlabel("training mesh  N")
    ax.set_ylabel("median GN iterations")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.text(0.02, 0.985, note, transform=ax.transAxes, fontsize=8.5,
            color=INK2, va="top")

fig.suptitle("Iterations to tolerance vs mesh — flat where the configuration "
             "is controlled (heat), growing along the shipped config family "
             "(poisson)", fontsize=12, color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"iters_vs_N.{ext}"), facecolor=SURFACE,
                bbox_inches="tight", dpi=300 if ext == "png" else None)
print("wrote iters_vs_N.png / .pdf")
