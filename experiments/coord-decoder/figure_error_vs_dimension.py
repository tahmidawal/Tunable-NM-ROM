"""Error vs reduced dimension: POD rank decay with the coord-net cutting across.

Panel (a): 2D Poisson blob family (this testbed, round-3 N=256 cell).
Panel (b): Heat 2D 1-blob family (heat2d testbed, N=256 cell).

The POD curve is the validation error of the train-fitted SVD basis vs rank —
the price any linear ROM pays per dimension (metric caveat in README.md: a
strong baseline, not a certified floor). The coord-net line shows the FiLM
coordinate decoder's error at its tiny input dimension; the grid-tied point
shows the CP-style control stuck at rank 24. Data straight from committed
JSONs; no new runs.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HEAT = os.path.normpath(os.path.join(
    HERE, "../../../2026-08-13-heat2d-coord-decoder/experiments/"
          "heat2d-coord-decoder/sweep/heat2d_results_N256.json"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"

pj = json.load(open(os.path.join(HERE, "round3", "results_2d_N256.json")))
hj = json.load(open(HEAT))

plt.rcParams.update({
    "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=150)
fig.patch.set_facecolor(SURFACE)

panels = [
    (axes[0], pj, "(a)  Poisson 2D blobs   (N=256 cell)",
     "coord_net", 4, "coord-net (4 inputs)"),
    (axes[1], hj, "(b)  Heat 2D blobs, space-time   (N=256 cell)",
     "film_coord", 6, "coord-net (5 params + t)"),
]
for ax, d, title, net_key, net_dim, net_label in panels:
    ax.set_facecolor(SURFACE)
    ranks = sorted(int(r) for r in d["pod"])
    errs = [d["pod"][str(r)] for r in ranks]
    ax.loglog(ranks, errs, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1)
    ax.annotate("POD (train-fitted SVD basis)", (ranks[2], errs[2]), (10, 6),
                textcoords="offset points", color=INK, fontsize=9.5)

    gt = d["grid_tied"]
    ax.plot([24], [gt], "D", color=AQUA, ms=9, mec=SURFACE, mew=1)
    ax.annotate("grid-tied CP control (rank 24)", (24, gt), (10, 4),
                textcoords="offset points", color=INK, fontsize=9.5)

    net = d[net_key]
    ax.axhline(net, color=ORANGE, lw=2)
    ax.plot([net_dim], [net], "o", color=ORANGE, ms=8, mec=SURFACE, mew=1)
    ax.annotate(f"{net_label}: {net:.1e}", (net_dim, net), (10, -14),
                textcoords="offset points", color=INK, fontsize=9.5)

    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_xlabel("reduced dimension (POD rank / net inputs)")
    ax.set_ylabel("val relative L2")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6)

fig.suptitle("The linear price of a dimension — and the coordinate decoder "
             "cutting across it", fontsize=12.5, color=INK, x=0.02, ha="left",
             y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"error_vs_dimension.{ext}"),
                facecolor=SURFACE, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
print("wrote error_vs_dimension.png / .pdf")
