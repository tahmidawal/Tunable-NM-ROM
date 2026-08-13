"""Hari's figure: online ROM solve cost depends on k only — n is not in the formula.

Panel (a): cost of one online solve vs mesh node count n = N^2 at fixed k=8.
           Both ROM arms are flat across a 256x range of n; the FOM rises.
Panel (b): cost per GN iteration vs latent dimension k at N=512 (log-log)
           with fitted power-law exponents — the cost curve rides on k alone.

Data: cp_timing_pax106.json (this dir) and coordnet_timing_pax106.json (the
coordnet worktree) — both measured sequentially on the same A100 (pax106),
median of 30 solves, fixed 10 GN iterations, JIT warm-up excluded.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CN_JSON = os.path.normpath(os.path.join(
    HERE, "../../../2026-08-13-cost-scaling-coordnet/experiments/"
          "cost-scaling-coordnet/coordnet_timing_pax106.json"))

# palette (validated, dataviz reference instance, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"

cp = json.load(open(os.path.join(HERE, "cp_timing_pax106.json")))
cn = json.load(open(CN_JSON))
GN_ITERS = cp["gn_iters"]
K_FIX, N_FIX = 8, 512


def series_vs_n(data, k):
    cells = sorted([c for c in data["cells"] if c["k"] == k],
                   key=lambda c: c["N"])
    n = np.array([c["N"] ** 2 for c in cells])
    t = np.array([c["median_s_per_gn_iter"] for c in cells]) * GN_ITERS
    return n, t


def series_vs_k(data, N):
    cells = sorted([c for c in data["cells"] if c["N"] == N],
                   key=lambda c: c["k"])
    k = np.array([c["k"] for c in cells])
    t = np.array([c["median_s_per_gn_iter"] for c in cells])
    return k, t


plt.rcParams.update({
    "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=150)
fig.patch.set_facecolor(SURFACE)

# ---------------- panel (a): cost vs n at fixed k ----------------
ax1.set_facecolor(SURFACE)
n_cp, t_cp = series_vs_n(cp, K_FIX)
n_cn, t_cn = series_vs_n(cn, K_FIX)
fom = sorted([(c["N"] ** 2, c["fom_median_s"]) for c in cp["cells"]
              if c.get("fom_median_s")])
n_f, t_f = np.array([x for x, _ in fom]), np.array([y for _, y in fom])

ax1.loglog(n_f, t_f * 1e3, "-o", color=AQUA, lw=2, ms=6, mec=SURFACE, mew=1)
ax1.loglog(n_cn, t_cn * 1e3, "-o", color=ORANGE, lw=2, ms=6, mec=SURFACE, mew=1)
ax1.loglog(n_cp, t_cp * 1e3, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1)

slope_f = np.polyfit(np.log(n_f), np.log(t_f), 1)[0]
ax1.set_ylim(top=t_f.max() * 1e3 * 4)
ax1.annotate(f"FOM CG solve  ~n^{slope_f:.2f}", (n_f[-2], t_f[-2] * 1e3),
             (-10, 4), textcoords="offset points", ha="right", color=INK,
             fontsize=10)
ax1.annotate("coord-net ROM (flat)", (n_cn[-1], t_cn[-1] * 1e3), (-8, 8),
             textcoords="offset points", ha="right", color=INK, fontsize=10)
ax1.annotate("ViT-CP ROM (flat)", (n_cp[-1], t_cp[-1] * 1e3), (-8, -16),
             textcoords="offset points", ha="right", color=INK, fontsize=10)
ax1.set_xlabel("mesh nodes  n = N²   (N = 32 … 512)")
ax1.set_ylabel(f"one online solve, {GN_ITERS} GN iterations  [ms]")
ax1.set_title(f"(a)  Online solve cost vs mesh size   (k = {K_FIX})",
              fontsize=11, color=INK, loc="left")
ax1.grid(True, which="major", color=GRID, lw=0.6)
ax1.text(0.02, 0.03,
         "XLA FLOPs per GN iteration: bit-identical across all N at fixed k\n"
         "(both ROM arms — n appears in no online tensor shape)",
         transform=ax1.transAxes, fontsize=8.5, color=INK2, va="bottom")

# ---------------- panel (b): cost vs k at N=512 ----------------
ax2.set_facecolor(SURFACE)
k_cp, tk_cp = series_vs_k(cp, N_FIX)
k_cn, tk_cn = series_vs_k(cn, N_FIX)
s_cp = np.polyfit(np.log(k_cp), np.log(tk_cp), 1)[0]
s_cn = np.polyfit(np.log(k_cn), np.log(tk_cn), 1)[0]

ax2.loglog(k_cn, tk_cn * 1e6, "-o", color=ORANGE, lw=2, ms=6, mec=SURFACE,
           mew=1)
ax2.loglog(k_cp, tk_cp * 1e6, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1)
ax2.set_ylim(top=tk_cn.max() * 1e6 * 1.6, bottom=tk_cp.min() * 1e6 * 0.75)
ax2.annotate(f"coord-net ROM  ~k^{s_cn:.2f}", (k_cn[2], tk_cn[2] * 1e6),
             (-6, 14), textcoords="offset points", ha="left", color=INK,
             fontsize=10)
ax2.annotate(f"ViT-CP ROM  ~k^{s_cp:.2f}", (k_cp[2], tk_cp[2] * 1e6),
             (2, -18), textcoords="offset points", ha="left", color=INK,
             fontsize=10)
ax2.set_xlabel("latent dimension  k")
ax2.set_ylabel("per GN iteration  [µs]")
ax2.set_title(f"(b)  Online cost vs latent dimension   (N = {N_FIX})",
              fontsize=11, color=INK, loc="left")
ax2.set_xticks([2, 4, 8, 16, 32])
ax2.set_xticklabels(["2", "4", "8", "16", "32"])
ax2.set_yticks([150, 200, 300, 500, 800])
ax2.set_yticklabels(["150", "200", "300", "500", "800"])
ax2.minorticks_off()
ax2.grid(True, which="major", color=GRID, lw=0.6)
ax2.text(0.02, 0.97,
         "shallow exponents: A100 latency-dominated at these sizes;\n"
         "FLOPs grow with k while N never enters",
         transform=ax2.transAxes, fontsize=8.5, color=INK2, va="top")

fig.suptitle("ROM online cost depends only on k — the mesh (n) is not in the "
             "formula", fontsize=12.5, color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"cost_scaling.{ext}"),
                facecolor=SURFACE, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
print("wrote cost_scaling.png / .pdf")

# console summary for the README
print("\nflatness check (max/min per-iter time across N, per k):")
for data, name in ((cp, "cp"), (cn, "coordnet")):
    for k in sorted({c["k"] for c in data["cells"]}):
        ts = [c["median_s_per_gn_iter"] for c in data["cells"] if c["k"] == k]
        print(f"  {name:9s} k={k:2d}: spread {max(ts)/min(ts)-1:6.2%}")
