"""Error and cost against the number of empirical-quadrature points.

Both ladders are at N=64, k=8, M=64 test modes, one GPU, median of 7.
Poisson from poisson2d-rom-objective/followup (FOLLOWUP_TABLES.md m-ladder);
Burgers from burgers2d-rom-latent-stepping/runs (blat_rom_N64_K8.json).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "talk_figs")
os.makedirs(OUT, exist_ok=True)

NINT = 62 * 62          # interior nodes of a 64x64 grid

# m, error, cost, NNLS relative fit residual
POISSON = dict(
    m=[64, 128, 256, 512, 1024, NINT],
    err=[5.48e-2, 1.80e-2, 8.66e-3, 7.68e-3, 7.65e-3, 7.65e-3],
    cost=[11.1, 13.9, 15.1, 15.2, 18.9, 35.6],          # ms per solve
    fit=[1.4e-1, 1.6e-2, 2.4e-3, 2.4e-4, 1.6e-5, 0.0],
    ceiling=7.11e-3, unit="ms per solve")
BURGERS = dict(
    m=[64, 128, 256, 512, 1024, NINT],
    err=[6.540e-2, 1.946e-2, 1.742e-2, 1.682e-2, 1.665e-2, 1.654e-2],
    cost=[3.352, 4.498, 6.248, 10.772, 20.669, 70.946],  # ms per time step
    fit=[2.09e-1, 4.89e-2, 6.21e-3, 1.04e-3, 1.45e-4, 0.0],
    ceiling=1.15e-2, unit="ms per step")

ERRC, COSTC, FLOOR = "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "axes.spines.top": False, "axes.grid": True, "grid.alpha": 0.22,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def panel(ax, d, tag):
    """Both curves normalised to the full-grid value, so one axis serves both."""
    m = np.array(d["m"], float)
    err = np.array(d["err"]) / d["err"][-1]
    cost = np.array(d["cost"]) / d["cost"][-1]

    ax.axhline(1.0, color="#bbb", lw=1.4, ls="-")
    ax.plot(m, err, "o-", color=ERRC, lw=2.6, ms=9, label="error")
    ax.plot(m, cost, "s-", color=COSTC, lw=2.4, ms=8, label="cost")
    ax.plot(m[-1], 1.0, "o", color="#666", ms=13, mfc="white", mew=2.2, zorder=5)

    knee = next(mm for mm, e in zip(d["m"], err) if e <= 1.05)
    ki = d["m"].index(knee)
    ax.annotate(f"m = {knee}  ({100*knee/NINT:.0f}% of the grid)\n"
                f"{100*(err[ki]-1):.0f}% more error, {100*cost[ki]:.0f}% of the cost",
                xy=(knee, cost[ki]), xytext=(0.5, 0.72), textcoords="axes fraction",
                fontsize=10.5, color="#444", ha="center",
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.3))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("quadrature points  m")
    ax.set_ylabel("relative to using every grid point")
    ax.set_title(tag)
    ax.text(m[-1], 1.28, "every\npoint", ha="center", fontsize=10, color="#666")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.26)
panel(axes[0], POISSON, "Poisson  (cost = one solve)")
panel(axes[1], BURGERS, "Burgers  (cost = one time step)")
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("Empirical quadrature: what you give up by not using every grid point",
             fontsize=15, y=1.02)
fig.text(0.5, -0.03, "N = 64, k = 8, M = 64 test modes, one GPU. "
         "Both curves are relative to evaluating the residual at all 3 844 interior nodes.",
         ha="center", fontsize=10, color="#888")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"9_eq_points.{ext}"), bbox_inches="tight")
print("wrote 9_eq_points.png / .pdf")

rows = ["| points m | share of grid | error | cost | quadrature fit |",
        "|---|---|---|---|---|"]
for tag, d in (("**Poisson**", POISSON), ("**Burgers**", BURGERS)):
    rows.append(f"| {tag} | | | | |")
    for mm, e, c, f in zip(d["m"], d["err"], d["cost"], d["fit"]):
        lbl = "every point" if mm == NINT else str(mm)
        rows.append(f"| {lbl} | {100*mm/NINT:.0f}% | {e:.2e} | {c:.1f} | "
                    f"{'exact' if f == 0 else f'{f:.1e}'} |")
open(os.path.join(HERE, "2026-08-18-quadrature-points-table.md"), "w").write("\n".join(rows) + "\n")
print("wrote eq_table.md")
