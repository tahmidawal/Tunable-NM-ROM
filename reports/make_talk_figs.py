"""Presentation figures for the 17 Aug 2026 results.

Deliberately separate from make_report_figs.py: these are sized and styled for
projection (large type, few elements, takeaway-as-title), not for a document.

Every number here is quoted in ROM-Cost-and-Accuracy-Findings.md. Figures whose
data is still provisional carry a visible PROVISIONAL stamp.
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

# Colourblind-safe (Okabe-Ito).
OURS = "#0072B2"     # coordinate ROM
POD = "#D55E00"      # POD
CEIL = "#009E73"     # decoder ceiling
FOM = "#555555"      # full-order solver
WARN = "#CC79A7"     # corrected values

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def stamp(ax, text="PROVISIONAL — final numbers pending the consolidation run"):
    """Below the axes, so it can never sit on top of a data line."""
    ax.text(1.0, -0.19, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#999999", style="italic")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------- 1. crossover
def fig_crossover():
    N = np.array([32, 64, 128])
    ours = np.array([0.40, 0.74, 1.42])
    pod = np.array([3.44, 4.82, 8.20])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhspan(1.0, 20, color="#e8f4ea", zorder=0)
    ax.axhline(1.0, color=FOM, lw=1.6, ls="--", zorder=2)

    ax.plot(N, ours, "o-", color=OURS, lw=3, ms=11, label="Coordinate ROM (ours)", zorder=4)
    ax.plot(N, pod, "s-", color=POD, lw=3, ms=10, label="POD", zorder=4)

    for x, y in zip(N, ours):
        ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=12.5, color=OURS, fontweight="bold")

    ax.text(33, 1.6, "faster than the full solver", fontsize=11.5, color="#2b7a43")
    ax.text(33, 0.62, "slower than the full solver", fontsize=11.5, color="#8a5a00")
    ax.annotate("we cross over\nhere", xy=(96, 1.0), xytext=(64, 0.42),
                fontsize=12, color=OURS, ha="center",
                arrowprops=dict(arrowstyle="->", color=OURS, lw=1.6))
    ax.text(128, 8.2, "POD is faster —\nbut 4× less accurate\n(see next slide)",
            fontsize=11, color=POD, ha="right", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xticks([], minor=True)
    ax.set_yticks([0.4, 1, 2, 4, 8]); ax.set_yticklabels(["0.4×", "1×", "2×", "4×", "8×"])
    ax.set_yticks([], minor=True)
    ax.set_ylim(0.33, 16)
    ax.set_xlabel("mesh resolution  N  (grid is N × N)")
    ax.set_ylabel("speed-up vs a full solver\nrun to the SAME accuracy")
    ax.set_title("Our advantage grows with mesh size — but only pays past N ≈ 100")
    ax.legend(loc="upper left", frameon=False)
    stamp(ax)
    save(fig, "1_crossover")


# ---------------------------------------------------------------- 2. accuracy
def fig_accuracy():
    pdes = ["Poisson", "Heat", "Burgers", "Wave"]
    ceil = [7.11e-3, 1.16e-2, 1.15e-2, 1.719e-1]
    ours = [7.65e-3, 1.87e-2, 1.65e-2, 8.783e-1]
    pod = [1.77e-1, 1.29e-1, 2.09e-1, 3.424e-1]

    x = np.arange(len(pdes)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(x - w, ceil, w, color=CEIL, label="best possible (decoder ceiling)")
    ax.bar(x, ours, w, color=OURS, label="coordinate ROM (ours)")
    ax.bar(x + w, pod, w, color=POD, label="POD, same k")

    for xi, (o, p) in enumerate(zip(ours, pod)):
        if p > o:
            ax.annotate(f"{p/o:.0f}× better", (xi + w, p), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=12.5,
                        color=OURS, fontweight="bold")

    ax.annotate("Wave fails —\nand we say so", xy=(3, 8.783e-1), xytext=(3.0, 3.0),
                fontsize=11.5, color="#a33", ha="center",
                arrowprops=dict(arrowstyle="->", color="#a33", lw=1.5))

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(pdes)
    ax.set_ylim(3e-3, 12)
    ax.set_ylabel("held-out error  (relative L2, lower is better)")
    ax.set_title("At the same latent dimension (k = 8) we are 7–23× more accurate than POD")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    save(fig, "2_accuracy")


# ---------------------------------------------------------------- 3. k ladder
def fig_kladder():
    k = np.array([2, 4, 6, 8, 12, 16, 24, 32])
    ceil = np.array([1.236e-1, 1.551e-2, 8.835e-3, 7.043e-3,
                     6.236e-3, 4.133e-3, 3.976e-3, 3.280e-3])
    mean = np.array([5.455e-1, 1.742e-2, 5.845e-2, 8.482e-3,
                     4.789e-2, 6.542e-3, 1.491e-2, 4.022e-2])
    kmed = np.array([4, 6, 8, 12, 16, 24, 32])
    med = np.array([1.55e-2, 8.52e-3, 7.25e-3, 7.58e-3, 5.05e-3, 4.93e-3, 3.66e-3])
    blown = {6: 2, 12: 3, 24: 1, 32: 5}

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(k, ceil, "o--", color=CEIL, lw=2.5, ms=9,
            label="best the decoder can do")
    ax.plot(k, mean, "o-", color="#bbbbbb", lw=2.4, ms=9,
            label="our error, mean of 16 cases")
    ax.plot(kmed, med, "o-", color=OURS, lw=3, ms=10,
            label="our error, median of 16 cases")

    for kk, n in blown.items():
        yi = mean[list(k).index(kk)]
        ax.annotate(f"{n} of 16\ndiverged", (kk, yi), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5, color="#a33")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(k); ax.set_xticklabels([str(v) for v in k])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("latent dimension  k")
    ax.set_ylabel("held-out error  (relative L2)")
    ax.set_title("The spikes are a few diverging cases, not a failure at particular k")
    ax.legend(loc="lower left", frameon=False)
    ax.text(0.985, 0.96,
            "The median tracks the ceiling at every k.\n"
            "A step-size limit removes the divergences\n"
            "and costs nothing.",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#666")
    save(fig, "3_k_ladder")


# ---------------------------------------------------------------- 4. correction
def fig_correction():
    N = ["32", "64", "128", "256"]
    old = np.array([0.72, 1.57, 4.46, 7.96])
    new = np.array([0.19, 0.36, 0.93, 1.83])

    x = np.arange(len(N)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, color=FOM, ls="--", lw=1.6)
    ax.bar(x - w / 2, old, w, color="#bbbbbb", label="what we published")
    ax.bar(x + w / 2, new, w, color=WARN, label="corrected (fair baseline)")

    for xi, (o, n) in enumerate(zip(old, new)):
        ax.text(xi - w / 2, o + 0.22, f"{o:.2f}×", ha="center", fontsize=11.5, color="#777")
        ax.text(xi + w / 2, n + 0.22, f"{n:.2f}×", ha="center", fontsize=11.5,
                color=WARN, fontweight="bold")

    ax.text(-0.62, 1.22, "break-even", fontsize=11, color=FOM, va="bottom")
    ax.annotate("N = 128 goes from\nclearly winning\nto break-even",
                xy=(2 + w / 2, 1.05), xytext=(0.72, 5.6), fontsize=11.5,
                color=WARN, ha="center",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.5,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_xticks(x); ax.set_xticklabels(N)
    ax.set_xlabel("mesh resolution  N")
    ax.set_ylabel("end-to-end speed-up vs the full solver")
    ax.set_ylim(0, 9.2)
    ax.set_title("Our old Burgers baseline did 4× more work than needed — so did our speed-ups")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "4_correction")


# ---------------------------------------------------------------- 5. warm start
def fig_warmstart():
    labels = ["previous time step\n(the standard trick)",
              "our ROM solution",
              "linear extrapolation\n(one line of code)"]
    iters = [770, 974, 442]
    cols = [FOM, OURS, CEIL]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(labels, iters, color=cols, width=0.55)
    for b, v in zip(bars, iters):
        ax.text(b.get_x() + b.get_width() / 2, v + 22, f"{v}",
                ha="center", fontsize=14, fontweight="bold",
                color=b.get_facecolor())

    ax.annotate("worse than doing nothing clever", xy=(1.28, 985), xytext=(1.62, 1140),
                fontsize=12, color="#a33", ha="center",
                arrowprops=dict(arrowstyle="->", color="#a33", lw=1.5))

    ax.set_ylabel("linear solver iterations to finish  (lower is better)")
    ax.set_ylim(0, 1250)
    ax.set_title("Starting the full solver from our ROM makes it SLOWER")
    ax.text(0.5, -0.235, "Burgers, N = 256, tolerance 1e-6.  The hybrid loses in all 12 "
            "configurations tested.",
            transform=ax.transAxes, ha="center", fontsize=11.5, color="#666")
    save(fig, "5_warm_start")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    fig_crossover()
    fig_accuracy()
    fig_kladder()
    fig_correction()
    fig_warmstart()
    print("done")
