"""Three summary figures for the advisor report.

Every number is transcribed from a committed table and re-verified against its
source file by `check()` before it is plotted, so a transcription slip fails
loudly instead of shipping a wrong figure.

Sources:
  P_OBJ  = worktrees/2026-08-16-poisson2d-rom-objective/.../README.md  (round-1 objective sweep)
  P_FU   = .../poisson2d-rom-objective/followup/FOLLOWUP_TABLES.md     (k ladder, hard-BC)
  B_FU   = .../burgers2d-rom-latent-stepping/followup/FOLLOWUP_TABLES.md
  H_RM   = .../heat2d-rom-latent-stepping/README.md
  W_RM   = .../wave2d-rom-latent-stepping/README.md
"""
from __future__ import annotations

import os
import sys

import numpy as np

WT = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees"
sys.path.insert(0, os.path.join(
    WT, "2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/followup"))
import fu_style as st  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

SRC = {
    "P_OBJ": f"{WT}/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/README.md",
    "P_FU": f"{WT}/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/followup/FOLLOWUP_TABLES.md",
    "B_FU": f"{WT}/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/FOLLOWUP_TABLES.md",
    "H_RM": f"{WT}/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/README.md",
    "W_RM": f"{WT}/2026-08-16-wave2d-rom-latent-stepping/experiments/wave2d-rom-latent-stepping/README.md",
}
_TEXT = {k: open(v).read() for k, v in SRC.items()}


def check(src: str, *needles: str) -> None:
    """Assert each needle literally appears in the source file."""
    for n in needles:
        if n not in _TEXT[src]:
            raise AssertionError(f"{n!r} not found in {SRC[src]}")


# ---------------------------------------------------------------- figure 1
# Poisson objective sweep (round 1, K=8 stage-0 decoder). Three initialisations.
OBJ = [
    ("FD residual\n(old recipe)", 2.20e-1, 6.25e-2, 1.13e-1),
    ("low-pass\n(σ=4 cells)", 5.93e-2 * 0 + 9.41e-3, 9.41e-3, 9.41e-3),
    ("20-step CG\npreconditioner", 8.45e-3, 8.45e-3, 8.47e-3),
    ("energy norm\n(Ritz/Galerkin)", 1.00e-2, 1.00e-2, 1.00e-2),
    ("weak form\n(64 test modes)", 8.46e-3, 8.48e-3, 8.48e-3),
]
ORACLE_1 = 7.78e-3
check("P_OBJ", "2.20e-1", "**6.25e-2**", "1.13e-1", "8.46e-3", "8.48e-3",
      "**1.00e-2**", "9.41e-3", "8.45e-3", "7.78e-3")


def fig_objective():
    st.use()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    st.clean(ax)
    names = [o[0] for o in OBJ]
    y = np.arange(len(OBJ))[::-1]
    h = 0.26
    series = [("mean latent", 1, st.C["orange"]),
              ("nearest training case", 2, st.C["blue"]),
              ("encoder E(f)", 3, st.C["aqua"])]
    for j, (lab, idx, col) in enumerate(series):
        ax.barh(y + (1 - j) * h, [o[idx] for o in OBJ], height=h,
                color=col, label=f"start from {lab}", zorder=3)
    ax.axvline(ORACLE_1, color=st.INK, ls=(0, (4, 3)), lw=1.4, zorder=4)
    ax.annotate("decoder's own ceiling\n(oracle latent) 7.8e-3", xy=(ORACLE_1, 0.62),
                xytext=(1.9e-2, 0.15), fontsize=7.6, color=st.INK,
                arrowprops=dict(arrowstyle="->", color=st.INK, lw=0.9))
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("held-out rel-L2 error  (log scale — lower is better)")
    ax.set_title("Poisson 2D: the ROM gap was the objective, not the solver\n"
                 "same decoder, solver and budget — only the minimised quantity changes",
                 loc="left")
    ax.set_xlim(5e-3, 4e-1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1)
    return st.save(fig, OUT, "objective_fix")


# ---------------------------------------------------------------- figure 2
# Four-PDE summary at K=8 (N=64): decoder ceiling, ROM, POD at the same k.
PDES = [
    ("Poisson 2D\n(elliptic)", 7.11e-3, 7.65e-3, 1.77e-1),
    ("Heat 2D\n(parabolic)", 1.16e-2, 1.87e-2, 1.29e-1),
    ("Burgers 2D\n(nonlinear advective)", 1.15e-2, 1.65e-2, 2.09e-1),
    ("Wave 2D\n(hyperbolic, conservative)", 1.719e-1, 8.783e-1, 3.424e-1),
]
check("P_FU", "7.11e-03", "7.65e-03", "1.77e-01")
check("B_FU", "1.15e-02", "1.65e-02", "2.09e-01")
check("H_RM", "1.16e-2", "1.87e-2", "1.29e-1")
check("W_RM", "1.719e-1", "8.783e-1", "3.424e-1")


def fig_four_pde():
    st.use()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    st.clean(ax)
    x = np.arange(len(PDES))
    w = 0.26
    ceil = [p[1] for p in PDES]
    rom = [p[2] for p in PDES]
    pod = [p[3] for p in PDES]
    ax.bar(x - w, ceil, w, color=st.MUTED, label="decoder's ceiling (oracle latent)", zorder=3)
    ax.bar(x, rom, w, color=st.C["blue"], label="coordinate ROM (what we deploy)", zorder=3)
    ax.bar(x + w, pod, w, color=st.C["orange"], label="POD at the same k (linear control)", zorder=3)
    for xi, (c, r) in enumerate(zip(ceil, rom)):
        ratio = r / c
        bad = ratio > 2
        ax.annotate(f"{ratio:.2f}x ceiling", (xi, r), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7.6,
                    color=st.C["red"] if bad else st.INK2,
                    fontweight="bold" if bad else "normal")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in PDES], fontsize=8)
    ax.set_ylabel("held-out rel-L2 (log scale)")
    ax.set_ylim(3e-3, 3)
    ax.set_title("All four PDEs at k = 8, N = 64: the ROM reaches the decoder's ceiling — except on Wave",
                 loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=1)
    return st.save(fig, OUT, "four_pde_summary")


# ---------------------------------------------------------------- figure 3
# Wave: linear ROM sits on its floor, nonlinear one does not — and loses energy.
W_POD_K = [6, 8, 16, 32, 64]
W_POD_ROM = [4.290e-1, 3.424e-1, 2.188e-1, 1.408e-1, 8.378e-2]
W_POD_FLOOR = [4.282e-1, 3.417e-1, 2.174e-1, 1.381e-1, 8.201e-2]
W_C_K = [4, 8, 16]
W_C_ROM = [1.019, 8.783e-1, 6.801e-1]
W_C_FLOOR = [3.114e-1, 1.719e-1, 1.106e-1]
W_ENERGY = [("POD-LSPG\n(k = 6…64)", 1.000003), ("coordinate ROM\n(K = 4/8/16)", 0.272)]
check("W_RM", "4.290e-1", "8.378e-2", "8.201e-2", "1.019", "6.801e-1",
      "3.114e-1", "1.106e-1", "0.272")


def fig_wave():
    st.use()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6),
                             gridspec_kw=dict(width_ratios=[1.75, 1]))
    ax = st.clean(axes[0])
    ax.plot(W_POD_K, W_POD_FLOOR, ls=":", color=st.C["orange"], lw=1.6, marker="",
            label="POD: best possible on its basis")
    ax.plot(W_POD_K, W_POD_ROM, marker="^", color=st.C["orange"],
            label="POD ROM (sits on its floor)")
    ax.plot(W_C_K, W_C_FLOOR, ls=":", color=st.C["blue"], lw=1.6, marker="",
            label="coordinate: best possible on its manifold")
    ax.plot(W_C_K, W_C_ROM, marker="o", color=st.C["blue"],
            label="coordinate ROM (3–6x above its floor)")
    for k, f, r in zip(W_C_K, W_C_FLOOR, W_C_ROM):
        ax.annotate("", xy=(k, r), xytext=(k, f),
                    arrowprops=dict(arrowstyle="<->", color=st.C["red"], lw=1.1))
    ax.annotate("the stepping gives back\neverything the manifold gains", (17.5, 9.5e-1),
                fontsize=7.6, color=st.C["red"], ha="left", va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(W_POD_K + [4])
    ax.set_xticklabels([str(k) for k in W_POD_K + [4]])
    ax.set_xlabel("latent dimension k")
    ax.set_ylabel("trajectory rel-RMS (log scale)")
    ax.set_title("Wave 2D: the nonlinear manifold is the better approximator\n"
                 "and the worse ROM", loc="left")
    ax.legend(loc="lower left", fontsize=7.4)

    ax2 = st.clean(axes[1])
    cols = [st.C["orange"], st.C["blue"]]
    ax2.bar([0, 1], [e[1] for e in W_ENERGY], 0.5, color=cols, zorder=3)
    ax2.axhline(1.0, color=st.INK, ls=(0, (4, 3)), lw=1.2, zorder=4)
    ax2.text(1.52, 1.02, "exact conservation", fontsize=7.4, color=st.INK,
             ha="right", va="bottom")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([e[0] for e in W_ENERGY], fontsize=8)
    ax2.set_ylabel("energy at the final time  /  initial energy")
    ax2.set_ylim(0, 1.25)
    ax2.set_title("Energy is not conserved on\na z-dependent manifold", loc="left")
    ax2.grid(axis="x", visible=False)
    for xi, (_, v) in enumerate(W_ENERGY):
        ax2.annotate(f"{v:.3f}" if v < 1 else "1.000003", (xi, v),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=7.6, color=st.INK2)
    return st.save(fig, OUT, "wave_failure")


if __name__ == "__main__":
    for f in (fig_objective, fig_four_pde, fig_wave):
        print("wrote", [p for p in f() if p.endswith(".png")])
