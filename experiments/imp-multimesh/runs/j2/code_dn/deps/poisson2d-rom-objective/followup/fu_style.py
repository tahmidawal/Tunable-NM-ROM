"""Shared figure style for the follow-up figures (paper figures: light mode only,
deliberately committed to one look).  Colours are slots 1/2/3/4/8 of the
validated categorical palette (blue / orange / aqua / yellow / red); identity is
never carried by colour alone -- every series is also direct-labelled or
distinguished by marker and dash pattern."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948")
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e2df"
SURFACE = "#ffffff"

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
    "lines.linewidth": 2.0, "lines.markersize": 5.5,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def use():
    plt.rcParams.update(RC)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def save(fig, outdir, name, extra_dirs=()):
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for d in (outdir,) + tuple(extra_dirs):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            p = os.path.join(d, f"{name}.{ext}")
            fig.savefig(p)
            paths.append(p)
    plt.close(fig)
    return paths
