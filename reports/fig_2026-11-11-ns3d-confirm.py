"""Figures for reports/2026-11-11-ns3d-confirm.md.

Reads reports/2026-11-11-ns3d-confirm.json (written by build_2026-11-11-ns3d-confirm.py) and
the small slices.npz of each run.  No number is typed here.

  python reports/fig_2026-11-11-ns3d-confirm.py <cohort>     # cohort = test or dev
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
RUNS = ROOT / "experiments" / "ns3d-pod-match" / "runs"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_COORD, C_POD, C_FRONT, C_SAME = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def pareto(d, cohort):
    meshes = sorted(d[cohort], key=int)
    fig, axes = plt.subplots(1, len(meshes), figsize=(4.3 * len(meshes), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, meshes):
        r = d[cohort][m]
        # the reference configuration itself (error 0 by definition) is left off the log axis
        fr = sorted((c for c in r["coarse_front"] if c["worst"] > 1e-6), key=lambda c: c["ms"])
        ax.step([c["ms"] for c in fr], [100 * c["worst"] for c in fr], where="post", color=C_FRONT,
                lw=2, marker="o", ms=4, mec=SURF, mew=0.8, label="classical solver, any mesh/step (front)",
                zorder=2)
        cn = [v for v in r["cnab2"].values() if not v.get("unstable") and 1e-6 < v["worst"] < 1]
        ax.plot([v["median_ms"] for v in cn], [100 * v["worst"] for v in cn], ls="none", marker="s",
                ms=6, color=C_SAME, mec=SURF, mew=0.8, label="CNAB2 on the ROM's mesh", zorder=3)
        co = {a: v for a, v in r["arms"].items() if a.startswith("coordnet") and v["finite"]}
        po = {a: v for a, v in r["arms"].items() if a.startswith("pod_head") and v["finite"]}
        ax.plot([v["median_ms"] for v in co.values()], [100 * v["worst"] for v in co.values()],
                ls="none", marker="D", ms=7, color=C_COORD, mec=SURF, mew=0.8,
                label="ROM, coordinate-network bank", zorder=4)
        ax.plot([v["median_ms"] for v in po.values()], [100 * v["worst"] for v in po.values()],
                ls="none", marker="^", ms=8, color=C_POD, mec=SURF, mew=0.8,
                label="ROM, POD bank + head", zorder=4)
        for a, v in co.items():
            if a.endswith("head_k8") or a in ("coordnet_span8", "coordnet_span64"):
                lab = "head k=8" if a.endswith("head_k8") else a.replace("coordnet_span", "span ")
                ax.annotate(lab, (v["median_ms"], 100 * v["worst"]), xytext=(6, 4),
                            textcoords="offset points", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([t for t in (2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000)
                       if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{m}$^3$ truth mesh   ({r['cases']} cases, {cohort} cohort)", loc="left",
                     fontsize=10, color=INK)
        ax.set_xlabel("median time per query, ms (same GPU, warm)")
    axes[0].set_ylabel("worst-case error over cases and frames, %")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("3D Navier-Stokes: ROM arms against classical solvers free to choose their mesh "
                 "(lower-left is better)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    out = REP / f"2026-11-11-ns3d-confirm-pareto-{cohort}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def slices(run, tag):
    f = RUNS / run / "output" / "mesh" / "slices.npz"
    if not f.exists():
        return None
    z = np.load(f)
    cases = sorted({k.rsplit("_c", 1)[1] for k in z.files}, key=int)
    c = cases[-1] if len(cases) > 1 else cases[0]
    plane = lambda a: np.asarray(a)[0]          # stored as (component, x, y) on the z = 0 plane
    truth = plane(z[f"truth_c{c}"])
    names = [k[: -len(f"_c{c}")] for k in z.files if k.endswith(f"_c{c}") and not k.startswith("truth")]
    order = [n for n in ("coordnet_head_k8", "pod_head_k8", "interp_head_k8", "coordnet_span32",
                         "coordnet_span8") if n in names]
    coarse = [n for n in names if n.startswith("COARSE_")]     # the two most accurate matched solvers
    order += sorted(coarse, key=lambda n: float(np.abs(plane(z[f"{n}_c{c}"]) - truth).max()))[:2][::-1]
    fig, axes = plt.subplots(2, len(order) + 1, figsize=(2.35 * (len(order) + 1), 5.0))
    vm = float(np.abs(truth).max())
    axes[0, 0].imshow(truth.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
    axes[0, 0].set_title("reference", fontsize=8.5)
    axes[1, 0].axis("off")
    errs = [np.abs(plane(z[f"{n}_c{c}"]) - truth) for n in order]
    em = max(float(e.max()) for e in errs)
    im_f = im_e = None
    for j, (n, e) in enumerate(zip(order, errs), start=1):
        im_f = axes[0, j].imshow(plane(z[f"{n}_c{c}"]).T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        axes[0, j].set_title(n.replace("COARSE_", "solver ").replace("_", " "), fontsize=8.5)
        im_e = axes[1, j].imshow(e.T, origin="lower", cmap="Blues", vmin=0, vmax=em)
        axes[1, j].set_title(f"|error|, max {e.max():.2e}", fontsize=8, color=INK2)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im_f, ax=axes[0, :], fraction=0.012, pad=0.01)
    fig.colorbar(im_e, ax=axes[1, :], fraction=0.012, pad=0.01)
    fig.suptitle(f"{tag}: x-velocity, final frame, z = 0 plane, case {c} "
                 f"(worst case of the coordinate-network head)", x=0.01, ha="left", fontsize=10)
    out = REP / f"2026-11-11-ns3d-confirm-slices-{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    d = json.load(open(REP / "2026-11-11-ns3d-confirm.json"))
    for cohort in sys.argv[1:] or ["test"]:
        if cohort in d and d[cohort]:
            print(pareto(d, cohort))
            for m, r in d[cohort].items():
                print(slices(r["run"], f"{cohort}{m}"))
