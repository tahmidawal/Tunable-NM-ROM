"""Figures for the cost-to-tolerance surface.

Every number is read from `runs/pareto_points.json` (written by ctol_tables.py
from the surface JSONs) -- nothing is transcribed by hand.  House style is the
project's `followup/fu_style.py` (light mode, validated categorical palette,
identity never carried by colour alone).

Figures
  1  ctol_cost_vs_k_<pde>       latent-solve cost vs k, one line per mesh N, one
                                panel per tau, plus a panel normalised to k=8 --
                                the direct test of whether the k dependence is
                                N-independent.
  2  ctol_iters_vs_k_<pde>      Jacobian evaluations to reach tau vs k.
  3  ctol_pareto_<pde>          the ISO-ERROR PARETO, one panel per N: faint
                                scatter of every configuration, solid
                                non-dominated envelope per method, dots labelled
                                with k, the FOM as a VERTICAL DASHED LINE (it is
                                the reference truth, so its error is 0 and it is
                                off-plot; the line reads "the price of exactness").
  4  ctol_scaling_<pde>         cheapest online time reaching a fixed error
                                target vs N, one line per method, the FOM for
                                comparison.  Mesh independence appears as a flat
                                line while the FOM's rises.  Timings come from
                                the SINGLE-GPU consolidation run when it exists.

Usage: python ctol_figs.py [--runs runs] [--out figs] [--extra <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(HERE, "deps", "poisson2d-rom-objective", "followup"),
           os.path.abspath(os.path.join(HERE, "..", "poisson2d-rom-objective", "followup"))):
    if os.path.isfile(os.path.join(_c, "fu_style.py")):
        sys.path.insert(0, _c)
        break
sys.path.insert(0, HERE)
import fu_style as st                                    # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402
import ctol_tables as T                                  # noqa: E402

PDE_LABEL = {"poisson2d": "Poisson 2D", "burgers2d": "Burgers 2D"}
N_COLOR = [st.C["blue"], st.C["orange"], st.C["aqua"], st.C["yellow"], st.C["violet"]]
N_MARK = ["o", "s", "^", "D", "v"]
M_COLOR = {"coord": st.C["blue"], "pod": st.C["orange"]}
M_LABEL = {"coord": "coordinate (INR)", "pod": "POD"}


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def sel(pts, **kw):
    out = pts
    for k, v in kw.items():
        out = [p for p in out if p.get(k) == v]
    return out


# --------------------------------------------------------------------------
def fig_cost_vs_k(pts, pde, taus, ns, ks, out, extra):
    prim = [p for p in pts if p["pde"] == pde and p.get("arm") == "primary"]
    if not prim:
        return None
    fig, axes = plt.subplots(2, len(taus), figsize=(3.5 * len(taus), 6.0), sharex=True,
                             squeeze=False)
    for j, tau in enumerate(taus):
        for row, norm in enumerate((False, True)):
            ax = st.clean(axes[row, j])
            for i, N in enumerate(ns):
                for meth, ls in (("coord", "-"), ("pod", "--")):
                    d = {p["k"]: p for p in sel(prim, N=N, tau=tau, method=meth)}
                    xs = [k for k in ks if k in d and finite(d[k].get("time_ms_solve"))]
                    if not xs:
                        continue
                    ys = np.array([d[k]["time_ms_solve"] for k in xs], float)
                    if norm:
                        if 8 not in d or not finite(d[8].get("time_ms_solve")):
                            continue
                        ys = ys / d[8]["time_ms_solve"]
                    ax.plot(xs, ys, ls, color=N_COLOR[i % len(N_COLOR)],
                            marker=N_MARK[i % len(N_MARK)], ms=4, lw=1.6,
                            alpha=1.0 if meth == "coord" else 0.75)
            ax.set_xscale("log", base=2); ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"$\\tau$ = {tau:.0e}")
                ax.set_ylabel("latent solve, ms" if j == 0 else "")
            else:
                ax.set_ylabel("cost / cost at $k$=8" if j == 0 else "")
                ax.axhline(1.0, color=st.MUTED, lw=0.8, ls=":")
                ax.set_xlabel("latent dimension $k$")
    handles = [plt.Line2D([], [], color=N_COLOR[i % len(N_COLOR)],
                          marker=N_MARK[i % len(N_MARK)], ms=4, lw=1.6, label=f"N = {N}")
               for i, N in enumerate(ns)]
    handles += [plt.Line2D([], [], color=st.INK2, ls="-", lw=1.6, label="coordinate (INR)"),
                plt.Line2D([], [], color=st.INK2, ls="--", lw=1.6, label="POD")]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)),
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(f"{PDE_LABEL[pde]}: cost of the latent solve vs $k$, overlaid across mesh"
                 f"\nbottom row normalised at $k$=8 -- curves collapsing means the $k$ "
                 f"dependence is mesh independent", y=1.02, color=st.INK)
    fig.tight_layout()
    return st.save(fig, out, f"ctol_cost_vs_k_{pde}", extra)


def fig_iters_vs_k(pts, pde, taus, ns, ks, out, extra):
    prim = [p for p in pts if p["pde"] == pde and p.get("arm") == "primary"]
    if not prim:
        return None
    fig, axes = plt.subplots(1, len(taus), figsize=(3.5 * len(taus), 3.2), sharey=True,
                             squeeze=False)
    for j, tau in enumerate(taus):
        ax = st.clean(axes[0][j])
        for i, N in enumerate(ns):
            for meth, ls in (("coord", "-"), ("pod", "--")):
                d = {p["k"]: p for p in sel(prim, N=N, tau=tau, method=meth)}
                xs = [k for k in ks if k in d and finite(d[k].get("jac_evals"))]
                if not xs:
                    continue
                ax.plot(xs, [d[k]["jac_evals"] for k in xs], ls,
                        color=N_COLOR[i % len(N_COLOR)], marker=N_MARK[i % len(N_MARK)],
                        ms=4, lw=1.6, alpha=1.0 if meth == "coord" else 0.75)
        ax.set_xscale("log", base=2); ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks]); ax.set_yscale("log")
        ax.set_title(f"$\\tau$ = {tau:.0e}")
        ax.set_xlabel("latent dimension $k$")
        if j == 0:
            ax.set_ylabel("Jacobian evaluations to $\\tau$")
    handles = [plt.Line2D([], [], color=N_COLOR[i % len(N_COLOR)],
                          marker=N_MARK[i % len(N_MARK)], ms=4, lw=1.6, label=f"N = {N}")
               for i, N in enumerate(ns)]
    handles += [plt.Line2D([], [], color=st.INK2, ls="-", lw=1.6, label="coordinate (INR)"),
                plt.Line2D([], [], color=st.INK2, ls="--", lw=1.6, label="POD")]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)),
               bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(f"{PDE_LABEL[pde]}: work to reach the tolerance vs $k$ "
                 f"(censored cells are plotted at the work they spent)", y=1.06, color=st.INK)
    fig.tight_layout()
    return st.save(fig, out, f"ctol_iters_vs_k_{pde}", extra)


def fig_pareto(pts, pde, ns, out, extra):
    prim = [p for p in pts if p["pde"] == pde and p.get("arm") in ("primary", "supp_m4M")]
    if not prim:
        return None
    ncol = min(3, len(ns))
    nrow = int(math.ceil(len(ns) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.5 * nrow), squeeze=False)
    for a in axes.ravel():
        a.set_visible(False)
    for idx, N in enumerate(ns):
        ax = st.clean(axes[idx // ncol][idx % ncol]); ax.set_visible(True)
        here = [p for p in prim if p["N"] == N]
        for p in here:
            if not (finite(p.get("time_ms")) and finite(p.get("err_rel_l2"))):
                continue
            cens = bool(p.get("censored") or p.get("censored_frac"))
            ax.plot(p["time_ms"], p["err_rel_l2"], "o" if not cens else "x", ms=3,
                    color=M_COLOR[p["method"]], alpha=0.22 if not cens else 0.35,
                    mec="none" if not cens else M_COLOR[p["method"]], mew=0.8)
        for meth in ("coord", "pod"):
            sel_m = [p for p in here if p["method"] == meth]
            # dashed = the AS-DEPLOYED frontier (censored cells allowed: set the knob
            # and take whatever the solver reaches); solid = the STRICT frontier
            # (only cells that actually reached their own tau)
            fd_ = T.nondominated(sel_m, require_uncensored=False)
            if fd_:
                ax.plot([p["time_ms"] for p in fd_], [p["err_rel_l2"] for p in fd_], "--",
                        color=M_COLOR[meth], lw=1.1, alpha=0.65, zorder=2)
            fr = T.nondominated(sel_m)
            if not fr:
                continue
            ax.plot([p["time_ms"] for p in fr], [p["err_rel_l2"] for p in fr], "-",
                    color=M_COLOR[meth], lw=1.8, marker="o", ms=5, zorder=3,
                    label=M_LABEL[meth])
            for p in fr:
                ax.annotate(f"{p['k']}", (p["time_ms"], p["err_rel_l2"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=6.5,
                            color=M_COLOR[meth])
        fom = next((p for p in pts if p["pde"] == pde and p["method"] == "fom"
                    and p["N"] == N and p.get("arm") == "fom"), None)
        if fom:
            ax.axvline(fom["time_ms"], color=st.C["red"], ls="--", lw=1.4, zorder=2)
            ax.text(fom["time_ms"], 0.97, "  FOM (exact)", transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="left", fontsize=7, color=st.C["red"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"N = {N}")
        ax.set_xlabel("online wall time, ms")
        if idx % ncol == 0:
            ax.set_ylabel("held-out rel-$L_2$")
        if idx == 0:
            ax.legend(loc="upper right")
    fig.suptitle(f"{PDE_LABEL[pde]}: iso-error Pareto frontier "
                 f"(all $k$ x $\\tau$ configurations; labels are $k$; the FOM is exact, so it "
                 f"is a vertical line -- the price of exactness)"
                 f"\nsolid = cells that reached their own $\\tau$;  dashed = as-deployed "
                 f"(censored cells, crosses, included)", y=1.04, color=st.INK)
    fig.tight_layout()
    return st.save(fig, out, f"ctol_pareto_{pde}", extra)


def fig_scaling(pts, pde, ns, out, extra):
    prim = [p for p in pts if p["pde"] == pde and p.get("arm") in ("primary", "supp_m4M")]
    if not prim:
        return None
    cons = {(p["method"], p["N"], p["k"], p["M"], p["m"], p["tau"]): p
            for p in pts if p["pde"] == pde and p.get("arm") == "consolidated"}
    cons_fom = {p["N"]: p for p in pts if p["pde"] == pde
                and p.get("arm") == "fom_consolidated"}
    targets = T.TARGETS[pde]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.3 * len(targets), 3.6),
                             squeeze=False)
    used_panel_times = False
    for j, target in enumerate(targets):
        ax = st.clean(axes[0][j])
        for meth in ("coord", "pod"):
            xs, ys, lab = [], [], []
            for N in ns:
                b = T.cheapest_reaching([p for p in prim if p["N"] == N
                                         and p["method"] == meth], target)
                if b is None:
                    continue
                key = (meth, N, b["k"], b["M"], b["m"], b["tau"])
                if key in cons:
                    t = cons[key]["time_ms"]
                else:
                    t = b["time_ms"]
                    used_panel_times = True
                xs.append(N); ys.append(t); lab.append(b["k"])
            if xs:
                ax.plot(xs, ys, "-o", color=M_COLOR[meth], lw=2.0, ms=5,
                        label=M_LABEL[meth])
                for x, y, k in zip(xs, ys, lab):
                    ax.annotate(f"k={k}", (x, y), textcoords="offset points",
                                xytext=(4, 5), fontsize=6.5, color=M_COLOR[meth])
        fx, fy = [], []
        for N in ns:
            f = cons_fom.get(N) or next((p for p in pts if p["pde"] == pde
                                         and p["method"] == "fom" and p["N"] == N
                                         and p.get("arm") == "fom"), None)
            if f:
                fx.append(N); fy.append(f["time_ms"])
        if fx:
            ax.plot(fx, fy, "-s", color=st.C["red"], lw=2.0, ms=5, label="FOM (exact)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("mesh $N$ (per side)")
        if j == 0:
            ax.set_ylabel("cheapest online time, ms")
            ax.legend(loc="upper left")
        ax.set_title(f"error target {target:.0e}")
    note = ("timings from the single-GPU consolidation run"
            if not used_panel_times else
            "WARNING: some timings fall back to panel jobs on different GPUs")
    fig.suptitle(f"{PDE_LABEL[pde]}: cheapest online time reaching a fixed accuracy, vs mesh"
                 f"\nflat = mesh independent; the FOM rises with the mesh   ({note})",
                 y=1.05, color=st.INK)
    fig.tight_layout()
    return st.save(fig, out, f"ctol_scaling_{pde}", extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--extra", default="/home/tahmid/Dev/pod-ae-nmrom/Plots")
    args = ap.parse_args()
    st.use()
    pts = json.load(open(os.path.join(args.runs, "pareto_points.json")))
    extra = (args.extra,) if args.extra else ()
    made = []
    for pde in ("poisson2d", "burgers2d"):
        here = [p for p in pts if p["pde"] == pde]
        if not here:
            continue
        ns = sorted({p["N"] for p in here})
        ks = sorted({p["k"] for p in here if p.get("k")})
        taus = sorted({p["tau"] for p in here if p.get("tau")}, reverse=True)
        for f in (fig_cost_vs_k(pts, pde, taus, ns, ks, args.out, extra),
                  fig_iters_vs_k(pts, pde, taus, ns, ks, args.out, extra),
                  fig_pareto(pts, pde, ns, args.out, extra),
                  fig_scaling(pts, pde, ns, args.out, extra)):
            if f:
                made += f
    for p in made:
        print("  wrote " + p)


if __name__ == "__main__":
    main()
