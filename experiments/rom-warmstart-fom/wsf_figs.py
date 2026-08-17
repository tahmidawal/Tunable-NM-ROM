"""Figures for the ROM-warm-started-FOM cell, in the reports-pipeline style
(`wsf_style.py`, a copy of the frozen `fu_style.py`).  PNG + PDF, written to
`figs/` and copied to /home/tahmid/Dev/pod-ae-nmrom/Plots/.

Every figure is drawn from `runs/hybrid_points.json` (written by wsf_summarize.py)
and the raw reports -- never from a number typed into this file.

Usage: python wsf_figs.py [runs_dir] [extra_outdir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wsf_style as st                                    # noqa: E402
from wsf_summarize import select_consolidated             # noqa: E402

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
FIGS = os.path.join(HERE, "figs")
EXTRA = sys.argv[2] if len(sys.argv) > 2 else "/home/tahmid/Dev/pod-ae-nmrom/Plots"
NCOL = [st.C["blue"], st.C["orange"], st.C["aqua"], st.C["yellow"], st.C["red"],
        st.C["violet"], st.C["magenta"]]
ARMC = {"prev": st.C["blue"], "extrap": st.C["yellow"], "rom": st.C["orange"]}
ARML = {"prev": "FOM warm start (u$_{n-1}$)", "extrap": "linear extrapolation",
        "rom": "ROM state (hybrid)"}


def load():
    p = os.path.join(RUNS, "hybrid_points.json")
    pts = json.load(open(p)) if os.path.isfile(p) else []
    reps = []
    for q in sorted(glob.glob(os.path.join(RUNS, "**", "*.json"), recursive=True)):
        if os.path.basename(q) == "hybrid_points.json":
            continue
        try:
            d = json.load(open(q))
        except Exception:
            continue
        if isinstance(d, dict) and "rows" in d:
            reps.append(d)
    return pts, reps


def taulab(t):
    return "conv." if not t else f"{t:g}"


# --------------------------------------------------------------- Poisson headline
def fig_poisson_total_vs_tau(P, paths):
    if not P:
        return
    fts = sorted({r["fom_tau"] for r in P}, reverse=True)
    Ns = sorted({r["N"] for r in P})
    rts = sorted({r["rom_tau"] for r in P}, reverse=True)
    fig, axes = plt.subplots(1, len(fts), figsize=(3.7 * len(fts), 3.6),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    x = np.arange(len(rts))
    for ax, ft in zip(axes, fts):
        for ci, n in enumerate(Ns):
            c = NCOL[ci % len(NCOL)]
            sub = {r["rom_tau"]: r for r in P if r["N"] == n and r["fom_tau"] == ft}
            y = [sub[t]["t_total_ms"] if t in sub else np.nan for t in rts]
            ax.plot(x, y, "-o", color=c, label=f"N={n}")
            if sub:
                b = list(sub.values())[0]["t_fom_baseline_ms"]
                ax.plot([x[0] - 0.35, x[-1] + 0.35], [b, b], "--", color=c,
                        lw=1.2, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([taulab(t) for t in rts])
        ax.set_yscale("log")
        ax.set_xlabel("ROM tolerance  $V(z)\\leq\\tau_{ROM}V(z_0)$")
        ax.set_title(f"$\\tau_{{FOM}}$ = {ft:g}")
        st.clean(ax)
    axes[0].set_ylabel("total time (ms)")
    axes[0].legend(ncol=2, loc="upper left", fontsize=7.5)
    axes[-1].text(0.98, 0.03, "dashed = pure FOM (CG from zero)", transform=axes[-1].transAxes,
                  ha="right", va="bottom", fontsize=7.5, color=st.MUTED)
    fig.suptitle("Poisson-2D: hybrid total cost vs the ROM's own stopping tolerance",
                 fontsize=10, color=st.INK)
    paths += st.save(fig, FIGS, "wsfom_poisson_total_vs_tau", (EXTRA,))


def fig_poisson_crossover(P, paths):
    if not P:
        return
    fts = sorted({r["fom_tau"] for r in P}, reverse=True)
    ftm = min(fts)
    Ns = sorted({r["N"] for r in P if r["fom_tau"] == ftm})
    if not Ns:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
    fom = [np.mean([r["t_fom_baseline_ms"] for r in P if r["N"] == n and r["fom_tau"] == ftm])
           for n in Ns]
    best = [min([r["t_total_ms"] for r in P if r["N"] == n and r["fom_tau"] == ftm])
            for n in Ns]
    romonly = [min([r["t_rom_ms"] + r.get("t_pre_ms", 0.0) + r["t_decode_ms"]
                    for r in P if r["N"] == n and r["fom_tau"] == ftm]) for n in Ns]
    direct = [np.mean([r.get("t_fom_direct_ms", np.nan) for r in P
                       if r["N"] == n and r["fom_tau"] == ftm]) for n in Ns]
    a1.plot(Ns, fom, "-o", color=st.C["blue"], label="pure FOM (CG, zero start)")
    a1.plot(Ns, best, "-s", color=st.C["orange"], label="best hybrid (ROM + CG)")
    a1.plot(Ns, romonly, "--^", color=st.C["violet"],
            label="ROM stage alone (solve + decode)")
    if np.isfinite(direct).any():
        a1.plot(Ns, direct, ":d", color=st.C["green"],
                label="direct solve (exact, sine diagonalisation)")
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xlabel("N"); a1.set_ylabel("time (ms)")
    a1.set_title(f"Poisson-2D cost vs mesh ($\\tau_{{FOM}}$={ftm:g})")
    a1.set_xticks(Ns); a1.set_xticklabels([str(n) for n in Ns])
    a1.legend(loc="upper left", fontsize=7.5); st.clean(a1)

    for ci, ft in enumerate(fts):
        nn = sorted({r["N"] for r in P if r["fom_tau"] == ft})
        sp = [max([r["speedup_vs_fom"] for r in P if r["N"] == n and r["fom_tau"] == ft],
                  default=np.nan) for n in nn]
        if nn:
            a2.plot(nn, sp, "-o", color=NCOL[ci % len(NCOL)], label=f"$\\tau_{{FOM}}$={ft:g}")
    a2.axhline(1.0, color=st.INK2, lw=1.0, ls="--")
    a2.text(0.02, 0.90, "hybrid wins above this line", transform=a2.transAxes,
            fontsize=7.5, color=st.MUTED, va="top")
    a2.set_xscale("log", base=2)          # linear y: the range is ~0.5-1.1 and a log
    a2.set_xticks(Ns)                     # axis there produces unreadable minor ticks
    a2.set_xticklabels([str(n) for n in Ns])
    a2.set_xlabel("N"); a2.set_ylabel("best hybrid speedup over the FOM")
    a2.set_title("Where does the hybrid start to win?")
    a2.legend(); st.clean(a2)
    paths += st.save(fig, FIGS, "wsfom_poisson_crossover", (EXTRA,))


def fig_poisson_iters(P, paths):
    if not P:
        return
    fts = sorted({r["fom_tau"] for r in P}, reverse=True)
    ftm = min(fts)
    Ns = sorted({r["N"] for r in P if r["fom_tau"] == ftm})
    rts = sorted({r["rom_tau"] for r in P}, reverse=True)
    if not Ns:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
    x = np.arange(len(rts))
    for ci, n in enumerate(Ns):
        sub = {r["rom_tau"]: r for r in P if r["N"] == n and r["fom_tau"] == ftm}
        y = [100.0 * sub[t]["iter_saving_frac"] if t in sub else np.nan for t in rts]
        a1.plot(x, y, "-o", color=NCOL[ci % len(NCOL)], label=f"N={n}")
    a1.axhline(0.0, color=st.INK2, lw=1.0, ls="--")
    a1.set_xticks(x); a1.set_xticklabels([taulab(t) for t in rts])
    a1.set_xlabel("ROM tolerance"); a1.set_ylabel("CG iterations saved (%)")
    a1.set_title(f"CG iterations saved by the warm start ($\\tau_{{FOM}}$={ftm:g})")
    a1.legend(ncol=2); st.clean(a1)

    for ci, n in enumerate(Ns):
        sub = {r["rom_tau"]: r for r in P if r["N"] == n and r["fom_tau"] == ftm}
        xs = [sub[t]["rom_err_Anorm_ratio"] for t in rts if t in sub
              and sub[t].get("rom_err_Anorm_ratio") is not None]
        ys = [100.0 * sub[t]["iter_saving_frac"] for t in rts if t in sub
              and sub[t].get("rom_err_Anorm_ratio") is not None]
        if xs:
            a2.plot(xs, ys, "o", color=NCOL[ci % len(NCOL)], label=f"N={n}")
    a2.set_xscale("log")
    a2.axhline(0.0, color=st.INK2, lw=1.0, ls="--")
    a2.set_xlabel("$\\|u_{ROM}-u^*\\|_A / \\|u^*\\|_A$  (what CG actually reduces)")
    a2.set_ylabel("CG iterations saved (%)")
    a2.set_title("Why the saving is what it is")
    a2.legend(ncol=2); st.clean(a2)
    paths += st.save(fig, FIGS, "wsfom_poisson_iters", (EXTRA,))


# --------------------------------------------------------------- Burgers
def fig_burgers_per_step(reps, paths):
    ps = []
    for d in reps:
        ps += d.get("per_step", [])
    # iteration counts are hardware-independent, so panels are fine here; prefer them
    if any((p.get("run_role") or "consolidated") == "panel" for p in ps):
        seen = {}
        for p in ps:
            k = (p["N"], p["fom_tau"])
            if k not in seen or (p.get("run_role") or "consolidated") == "panel":
                seen[k] = p
        ps = list(seen.values())
    if not ps:
        return
    Ns = sorted({p["N"] for p in ps})
    fts = sorted({p["fom_tau"] for p in ps}, reverse=True)
    ftm = min(fts)
    nmax = max(Ns)
    sel = [p for p in ps if p["N"] == nmax and p["fom_tau"] == ftm]
    if not sel:
        return
    p = sel[0]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.0, 3.6), layout="constrained")
    steps = np.arange(1, len(p["newton_per_step"]["prev"]) + 1)
    for arm in ("prev", "extrap", "rom"):
        a1.plot(steps, p["newton_per_step"][arm], "-", color=ARMC[arm], label=ARML[arm])
        a2.plot(steps, p["lin_per_step"][arm], "-", color=ARMC[arm], label=ARML[arm])
    a1.set_xlabel("time step"); a1.set_ylabel("Newton iterations")
    a1.set_title(f"Newton per step (N={nmax}, $\\tau_{{FOM}}$={ftm:g})")
    a1.legend(); st.clean(a1)
    a2.set_xlabel("time step"); a2.set_ylabel("BiCGStab iterations")
    a2.set_title("Inner linear iterations per step")
    st.clean(a2)
    a3.plot(steps, p["rom_err_per_step"], "-", color=st.C["orange"])
    a3.set_yscale("log")
    a3.set_xlabel("time step"); a3.set_ylabel("ROM rel-L2 vs the FOM")
    a3.set_title("Quality of the guess the ROM supplies")
    st.clean(a3)
    paths += st.save(fig, FIGS, "wsfom_burgers_per_step", (EXTRA,))


def fig_burgers_cost(B, paths):
    if not B:
        return
    Ns = sorted({r["N"] for r in B})
    fts = sorted({r["fom_tau"] for r in B}, reverse=True)
    ftm = min(fts)
    rows = {n: [r for r in B if r["N"] == n and r["fom_tau"] == ftm][0]
            for n in Ns if [r for r in B if r["N"] == n and r["fom_tau"] == ftm]}
    Ns = sorted(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
    a1.plot(Ns, [rows[n]["t_fom_baseline_ms"] for n in Ns], "-o", color=ARMC["prev"],
            label="pure FOM (" + ARML["prev"] + ")")
    a1.plot(Ns, [rows[n]["t_fom_extrap_ms"] for n in Ns], "-^", color=ARMC["extrap"],
            label="pure FOM (extrapolation)")
    a1.plot(Ns, [rows[n]["t_total_ms"] for n in Ns], "-s", color=ARMC["rom"],
            label="hybrid total (ROM + FOM)")
    a1.plot(Ns, [rows[n]["t_fom_ms"] for n in Ns], "--s", color=ARMC["rom"], alpha=0.6,
            label="hybrid: FOM stage only")
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xticks(Ns); a1.set_xticklabels([str(n) for n in Ns])
    a1.set_xlabel("N"); a1.set_ylabel("time for the 50-step trajectory (ms)")
    a1.set_title(f"Burgers-2D cost vs mesh ($\\tau_{{FOM}}$={ftm:g})")
    a1.legend(); st.clean(a1)

    w = 0.26
    x = np.arange(len(Ns))
    for k, arm in enumerate(("prev", "extrap", "rom")):
        key = {"prev": "iters_from_baseline", "extrap": "iters_from_extrap",
               "rom": "iters_from_rom"}[arm]
        a2.bar(x + (k - 1) * w, [rows[n][key] for n in Ns], w, color=ARMC[arm],
               label=ARML[arm])
    a2.set_xticks(x); a2.set_xticklabels([str(n) for n in Ns])
    a2.set_xlabel("N"); a2.set_ylabel("Newton iterations per trajectory")
    a2.set_title("Total Newton iterations (50 steps)")
    a2.legend(); st.clean(a2)
    paths += st.save(fig, FIGS, "wsfom_burgers_cost_vs_N", (EXTRA,))


def main():
    st.use()
    pts, reps = load()
    # CROSS-N WALL CLOCK MAY ONLY COME FROM THE CONSOLIDATED RUN (every N measured
    # sequentially in one job on one GPU).  Panel rows are fanned out across GPUs and
    # their wall clock is not comparable across N; they are used only for the
    # iteration-count panel, which is hardware-independent.
    role = lambda r: r.get("run_role") or "consolidated"   # the script default
    # the cross-N axes may only carry ONE consolidated run (one process, one GPU);
    # select_consolidated enforces that and reports what it dropped
    P, Pprov = select_consolidated([r for r in pts if r["pde"] == "poisson2d"],
                                   lambda r: (r["N"], r["rom_tau"], r["fom_tau"]))
    B, Bprov = select_consolidated([r for r in pts if r["pde"] == "burgers2d"],
                                   lambda r: (r["N"], r["fom_tau"]))
    for nm, pv in (("poisson", Pprov), ("burgers", Bprov)):
        if pv:
            print(f"  {nm} cross-N times from {pv['source_json']} job "
                  f"{pv['slurm_job_id']} gpu {pv['gpu']} meshes {pv['meshes']} "
                  f"({len(pv['dropped_groups'])} other group(s) dropped)")
    Piter = {}
    for r in [q for q in pts if q["pde"] == "poisson2d"]:
        k = (r["N"], r["rom_tau"], r["fom_tau"])
        if k not in Piter or role(r) == "panel":
            Piter[k] = r
    Piter = list(Piter.values())
    if not P:
        print("WARNING: no consolidated Poisson rows -- cross-N figures skipped")
    if not B:
        print("WARNING: no consolidated Burgers rows -- cross-N figures skipped")
    paths = []
    fig_poisson_total_vs_tau(P, paths)
    fig_poisson_crossover(P, paths)
    fig_poisson_iters(Piter, paths)
    fig_burgers_per_step(reps, paths)
    fig_burgers_cost(B, paths)
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
