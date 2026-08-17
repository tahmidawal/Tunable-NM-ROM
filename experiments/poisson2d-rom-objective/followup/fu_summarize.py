"""Markdown tables + figures for the Poisson-2D FOLLOW-UP cells.

Reads runs/followup/<cell>/*.json (pulled from the cluster) and writes
followup/FOLLOWUP_TABLES.md plus PNG+PDF figures into followup/figs/ and the
shared Plots directory.  Every number printed here comes straight from a JSON;
nothing is recomputed.

Usage: python followup/fu_summarize.py [<experiment dir>]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
RUNS = os.path.join(EXP, "runs", "followup")
FIGS = os.path.join(HERE, "figs")
PLOTS = "/home/tahmid/Dev/pod-ae-nmrom/Plots"
sys.path.insert(0, HERE)
import fu_style as fs  # noqa: E402

KS = [2, 4, 6, 8, 12, 16, 24, 32]
SEEDS = [0, 1, 2]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cell(name, pattern):
    hits = sorted(glob.glob(os.path.join(RUNS, name, pattern)))
    return load(hits[0]) if hits else None


def pick(rows, **kw):
    """First row matching all key=value constraints."""
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def fmt(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}e}"


# ------------------------------------------------------------------ gather
def gather():
    D = dict(kladder=[], seeds=[], mladder=None, Mladder=[], pod=None, timing=None)
    for K in KS:
        rom = cell(f"pk_K{K}", f"rom_K{K}.json")
        fd = cell(f"pk_K{K}", f"fd_K{K}.json")
        tr = cell(f"pk_K{K}", f"autodec_K{K}_N64_hbc_report.json")
        if rom is None:
            continue
        D["kladder"].append(dict(K=K, rom=rom, fd=fd, train=tr))
    for S in SEEDS:
        if S == 0:
            rom = cell("pk_K8", "rom_K8.json"); fd = cell("pk_K8", "fd_K8.json")
            tr = cell("pk_K8", "autodec_K8_N64_hbc_report.json")
        else:
            rom = cell(f"ps_S{S}", f"rom_S{S}.json"); fd = cell(f"ps_S{S}", f"fd_S{S}.json")
            tr = cell(f"ps_S{S}", f"autodec_K8_N64_hbc_S{S}_report.json")
        if rom is None:
            continue
        D["seeds"].append(dict(seed=S, rom=rom, fd=fd, train=tr))
    D["mladder"] = cell("pm_m", "mlad_K8.json")
    for M in (16, 32, 64, 128, 256):
        r = cell(f"pM_M{M}", f"Mlad_M{M}.json")
        if r:
            D["Mladder"].append(dict(M=M, rom=r))
    D["pod"] = cell("pp_pod", "pod_ladder.json")
    D["timing"] = cell("pt_n", "timing_n.json")
    return D


# ------------------------------------------------------------------ tables
def tab_kladder(D, out):
    if not D["kladder"]:
        return
    pod = D["pod"]
    prow = {r["k"]: r for r in pod["rows"]} if pod else {}
    out.append("### k ladder — coordinate ROM vs POD at the same k "
               "(N=64, hard-BC, weak_a1_M64, NNLS-EQ grid m=256, nearest init)\n")
    out.append("| k | coord train recon | coord ORACLE (inferred latent, val) | coord ROM full grid | coord ROM EQ m=256 (grid) | coord ROM EQ m=256 (meshfree) | coord FD-LSPG control | POD proj floor | POD Galerkin | POD weak_a1_M64 | POD FD-LSPG |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in D["kladder"]:
        K = e["K"]; rows = e["rom"]["rows"]
        f_ = pick(rows, scheme="full", init="nearest")
        g_ = pick(rows, scheme="nnls", init="nearest")
        o_ = pick(rows, scheme="nnlsoff", init="nearest")
        fdr = pick(e["fd"]["rows"], scheme="full", init="nearest") if e["fd"] else None
        tr = e["train"]
        p = prow.get(K)
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            K,
            fmt(tr["train_mean_rel_l2"]) if tr else "—",
            fmt(tr["val_lm_inferred_mean_rel_l2"]["best"]) if tr else "—",
            fmt(f_["rom_rel_l2_mean"]) if f_ else "—",
            fmt(g_["rom_rel_l2_mean"]) if g_ else "—",
            fmt(o_["rom_rel_l2_mean"]) if o_ else "—",
            fmt(fdr["rom_rel_l2_mean"]) if fdr else "—",
            fmt(p["proj"]["mean"]) if p else "—",
            fmt(p["galerkin"]["mean"]) if p else "—",
            fmt(p["weak_a1_M64"]["mean"]) if p else "—",
            fmt(p["fd"]["mean"]) if p else "—"))
    out.append("")
    out.append("Solver work at each k (median over the 16 test sources, LM budget 60): "
               + ", ".join(f"k={e['K']}: {pick(e['rom']['rows'], scheme='nnls', init='nearest')['lm_accepted_med']:.0f} accepted / "
                           f"{pick(e['rom']['rows'], scheme='nnls', init='nearest')['lm_rejected_med']:.0f} rejected"
                           for e in D["kladder"] if pick(e["rom"]["rows"], scheme="nnls", init="nearest")) + ".\n")


def tab_seeds(D, out):
    if len(D["seeds"]) < 2:
        return
    out.append("### Multi-seed (K=8 hard-BC; TRAIN_SEED varies the net/latent init and the "
               "batch order only — the data draw, the split and the test set are fixed)\n")
    keys = [("coord ROM, full grid", lambda e: pick(e["rom"]["rows"], scheme="full", init="nearest")["rom_rel_l2_mean"]),
            ("coord ROM, EQ m=256 (grid)", lambda e: pick(e["rom"]["rows"], scheme="nnls", init="nearest")["rom_rel_l2_mean"]),
            ("coord ROM, EQ m=256 (meshfree)", lambda e: pick(e["rom"]["rows"], scheme="nnlsoff", init="nearest")["rom_rel_l2_mean"]),
            ("ORACLE inferred-latent floor (test)", lambda e: pick(e["rom"]["rows"], scheme="full", init="nearest")["oracle_rel_l2_mean"]),
            ("FD-LSPG control", lambda e: pick(e["fd"]["rows"], scheme="full", init="nearest")["rom_rel_l2_mean"]),
            ("train recon", lambda e: e["train"]["train_mean_rel_l2"]),
            ("val inferred-latent floor", lambda e: e["train"]["val_lm_inferred_mean_rel_l2"]["best"])]
    hdr = "| quantity | " + " | ".join(f"seed {e['seed']}" for e in D["seeds"]) + " | mean ± std |"
    out.append(hdr); out.append("|" + "---|" * (len(D["seeds"]) + 2))
    for name, fn in keys:
        vals = []
        for e in D["seeds"]:
            try:
                vals.append(float(fn(e)))
            except (TypeError, KeyError):
                vals.append(float("nan"))
        v = np.array(vals)
        out.append(f"| {name} | " + " | ".join(fmt(x) for x in v)
                   + f" | {np.nanmean(v):.2e} ± {np.nanstd(v, ddof=1):.1e} |")
    if D["pod"]:
        p = {r["k"]: r for r in D["pod"]["rows"]}.get(8)
        if p:
            out.append(f"| POD control k=8 (Galerkin / weak_a1_M64 / FD) | "
                       + " | ".join([f"{p['galerkin']['mean']:.2e} / {p['weak_a1_M64']['mean']:.2e} / {p['fd']['mean']:.2e}"] * len(D["seeds"]))
                       + " | std 0 by construction |")
    out.append("\nThe POD control is a deterministic function of the training snapshots, so it "
               "carries no training-seed variance; its row is constant by construction.\n")


def tab_ladders(D, out):
    if D["mladder"]:
        rows = D["mladder"]["rows"]
        ms = sorted({r["m"] for r in rows if r["scheme"] != "full"})
        full = pick(rows, scheme="full", init="nearest")
        out.append("### m ladder at fixed (K=8, M=64) — NNLS-EQ on grid nodes vs a meshfree pool\n")
        out.append("| scheme | " + " | ".join(f"m={m}" for m in ms) + " | full grid |")
        out.append("|" + "---|" * (len(ms) + 2))
        for sch, label in (("nnls", "NNLS-EQ, grid nodes"), ("nnlsoff", "NNLS-EQ, meshfree pool")):
            cells_ = [pick(rows, scheme=sch, m=m, init="nearest") for m in ms]
            out.append(f"| {label} | " + " | ".join(fmt(c["rom_rel_l2_mean"]) if c else "—" for c in cells_)
                       + f" | {fmt(full['rom_rel_l2_mean']) if full else '—'} |")
            out.append(f"| ↳ NNLS relative fit residual | " + " | ".join(
                (f"{c['eq_info']['rnorm_final']/c['eq_info']['b_norm']:.1e}" if c and c.get("eq_info") else "—")
                for c in cells_) + " | 0 |")
        out.append(f"\nOracle (finite-budget inferred latent) on the same test set: "
                   f"{fmt(full['oracle_rel_l2_mean']) if full else '—'}.\n")
    if D["Mladder"]:
        out.append("### M ladder at m ≈ 4M (K=8)\n")
        out.append("| M (requested) | modes retained M' | m | full grid | NNLS-EQ grid | NNLS-EQ meshfree |")
        out.append("|---|---|---|---|---|---|")
        for e in D["Mladder"]:
            rows = e["rom"]["rows"]
            f_ = pick(rows, scheme="full", init="nearest")
            g_ = pick(rows, scheme="nnls", init="nearest")
            o_ = pick(rows, scheme="nnlsoff", init="nearest")
            out.append(f"| {e['M']} | {f_['n_modes_retained'] if f_ else '—'} | {g_['m'] if g_ else '—'} | "
                       f"{fmt(f_['rom_rel_l2_mean']) if f_ else '—'} | {fmt(g_['rom_rel_l2_mean']) if g_ else '—'} | "
                       f"{fmt(o_['rom_rel_l2_mean']) if o_ else '—'} |")
        out.append("")


def tab_timing(D, out):
    t = D["timing"]
    if not t:
        return
    c = t["config"]
    out.append(f"### Online cost vs N on ONE GPU ({t['config']['device']}, all N sequential in one "
               f"process; K={c['K']}, M={c['M']}, m={c['m']}, median of {c['time_reps']} after 2 warm-ups, "
               f"`block_until_ready`)\n")
    out.append("| N | interior DOF | FOM (CG, f64) | ROM solve (jitted LM) | ROM iters | ms / iteration | preprocessing Λ⁻¹Φᵀf | speedup (solve) | speedup (+preproc) | ROM rel-L2 vs FD at this N |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in t["rows"]:
        out.append(f"| {r['N']} | {r['n_dof']} | {r['fom_cg_s']*1e3:.1f} ms | {r['rom_solve_s']*1e3:.2f} ms | "
                   f"{r['rom_iters_src0']} | {r['rom_s_per_iter']*1e3:.3f} | {r['preprocess_s']*1e3:.2f} ms | "
                   f"**{r['speedup_solve_only']:.1f}x** | {r['speedup_with_preprocess']:.1f}x | {fmt(r['rom_rel_l2_mean'])} |")
    out.append("")


# ------------------------------------------------------------------ figures
def fig_kladder(D):
    if not D["kladder"] or not D["pod"]:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    prow = {r["k"]: r for r in D["pod"]["rows"]}
    ks, coord, coord_eq, orac, podg, podw, podp = [], [], [], [], [], [], []
    for e in D["kladder"]:
        K = e["K"]
        f_ = pick(e["rom"]["rows"], scheme="full", init="nearest")
        g_ = pick(e["rom"]["rows"], scheme="nnls", init="nearest")
        if f_ is None or K not in prow:
            continue
        ks.append(K); coord.append(f_["rom_rel_l2_mean"])
        coord_eq.append(g_["rom_rel_l2_mean"] if g_ else np.nan)
        orac.append(f_["oracle_rel_l2_mean"])
        podg.append(prow[K]["galerkin"]["mean"]); podw.append(prow[K]["weak_a1_M64"]["mean"])
        podp.append(prow[K]["proj"]["mean"])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    fs.clean(ax)
    ax.plot(ks, coord, "o-", color=fs.C["blue"], label="coordinate ROM (full grid)", zorder=5)
    ax.plot(ks, coord_eq, "s--", color=fs.C["blue"], alpha=0.75, label="coordinate ROM (NNLS-EQ, m=256)", zorder=4)
    ax.plot(ks, orac, ":", color=fs.MUTED, lw=1.6, label="coord. inferred-latent floor (oracle)", zorder=3)
    # POD-Galerkin and the same weak objective agree to <1% with the projection floor at
    # every k (POD sits ON its linear ceiling), so one solid POD line + the floor is plotted;
    # all three are in the table.
    ax.plot(ks, podw, "^-", color=fs.C["orange"], label="POD, same weak objective", zorder=5)
    ax.plot(ks, podp, ":", color=fs.C["orange"], lw=1.4, alpha=0.7, label="POD projection floor", zorder=3)
    ax.axvline(4, color=fs.MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    fin = np.array([v for v in coord + coord_eq + orac + podg + podw + podp
                    if v is not None and np.isfinite(v)], dtype=float)
    ax.set_ylim(fin.min() / 1.7, fin.max() * 1.5)
    ax.annotate("intrinsic dim 4", (4, ax.get_ylim()[1]), xytext=(3, -14),
                textcoords="offset points", color=fs.INK2, fontsize=8, va="top")
    ax.set_xlabel("latent dimension k"); ax.set_ylabel("held-out rel-L2 error")
    ax.set_title("Poisson 2D — where the nonlinear manifold stops paying", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    # the headline gap, stated on the figure rather than left to the reader
    try:
        i8 = ks.index(8)
        ax.annotate(f"{podw[i8] / coord[i8]:.0f}x", xy=(8, np.sqrt(podw[i8] * coord[i8])),
                    xytext=(11, np.sqrt(podw[i8] * coord[i8])), color=fs.INK2, fontsize=8,
                    va="center", arrowprops=dict(arrowstyle="-", lw=0.7, color=fs.MUTED))
        ax.annotate("", xy=(8, podw[i8]), xytext=(8, coord[i8]),
                    arrowprops=dict(arrowstyle="<->", lw=0.9, color=fs.MUTED))
    except (ValueError, TypeError):
        pass
    return fs.save(fig, FIGS, "poisson_k_ladder", (PLOTS,))


def fig_timing(D):
    t = D["timing"]
    if not t:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    r = t["rows"]
    N = [x["N"] for x in r]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    fs.clean(ax)
    ax.plot(N, [x["fom_cg_s"] * 1e3 for x in r], "o-", color=fs.C["red"], label="FOM (CG solve)")
    ax.plot(N, [x["rom_solve_s"] * 1e3 for x in r], "s-", color=fs.C["blue"],
            label=f"ROM latent solve (k={t['config']['K']}, M={t['config']['M']}, m={t['config']['m']})")
    ax.plot(N, [x["preprocess_s"] * 1e3 for x in r], "^--", color=fs.C["aqua"],
            label=r"input preprocessing $\Lambda^{-1}\Phi^{T}f$")
    for x in r:
        ax.annotate(f"{x['speedup_solve_only']:.0f}x", (x["N"], x["rom_solve_s"] * 1e3),
                    xytext=(0, -13), textcoords="offset points", ha="center",
                    fontsize=8, color=fs.INK2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("mesh N (interior DOF $(N-2)^2$)"); ax.set_ylabel("wall time (ms)")
    ax.set_title("Poisson 2D — online cost is independent of the mesh", loc="left")
    ax.legend(loc="center left")
    return fs.save(fig, FIGS, "poisson_cost_vs_N", (PLOTS,))


def fig_ladders(D):
    if not D["mladder"]:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    rows = D["mladder"]["rows"]
    ms = sorted({r["m"] for r in rows if r["scheme"] != "full"})
    full = pick(rows, scheme="full", init="nearest")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = fs.clean(axes[0])
    for sch, label, col, mk in (("nnls", "NNLS-EQ, grid nodes", fs.C["blue"], "o"),
                                ("nnlsoff", "NNLS-EQ, meshfree pool", fs.C["aqua"], "s")):
        y = [pick(rows, scheme=sch, m=m, init="nearest") for m in ms]
        ax.plot(ms, [c["rom_rel_l2_mean"] if c else np.nan for c in y], mk + "-", color=col, label=label)
    if full:
        ax.axhline(full["rom_rel_l2_mean"], color=fs.INK2, ls="--", lw=1.3, label="full grid (3844 nodes)")
        ax.axhline(full["oracle_rel_l2_mean"], color=fs.MUTED, ls=":", lw=1.3, label="inferred-latent floor")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("quadrature points m"); ax.set_ylabel("held-out rel-L2 error")
    ax.set_title("m ladder at M=64 (m ≈ 4M dashed marker)", loc="left")
    ax.legend(loc="upper right")
    ax = fs.clean(axes[1])
    if D["Mladder"]:
        Ms = [e["M"] for e in D["Mladder"]]
        for sch, label, col, mk in (("full", "full grid", fs.INK2, "d"),
                                    ("nnls", "NNLS-EQ grid, m=4M", fs.C["blue"], "o"),
                                    ("nnlsoff", "NNLS-EQ meshfree, m=4M", fs.C["aqua"], "s")):
            y = [pick(e["rom"]["rows"], scheme=sch, init="nearest") for e in D["Mladder"]]
            ax.plot(Ms, [c["rom_rel_l2_mean"] if c else np.nan for c in y], mk + "-", color=col, label=label)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("test modes M (requested)"); ax.set_ylabel("held-out rel-L2 error")
    ax.set_title("M ladder at m ≈ 4M", loc="left")
    ax.legend(loc="upper right")
    fig.suptitle("Poisson 2D — the hyper-reduction knobs", x=0.005, ha="left", fontsize=10, color=fs.INK)
    fig.tight_layout()
    return fs.save(fig, FIGS, "poisson_eq_knobs", (PLOTS,))


def main():
    D = gather()
    out = ["# Poisson-2D follow-up — tables (generated by `followup/fu_summarize.py`)", ""]
    tab_kladder(D, out)
    tab_seeds(D, out)
    tab_ladders(D, out)
    tab_timing(D, out)
    p = os.path.join(HERE, "FOLLOWUP_TABLES.md")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", p)
    for f in (fig_kladder, fig_timing, fig_ladders):
        try:
            r = f(D)
            print("figure:", r)
        except Exception as e:                    # a missing cell must not kill the rest
            print(f"  figure {f.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
