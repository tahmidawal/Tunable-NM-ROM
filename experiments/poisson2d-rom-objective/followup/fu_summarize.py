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
    D = dict(kladder=[], seeds=[], mladder=None, Mladder=[], pod=None, timing=None,
             timing_k=None, timing_m=None, family=[])
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
    D["timing_k"] = cell("pt_k", "timing_k.json")
    D["timing_m"] = cell("pt_m", "timing_m.json")
    for nb in (1, 2, 3):
        r = cell(f"pc_nb{nb}", f"family_NB{nb}.json")
        if r:
            D["family"].append(dict(nb=nb, rep=r))
    return D


# ------------------------------------------------------------------ tables
def tab_kladder(D, out):
    if not D["kladder"]:
        return
    pod = D["pod"]
    prow = {r["k"]: r for r in pod["rows"]} if pod else {}
    out.append("### k ladder — coordinate ROM vs POD at the same k "
               "(N=64, hard-BC, weak_a1_M64, NNLS-EQ m=256)\n")
    out.append("ACCURACY protocol: nearest init, mean rel-L2 over the 16 held-out sources, LM "
               "budget 60.  The POD columns are FULL-GRID objectives (exact minimisers, the "
               "problem being quadratic in the POD coefficients) and are therefore comparable to "
               "the coordinate ROM's full-grid column; the coordinate EQ columns use a "
               "hyper-reduced quadrature.\n")
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
    out.append("Medians over the 16 sources (the means above carry heavy tails): "
               + ", ".join(
                   "k={}: full {} / EQ-grid {} / EQ-meshfree {}".format(
                       e["K"],
                       fmt(pick(e["rom"]["rows"], scheme="full", init="nearest")["rom_rel_l2_med"]),
                       fmt(pick(e["rom"]["rows"], scheme="nnls", init="nearest")["rom_rel_l2_med"]),
                       fmt(pick(e["rom"]["rows"], scheme="nnlsoff", init="nearest")["rom_rel_l2_med"]))
                   for e in D["kladder"]
                   if pick(e["rom"]["rows"], scheme="full", init="nearest")) + ".\n")
    out.append("Solves that ended on the LM BUDGET rather than converging (out of 16, full grid / "
               "EQ grid / EQ meshfree / FD control): " + ", ".join(
                   "k={}: {}/{}/{}/{}".format(
                       e["K"],
                       pick(e["rom"]["rows"], scheme="full", init="nearest")["lm_reasons"].get("budget", 0),
                       pick(e["rom"]["rows"], scheme="nnls", init="nearest")["lm_reasons"].get("budget", 0),
                       pick(e["rom"]["rows"], scheme="nnlsoff", init="nearest")["lm_reasons"].get("budget", 0),
                       (pick(e["fd"]["rows"], scheme="full", init="nearest")["lm_reasons"].get("budget", 0)
                        if e["fd"] else "-"))
                   for e in D["kladder"]
                   if pick(e["rom"]["rows"], scheme="full", init="nearest")) +
               ".  Budget terminations are INCLUDED in the means above (nothing is dropped).\n")
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
            out.append(f"| ↳ ms per solve (median of 7, one GPU) | " + " | ".join(
                (f"{cost_m(D, 64, m, 'offgrid' if sch == 'nnlsoff' else 'grid')['rom_solve_s']*1e3:.1f}"
                 if cost_m(D, 64, m, "offgrid" if sch == "nnlsoff" else "grid") else "—") for m in ms)
                + (f" | {cost_m(D, 64, None, 'full')['rom_solve_s']*1e3:.1f} |"
                   if cost_m(D, 64, None, "full") else " | — |"))
            out.append(f"| ↳ NNLS relative fit residual | " + " | ".join(
                (f"{c['eq_info']['rnorm_final']/c['eq_info']['b_norm']:.1e}" if c and c.get("eq_info") else "—")
                for c in cells_) + " | 0 |")
        out.append(f"\nOracle (finite-budget inferred latent) on the same test set: "
                   f"{fmt(full['oracle_rel_l2_mean']) if full else '—'}.\n")
    if D["Mladder"]:
        out.append("### M ladder at m ≈ 4M (K=8)\n")
        out.append("Accuracy columns: nearest init, mean over the 16 held-out sources (medians in "
                   "brackets).  Cost columns: the `pt_m` cell, mean init, timed on source 0.  The "
                   "retained mode count M' differs between the on-grid (discrete eigenvalues) and "
                   "meshfree (continuum eigenvalues) arms because their degeneracy patterns differ, "
                   "so it is listed per arm.\n")
        out.append("| M (requested) | M' (full/grid, meshfree) | m | full grid | NNLS-EQ grid | NNLS-EQ meshfree | ms/solve full | ms/solve EQ (meshfree) |")
        out.append("|---|---|---|---|---|---|---|---|")
        for e in D["Mladder"]:
            rows = e["rom"]["rows"]
            f_ = pick(rows, scheme="full", init="nearest")
            g_ = pick(rows, scheme="nnls", init="nearest")
            o_ = pick(rows, scheme="nnlsoff", init="nearest")
            mm = f"{f_['n_modes_retained']}, {o_['n_modes_retained']}" if (f_ and o_) else "—"
            cf = cost_m(D, e["M"], None, "full"); ce = cost_m(D, e["M"], 4 * e["M"], "offgrid")
            def cell_(r_):
                return (f"{fmt(r_['rom_rel_l2_mean'])} [{fmt(r_['rom_rel_l2_med'])}]" if r_ else "—")
            tf = f"{cf['rom_solve_s']*1e3:.1f}" if cf else "—"
            te = f"{ce['rom_solve_s']*1e3:.1f}" if ce else "—"
            out.append(f"| {e['M']} | {mm} | {g_['m'] if g_ else '—'} | {cell_(f_)} | {cell_(g_)} | "
                       f"{cell_(o_)} | {tf} | {te} |")
        out.append("")


def cost_m(D, M, m, pool):
    t = D.get("timing_m")
    if not t:
        return None
    for r in t["rows"]:
        if r["M"] == M and r["pool"] == pool and (pool == "full" or r["m"] == m):
            return r
    return None


def tab_timing_k(D, out):
    t = D.get("timing_k")
    if not t:
        return
    c = t["config"]
    out.append(f"### Per-iteration cost and iterations vs k on ONE GPU ({c['device']}, N=64, "
               f"M={c['M']}, m={c['m']}, meshfree EQ; FOM CG {t['fom_cg_s']*1e3:.2f} ms). "
               f"The reference LM has no absolute residual tolerance, so these are iterations "
               f"to TERMINATION (reason histogram in the JSON); the POD ROM is linear, so its "
               f"online solve is one precomputed pseudo-inverse matvec.\n")
    out.append("| manifold | k | ROM solve | Jacobian evals | LM attempts | ms / iteration | speedup vs FOM | rel-L2 (cross-check) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in t["rows"]:
        if r["kind"] == "coord":
            out.append(f"| coordinate | {r['k']} | {r['rom_solve_s']*1e3:.2f} ms | {r['rom_iters_mean']:.1f} | "
                       f"{r['rom_attempts_mean']:.1f} | {r['rom_s_per_iter']*1e3:.3f} | "
                       f"{r['speedup_solve_only']:.2f}x | {fmt(r['rom_rel_l2_mean'])} |")
    for r in t["rows"]:
        if r["kind"] == "pod":
            sq = " (square)" if r.get("square_system") else ""
            out.append(f"| POD{sq} | {r['k']} | {r['rom_solve_s']*1e6:.0f} us | 1 (direct) | — | — | "
                       f"{r['speedup_solve_only']:.0f}x | {fmt(r['rom_rel_l2_mean'])} |")
    out.append("\nThe POD solve is a single small matvec and is dispatch-bound, so its time is a "
               "floor on the measurement, not a property of the method.\n")


def tab_family(D, out):
    if not D["family"]:
        return
    out.append("### Complexity ladder — NB independent bump sources (intrinsic dimension 4·NB)\n")
    out.append("NB=1 is the main study's family (`ms_parametric.sample_params` verbatim). Same "
               "training recipe and budget at every (NB, k); coordinate ROM = `weak_a1_M64`, "
               "nearest init; POD rows are the exact minimisers of the same full-grid objectives.\n")
    out.append("| NB (intrinsic dim) | k | coord train | coord ORACLE | coord ROM full | coord ROM EQ m=256 | coord FD-LSPG | POD proj | POD weak |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for e in D["family"]:
        pod = {r["k"]: r for r in e["rep"]["pod"]}
        for r in e["rep"]["rows"]:
            p = pod.get(r["K"], {})
            out.append(f"| {e['nb']} ({4*e['nb']}) | {r['K']} | {fmt(r['train_mean_rel_l2'])} | "
                       f"{fmt(r['oracle']['nearest'])} | {fmt(r['rom']['weak_full']['rel_l2_mean'])} | "
                       f"{fmt(r['rom']['weak_eq_meshfree']['rel_l2_mean'])} | "
                       f"{fmt(r['rom']['fd_full']['rel_l2_mean'])} | {fmt(p.get('proj'))} | "
                       f"{fmt(p.get('weak'))} |")
    out.append("")


def tab_timing(D, out):
    t = D["timing"]
    if not t:
        return
    c = t["config"]
    out.append(f"### Online cost vs N on ONE GPU ({c['device']}, all N sequential in one "
               f"process; k={c['K']}, M={c['M']}, m={c['m']}, meshfree NNLS-EQ REFIT ON EACH N's "
               f"GRID, median of {c['time_reps']} after {c['time_warm']} warm-ups, `block_until_ready`)\n")
    out.append("| N | interior DOF | FOM (CG, f64) | ROM latent solve | ROM iters | ms / iteration | "
               "input projection Λ⁻¹Φᵀf | full-field decode | speedup (solve) | speedup (end to end) | ROM rel-L2 vs FD at this N |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t["rows"]:
        out.append(f"| {r['N']} | {r['n_dof']} | {r['fom_cg_s']*1e3:.2f} ms | {r['rom_solve_s']*1e3:.2f} ms | "
                   f"{r['rom_iters_mean']:.1f} | {r['rom_s_per_iter']*1e3:.3f} | "
                   f"{r['preprocess_s']*1e3:.2f} ms | {r['decode_full_field_s']*1e3:.2f} ms | "
                   f"**{r['speedup_solve_only']:.1f}x** | {r['speedup_end_to_end']:.1f}x | "
                   f"{fmt(r['rom_rel_l2_mean'])} |")
    out.append("Protocol split: `ROM iters` is the mean over the 16 sources, while `ms/iteration` "
               "is the TIMED source's own solve divided by ITS iteration count (source 0, mean "
               "init) -- e.g. at N=32 the timed source took "
               f"{t['rows'][0]['rom_solve_s']/t['rows'][0]['rom_s_per_iter']:.0f} iterations against "
               f"a 16-source mean of {t['rows'][0]['rom_iters_mean']:.1f}.  Accuracy in the last "
               "column is the 16-source mean.\n")
    tbl = ", ".join(f"N={r['N']}: {r['preprocess_offline_table_s']*1e3:.0f} ms" for r in t["rows"])
    agree = max(r["preprocess_vs_reference_maxabs"] for r in t["rows"])
    out.append(f"\nThe mode table Phi is a per-mesh constant and is built offline ({tbl}); the online "
               f"projection above is the one (M' x n) matvec, verified equal to "
               f"`pro_common.weak_source_term` to {agree:.0e} absolute.\n")


def cost_m(D, M, m, pool):
    t = D.get("timing_m")
    if not t:
        return None
    for r in t["rows"]:
        if r["M"] == M and r["pool"] == pool and (pool == "full" or r["m"] == m):
            return r
    return None


def tab_timing_k(D, out):
    t = D.get("timing_k")
    if not t:
        return
    c = t["config"]
    out.append(f"### Per-iteration cost and iterations vs k on ONE GPU ({c['device']}, N=64, "
               f"M={c['M']}, m={c['m']}, meshfree EQ; FOM CG {t['fom_cg_s']*1e3:.2f} ms). "
               f"The reference LM has no absolute residual tolerance, so these are iterations "
               f"to TERMINATION (reason histogram in the JSON); the POD ROM is linear, so its "
               f"online solve is one precomputed pseudo-inverse matvec.\n")
    out.append("| manifold | k | ROM solve | Jacobian evals | LM attempts | ms / iteration | speedup vs FOM | rel-L2 (cross-check) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in t["rows"]:
        if r["kind"] == "coord":
            out.append(f"| coordinate | {r['k']} | {r['rom_solve_s']*1e3:.2f} ms | {r['rom_iters_mean']:.1f} | "
                       f"{r['rom_attempts_mean']:.1f} | {r['rom_s_per_iter']*1e3:.3f} | "
                       f"{r['speedup_solve_only']:.2f}x | {fmt(r['rom_rel_l2_mean'])} |")
    for r in t["rows"]:
        if r["kind"] == "pod":
            sq = " (square)" if r.get("square_system") else ""
            out.append(f"| POD{sq} | {r['k']} | {r['rom_solve_s']*1e6:.0f} us | 1 (direct) | — | — | "
                       f"{r['speedup_solve_only']:.0f}x | {fmt(r['rom_rel_l2_mean'])} |")
    out.append("\nThe POD solve is a single small matvec and is dispatch-bound, so its time is a "
               "floor on the measurement, not a property of the method.\n")


def tab_family(D, out):
    if not D["family"]:
        return
    out.append("### Complexity ladder — NB independent bump sources (intrinsic dimension 4·NB)\n")
    out.append("NB=1 is the main study's family (`ms_parametric.sample_params` verbatim). Same "
               "training recipe and budget at every (NB, k); coordinate ROM = `weak_a1_M64`, "
               "nearest init; POD rows are the exact minimisers of the same full-grid objectives.\n")
    out.append("| NB (intrinsic dim) | k | coord train | coord ORACLE | coord ROM full | coord ROM EQ m=256 | coord FD-LSPG | POD proj | POD weak |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for e in D["family"]:
        pod = {r["k"]: r for r in e["rep"]["pod"]}
        for r in e["rep"]["rows"]:
            p = pod.get(r["K"], {})
            out.append(f"| {e['nb']} ({4*e['nb']}) | {r['K']} | {fmt(r['train_mean_rel_l2'])} | "
                       f"{fmt(r['oracle']['nearest'])} | {fmt(r['rom']['weak_full']['rel_l2_mean'])} | "
                       f"{fmt(r['rom']['weak_eq_meshfree']['rel_l2_mean'])} | "
                       f"{fmt(r['rom']['fd_full']['rel_l2_mean'])} | {fmt(p.get('proj'))} | "
                       f"{fmt(p.get('weak'))} |")
    out.append("")


def tab_timing(D, out):
    t = D["timing"]
    if not t:
        return
    c = t["config"]
    out.append(f"### Online cost vs N on ONE GPU ({c['device']}, all N sequential in one "
               f"process; k={c['K']}, M={c['M']}, m={c['m']}, meshfree NNLS-EQ REFIT ON EACH N's "
               f"GRID, median of {c['time_reps']} after {c['time_warm']} warm-ups, `block_until_ready`)\n")
    out.append("| N | interior DOF | FOM (CG, f64) | ROM latent solve | ROM iters | ms / iteration | "
               "input projection Λ⁻¹Φᵀf | full-field decode | speedup (solve) | speedup (end to end) | ROM rel-L2 vs FD at this N |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t["rows"]:
        out.append(f"| {r['N']} | {r['n_dof']} | {r['fom_cg_s']*1e3:.2f} ms | {r['rom_solve_s']*1e3:.2f} ms | "
                   f"{r['rom_iters_mean']:.1f} | {r['rom_s_per_iter']*1e3:.3f} | "
                   f"{r['preprocess_s']*1e3:.2f} ms | {r['decode_full_field_s']*1e3:.2f} ms | "
                   f"**{r['speedup_solve_only']:.1f}x** | {r['speedup_end_to_end']:.1f}x | "
                   f"{fmt(r['rom_rel_l2_mean'])} |")
    r0 = t["rows"][0]
    out.append(f"\nThe mode table Φ is a per-mesh constant and is built offline "
               f"({', '.join(f'N={r[chr(39)+chr(39)] if False else r[chr(78)]}: {r[chr(112)+chr(114)+chr(101)+chr(112)+chr(114)+chr(111)+chr(99)+chr(101)+chr(115)+chr(115)+chr(95)+chr(111)+chr(102)+chr(102)+chr(108)+chr(105)+chr(110)+chr(101)+chr(95)+chr(116)+chr(97)+chr(98)+chr(108)+chr(101)+chr(95)+chr(115)]*1e3:.0f} ms' for r in t['rows'])}); "
               f"the online projection above is the one (M'×n) matvec, verified equal to "
               f"`pro_common.weak_source_term` to {max(r['preprocess_vs_reference_maxabs'] for r in t['rows']):.0e} absolute.\n")


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
            label=r"input projection $\Lambda^{-1}\Phi^{T}f$ (one matvec)")
    ax.plot(N, [x["decode_full_field_s"] * 1e3 for x in r], "v:", color=fs.C["yellow"],
            label="full-field decode (output stage)")
    for x in r:
        ax.annotate(f"{x['speedup_solve_only']:.1f}x", (x["N"], x["rom_solve_s"] * 1e3),
                    xytext=(0, -13), textcoords="offset points", ha="center",
                    fontsize=8, color=fs.INK2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("mesh N (interior DOF $(N-2)^2$)"); ax.set_ylabel("wall time (ms)")
    ax.set_title("Poisson 2D — the latent solve does not see the mesh", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
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
    ax.set_title("m ladder at M=64 (dashed: full grid, 3844 nodes)", loc="left")
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


def fig_family(D):
    """Small multiples, one per family: the coordinate ROM's knee should track the
    intrinsic dimension 4*NB while POD keeps improving slowly with k."""
    if not D["family"]:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    n = len(D["family"])
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 0.6, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, D["family"]):
        fs.clean(ax)
        rows = e["rep"]["rows"]; pod = {r["k"]: r for r in e["rep"]["pod"]}
        ks = [r["K"] for r in rows]
        ax.plot(ks, [r["rom"]["weak_full"]["rel_l2_mean"] for r in rows], "o-",
                color=fs.C["blue"], label="coordinate ROM")
        ax.plot(ks, [r["rom"]["weak_eq_meshfree"]["rel_l2_mean"] for r in rows], "s--",
                color=fs.C["blue"], alpha=0.7, label="coord. ROM, NNLS-EQ m=256")
        ax.plot(ks, [r["oracle"]["nearest"] for r in rows], ":", color=fs.MUTED, lw=1.5,
                label="inferred-latent floor")
        pk = sorted(k for k in pod if not pod[k]["square_or_underdetermined"])
        ax.plot(pk, [pod[k]["weak"] for k in pk], "^-", color=fs.C["orange"],
                label="POD, same objective")
        d0 = 4 * e["nb"]
        ax.axvline(d0, color=fs.MUTED, lw=0.9, ls=(0, (2, 3)))
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        tk = sorted(set(ks) | set(pk))
        ax.set_xticks(tk); ax.set_xticklabels([str(k) for k in tk], fontsize=7)
        ax.set_title(f"{e['nb']} bump source{'s' if e['nb'] > 1 else ''} — intrinsic dim {d0}",
                     loc="left", fontsize=9)
        ax.set_xlabel("latent dimension k")
    axes[0].set_ylabel("held-out rel-L2 error")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Poisson 2D — the nonlinear manifold's advantage shrinks as the family's "
                 "intrinsic dimension grows\n(512 training sources and one training budget at "
                 "every point; the ROM stays within 1.0-1.6x of its own inferred-latent floor)",
                 x=0.005, ha="left", fontsize=9.5, color=fs.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fs.save(fig, FIGS, "poisson_complexity_ladder", (PLOTS,))


def fig_cost_vs_k(D):
    t = D.get("timing_k")
    if not t:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    co = [r for r in t["rows"] if r["kind"] == "coord"]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    fs.clean(ax)
    ks = [r["k"] for r in co]
    ax.plot(ks, [r["rom_solve_s"] * 1e3 for r in co], "o-", color=fs.C["blue"],
            label="ROM latent solve (total)")
    ax.plot(ks, [r["rom_s_per_iter"] * 1e3 for r in co], "s--", color=fs.C["aqua"],
            label="per Gauss-Newton iteration")
    ax.axhline(t["fom_cg_s"] * 1e3, color=fs.C["red"], ls="-", lw=1.6, label="FOM (CG solve)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    # iteration counts as direct labels on the total-cost markers (no second scale)
    for r_ in co:
        ax.annotate(f"{r_['rom_iters_mean']:.0f} it", (r_["k"], r_["rom_solve_s"] * 1e3),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5,
                    color=fs.INK2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("latent dimension k"); ax.set_ylabel("wall time (ms)")
    ax.set_title("Poisson 2D — online cost vs latent dimension", loc="left")
    ax.legend(loc="center left")
    fig.tight_layout()
    return fs.save(fig, FIGS, "poisson_cost_vs_k", (PLOTS,))


def main():
    D = gather()
    out = ["# Poisson-2D follow-up — tables (generated by `followup/fu_summarize.py`)", ""]
    tab_kladder(D, out)
    tab_seeds(D, out)
    tab_ladders(D, out)
    tab_timing(D, out)
    tab_timing_k(D, out)
    tab_family(D, out)
    p = os.path.join(HERE, "FOLLOWUP_TABLES.md")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", p)
    for f in (fig_kladder, fig_timing, fig_ladders, fig_family, fig_cost_vs_k):
        try:
            r = f(D)
            print("figure:", r)
        except Exception as e:                    # a missing cell must not kill the rest
            print(f"  figure {f.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
