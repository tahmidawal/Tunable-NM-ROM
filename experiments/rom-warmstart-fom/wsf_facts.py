"""Every number quoted in README.md, derived from the JSONs.

The previous round's results audit re-derived 1311 generated table cells and found
no error in any of them -- but 19 errors in HAND-WRITTEN PROSE.  So the prose here
is not hand-written either: `README.md` is rendered from `README.tmpl.md` by
substituting the facts this module computes, and `wsf_render_readme.py` fails if
the template references a fact that does not exist.

Usage: python wsf_facts.py            # print every fact
       from wsf_facts import facts    # dict of name -> string
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wsf_summarize import load_reports, flat_points, split_roles, select_consolidated, breakeven  # noqa: E402


def g(x, spec=".3g"):
    return "--" if x is None else format(x, spec)


def build(runs=None):
    global RUNS
    import wsf_summarize as ws
    if runs:
        ws.RUNS = runs
    reports, skipped = ws.load_reports()
    pts = ws.flat_points(reports)
    P = [r for r in pts if r["pde"] == "poisson2d"]
    B = [r for r in pts if r["pde"] == "burgers2d"]
    pkey = lambda r: (r["N"], r["rom_tau"], r["fom_tau"])
    bkey = lambda r: (r["N"], r["fom_tau"])
    Pc, Pprov = ws.select_consolidated(P, pkey)
    Bc, Bprov = ws.select_consolidated(B, bkey)
    _, Pm = ws.split_roles(P, pkey)
    _, Bm = ws.split_roles(B, bkey)
    f = {}
    f["n_reports"] = str(len(reports))
    f["n_rows"] = str(len(pts))
    f["n_skipped"] = str(len(skipped))

    # ---------------- Poisson ----------------
    if Pc:
        Ns = sorted({r["N"] for r in Pc})
        fts = sorted({r["fom_tau"] for r in Pc}, reverse=True)
        f["p_meshes"] = ", ".join(str(n) for n in Ns)
        f["p_fom_taus"] = ", ".join(f"{t:g}" for t in fts)
        f["p_gpu"] = str(Pprov["gpu"])
        f["p_job"] = str(Pprov["slurm_job_id"])
        # per-tolerance best hybrid and crossover
        for ft in fts:
            tag = f"{ft:g}".replace("-", "m").replace("+", "").replace(".", "")
            best = {}
            for n in Ns:
                sub = [r for r in Pc if r["N"] == n and r["fom_tau"] == ft]
                if sub:
                    best[n] = min(sub, key=lambda r: r["t_total_ms"])
            wins = [n for n, b in best.items() if b["speedup_vs_fom"] > 1.0]
            f[f"p_cross_{tag}"] = str(min(wins)) if wins else "none in the ladder"
            for n, b in best.items():
                f[f"p_best_{tag}_N{n}"] = g(b["speedup_vs_fom"], ".3f")
                f[f"p_besttau_{tag}_N{n}"] = ("ref. stops" if not b["rom_tau"]
                                              else f"{b['rom_tau']:g}")
                f[f"p_total_{tag}_N{n}"] = g(b["t_total_ms"], ".4g")
                f[f"p_fom_{tag}_N{n}"] = g(b["t_fom_baseline_ms"], ".4g")
        # per-mesh structural quantities (at the tightest tolerance)
        ftm = min(fts)
        tag = f"{ftm:g}".replace("-", "m").replace("+", "").replace(".", "")
        for n in Ns:
            sub = [r for r in Pc if r["N"] == n and r["fom_tau"] == ftm]
            if not sub:
                continue
            r0 = sub[0]
            f[f"p_direct_N{n}"] = g(r0.get("t_fom_direct_ms"), ".3g")
            f[f"p_directerr_N{n}"] = g(r0.get("direct_rel_err"), ".2e")
            f[f"p_cgdirect_N{n}"] = g(r0["t_fom_baseline_ms"] / r0["t_fom_direct_ms"], ".0f")
            f[f"p_decode_N{n}"] = g(r0["t_decode_ms"], ".3g")
            f[f"p_dof_N{n}"] = str(r0["n_dof"])
            f[f"p_ref_res_N{n}"] = g(r0.get("reference_true_rel_residual"), ".2e")
            bb = max(sub, key=lambda r: r["iter_saving_frac"])
            f[f"p_maxsave_{tag}_N{n}"] = g(100 * bb["iter_saving_frac"], ".3g")
        # saving fraction vs tolerance, at the largest mesh
        nmax = Ns[-1]
        for ft in fts:
            tg = f"{ft:g}".replace("-", "m").replace("+", "").replace(".", "")
            sub = [r for r in Pc if r["N"] == nmax and r["fom_tau"] == ft]
            if sub:
                bb = max(sub, key=lambda r: r["iter_saving_frac"])
                f[f"p_maxsave_{tg}_Nmax"] = g(100 * bb["iter_saving_frac"], ".3g")
        f["p_nmax"] = str(nmax)
        # ROM quality at the converged end
        conv = [r for r in Pc if r["rom_tau"] == 0 and r["fom_tau"] == ftm]
        if conv:
            c0 = conv[0]
            f["p_rom_err"] = g(np.mean([r["err_rel_l2_rom"] for r in conv]), ".3g")
            f["p_rom_resid"] = g(np.mean([r["rom_rel_residual"] for r in conv]), ".3g")
            f["p_rom_anorm"] = g(np.mean([r["rom_err_Anorm_ratio"] for r in conv]), ".3g")
        for n in Ns:
            s2 = [r for r in Pc if r["N"] == n and r["rom_tau"] == 0 and r["fom_tau"] == ftm]
            if s2:
                f[f"p_worth_N{n}"] = g(s2[0].get("cg_iters_equivalent_to_rom"), ".0f")
                f[f"p_baseiters_N{n}"] = g(s2[0]["iters_from_baseline"], ".0f")
        # negative-saving evidence
        neg = [r for r in Pc if r.get("iter_saving_frac", 0) < 0]
        f["p_n_negative"] = str(len(neg))
        if neg:
            w = min(neg, key=lambda r: r["iter_saving_frac"])
            f["p_worst_negative"] = g(100 * w["iter_saving_frac"], ".3g")
            f["p_worst_negative_where"] = (f"N={w['N']}, rom_tau={w['rom_tau']:g}, "
                                           f"tau_FOM={w['fom_tau']:g}")
        be = [b for b in (ws.breakeven(Pc, ft) for ft in fts) if b]
        for b in be:
            tg = f"{b['fom_tau']:g}".replace("-", "m").replace("+", "").replace(".", "")
            f[f"p_break_{tg}"] = ("already wins" if b.get("already_wins")
                                  else g(b.get("breakeven_N"), ".0f"))
            f[f"p_exponent_{tg}"] = g(b.get("exponent"), ".2f")
        # health
        f["p_max_final_resid_ratio"] = g(max(
            r["final_rel_residual"] / r["fom_tau"] for r in Pc), ".3g")

    # ---------------- Burgers ----------------
    if Bc:
        Ns = sorted({r["N"] for r in Bc})
        fts = sorted({r["fom_tau"] for r in Bc}, reverse=True)
        f["b_meshes"] = ", ".join(str(n) for n in Ns)
        f["b_gpu"] = str(Bprov["gpu"])
        f["b_job"] = str(Bprov["slurm_job_id"])
        f["b_variant"] = str(Bc[0].get("variant"))
        f["b_m"] = str(Bc[0].get("m"))
        for r in Bc:
            tg = f"{r['fom_tau']:g}".replace("-", "m").replace("+", "").replace(".", "")
            k = f"{tg}_N{r['N']}"
            f[f"b_speed_{k}"] = g(r["speedup_vs_fom"], ".3f")
            f[f"b_total_{k}"] = g(r["t_total_ms"], ".4g")
            f[f"b_fom_{k}"] = g(r["t_fom_baseline_ms"], ".4g")
            f[f"b_fomwarm_{k}"] = g(r["t_fom_ms"], ".4g")
            f[f"b_fomextrap_{k}"] = g(r["t_fom_extrap_ms"], ".4g")
            f[f"b_testbed_{k}"] = g(r.get("t_fom_testbed_ms"), ".4g")
            f[f"b_newt_prev_{k}"] = g(r["iters_from_baseline"], ".1f")
            f[f"b_newt_rom_{k}"] = g(r["iters_from_rom"], ".1f")
            f[f"b_newt_ext_{k}"] = g(r["iters_from_extrap"], ".1f")
            f[f"b_lin_prev_{k}"] = g(r["lin_iters_from_baseline"], ".0f")
            f[f"b_lin_rom_{k}"] = g(r["lin_iters_from_rom"], ".0f")
            f[f"b_lin_ext_{k}"] = g(r["lin_iters_from_extrap"], ".0f")
            f[f"b_rom_{k}"] = g(r["t_rom_ms"], ".4g")
            f[f"b_dec_{k}"] = g(r["t_decode_ms"], ".4g")
            f[f"b_err_{k}"] = g(r["err_rel_l2_rom"], ".3g")
        f["b_breakdowns"] = str(sum(r.get("bicgstab_breakdowns", 0) for r in Bc))
        f["b_flags"] = str(sum(r.get("newton_flags_nonzero", 0) for r in Bc))
        f["b_warnings"] = str(sum(1 for r in Bc if r.get("health_warning")))
        f["b_nmax"] = str(max(Ns))
        # how often the ROM start beat the previous-step start
        wins = [r for r in Bc if r["iters_from_rom"] < r["iters_from_baseline"]]
        f["b_newton_wins"] = f"{len(wins)} of {len(Bc)}"
        lwins = [r for r in Bc if r["lin_iters_from_rom"] < r["lin_iters_from_baseline"]]
        f["b_lin_wins"] = f"{len(lwins)} of {len(Bc)}"
        twins = [r for r in Bc if r["speedup_vs_fom"] > 1.0]
        f["b_time_wins"] = f"{len(twins)} of {len(Bc)}"
        ewins = [r for r in Bc if r["iters_from_extrap"] < r["iters_from_baseline"]]
        f["b_extrap_wins"] = f"{len(ewins)} of {len(Bc)}"
        f["b_max_resid_ratio"] = g(max(
            max(r["max_rel_newton_residual"].values()) / r["fom_tau"] for r in Bc
            if isinstance(r.get("max_rel_newton_residual"), dict)), ".3g")

    # ---------------- consistency ----------------
    from wsf_summarize import role_consistency
    cc = ws.role_consistency(P, pkey, ["iters_from_rom", "iters_from_baseline",
                                       "err_rel_l2_rom"])
    cb = ws.role_consistency(B, bkey, ["iters_from_rom", "iters_from_baseline",
                                       "iters_from_extrap"])
    f["consistency_checked"] = str(len(cc) + len(cb))
    bad = [c for c in cc + cb
           if c["abs_diff"] > 1e-9 * max(abs(c["consolidated"]), 1.0)]
    f["consistency_bad"] = str(len(bad))
    return f


facts = None


if __name__ == "__main__":
    d = build(sys.argv[1] if len(sys.argv) > 1 else None)
    for k in sorted(d):
        print(f"{k} = {d[k]}")
