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
        # ---- generated verdict sentences: a hand-written "the hybrid only just wins"
        # would silently become false if the data changed, so the claim itself is derived.
        ftm = min(fts)
        tagm = f"{ftm:g}".replace("-", "m").replace("+", "").replace(".", "")
        crossed = [ft for ft in fts
                   if f[f"p_cross_{f'{ft:g}'.replace('-', 'm').replace('+', '').replace('.', '')}"]
                   != "none in the ladder"]
        bestall = max(Pc, key=lambda r: r["speedup_vs_fom"])
        f["p_best_overall"] = g(bestall["speedup_vs_fom"], ".3f")
        f["p_best_overall_where"] = (f"N={bestall['N']}, rom_tau="
                                     + ("ref. stops" if not bestall["rom_tau"]
                                        else f"{bestall['rom_tau']:g}")
                                     + f", tau_FOM={bestall['fom_tau']:g}")
        if crossed:
            cn = min(int(f[f"p_cross_{f'{ft:g}'.replace('-', 'm').replace('+', '').replace('.', '')}"])
                     for ft in crossed)
            f["p_headline"] = (f"The hybrid first breaks even at **N = {cn}**, and the best "
                               f"speedup anywhere in the ladder is "
                               f"**{f['p_best_overall']}x** ({f['p_best_overall_where']}). "
                               f"So an FOM-exact hybrid pays only just, and only at the "
                               f"largest meshes tested.")
            f["p_verdict"] = (f"Poisson breaks even only at N = {cn} and only by "
                              f"{f['p_best_overall']}x")
            f["p_crossover"] = str(cn)
        else:
            f["p_headline"] = ("The hybrid **never breaks even** anywhere in the ladder: "
                               f"its best result is {f['p_best_overall']}x "
                               f"({f['p_best_overall_where']}), i.e. it is always slower "
                               f"than simply running the FOM.")
            f["p_verdict"] = (f"Poisson never breaks even (best {f['p_best_overall']}x)")
            f["p_crossover"] = "none in the ladder"

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
        f["b_lin_total_configs"] = str(len(Bc))
        twins = [r for r in Bc if r["speedup_vs_fom"] > 1.0]
        f["b_time_wins"] = f"{len(twins)} of {len(Bc)}"
        ewins = [r for r in Bc if r["iters_from_extrap"] < r["iters_from_baseline"]]
        f["b_extrap_wins"] = f"{len(ewins)} of {len(Bc)}"
        elin = [r for r in Bc if r["lin_iters_from_extrap"] < r["lin_iters_from_baseline"]]
        f["b_extrap_lin_wins"] = f"{len(elin)} of {len(Bc)}"
        # The FOM STAGE alone (excluding the ROM stage) beating the previous-step start
        # is a different, weaker claim than the hybrid TOTAL winning.  Report it, and
        # name the exceptions rather than averaging them away.
        fs = [r for r in Bc if r["t_fom_ms"] < r["t_fom_baseline_ms"]]
        f["b_fomstage_wins"] = f"{len(fs)} of {len(Bc)}"
        f["b_fomstage_where"] = ("; ".join(f"N={r['N']}, tau_FOM={r['fom_tau']:g} "
                                           f"({r['t_fom_ms']:.1f} vs "
                                           f"{r['t_fom_baseline_ms']:.1f} ms)" for r in fs)
                                 if fs else "none")
        nb = len(Bc)
        nw = len([r for r in Bc if r["speedup_vs_fom"] > 1.0])
        ni = len([r for r in Bc if r["iters_from_rom"] < r["iters_from_baseline"]])
        nl = len([r for r in Bc if r["lin_iters_from_rom"] < r["lin_iters_from_baseline"]])
        if nw == 0:
            f["b_headline"] = ("The ROM warm start **loses in every configuration measured**: "
                               f"it beat the previous-step start on Newton iterations in "
                               f"{ni} of {nb}, on inner BiCGStab iterations in {nl} of {nb}, "
                               f"and on wall clock in 0 of {nb}.")
            f["b_verdict"] = "Burgers loses outright, to a baseline the FOM already had for free"
        else:
            f["b_headline"] = (f"The ROM warm start beats the previous-step start on wall "
                               f"clock in {nw} of {nb} configurations (Newton iterations "
                               f"{ni} of {nb}, inner BiCGStab iterations {nl} of {nb}).")
            f["b_verdict"] = f"Burgers wins in {nw} of {nb} configurations"
        f["b_max_resid_ratio"] = g(max(
            max(r["max_rel_newton_residual"].values()) / r["fom_tau"] for r in Bc
            if isinstance(r.get("max_rel_newton_residual"), dict)), ".3g")

    # ---------------- OVER-CONVERGENCE AUDIT ----------------
    # Both of this project's previously reported speedup families divide by a FOM
    # baseline that was run far past any stated tolerance.  Correct the record by
    # combining the archived timing JSONs (read here, not retyped) with the
    # tolerance-based baselines measured in this cell.
    OLD_B = os.path.join(HERE, "..", "burgers2d-rom-latent-stepping",
                         "runs", "followup", "bt_n", "timing_n.json")
    OLD_P = os.path.join(HERE, "..", "poisson2d-rom-objective",
                         "runs", "followup", "pt_n", "timing_n.json")
    tg = lambda t: f"{t:g}".replace("-", "m").replace("+", "").replace(".", "")

    if Bc and os.path.isfile(OLD_B):
        old = json.load(open(OLD_B))
        f["oc_b_old_json"] = os.path.relpath(OLD_B, HERE)
        f["oc_b_newton_fixed"] = g(Bc[0].get("fom_testbed_newton_iters"), ".0f")
        for r in Bc:
            k = f"{tg(r['fom_tau'])}_N{r['N']}"
            f[f"oc_b_resid_N{r['N']}"] = g(r.get("fom_testbed_rel_newton_residual"), ".2e")
            # Is this rung ACCURACY-MATCHED to the archived baseline, or merely the
            # tightest rung measured?  Two different over-convergence questions hang on
            # this and they give different multipliers (see the README).
            mr = r.get("max_rel_newton_residual")
            ach = mr.get("prev") if isinstance(mr, dict) else None
            arch = r.get("fom_testbed_rel_newton_residual")
            f[f"oc_b_ladder_ach_{k}"] = g(ach, ".2e")
            if ach and arch:
                f[f"oc_b_looser_{k}"] = g(ach / arch, ".0f")
                f[f"oc_b_matched_{k}"] = "yes" if ach <= arch else "no"
            f[f"oc_b_t_fixed_N{r['N']}"] = g(r.get("t_fom_testbed_ms"), ".4g")
            f[f"oc_b_t_tol_{k}"] = g(r["t_fom_baseline_ms"], ".4g")
            f[f"oc_b_newton_tol_{k}"] = g(r["iters_from_baseline"], ".0f")
            fac = r.get("overconvergence_factor")
            f[f"oc_b_factor_{k}"] = g(fac, ".2f")
            f[f"oc_b_mult_{k}"] = g(1.0 / fac if fac else None, ".3f")
            # HARDWARE-FREE multiplier: the ratio of Newton steps actually performed.
            # The time-based multiplier is a ratio measured on ONE gpu; this one is a
            # pure work count and carries across machines.
            nf = r.get("fom_testbed_newton_iters")
            f[f"oc_b_multit_{k}"] = g(r["iters_from_baseline"] / nf if nf else None, ".3f")
        for orow in old["rows"]:
            v = orow["rom"].get("lspg:eq256:weak64")
            if not v:
                continue
            n = orow["N"]
            f[f"oc_b_old_speed_N{n}"] = g(v["speedup_rollout_only"], ".2f")
            e = v.get("rollout_from_eq_start") or {}
            if e.get("speedup_end_to_end_no_decode"):
                f[f"oc_b_old_e2e_N{n}"] = g(e["speedup_end_to_end_no_decode"], ".2f")
            f[f"oc_b_old_fom_N{n}"] = g(orow["fom_rollout_s"] * 1e3, ".4g")
            # INSTRUMENT VALIDATION: this cell re-times the archived baseline function
            # itself, so the two should agree.  If they do, the correction below is
            # anchored to the same denominator the archived speedups actually used.
            mine = [r.get("t_fom_testbed_ms") for r in Bc if r["N"] == n]
            if mine and mine[0]:
                f[f"oc_b_archcheck_N{n}"] = g(
                    100.0 * abs(mine[0] / (orow["fom_rollout_s"] * 1e3) - 1.0), ".2g")
            for r in Bc:
                if r["N"] != n:
                    continue
                k = f"{tg(r['fom_tau'])}_N{n}"
                fac = r.get("overconvergence_factor")
                if fac:
                    f[f"oc_b_new_speed_{k}"] = g(v["speedup_rollout_only"] / fac, ".2f")
                    if e.get("speedup_end_to_end_no_decode"):
                        f[f"oc_b_new_e2e_{k}"] = g(
                            e["speedup_end_to_end_no_decode"] / fac, ".2f")

    if Pc and os.path.isfile(OLD_P):
        oldp = json.load(open(OLD_P))
        f["oc_p_old_json"] = os.path.relpath(OLD_P, HERE)
        f["oc_p_cg_tol"] = g(Pc[0].get("fom_testbed_cg_tol"), ".0e")
        for r in Pc:
            k = f"{tg(r['fom_tau'])}_N{r['N']}"
            f[f"oc_p_t_fixed_N{r['N']}"] = g(r.get("t_fom_testbed_ms"), ".4g")
            f[f"oc_p_iters_fixed_N{r['N']}"] = g(r.get("fom_testbed_iters"), ".0f")
            f[f"oc_p_resid_N{r['N']}"] = g(r.get("fom_testbed_true_rel_res"), ".2e")
            ach = r.get("final_rel_residual_baseline")
            arch = r.get("fom_testbed_true_rel_res")
            f[f"oc_p_ladder_ach_{k}"] = g(ach, ".2e")
            if ach and arch:
                f[f"oc_p_looser_{k}"] = g(ach / arch, ".0f")
                f[f"oc_p_matched_{k}"] = "yes" if ach <= arch else "no"
            f[f"oc_p_t_tol_{k}"] = g(r["t_fom_baseline_ms"], ".4g")
            f[f"oc_p_t_tol_native_{k}"] = g(r.get("t_fom_baseline_native_ms"), ".4g")
            f[f"oc_p_iters_tol_{k}"] = g(r["iters_from_baseline"], ".0f")
            # LIKE-FOR-LIKE time factor: the testbed baseline is jax.scipy CG, so it must
            # be compared against jax.scipy CG at the looser tolerance, NOT against this
            # cell's counting CG -- otherwise the ratio conflates "tighter tolerance" with
            # "different solver implementation" (the two differ by ~15% per iteration).
            tn = r.get("t_fom_baseline_native_ms")
            tb = r.get("t_fom_testbed_ms")
            facn = (tb / tn) if (tb and tn) else None
            f[f"oc_p_factor_{k}"] = g(facn, ".2f")
            f[f"oc_p_mult_{k}"] = g(1.0 / facn if facn else None, ".3f")
            f[f"oc_p_factor_mixed_{k}"] = g(r.get("overconvergence_factor"), ".2f")
            nfi = r.get("fom_testbed_iters")
            f[f"oc_p_multit_{k}"] = g(r["iters_from_baseline"] / nfi if nfi else None, ".3f")
        for orow in oldp["rows"]:
            n = orow["N"]
            f[f"oc_p_old_speed_N{n}"] = g(orow["speedup_solve_only"], ".2f")
            f[f"oc_p_old_e2e_N{n}"] = g(orow["speedup_end_to_end"], ".2f")
            f[f"oc_p_old_fom_N{n}"] = g(orow["fom_cg_s"] * 1e3, ".4g")
            mine = [r.get("t_fom_testbed_ms") for r in Pc if r["N"] == n]
            if mine and mine[0]:
                f[f"oc_p_archcheck_N{n}"] = g(
                    100.0 * abs(mine[0] / (orow["fom_cg_s"] * 1e3) - 1.0), ".2g")
            for r in Pc:
                if r["N"] != n:
                    continue
                k = f"{tg(r['fom_tau'])}_N{n}"
                tn = r.get("t_fom_baseline_native_ms"); tb = r.get("t_fom_testbed_ms")
                facn = (tb / tn) if (tb and tn) else None
                if facn:
                    f[f"oc_p_new_speed_{k}"] = g(orow["speedup_solve_only"] / facn, ".2f")
                    f[f"oc_p_new_e2e_{k}"] = g(orow["speedup_end_to_end"] / facn, ".2f")

    # Does ANY rung in this cell's ladder reach what the archived baselines achieved?
    for tagp, C, key in (("p", Pc, "final_rel_residual_baseline"),
                         ("b", Bc, None)):
        if not C:
            continue
        anym = False
        for r in C:
            if tagp == "p":
                ach, arch = r.get(key), r.get("fom_testbed_true_rel_res")
            else:
                mr = r.get("max_rel_newton_residual")
                ach = mr.get("prev") if isinstance(mr, dict) else None
                arch = r.get("fom_testbed_rel_newton_residual")
            if ach and arch and ach <= arch:
                anym = True
        f[f"oc_{tagp}_any_matched_rung"] = "yes" if anym else "no"
        f[f"oc_{tagp}_factor_kind"] = ("accuracy-matched" if anym else "engineering")

    # EXTERNALLY REPORTED, relayed from the cost-to-tolerance cell (agent
    # a25b45872e6e0bec4) via the coordinator -- NOT measured here.  Kept in the fact
    # table so it carries its provenance wherever it is quoted.
    f["xagent_p_arch_N128"] = "15.145"
    f["xagent_p_mine_N128"] = "14.86"
    f["xagent_p_theirs_N128"] = "14.795"
    f["xagent_p_spread_N128"] = "2.3"

    # ---------------- consistency ----------------
    from wsf_summarize import role_consistency
    cc = ws.role_consistency(P, pkey, ["iters_from_rom", "iters_from_baseline",
                                       "err_rel_l2_rom"])
    cb = ws.role_consistency(B, bkey, ["iters_from_rom", "iters_from_baseline",
                                       "iters_from_extrap"])
    f["consistency_checked"] = str(len(cc) + len(cb))
    allc = cc + cb
    if allc:
        w = max(allc, key=lambda c: c["abs_diff"])
        f["consistency_worst_diff"] = g(w["abs_diff"], ".3g")
        f["consistency_worst_where"] = f"{w['field']} at {w['key']}"
    bad = [c for c in cc + cb
           if c["abs_diff"] > 1e-9 * max(abs(c["consolidated"]), 1.0)]
    f["consistency_bad"] = str(len(bad))
    return f


facts = None


if __name__ == "__main__":
    d = build(sys.argv[1] if len(sys.argv) > 1 else None)
    for k in sorted(d):
        print(f"{k} = {d[k]}")
