#!/usr/bin/env python3
"""Round-3 summary tables FROM THE RUN JSONS ONLY (never hand-typed).

    python runs/summarize_r3.py runs/push_r3*/out/*.json > runs/SUMMARY-R3.md

Every number printed here is read out of a committed run JSON or the committed
POD-floor diagnostic; nothing is computed from memory or typed in.
"""
import json
import os
import sys

import numpy as np

POD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pod_floor", "pod_floor_N256_burgers.json")


def nm(s):
    """Arm names contain '|', which would break a markdown table cell."""
    return s.replace("|", "/")


def f(x, p=3):
    return "--" if x is None else f"{x:.{p}e}"


def pod_floors():
    if not os.path.isfile(POD):
        return {}, {}
    d = json.load(open(POD))["burgers"]
    return d["family_floor"], d["test_states"]


def arms(d):
    return [r for r in d["rows"] if r.get("method", "").startswith(("cach", "mesh"))]


def champ_row(d):
    want = "cach|ctrl|ic_enc|roll_extrap|t1e-9"
    for r in arms(d):
        if r["method"] == want:
            return r
    return min(arms(d), key=lambda r: r["err_traj_rel_mean"])


def base_rows(d):
    return [r for r in d["rows"] if r.get("method") == "fom_newton_tol_pc"]


def per_time_stats(per_time, t_early):
    a = np.asarray(per_time, dtype=float)
    return dict(t0=float(a[0]),
                early=float(np.mean(a[1:t_early + 1])) if len(a) > t_early else None,
                late=float(np.mean(a[t_early + 1:])) if len(a) > t_early + 1 else None,
                mx=float(np.max(a)), end=float(a[-1]))


def block(path, d, ff, tf):
    c, t = d["config"], d.get("train", {})
    ar = c.get("arch_overrides", {})
    N, R, K = c["N"], c["r"], c["k"]
    te = c.get("t_early", 5)
    print(f"## `{os.path.basename(path)}` -- N={N} K={K} R={R} "
          f"g_hidden={ar.get('g_hidden', 128)} h_hidden={ar.get('h_hidden', 128)} "
          f"n_ff={ar.get('n_ff', 64)} "
          f"ff={'multi ' + str(ar['ff_scales']) if ar.get('ff_scales') else 'single-scale'} "
          f"snap_norm={c.get('snap_norm')}\n")
    print(f"- job `{c.get('slurm_job')}` on {c.get('gpu')} ({c.get('node')}), "
          f"backend `{c.get('backend')}`, matmul `{c.get('matmul_precision')}`, "
          f"complete={d.get('complete')}, {d.get('total_seconds', 0)/3600:.2f} h")
    print(f"- training: {t.get('steps_done')}/{t.get('steps')} steps "
          f"({t.get('seconds', 0)/3600:.2f} h, time_capped={t.get('time_capped')}, "
          f"used_ema={t.get('used_ema')})")
    da = d.get("data", {})
    print(f"- data: {da.get('n_traj')} trajectories, {da.get('n_states_trained')} "
          f"training states of {da.get('n_states_total')} "
          f"({da.get('n_early_states_in_pick')} with t<={te}); point pool "
          f"{da.get('n_pool')}/{da.get('n_i2')} "
          f"(full interior: {da.get('pool_is_full_interior')}); worst FOM rel "
          f"residual {f(da.get('max_fom_rel_residual'))} train / "
          f"{f(da.get('max_fom_rel_residual_test'))} test\n")

    print("### Gates\n")
    print("| gate | bar | value |")
    print("|---|---|---|")
    for k, v in d.get("eq", {}).items():
        print(f"| gate 0, EQ set `{k}` (cached vs incumbent meshfree) | <=1e-12 "
              f"| {f(v.get('gate0'))} |")
    for k, v in d.get("gates", {}).items():
        bar = "<=1e-12" if "bank" in k else "<1e-6"
        print(f"| {k} | {bar} | {f(v)} |")
    print(f"| IC solver jit vs incumbent `fit_ic` | <=1e-9 | "
          f"{f(d.get('ic', {}).get('jit_vs_incumbent_rel_dev'))} |")
    devs = [r["timed_vs_untimed_max_latent_dev"]
            for a in arms(d) for r in a["per_traj"]]
    if devs:
        print(f"| timed-call vs untimed latent deviation (max over arms/traj) "
              f"| -- | {f(max(devs))} |")
    for b in d.get("batched", []):
        if "batched_vs_single_max_pertime_dev" in b:
            print(f"| batched vs single-query per-step error, `{nm(b['subject'])}` "
                  f"| -- | {f(b['batched_vs_single_max_pertime_dev'])} |")
    print()

    sp = d.get("span")
    if sp:
        print("### Span (DIAGNOSTIC -- SVD/least-squares never enter the model)\n")
        svs = sp["gram_sv_ratio"]
        print("f64-Gram spectrum of the trained bank: "
              + "  ".join(f"`sv[{i}]/sv0={float(v):.2e}`"
                          for i, v in sorted(svs.items(), key=lambda kv: int(kv[0]))))
        print(f"\n- numerical rank (>1e-8): **{sp['numerical_rank_1e8']} / {R}** "
              f"(g_hidden={sp['g_hidden']}; the g-track's last layer is linear, so "
              f"the bank rank cannot exceed g_hidden+1)")
        print(f"- unconstrained R-coefficient least-squares floor of the LEARNED "
              f"span on fresh test states: **{f(sp.get('ls_floor_mean'))}**")
        if str(R) in tf:
            print(f"- rank-{R} POD floor of the training family (separate diagnostic, "
                  f"its own seed-777 cohort): {f(tf[str(R)]['mean'])} mean / "
                  f"{f(tf[str(R)]['max'])} worst state; family floor "
                  f"{f(ff.get(str(R)))}")
        print()

    cr = champ_row(d)
    orc = d.get("oracle", [])
    o_mean = float(np.mean([o["mean"] for o in orc])) if orc else None
    o_max = float(np.max([o["max"] for o in orc])) if orc else None
    o_t0 = float(np.mean([o["t0"] for o in orc])) if orc else None
    print("### The error ladder\n")
    print("| rung | mean | worst |")
    print("|---|---|---|")
    print(f"| training reconstruction (on the training point pool) | "
          f"{f(t.get('recon_rel_l2_mean'))} | {f(t.get('recon_rel_l2_max'))} |")
    if "recon_fullgrid_subset_mean" in t:
        print(f"| the same states re-checked on the FULL interior "
              f"({t.get('recon_fullgrid_subset_n')} states) | "
              f"{f(t['recon_fullgrid_subset_mean'])} | "
              f"{f(t['recon_fullgrid_subset_max'])} |")
    if sp:
        print(f"| learned-span LS floor, fresh test states (bound on any h) | "
              f"{f(sp.get('ls_floor_mean'))} | -- |")
    print(f"| per-state representation oracle, fresh test trajectories | "
          f"{f(o_mean)} | {f(o_max)} |")
    for name, rows in d.get("single_step_weak_opt", {}).items():
        rat = [r["err"] / r["oracle_err"] for r in rows if r["oracle_err"] > 0]
        print(f"| single-step weak-EQ optimum / oracle, EQ set `{name}` "
              f"({len(rows)} probes) | {np.mean(rat):.4f}x | {np.max(rat):.4f}x |")
    print(f"| solver output, 50-step trajectory (`{nm(cr['method'])}`) | "
          f"{f(cr['err_traj_rel_mean'])} | {f(cr['err_traj_rel_max'])} |")
    print()
    if orc:
        print("Per-step structure (mean over fresh test trajectories):\n")
        print("| curve | t=0 | mean 1<=t<=%d | mean t>%d | max | t=50 |" % (te, te))
        print("|---|---|---|---|---|---|")
        for lbl, curves in (("representation oracle",
                             [o["per_time"] for o in orc]),
                            (f"solver ({nm(cr['method'])})",
                             [r["per_time"] for r in cr["per_traj"]])):
            st = [per_time_stats(cv, te) for cv in curves]
            g = lambda k: f(float(np.mean([s[k] for s in st])) if st[0][k] is not None else None)
            print(f"| {lbl} | {g('t0')} | {g('early')} | {g('late')} | "
                  f"{f(float(np.max([s['mx'] for s in st])))} | {g('end')} |")
        print()

    print("### ROM arms (end-to-end: IC latent fit + 50 implicit steps + full-grid decode)\n")
    print("| arm | e2e ms (median) | traj err mean | traj err max | IC rel mean | "
          "jacobians | stop reasons | blowups |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(arms(d), key=lambda r: r["e2e_ms_median"]):
        print(f"| `{nm(r['method'])}` | {r['e2e_ms_median']:.1f} | "
              f"{f(r['err_traj_rel_mean'])} | {f(r['err_traj_rel_max'])} | "
              f"{f(r['ic_rel_mean'])} | {r['jac_total_mean']:.0f} | "
              f"{r['stop_reasons']} | {r['n_blowups']} |")
    sp_l = d.get("splits", [])
    if sp_l:
        print(f"\nSplit (median over {len(sp_l)} trajectories): IC fit "
              f"{np.median([s['ic_ctrl_ms'] for s in sp_l]):.2f} ms (candidate "
              f"search) / {np.median([s['ic_enc_ms'] for s in sp_l]):.2f} ms "
              f"(encoder init); full-grid decode of all 51 states "
              f"{np.median([s['decode_ms'] for s in sp_l]):.2f} ms.\n")

    print("### Classical baseline ladder (SAME job, SAME GPU)\n")
    print("Tolerance-terminated Newton, exact-Helmholtz-preconditioned BiCGStab, "
          "identical discrete residual to the truth generator.\n")
    print("| newton_tol | lin_tol | ms (median) | traj err mean | traj err max | "
          "Newton its / rollout | steps hitting tol |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(base_rows(d), key=lambda r: r["time_ms_median"]):
        print(f"| {r['newton_tol']:.0e} | {r['lin_tol']:.0e} | "
              f"{r['time_ms_median']:.1f} | {f(r['err_traj_rel_mean'])} | "
              f"{f(r['err_traj_rel_max'])} | {r['newton_iters_mean']:.1f} | "
              f"{100*r['steps_converged_frac']:.0f}% |")
    tg = [r for r in d["rows"] if r.get("method") == "fom_newton8_truthgen"]
    if tg:
        print(f"\n`fom_newton8_truthgen` (OVER-SOLVED, never a headline "
              f"comparator): {tg[0]['time_ms_median']:.0f} ms.\n")

    ma = d.get("matched_accuracy")
    if ma:
        print("### Matched accuracy\n")
        print(f"ROM arm `{nm(ma['rom_arm'])}`: {f(ma['rom_err'])} at "
              f"{ma['rom_e2e_ms']:.1f} ms.")
        if ma.get("matched"):
            m = ma["matched"]
            print(f"\nCheapest classical rung at least as accurate: "
                  f"newton_tol={m['newton_tol']:.0e}, lin_tol={m['lin_tol']:.0e} "
                  f"-> {m['ms']:.1f} ms at {f(m['err'])}.")
            p = ma.get("paired")
            if p:
                rr = p["base_ms"] / p["rom_ms"]
                print(f"\n**Paired AB/BA head-to-head** (order "
                      f"`{p['per_traj'][0]['order']}`): ROM {p['rom_ms']:.1f} ms "
                      f"vs classical {p['base_ms']:.1f} ms -> "
                      f"**{rr:.2f}x** ({'ROM wins' if rr > 1 else 'classical wins'}).")
        else:
            print("\nNo classical rung measured reaches the ROM's accuracy "
                  "(the ROM is the more accurate side).")
        print()

    if d.get("batched"):
        print(f"### Batched multi-query ({d['batched'][0]['n_queries']} queries "
              f"in one vmapped call)\n")
        print("| subject | total ms | amortized ms/query | traj err mean |")
        print("|---|---|---|---|")
        for b in sorted(d["batched"], key=lambda b: b["amortized_ms"]):
            print(f"| `{nm(b['subject'])}` | {b['total_ms_median']:.1f} | "
                  f"{b['amortized_ms']:.2f} | {f(b['err_traj_rel_mean'])} |")
        print()


def cross(runs, ff, tf):
    print("# Cross-run tables\n")
    print("## X1. Error ladder across the round-3 cells\n")
    print("| run | N | R | g_hidden | snap_norm | recon | span LS floor | "
          "oracle (fresh) | trajectory | POD floor at R |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p, d in runs:
        c, t = d["config"], d.get("train", {})
        orc = d.get("oracle", [])
        cr = champ_row(d)
        pf = tf.get(str(c["r"]), {}).get("mean")
        print(f"| `{os.path.basename(p)}` | {c['N']} | {c['r']} | "
              f"{c.get('arch_overrides', {}).get('g_hidden', 128)} | "
              f"{c.get('snap_norm')} | {f(t.get('recon_rel_l2_mean'))} | "
              f"{f(d.get('span', {}).get('ls_floor_mean'))} | "
              f"{f(float(np.mean([o['mean'] for o in orc])) if orc else None)} | "
              f"{f(cr['err_traj_rel_mean'])} | {f(pf)} |")
    print()
    print("## X2. Matched-accuracy Pareto (single query and batched)\n")
    print("| run | N | ROM err | ROM e2e ms | cheapest classical >= ROM accuracy | "
          "its ms | paired ROM/classical | batched ROM ms/query | "
          "batched classical ms/query |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p, d in runs:
        c = d["config"]
        ma = d.get("matched_accuracy", {})
        m, pr = ma.get("matched"), ma.get("paired")
        bt = d.get("batched", [])
        br = [b for b in bt if b["subject"].startswith("batched|cach")]
        bf_ = [b for b in bt if b["subject"].startswith("batched|fom")]
        print(f"| `{os.path.basename(p)}` | {c['N']} | {f(ma.get('rom_err'))} | "
              f"{ma.get('rom_e2e_ms', float('nan')):.1f} | "
              f"{('nt=%.0e lt=%.0e' % (m['newton_tol'], m['lin_tol'])) if m else 'none reaches it'} | "
              f"{('%.1f' % m['ms']) if m else '--'} | "
              f"{('%.2fx' % (pr['base_ms']/pr['rom_ms'])) if pr else '--'} | "
              f"{('%.2f' % min(b['amortized_ms'] for b in br)) if br else '--'} | "
              f"{('%.2f' % min(b['amortized_ms'] for b in bf_)) if bf_ else '--'} |")
    print()


def main(paths):
    ff, tf = pod_floors()
    runs = []
    for p in sorted(paths):
        d = json.load(open(p))
        if d.get("config", {}).get("round") != 3:
            continue
        runs.append((p, d))
    print("# Separable EQ-decoder, N=256 push -- ROUND 3\n")
    print("Generated by `runs/summarize_r3.py` from the committed run JSONs. "
          "Nothing here is hand-typed.\n")
    print("POD floors quoted below are the DIAGNOSTIC bound from "
          "`runs/pod_floor/pod_floor_N256_burgers.json` (its own fresh cohort, "
          "seed 777, N=256); they bound any architecture whose online field is "
          "a fixed spatial bank times latent coefficients. No POD enters any "
          "model.\n")
    if len(runs) > 1:
        cross(runs, ff, tf)
    print("# Per-run detail\n")
    for p, d in runs:
        block(p, d, ff, tf)


if __name__ == "__main__":
    main(sys.argv[1:])
