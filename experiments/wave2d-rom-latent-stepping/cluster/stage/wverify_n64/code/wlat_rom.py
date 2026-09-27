"""Stage 2: latent-stepping ROM evaluation on HELD-OUT Wave-2D trajectories.

Inputs : wlat_ad_N{N}_K{K}.pkl (auto-decoder + POD basis) from wlat_train_ad.py;
         data regenerated from seed (fingerprint asserted equal).
The ROM knows: the initial condition u0 (and u_t(.,0) = 0), the wave speed c,
the PDE.  It never sees the held-out trajectory (oracle floors are computed
separately and labelled).

Per variant "solver:colloc:objective" (env VARIANTS, comma-separated):
  solver    lspg | galerkin
  objective fd (strong Newmark residual; colloc full | rand<m> | biased<m> | offgrid<m>)
            weak<M> | weakl<M> (weak Galerkin, M sine test modes; weakl = extra
            lam^-1/2 weighting; colloc full | eq<m> grid NNLS | eqoff<m> meshfree NNLS)
POD control: POD_KS x POD_VARIANTS (same solver).

Reports per variant: traj-RMS error (PRIMARY) and per-snapshot error vs the
FOM (80-substep CN) AND vs the same-dt u-only Newmark FOM (isolates the ROM's
CN time-discretisation error at RS < 80), per-time curves, energy drift,
iterations (cold/warm), blow-ups, per-step wall time; ROM-vs-FOM timing
(warm-up 2, median of 7, block_until_ready, same device; FOM = wave2d CN/CG
rollout at 80 substeps, batch 1, jitted; also the same-dt Newmark FOM); the
JITTED IC solve timed separately.

Usage: N=64 K_LAT=8 [ROM_SUBSTEPS=20] [N_TEST=16] [VARIANTS=...] python wlat_rom.py <ad.pkl> <outdir> [tag]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import wlat_common as wc
from wlat_common import F64, log, lm_solve

AD_PKL = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
N = wc.N
T1 = wc.NUM_STEPS + 1
DEFAULT_VARIANTS = ("lspg:full:fd,galerkin:full:fd,lspg:rand512:fd,lspg:offgrid512:fd,"
                    "lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64,lspg:eqoff256:weak64,"
                    "lspg:full:weak144,lspg:eq576:weak144,"
                    "lspg:full:weak256,lspg:eq1024:weak256,"
                    "lspg:full:weakl64,galerkin:full:weak64")
VARIANTS = [v for v in os.environ.get("VARIANTS", DEFAULT_VARIANTS).split(",") if v]
POD_KS = [int(k) for k in os.environ.get("POD_KS", "6,8,16,32,64").split(",") if k]
POD_VARIANTS = [v for v in os.environ.get(
    "POD_VARIANTS", "lspg:full:fd,lspg:full:weak256,lspg:eq1024:weak256").split(",") if v]
EQ_RNG_SEED = 4321
FLOOR_BUDGET = int(os.environ.get("FLOOR_BUDGET", "60"))
DO_TIMING = int(os.environ.get("DO_TIMING", "1"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
DO_ENERGY = int(os.environ.get("DO_ENERGY", "1"))


def summarize(runs):
    ok = [r for r in runs if r["complete"]]
    tr = np.array([r["traj_rel"] for r in ok]) if ok else np.array([np.nan])
    sn = np.array([r["snap_rel"] for r in ok]) if ok else np.array([np.nan])
    trd = np.array([r["traj_rel_samedt"] for r in ok]) if ok else np.array([np.nan])
    per = np.array([r["per_time"] for r in runs])
    reasons = {}
    for r in runs:
        for k_, v_ in r["reasons"].items():
            reasons[k_] = reasons.get(k_, 0) + v_
    ed = np.array([r.get("energy_drift_max", np.nan) for r in ok]) if ok else np.array([np.nan])
    ef = np.array([r.get("energy_final_ratio", np.nan) for r in ok]) if ok else np.array([np.nan])
    return dict(n_total=len(runs), n_completed=len(ok), n_blowup=len(runs) - len(ok),
                traj_rel_mean=float(np.mean(tr)), traj_rel_median=float(np.median(tr)),
                traj_rel_max=float(np.max(tr)), snap_rel_mean=float(np.mean(sn)),
                traj_rel_vs_samedt_fom_mean=float(np.mean(trd)),
                per_time_mean=np.nanmean(per, axis=0).tolist(),
                per_time_survivors=np.sum(np.isfinite(per), axis=0).tolist(),
                energy_drift_max_mean=float(np.nanmean(ed)), energy_drift_max_max=float(np.nanmax(ed)),
                energy_final_ratio_mean=float(np.nanmean(ef)),
                iters_cold_step0=float(np.nanmean([r["iters_cold"] for r in runs])),
                iters_warm_mean=float(np.nanmean([r["iters_warm_mean"] for r in runs])),
                iters_warm_max=float(np.nanmax([r["iters_warm_max"] for r in runs])),
                res_step_mean=float(np.nanmean([np.mean(r["res"]) if r["res"] else np.nan for r in runs])),
                reasons=reasons,
                step_time_ms_median=float(np.median([r["step_time_ms"] for r in runs])),
                rollout_wall_s_median=float(np.median([r["wall_s"] for r in runs])),
                ic_misfit_mean=float(np.mean([r["ic_rel"] for r in runs])),
                n_done_min=int(min(r["n_done"] for r in runs)))


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  RS={wc.RS} dt={wc.DT:g}  variants={VARIANTS}")
    with open(AD_PKL, "rb") as f:
        ck = pickle.load(f)
    K = ck["k_lat"]
    for key_, val_ in (("bc_mode", wc.BC_MODE), ("N", N), ("ad_hidden", wc.AD_HIDDEN),
                       ("ad_layers", wc.AD_LAYERS), ("n_train", wc.N_TRAIN), ("seed", wc.SEED)):
        if ck["config"][key_] != val_:
            raise SystemExit(f"checkpoint/config mismatch on {key_}: {ck['config'][key_]} vs {val_}")
    dec = wc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]), ck["n_freq"], ck["eps"], K)
    Ztr = ck["Z_train"]                                   # (n_tr, T1, K)
    V = ck["V"]
    if max(POD_KS) > V.shape[1]:
        raise SystemExit(f"POD_KS {POD_KS} exceeds stored basis rank {V.shape[1]}")
    d = wc.build_data(N)
    fp = wc.data_fingerprint(d["U"])
    fp_rel = max(abs(fp["sum"] - ck["data_fingerprint"]["sum"]) / abs(ck["data_fingerprint"]["sum"]),
                 abs(fp["sumsq"] - ck["data_fingerprint"]["sumsq"]) / ck["data_fingerprint"]["sumsq"])
    if fp_rel > 1e-6:
        raise SystemExit(f"data fingerprint mismatch ({fp_rel:.2e})")
    U = d["U"]
    U_te, c_te = d["U_test"], d["c_test"]
    U_tr = U[:wc.N_TRAIN]
    interior = wc.interior_indices(N)
    u0_rms = np.sqrt(np.mean(U_te[:, 0][:, interior] ** 2, axis=1))
    n2 = N * N
    coords = jnp.asarray(wc.grid_coords(N))
    report = dict(config=dict(wc.CONFIG, variants=VARIANTS, pod_ks=POD_KS, pod_variants=POD_VARIANTS,
                              floor_budget=FLOOR_BUDGET, ad_pkl=os.path.basename(AD_PKL),
                              ad_config=ck["config"], tag=TAG),
                  backend=jax.default_backend(), data_fingerprint=fp, test_seed=wc.TEST_SEED,
                  test_energy_drift=d["test_energy_drift"], train_rel_mean=ck["train_rel_mean"],
                  train_traj_rel_mean=ck.get("train_traj_rel_mean"),
                  oracle_pod_projection_floor_val=ck["pod_floors"])

    # ---------------- checks ----------------
    chk = wc.verify_residual_ops(N, c=float(c_te[0]), M=32)
    log(f"  residual-operator checks (identity decoder on exact Newmark states): {chk}")
    if max(chk["strong_full"], chk["weak_full"]) > 1e-9 or chk["weak_nonsolution"] < 1e-12:
        raise SystemExit("residual operator check failed")
    report["checks"] = chk
    # same-dt Newmark FOM for every test trajectory (isolates the CN time error at RS)
    nm = wc.make_newmark_fom(N, wc.RS)
    U_te_dt = np.zeros_like(U_te)
    e_dt = []
    for i in range(wc.N_TEST):
        S, E = nm(jnp.asarray(U_te[i, 0]), float(c_te[i]))
        U_te_dt[i] = np.asarray(S); E = np.asarray(E)
        e_dt.append(float(np.max(np.abs(E - E[0]) / E[0])))
    td = [wc.traj_metrics(U_te_dt[i], U_te[i])[2] for i in range(wc.N_TEST)]
    report["samedt_fom"] = dict(rs=wc.RS, traj_rel_vs_fom_mean=float(np.mean(td)),
                                traj_rel_vs_fom_max=float(np.max(td)),
                                energy_drift_max=float(np.max(e_dt)))
    log(f"  same-dt Newmark FOM (RS={wc.RS}) vs 80-substep FOM: traj rel {np.mean(td):.3e} "
        f"(max {np.max(td):.3e}), energy drift {np.max(e_dt):.1e}")

    # ---------------- floors on the test trajectories ----------------
    Ute = U_te.reshape(-1, n2)
    rms_te = np.repeat(np.sqrt(np.mean(np.sum(U_te ** 2, axis=2), axis=1)), T1)
    pf = {}
    for k in POD_KS:
        rec = (Ute @ V[:, :k]) @ V[:, :k].T
        pf[k] = float(np.mean(np.linalg.norm(rec - Ute, axis=1) / rms_te))
    report["oracle_pod_projection_floor_test"] = pf
    log("  POD projection floors (test, traj metric): " + " ".join(f"k{k}={v:.3e}" for k, v in pf.items()))
    # ORACLE per-snapshot inferred latents (LM on the held-out field; NOT available to the ROM)
    zmean_t = Ztr.mean(axis=0)
    f_mis = lambda z, u: dec(z, coords) - u
    rJ_mis = jax.jit(lambda z, u: (f_mis(z, u), jax.jacfwd(f_mis)(z, u)))
    rn_mis = jax.jit(lambda z, u: jnp.linalg.norm(f_mis(z, u)))
    Utr_flat = U_tr.reshape(-1, n2)
    Utr_sq = np.sum(Utr_flat ** 2, axis=1)
    t0 = time.time()
    orc = np.zeros((wc.N_TEST, T1)); orc_s = np.zeros((wc.N_TEST, T1))
    for i in range(wc.N_TEST):
        rms_i = np.sqrt(np.mean(np.sum(U_te[i] ** 2, axis=1)))
        for n in range(T1):
            u = jnp.asarray(U_te[i, n])
            j = int(np.argmin(Utr_sq - 2.0 * (Utr_flat @ U_te[i, n])))
            best = None
            for z0 in (zmean_t[n], Ztr.reshape(-1, K)[j]):
                z, r, info = lm_solve(lambda zz: rJ_mis(zz, u), lambda zz: rn_mis(zz, u),
                                      jnp.asarray(z0), FLOOR_BUDGET)
                if best is None or r < best:
                    best = r
            orc[i, n] = best / rms_i
            orc_s[i, n] = best / max(float(jnp.linalg.norm(u)), 1e-300)
    report["oracle_inferred_latent_test"] = dict(
        traj_rel_mean=float(orc.mean()), snap_rel_mean=float(orc_s.mean()),
        per_time_mean=orc.mean(axis=0).tolist(), secs=time.time() - t0)
    log(f"  ORACLE inferred-latent floor (held-out, budget {FLOOR_BUDGET}): traj {orc.mean():.3e} "
        f"snap {orc_s.mean():.3e} [{time.time()-t0:.0f}s]")

    # ---------------- cold starts from the KNOWN u0 (jitted LM) ----------------
    ic_fit = wc.make_ic_solver(dec, N)
    Utr0 = U_tr[:, 0]
    ics = []
    for i in range(wc.N_TEST):
        j = int(np.argmin(np.linalg.norm(Utr0 - U_te[i, 0], axis=1)))
        z0, rel, which = ic_fit(jnp.asarray(U_te[i, 0]), jnp.asarray(np.stack([zmean_t[0], Ztr[j, 0]])))
        ics.append((np.asarray(z0), float(rel), int(which)))
    report["ic_fit"] = dict(rel_mean=float(np.mean([r for _, r, _ in ics])),
                            rel_max=float(np.max([r for _, r, _ in ics])),
                            init_used={"mean_t0": sum(w == 0 for _, _, w in ics),
                                       "nearest_ic": sum(w == 1 for _, _, w in ics)})
    log(f"  IC fit (u0 misfit, jitted LM): mean {report['ic_fit']['rel_mean']:.3e}")

    # ---------------- ROM variants ----------------
    eq_cache = {}
    Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
        Ztr.shape[0] * T1, wc.EQ_SNAPS, replace=False)]

    def build_ops(decoder, var, i, cache):
        solver, colloc_name, objective = var.split(":")
        if objective.startswith("weak"):
            beta = 1.0 if objective.startswith("weakl") else wc.WEAK_BETA
            M = int(objective[5:] if objective.startswith("weakl") else objective[4:])
            if colloc_name == "full":
                col = dict(kind="grid", idx=interior, w=None)
            else:
                pool = "off" if colloc_name.startswith("eqoff") else "grid"
                m = int(colloc_name[5:] if pool == "off" else colloc_name[2:])
                key = (decoder.kind, decoder.k, M, m, pool)
                if key not in cache:
                    zs = Z_snap if decoder.kind == "coord" else (
                        (U_tr.reshape(-1, n2) @ np.asarray(decoder.V))[
                            np.random.default_rng(EQ_RNG_SEED).choice(Ztr.shape[0] * T1, wc.EQ_SNAPS, replace=False)])
                    cache[key] = wc.fit_eq_weights(decoder, N, M, m, zs, pool=pool,
                                                   rng=np.random.default_rng(EQ_RNG_SEED))
                col = cache[key]
            return wc.make_weak_ops(decoder, N, col, M=M, solver=solver, beta=beta)
        col = wc.make_collocation(colloc_name, N, np.random.default_rng(1234 + i), u0=U_te[i, 0])
        return wc.make_strong_ops(decoder, N, col, solver=solver)

    def run_variants(decoder, variants, ics_, label, res_dict, timing, time_key):
        for var in variants:
            solver, colloc_name, objective = var.split(":")
            runs = []
            t0 = time.time()
            ops = None
            for i in range(wc.N_TEST):
                if colloc_name.startswith("biased") or ops is None:
                    ops = build_ops(decoder, var, i, eq_cache)
                z0, ic_rel, _ = ics_[i]
                r = wc.rollout(decoder, N, ops, z0, float(c_te[i]), float(u0_rms[i]), U_true=U_te[i],
                               energies=bool(DO_ENERGY))
                r["ic_rel"] = ic_rel
                # error vs the same-dt Newmark FOM (isolates the manifold/solver error)
                if r["complete"]:
                    r["traj_rel_samedt"] = wc.traj_metrics(r["fields"], U_te_dt[i])[2]
                else:
                    r["traj_rel_samedt"] = float("nan")
                del r["fields"]; r.pop("Z_snap", None); r.pop("energy", None)
                runs.append(r)
            s = summarize(runs); s["m"] = int(ops["m"]); s["M"] = ops.get("M"); s["secs"] = time.time() - t0
            s["eq_info"] = ops.get("colloc_info")
            res_dict[f"{label}{var}"] = s
            log(f"  {label}{var:24s} m={ops['m']:5d}  traj rel {s['traj_rel_mean']:.3e} "
                f"(med {s['traj_rel_median']:.3e}, max {s['traj_rel_max']:.3e}; vs same-dt FOM "
                f"{s['traj_rel_vs_samedt_fom_mean']:.3e}) blowups {s['n_blowup']}  E-drift "
                f"{s['energy_drift_max_mean']:.2e}  iters cold {s['iters_cold_step0']:.1f} warm "
                f"{s['iters_warm_mean']:.2f}  {s['step_time_ms_median']:.2f} ms/step  [{s['secs']:.0f}s]")
            if DO_TIMING and solver == "lspg" and (time_key is None or var == time_key):
                i = 0
                if colloc_name.startswith("biased"):
                    ops = build_ops(decoder, var, 0, eq_cache)
                z0 = jnp.asarray(ics_[0][0])
                tol_abs = wc.GN_TOL * float(u0_rms[0]) * ops["tol_scale"]
                n_steps = wc.NUM_STEPS * wc.RS
                def run_once():
                    Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, float(c_te[0]), tol_abs, wc.GN_BUDGET, n_steps)
                    Z_.block_until_ready()
                Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, float(c_te[0]), tol_abs, wc.GN_BUDGET, n_steps)
                nj_total = int(jnp.sum(nj_))
                med, ts = wc.time_fn(run_once, reps=TIME_REPS, warm=2)
                ent = dict(rollout_s_median=med, all=ts, iters_total=nj_total, n_steps=n_steps,
                           ms_per_step=1e3 * med / n_steps)
                if decoder.kind == "coord":
                    j0 = int(np.argmin(np.linalg.norm(Utr0 - U_te[0, 0], axis=1)))
                    zi = jnp.asarray(np.stack([zmean_t[0], Ztr[j0, 0]]))
                    u00 = jnp.asarray(U_te[0, 0])
                    def ic_once():
                        ic_fit(u00, zi)[0].block_until_ready()
                    ent["ic_fit_s"], _ = wc.time_fn(ic_once, reps=TIME_REPS, warm=2)
                    Zt = jnp.asarray(np.stack([ics_[0][0]] * T1))
                    def dec_once():
                        ops["full_batch"](Zt).block_until_ready()
                    ent["decode_all_slices_s"], _ = wc.time_fn(dec_once, reps=TIME_REPS, warm=2)
                timing[f"{label}{var}"] = ent

    results, timing = {}, {}
    run_variants(dec, VARIANTS, ics, "", results, timing, None)
    report["rom"] = results

    # ---------------- POD control (same solver) ----------------
    podres = {}
    for k in POD_KS:
        pdec = wc.PODDecoder(V[:, :k])
        pfit = wc.make_ic_solver(pdec, N)
        pics = []
        for i in range(wc.N_TEST):
            z0, rel, _ = pfit(jnp.asarray(U_te[i, 0]), jnp.zeros((1, k)))
            pics.append((np.asarray(z0), float(rel), -1))
        run_variants(pdec, POD_VARIANTS, pics, f"pod_k{k}:", podres, timing, POD_VARIANTS[0])
    report["pod_rom"] = podres

    # ---------------- FOM timing ----------------
    if DO_TIMING:
        roll, _ = wc.wf.make_rollout(N)
        U0 = jnp.asarray(U_te[0, 0])[None]
        c1 = jnp.asarray([c_te[0]])
        def fom_once():
            s, e = roll(U0, c1)
            s.block_until_ready()
        med, ts = wc.time_fn(fom_once, reps=TIME_REPS, warm=2)
        timing["fom_rollout_cn80"] = dict(rollout_s_median=med, all=ts)
        u00 = jnp.asarray(U_te[0, 0])
        def nm_once():
            s, e = nm(u00, float(c_te[0]))
            s.block_until_ready()
        med2, ts2 = wc.time_fn(nm_once, reps=TIME_REPS, warm=2)
        timing[f"fom_rollout_newmark_rs{wc.RS}"] = dict(rollout_s_median=med2, all=ts2)
        for kk, v in timing.items():
            if not kk.startswith("fom_"):
                v["speedup_vs_fom_rollout_only"] = med / v["rollout_s_median"]
                v["speedup_vs_samedt_fom_rollout_only"] = med2 / v["rollout_s_median"]
                if "ic_fit_s" in v:
                    v["speedup_vs_fom_end_to_end"] = med / (
                        v["rollout_s_median"] + v["ic_fit_s"] + v["decode_all_slices_s"])
        log("  timing: " + "  ".join(f"{k}={v['rollout_s_median']*1e3:.0f}ms" for k, v in timing.items()))
    report["timing"] = timing

    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"N{N}_K{K}" + (f"_{TAG}" if TAG else "")
    with open(os.path.join(OUTDIR, f"wlat_rom_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"wrote wlat_rom_{tag}.json")


if __name__ == "__main__":
    main()
