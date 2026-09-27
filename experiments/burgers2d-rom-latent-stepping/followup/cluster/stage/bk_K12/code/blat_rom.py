"""Stage 2: latent-stepping ROM evaluation on HELD-OUT Burgers-2D trajectories.

Inputs : blat_ad_N{N}_K{K}.pkl (auto-decoder + POD basis) from blat_train_ad.py;
         data regenerated from seed (fingerprint asserted equal).
The ROM knows: the initial condition u0, the viscosity nu, the PDE.  It never
sees the held-out trajectory (oracle floors are computed separately and
labelled).

Per variant "solver:colloc:objective" (env VARIANTS, comma-separated):
  solver    lspg | galerkin
  colloc    full | rand<m> | biased<m> | offgrid<m>
  objective fd | lowpass<sigma> | ihelm<K>        (non-fd need colloc=full)
POD control: POD_KS (default 8,16,32,64) x POD_VARIANTS.

Reports per variant: trajectory rel-L2 (mean over the 51 slices) mean/median/
max over N_TEST trajectories, per-time mean curve, iterations (cold step 0 vs
warm), termination reasons, blow-ups, per-step wall time; ROM-vs-FOM timing
(warm-up 2, median of 7, block_until_ready, same device, FOM = the same
jitted implicit solver at batch 1).

Usage: N=64 K_LAT=8 [N_TEST=16] [VARIANTS=...] python blat_rom.py <ad.pkl> <outdir>
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

import blat_common as bc
from blat_common import F64, log, lm_solve

AD_PKL = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
N = bc.N
T1 = bc.NUM_STEPS + 1
# solver:colloc:objective.  objective weak<M> = weak-form Galerkin with the FOM's
# upwind advection (colloc full | eq<m> NNLS-EQ grid nodes); weakc<M> = continuum
# weak form (colloc full | eq<m> | eqoff<m> meshfree pool); fd = strong FD residual
# control (full | rand<m> | biased<m> | offgrid<m>).
DEFAULT_VARIANTS = ("lspg:full:fd,galerkin:full:fd,lspg:rand512:fd,lspg:offgrid512:fd,"
                    "lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64,"
                    "lspg:full:weak256,lspg:eq512:weak256,lspg:eq1024:weak256,"
                    "galerkin:full:weak64,"
                    "lspg:full:weakc64,lspg:eq512:weakc64,lspg:eqoff512:weakc64")
VARIANTS = [v for v in os.environ.get("VARIANTS", DEFAULT_VARIANTS).split(",") if v]
POD_KS = [int(k) for k in os.environ.get("POD_KS", "8,16,32,64").split(",") if k]
POD_VARIANTS = [v for v in os.environ.get(
    "POD_VARIANTS", "lspg:full:fd,galerkin:full:fd,lspg:full:weak64,lspg:eq512:weak64").split(",") if v]
EQ_RNG_SEED = 4321
FLOOR_BUDGET = int(os.environ.get("FLOOR_BUDGET", "60"))
DO_TIMING = int(os.environ.get("DO_TIMING", "1"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))


def summarize(runs):
    """Accuracy statistics are over COMPLETED rollouts only (n_completed of
    n_total); blow-ups are counted separately and NOT averaged in."""
    ok = [r for r in runs if r["n_done"] == bc.NUM_STEPS]
    tr = np.array([r["traj_rel"] for r in ok]) if ok else np.array([np.nan])
    it = np.array([r["iters"] for r in ok], dtype=float)
    at = np.array([r["attempts"] for r in ok], dtype=float)
    per = np.array([r["per_time"] for r in runs])
    reasons = {}
    for r in runs:
        for s_ in r["reasons"]:
            reasons[s_] = reasons.get(s_, 0) + 1
    out = dict(n_total=len(runs), n_completed=len(ok),
               n_blowup=int(sum(r["n_done"] < bc.NUM_STEPS for r in runs)),
               traj_rel_mean=float(np.mean(tr)), traj_rel_median=float(np.median(tr)),
               traj_rel_max=float(np.max(tr)),
               per_time_mean=np.nanmean(per, axis=0).tolist(),
               per_time_survivors=np.sum(np.isfinite(per), axis=0).tolist(),
               iters_cold_step0=float(it[:, 0].mean()) if it.size else float("nan"),
               iters_warm_mean=float(it[:, 1:].mean()) if it.size else float("nan"),
               iters_warm_max=float(it[:, 1:].max()) if it.size else float("nan"),
               attempts_cold_step0=float(at[:, 0].mean()) if at.size else float("nan"),
               attempts_warm_mean=float(at[:, 1:].mean()) if at.size else float("nan"),
               res_step_mean=float(np.mean([np.mean(r["res"]) for r in runs])),
               res_final_mean=float(np.mean([r["res"][-1] for r in runs])),
               reasons=reasons,
               step_time_ms_median=float(1e3 * np.median(np.concatenate(
                   [r["step_time"] for r in runs]))),
               ic_misfit_mean=float(np.mean([r["ic_rel"] for r in runs])),
               ic_init_used={n: sum(r["ic_init"] == n for r in runs)
                             for n in set(r["ic_init"] for r in runs)})
    return out


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  variants={VARIANTS}")
    with open(AD_PKL, "rb") as f:
        ck = pickle.load(f)
    K = ck["k_lat"]
    for key_, val_ in (("bc_mode", bc.BC_MODE), ("N", N), ("ad_hidden", bc.AD_HIDDEN),
                       ("ad_layers", bc.AD_LAYERS), ("n_train", bc.N_TRAIN), ("seed", bc.SEED)):
        if ck["config"][key_] != val_:
            raise SystemExit(f"checkpoint/config mismatch on {key_}: {ck['config'][key_]} vs {val_}")
    dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], K)
    Ztr = ck["Z_train"]                                   # (n_tr, T1, K)
    V = ck["V"]
    if max(POD_KS) > V.shape[1]:
        raise SystemExit(f"POD_KS {POD_KS} exceeds stored basis rank {V.shape[1]}")
    d = bc.build_data(N)
    fp = bc.data_fingerprint(d["U"])
    fp_rel = max(abs(fp["sum"] - ck["data_fingerprint"]["sum"]) / abs(ck["data_fingerprint"]["sum"]),
                 abs(fp["sumsq"] - ck["data_fingerprint"]["sumsq"]) / ck["data_fingerprint"]["sumsq"])
    if fp_rel > 1e-6:
        raise SystemExit(f"data fingerprint mismatch ({fp_rel:.2e}): {fp} vs {ck['data_fingerprint']}")
    if fp_rel > 1e-12:
        log(f"  WARNING: data fingerprint differs by {fp_rel:.2e} (cross-machine rounding?)")
    U = d["U"]
    U_te = d["U_test"]                                     # fresh TEST_SEED trajectories
    nu_te = d["nu_test"]
    U_tr = U[:bc.N_TRAIN]
    interior_np = bc.interior_indices(N)
    u0_rms = np.sqrt(np.mean(U_te[:, 0][:, interior_np] ** 2, axis=1))   # tolerance scales
    n2 = N * N
    coords = jnp.asarray(bc.grid_coords(N))
    interior = bc.interior_indices(N)
    report = dict(config=dict(bc.CONFIG, variants=VARIANTS, pod_ks=POD_KS,
                              pod_variants=POD_VARIANTS, floor_budget=FLOOR_BUDGET,
                              ad_pkl=os.path.basename(AD_PKL), ad_config=ck["config"]),
                  backend=jax.default_backend(), data_fingerprint=fp,
                  test_seed=bc.TEST_SEED, max_fom_rel_residual=d["max_fom_rel_residual"],
                  train_rel_mean=ck["train_rel_mean"],
                  oracle_pod_projection_floor_val=ck["pod_floors"])

    # ---------------- checks: point-local residual == FOM residual ----------------
    _, fom_res = bc.bf.make_rollout(N)
    st = jnp.asarray(bc.stencil_indices(interior, N))
    u1, u0 = jnp.asarray(U_te[0, 1]), jnp.asarray(U_te[0, 0])
    r_local = bc.be_residual_from_stencil(u1[st], u0[jnp.asarray(interior)],
                                          float(nu_te[0]), N)
    r_fom = fom_res(u1, u0, float(nu_te[0]))[jnp.asarray(interior)]
    chk = dict(local_vs_fom_maxabs=float(jnp.max(jnp.abs(r_local - r_fom))),
               fom_traj_step1_rel_res=float(jnp.linalg.norm(r_fom) / jnp.linalg.norm(u0)))
    log(f"  checks: {chk}")
    report["checks"] = chk

    # ---------------- floors on the test trajectories ----------------
    # (a) POD projection floors (linear ceiling of the control)
    Ute = U_te.reshape(-1, n2)
    pf = {}
    for k in POD_KS:
        rec = (Ute @ V[:, :k]) @ V[:, :k].T
        pf[k] = float(np.mean(np.linalg.norm(rec - Ute, axis=1) / np.linalg.norm(Ute, axis=1)))
    report["oracle_pod_projection_floor_test"] = pf
    # (b) ORACLE per-snapshot inferred latents (LM on the data misfit to the
    #     held-out field; NOT available to the ROM) -- floor 2 of the Poisson study
    zmean_t = Ztr.mean(axis=0)                            # (T1, K)
    f_mis = lambda z, u: dec(z, coords) - u
    rJ_mis = jax.jit(lambda z, u: (f_mis(z, u), jax.jacfwd(f_mis)(z, u)))
    rn_mis = jax.jit(lambda z, u: jnp.linalg.norm(f_mis(z, u)))
    Utr_flat = U_tr.reshape(-1, n2)
    Utr_sq = np.sum(Utr_flat ** 2, axis=1)
    t0 = time.time()
    orc = np.zeros((bc.N_TEST, T1)); orc_Z = np.zeros((bc.N_TEST, T1, K))
    for i in range(bc.N_TEST):
        for n in range(T1):
            u = jnp.asarray(U_te[i, n])
            # nearest TRAIN snapshot by field distance (oracle: uses held-out field)
            j = int(np.argmin(Utr_sq - 2.0 * (Utr_flat @ U_te[i, n])))   # argmin ||u_j - u||
            best = None
            for name, z0 in (("mean", zmean_t[n]), ("nearest", Ztr.reshape(-1, K)[j])):
                z, r, info = lm_solve(lambda zz: rJ_mis(zz, u), lambda zz: rn_mis(zz, u),
                                      jnp.asarray(z0), FLOOR_BUDGET)
                rel = r / float(jnp.linalg.norm(u))
                if best is None or rel < best[0]:
                    best = (rel, np.asarray(z))
            orc[i, n], orc_Z[i, n] = best
    report["oracle_inferred_latent_test"] = dict(
        traj_rel_mean=float(orc.mean()), per_time_mean=orc.mean(axis=0).tolist(),
        secs=time.time() - t0)
    log(f"  ORACLE inferred-latent floor (held-out, budget {FLOOR_BUDGET}): "
        f"{orc.mean():.3e} [{time.time()-t0:.0f}s]")

    # ---------------- cold starts from the KNOWN u0 ----------------
    Utr0 = U_tr[:, 0]
    ics = []
    for i in range(bc.N_TEST):
        j = int(np.argmin(np.linalg.norm(Utr0 - U_te[i, 0], axis=1)))   # legit: u0 known
        z0, rel, info = bc.fit_ic(dec, N, U_te[i, 0],
                                  {"mean_t0": zmean_t[0], "nearest_ic": Ztr[j, 0]},
                                  coords=coords)
        ics.append((np.asarray(z0), rel, info.get("init", "?")))
    report["ic_fit"] = dict(rel_mean=float(np.mean([r for _, r, _ in ics])),
                            rel_max=float(np.max([r for _, r, _ in ics])))
    log(f"  IC fit (u0 misfit): mean {report['ic_fit']['rel_mean']:.3e}")

    # ---------------- ROM variants (coord decoder) ----------------
    eq_cache = {}
    Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
        Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)]

    def build_ops(decoder, var, i, cache):
        solver, colloc_name, objective = var.split(":")
        if objective.startswith("weak"):
            kind = "weakc" if objective.startswith("weakc") else "weak"
            M = int(objective[len(kind):])
            if colloc_name == "full":
                col = dict(kind="grid", idx=interior, w=None)
            else:
                pool = "off" if colloc_name.startswith("eqoff") else "grid"
                m = int(colloc_name[5:] if pool == "off" else colloc_name[2:])
                key = (decoder.kind, decoder.k, kind, M, m, pool)
                if key not in cache:
                    zs = Z_snap if decoder.kind == "coord" else (
                        (U_tr.reshape(-1, n2) @ np.asarray(decoder.V))[
                            np.random.default_rng(EQ_RNG_SEED).choice(
                                Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)])
                    cache[key] = bc.fit_eq_weights(decoder, N, M, m, zs, kind=kind, pool=pool,
                                                   rng=np.random.default_rng(EQ_RNG_SEED))
                col = cache[key]
            return bc.make_weak_ops(decoder, N, col, kind=kind, M=M, solver=solver)
        col = bc.make_collocation(colloc_name, N, np.random.default_rng(1234 + i),
                                  data_row=dict(u0=U_te[i, 0]))
        return bc.make_step_ops(decoder, N, col, objective, solver)

    results = {}
    timing = {}
    for var in VARIANTS:
        solver, colloc_name, objective = var.split(":")
        runs = []
        t0 = time.time()
        ops = None
        for i in range(bc.N_TEST):
            if colloc_name.startswith("biased") or ops is None:
                ops = build_ops(dec, var, i, eq_cache)
            z0, ic_rel, ic_init = ics[i]
            r = bc.rollout(dec, N, ops, z0, float(nu_te[i]), float(u0_rms[i]), U_true=U_te[i])
            r["ic_rel"], r["ic_init"] = ic_rel, ic_init
            del r["fields"]
            runs.append(r)
        s = summarize(runs)
        s["m"] = int(ops["m"]); s["secs"] = time.time() - t0
        s["eq_info"] = ops.get("colloc_info")
        results[var] = s
        log(f"  {var:24s} m={ops['m']:5d}  traj rel mean {s['traj_rel_mean']:.3e} "
            f"(med {s['traj_rel_median']:.3e}, max {s['traj_rel_max']:.3e}) "
            f"blowups {s['n_blowup']}  iters cold {s['iters_cold_step0']:.1f} "
            f"warm {s['iters_warm_mean']:.2f}  step {s['step_time_ms_median']:.1f} ms  "
            f"[{s['secs']:.0f}s]")
        if DO_TIMING:
            i = 0
            if colloc_name.startswith("biased"):     # rebuild for test case 0
                ops = build_ops(dec, var, 0, eq_cache)
            z0 = jnp.asarray(ics[i][0]); us0 = float(u0_rms[i])
            if solver == "lspg":
                usc = jnp.full((bc.NUM_STEPS,), bc.GN_TOL * us0 * np.sqrt(ops["m"]))
                def run_once():
                    Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, float(nu_te[i]), usc, bc.GN_BUDGET)
                    Z_.block_until_ready()
                Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, float(nu_te[i]), usc, bc.GN_BUDGET)
                nj_total = int(jnp.sum(nj_))
                impl = "device_scan"
            else:
                def run_once():
                    bc.rollout(dec, N, ops, z0, float(nu_te[i]), us0)
                nj_total = None
                impl = "python_loop"
            med, ts = bc.time_fn(run_once, reps=TIME_REPS, warm=2)
            # online IC solve (python LM on the u0 misfit) and full-field decode of
            # all 51 slices, timed separately so the reader can compose them
            j0 = int(np.argmin(np.linalg.norm(Utr0 - U_te[i, 0], axis=1)))
            def ic_once():
                bc.fit_ic(dec, N, U_te[i, 0], {"mean_t0": zmean_t[0], "nearest_ic": Ztr[j0, 0]},
                          coords=coords)
            ic_med, _ = bc.time_fn(ic_once, reps=3, warm=1)
            Zt = jnp.asarray(np.stack([ics[i][0]] * (bc.NUM_STEPS + 1)))
            dec_all = jax.jit(lambda ZZ: jax.vmap(lambda zz: dec(zz, coords))(ZZ))
            def dec_once():
                dec_all(Zt).block_until_ready()
            dec_med, _ = bc.time_fn(dec_once, reps=TIME_REPS, warm=2)
            timing[var] = dict(rollout_s_median=med, all=ts, impl=impl, iters_total=nj_total,
                               ic_fit_s=ic_med, decode_all_slices_s=dec_med)
    report["rom"] = results

    # ---------------- POD control (same solver) ----------------
    podres = {}
    for k in POD_KS:
        pdec = bc.PODDecoder(V[:, :k])
        for var in POD_VARIANTS:
            solver, colloc_name, objective = var.split(":")
            ops = build_ops(pdec, var, 0, eq_cache)
            runs = []
            t0 = time.time()
            for i in range(bc.N_TEST):
                z0, ic_rel, _ = bc.fit_ic(pdec, N, U_te[i, 0], {})
                r = bc.rollout(pdec, N, ops, z0, float(nu_te[i]), float(u0_rms[i]), U_true=U_te[i])
                r["ic_rel"], r["ic_init"] = ic_rel, "projection"
                del r["fields"]
                runs.append(r)
            s = summarize(runs); s["m"] = int(ops["m"]); s["secs"] = time.time() - t0
            s["eq_info"] = ops.get("colloc_info")
            podres[f"k{k}:{var}"] = s
            log(f"  POD k={k:3d} {var:20s} traj rel mean {s['traj_rel_mean']:.3e} "
                f"(med {s['traj_rel_median']:.3e}) blowups {s['n_blowup']} "
                f"iters warm {s['iters_warm_mean']:.2f} step {s['step_time_ms_median']:.1f} ms")
            if DO_TIMING and var == POD_VARIANTS[0]:
                z0 = jnp.asarray(bc.fit_ic(pdec, N, U_te[0, 0], {})[0])
                usc = jnp.full((bc.NUM_STEPS,), bc.GN_TOL * float(u0_rms[0]) * np.sqrt(ops["m"]))
                def pod_once():
                    Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, float(nu_te[0]), usc, bc.GN_BUDGET)
                    Z_.block_until_ready()
                med, ts = bc.time_fn(pod_once, reps=TIME_REPS, warm=2)
                timing[f"pod_k{k}:{var}"] = dict(rollout_s_median=med, all=ts, impl="device_scan")
    report["pod_rom"] = podres

    # ---------------- FOM timing (same jitted solver, batch 1) ----------------
    if DO_TIMING:
        roll, _ = bc.bf.make_rollout(N)
        U0 = jnp.asarray(U_te[0, 0])[None]
        nu1 = jnp.asarray([nu_te[0]])
        def fom_once():
            s, r = roll(U0, nu1)
            s.block_until_ready()
        med, ts = bc.time_fn(fom_once, reps=TIME_REPS, warm=2)
        timing["fom_rollout"] = dict(rollout_s_median=med, all=ts)
        for kk, v in timing.items():
            if kk != "fom_rollout":
                v["speedup_vs_fom_rollout_only"] = med / v["rollout_s_median"]
                if "ic_fit_s" in v:
                    v["speedup_vs_fom_end_to_end"] = med / (
                        v["rollout_s_median"] + v["ic_fit_s"] + v["decode_all_slices_s"])
        log("  timing: " + "  ".join(f"{k}={v['rollout_s_median']*1e3:.0f}ms"
                                    for k, v in timing.items()))
    report["timing"] = timing

    os.makedirs(OUTDIR, exist_ok=True)
    ts = ck["config"].get("train_seed", bc.SEED)
    tag = f"N{N}_K{K}" + (f"_S{ts}" if ts != bc.SEED else "")
    with open(os.path.join(OUTDIR, f"blat_rom_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"wrote blat_rom_{tag}.json")


if __name__ == "__main__":
    main()
