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
DEFAULT_VARIANTS = ("lspg:full:fd,galerkin:full:fd,lspg:rand256:fd,lspg:rand512:fd,"
                    "lspg:rand1024:fd,lspg:biased512:fd,lspg:offgrid512:fd,"
                    "lspg:full:lowpass2,lspg:full:ihelm20")
VARIANTS = [v for v in os.environ.get("VARIANTS", DEFAULT_VARIANTS).split(",") if v]
POD_KS = [int(k) for k in os.environ.get("POD_KS", "8,16,32,64").split(",") if k]
POD_VARIANTS = [v for v in os.environ.get(
    "POD_VARIANTS", "lspg:full:fd,galerkin:full:fd,lspg:rand512:fd").split(",") if v]
FLOOR_BUDGET = int(os.environ.get("FLOOR_BUDGET", "60"))
DO_TIMING = int(os.environ.get("DO_TIMING", "1"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))


def summarize(runs):
    tr = np.array([r["traj_rel"] for r in runs])
    it = np.array([r["iters"] for r in runs if r["n_done"] == bc.NUM_STEPS], dtype=float)
    per = np.array([r["per_time"] for r in runs])
    reasons = {}
    for r in runs:
        for s in r["reasons"]:
            reasons[s] = reasons.get(s, 0) + 1
    out = dict(traj_rel_mean=float(np.nanmean(tr)), traj_rel_median=float(np.nanmedian(tr)),
               traj_rel_max=float(np.nanmax(tr)),
               n_blowup=int(sum(r["n_done"] < bc.NUM_STEPS for r in runs)),
               per_time_mean=np.nanmean(per, axis=0).tolist(),
               iters_cold_step0=float(it[:, 0].mean()) if it.size else float("nan"),
               iters_warm_mean=float(it[:, 1:].mean()) if it.size else float("nan"),
               iters_warm_max=float(it[:, 1:].max()) if it.size else float("nan"),
               res_final_mean=float(np.mean([np.mean(r["res"]) for r in runs])),
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
    dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], K)
    Ztr = ck["Z_train"]                                   # (n_tr, T1, K)
    V = ck["V"]
    d = bc.build_data(N)
    fp = bc.data_fingerprint(d["U"])
    if not np.allclose([fp["sum"], fp["sumsq"]],
                       [ck["data_fingerprint"]["sum"], ck["data_fingerprint"]["sumsq"]],
                       rtol=1e-10):
        raise SystemExit(f"data fingerprint mismatch: {fp} vs {ck['data_fingerprint']}")
    U = d["U"]; nu_all = d["nu"]
    U_te = U[bc.N_TRAIN:bc.N_TRAIN + bc.N_TEST]
    nu_te = nu_all[bc.N_TRAIN:bc.N_TRAIN + bc.N_TEST]
    U_tr = U[:bc.N_TRAIN]
    n2 = N * N
    coords = jnp.asarray(bc.grid_coords(N))
    interior = bc.interior_indices(N)
    report = dict(config=dict(bc.CONFIG, variants=VARIANTS, pod_ks=POD_KS,
                              pod_variants=POD_VARIANTS, floor_budget=FLOOR_BUDGET,
                              ad_pkl=os.path.basename(AD_PKL), ad_config=ck["config"]),
                  backend=jax.default_backend(), data_fingerprint=fp,
                  train_rel_mean=ck["train_rel_mean"], pod_floors_val=ck["pod_floors"])

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
    report["pod_projection_floor_test"] = pf
    # (b) ORACLE per-snapshot inferred latents (LM on the data misfit to the
    #     held-out field; NOT available to the ROM) -- floor 2 of the Poisson study
    zmean_t = Ztr.mean(axis=0)                            # (T1, K)
    f_mis = lambda z, u: dec(z, coords) - u
    rJ_mis = jax.jit(lambda z, u: (f_mis(z, u), jax.jacfwd(f_mis)(z, u)))
    rn_mis = jax.jit(lambda z, u: jnp.linalg.norm(f_mis(z, u)))
    Utr_flat = U_tr.reshape(-1, n2)
    t0 = time.time()
    orc = np.zeros((bc.N_TEST, T1)); orc_Z = np.zeros((bc.N_TEST, T1, K))
    for i in range(bc.N_TEST):
        for n in range(T1):
            u = jnp.asarray(U_te[i, n])
            # nearest TRAIN snapshot by field distance (oracle: uses held-out field)
            j = int(np.argmin(np.linalg.norm(Utr_flat - U_te[i, n], axis=1)))
            best = None
            for name, z0 in (("mean", zmean_t[n]), ("nearest", Ztr.reshape(-1, K)[j])):
                z, r, info = lm_solve(lambda zz: rJ_mis(zz, u), lambda zz: rn_mis(zz, u),
                                      jnp.asarray(z0), FLOOR_BUDGET)
                rel = r / float(jnp.linalg.norm(u))
                if best is None or rel < best[0]:
                    best = (rel, np.asarray(z))
            orc[i, n], orc_Z[i, n] = best
    report["oracle_inferred_latent"] = dict(
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
    rng = np.random.default_rng(1234)
    results = {}
    timing = {}
    for var in VARIANTS:
        solver, colloc_name, objective = var.split(":")
        runs = []
        t0 = time.time()
        ops = None
        for i in range(bc.N_TEST):
            if colloc_name.startswith("biased") or ops is None:
                col = bc.make_collocation(colloc_name, N, np.random.default_rng(1234 + i),
                                          data_row=dict(u0=U_te[i, 0]))
                ops = bc.make_step_ops(dec, N, col, objective, solver)
            z0, ic_rel, ic_init = ics[i]
            r = bc.rollout(dec, N, ops, z0, float(nu_te[i]), U_true=U_te[i],
                           u_scale=float(np.linalg.norm(U_te[i, 0])))
            r["ic_rel"], r["ic_init"] = ic_rel, ic_init
            del r["fields"]
            runs.append(r)
        s = summarize(runs)
        s["m"] = int(ops["m"]); s["secs"] = time.time() - t0
        results[var] = s
        log(f"  {var:24s} m={ops['m']:5d}  traj rel mean {s['traj_rel_mean']:.3e} "
            f"(med {s['traj_rel_median']:.3e}, max {s['traj_rel_max']:.3e}) "
            f"blowups {s['n_blowup']}  iters cold {s['iters_cold_step0']:.1f} "
            f"warm {s['iters_warm_mean']:.2f}  step {s['step_time_ms_median']:.1f} ms  "
            f"[{s['secs']:.0f}s]")
        if DO_TIMING:
            i = 0
            z0 = jnp.asarray(ics[i][0]); us0 = float(np.linalg.norm(U_te[i, 0]))
            if solver == "lspg":
                usc = jnp.full((bc.NUM_STEPS,), us0)
                box = {}
                def run_once():
                    Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, float(nu_te[i]), usc, bc.GN_BUDGET)
                    Z_.block_until_ready(); box["nj"] = int(jnp.sum(nj_))
                impl = "device_scan"
            else:
                def run_once():
                    bc.rollout(dec, N, ops, z0, float(nu_te[i]), u_scale=us0)
                impl = "python_loop"
            med, ts = bc.time_fn(run_once, reps=TIME_REPS, warm=2)
            timing[var] = dict(rollout_s_median=med, all=ts, impl=impl,
                               iters_total=box.get("nj") if solver == "lspg" else None)
    report["rom"] = results

    # ---------------- POD control (same solver) ----------------
    podres = {}
    for k in POD_KS:
        pdec = bc.PODDecoder(V[:, :k])
        for var in POD_VARIANTS:
            solver, colloc_name, objective = var.split(":")
            col = bc.make_collocation(colloc_name, N, np.random.default_rng(1234))
            ops = bc.make_step_ops(pdec, N, col, objective, solver)
            runs = []
            t0 = time.time()
            for i in range(bc.N_TEST):
                z0, ic_rel, _ = bc.fit_ic(pdec, N, U_te[i, 0], {})
                r = bc.rollout(pdec, N, ops, z0, float(nu_te[i]), U_true=U_te[i],
                               u_scale=float(np.linalg.norm(U_te[i, 0])))
                r["ic_rel"], r["ic_init"] = ic_rel, "projection"
                del r["fields"]
                runs.append(r)
            s = summarize(runs); s["m"] = int(ops["m"]); s["secs"] = time.time() - t0
            podres[f"k{k}:{var}"] = s
            log(f"  POD k={k:3d} {var:20s} traj rel mean {s['traj_rel_mean']:.3e} "
                f"(med {s['traj_rel_median']:.3e}) blowups {s['n_blowup']} "
                f"iters warm {s['iters_warm_mean']:.2f} step {s['step_time_ms_median']:.1f} ms")
            if DO_TIMING and var == POD_VARIANTS[0]:
                z0 = jnp.asarray(bc.fit_ic(pdec, N, U_te[0, 0], {})[0])
                usc = jnp.full((bc.NUM_STEPS,), float(np.linalg.norm(U_te[0, 0])))
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
                v["speedup_vs_fom"] = med / v["rollout_s_median"]
        log("  timing: " + "  ".join(f"{k}={v['rollout_s_median']*1e3:.0f}ms"
                                    for k, v in timing.items()))
    report["timing"] = timing

    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"N{N}_K{K}"
    with open(os.path.join(OUTDIR, f"blat_rom_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"wrote blat_rom_{tag}.json")


if __name__ == "__main__":
    main()
