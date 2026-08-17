"""Per-interval ERROR-INJECTION diagnostic: does the wave ROM fail because a
single latent step is inaccurate, or because a small per-step error accumulates
undamped over the horizon?

The Stage-2 rollout starts at the IC and runs NUM_STEPS*RS latent steps, so its
error mixes (a) how much error one step injects and (b) how that error grows.
This script separates them.  For each test trajectory and each of several START
snapshots n0 it:

  1. builds the EXACT u-only Newmark sub-step trajectory at the ROM's own dt
     (`newmark_substeps`), so the two-level warm start is exact;
  2. fits ORACLE latents to the exact fields at sub-steps k0-1 and k0 (LM on the
     full grid -- this uses the held-out trajectory and is therefore labelled an
     ORACLE diagnostic, never a ROM);
  3. runs H*RS latent steps from that oracle start with the ROM's own operator;
  4. reports, at each of the H snapshot horizons, the error of the ROM state
     against the FOM AND against the oracle latent fit of the same FOM state.

The difference between (4) at H=1 and the oracle floor is the per-interval
INJECTION; its growth with H is the ACCUMULATION law.  A `hold` control repeats
the measurement with the ROM replaced by "do nothing" (freeze the latent), which
bounds how much of the H-step error is simply the wave moving on.

Usage: N=64 K_LAT=8 [ROM_SUBSTEPS=20] [SD_STARTS=0,10,20,30,40] [SD_H=1,2,5,10]
       [SD_NTEST=8] [SD_VARIANTS=lspg:eq256:weak64,lspg:full:fd]
       python wlat_stepdiag.py <ad.pkl> <outdir> [tag]
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
RS = wc.RS
STARTS = [int(s) for s in os.environ.get("SD_STARTS", "0,10,20,30,40").split(",") if s]
HS = [int(s) for s in os.environ.get("SD_H", "1,2,5,10").split(",") if s]
NTEST = int(os.environ.get("SD_NTEST", "8"))
VARIANTS = [v for v in os.environ.get(
    "SD_VARIANTS", "lspg:eq256:weak64,lspg:full:weak64,lspg:full:fd,galerkin:full:fd").split(",") if v]
FIT_BUDGET = int(os.environ.get("SD_FIT_BUDGET", "80"))
EQ_RNG_SEED = 4321


def main():
    log(f"jax_backend={jax.default_backend()}  N={N} RS={RS}  starts={STARTS} H={HS} "
        f"n_test={NTEST}  variants={VARIANTS}")
    with open(AD_PKL, "rb") as f:
        ck = pickle.load(f)
    K = ck["k_lat"]
    dec = wc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], K)
    Ztr = ck["Z_train"]
    d = wc.build_data(N)
    fp = wc.data_fingerprint(d["U"])
    if fp["sha256"] != ck["data_fingerprint"].get("sha256", fp["sha256"]):
        raise SystemExit("data fingerprint mismatch")
    U_te, c_te = d["U_test"], d["c_test"]
    interior = wc.interior_indices(N)
    coords = jnp.asarray(wc.grid_coords(N))
    n_test = min(NTEST, wc.N_TEST)
    Hmax = max(HS)

    report = dict(config=dict(wc.CONFIG, starts=STARTS, hs=HS, sd_n_test=n_test,
                              variants=VARIANTS, fit_budget=FIT_BUDGET, tag=TAG,
                              ad_pkl=os.path.basename(AD_PKL), ad_config=ck["config"]),
                  backend=jax.default_backend(), data_fingerprint=fp,
                  train_traj_rel_mean=ck.get("train_traj_rel_mean"))

    # ---- ORACLE latent fit on an arbitrary field (uses the held-out data) ----
    f_mis = lambda z, u: dec(z, coords) - u
    rJ = jax.jit(lambda z, u: (f_mis(z, u), jax.jacfwd(f_mis)(z, u)))
    rn = jax.jit(lambda z, u: jnp.linalg.norm(f_mis(z, u)))
    zmean_t = Ztr.mean(axis=0)

    def fit(u, z_init):
        z, r, info = lm_solve(lambda zz: rJ(zz, jnp.asarray(u)),
                              lambda zz: rn(zz, jnp.asarray(u)),
                              jnp.asarray(z_init, F64), FIT_BUDGET)
        return np.asarray(z), float(r)

    # ---- EQ weights (fitted on TRAINING latents, exactly as in wlat_rom) -----
    T1 = wc.NUM_STEPS + 1
    Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
        Ztr.shape[0] * T1, wc.EQ_SNAPS, replace=False)]
    eq_cache = {}

    def build_ops(var):
        solver, colloc_name, objective = var.split(":")
        if objective.startswith("weak"):
            M = int(objective[5:] if objective[4] in "lu" else objective[4:])
            beta = 1.0 if objective.startswith("weakl") else wc.WEAK_BETA
            alpha_w = 0.0 if objective.startswith("weaku") else wc.WEAK_ALPHA
            if colloc_name == "full":
                col = dict(kind="grid", idx=interior, w=None)
            else:
                pool = "off" if colloc_name.startswith("eqoff") else "grid"
                m = int(colloc_name[5:] if pool == "off" else colloc_name[2:])
                key = (M, m, pool)
                if key not in eq_cache:
                    eq_cache[key] = wc.fit_eq_weights(dec, N, M, m, Z_snap, pool=pool,
                                                      rng=np.random.default_rng(EQ_RNG_SEED))
                col = eq_cache[key]
            return wc.make_weak_ops(dec, N, col, M=M, solver=solver, beta=beta, alpha_w=alpha_w)
        col = wc.make_collocation(colloc_name, N, np.random.default_rng(1234), u0=U_te[0, 0])
        return wc.make_strong_ops(dec, N, col, solver=solver)

    # ---- per-trajectory exact sub-step trajectories --------------------------
    log("  building exact Newmark sub-step trajectories at the ROM dt ...")
    t0 = time.time()
    SUB = [wc.newmark_substeps(N, RS, U_te[i, 0], float(c_te[i])) for i in range(n_test)]
    log(f"    {n_test} x {SUB[0].shape[0]} sub-step fields [{time.time()-t0:.0f}s]")
    # sanity: the sub-step trajectory must hit the stored snapshots (up to the
    # RS time-discretisation error already quantified by wlat_verify V5)
    chk = [wc.traj_metrics(SUB[i][::RS], U_te[i])[2] for i in range(n_test)]
    report["substeps_vs_stored_snapshots_traj_rel"] = float(np.mean(chk))
    log(f"  exact sub-step trajectory vs the 80-substep FOM snapshots: {np.mean(chk):.3e} "
        f"(= the RS={RS} time-discretisation floor)")

    # ---- oracle latents at every needed sub-step ----------------------------
    need = sorted({k for n0 in STARTS for k in (n0 * RS - 1, n0 * RS)} |
                  {(n0 + h) * RS for n0 in STARTS for h in HS})
    need = [k for k in need if 0 <= k <= wc.NUM_STEPS * RS]
    log(f"  oracle latent fits at {len(need)} sub-steps x {n_test} trajectories ...")
    t0 = time.time()
    ZO = {}                                   # (i, k) -> (z, rel)
    for i in range(n_test):
        rms_i = np.sqrt(np.mean(np.sum(U_te[i] ** 2, axis=1)))
        z_prev = zmean_t[0]
        for k in need:
            u = SUB[i][max(k, 0)]
            best = None
            for zi in (z_prev, zmean_t[min(k // RS, wc.NUM_STEPS)]):
                z, r = fit(u, zi)
                if best is None or r < best[1]:
                    best = (z, r)
            ZO[(i, k)] = (best[0], best[1] / rms_i)
            z_prev = best[0]
    log(f"    done [{time.time()-t0:.0f}s]; mean oracle rel "
        f"{np.mean([v[1] for v in ZO.values()]):.3e}")
    report["oracle_fit_rel_mean"] = float(np.mean([v[1] for v in ZO.values()]))

    # ---- the diagnostic ------------------------------------------------------
    res = {}
    for var in VARIANTS + ["hold"]:
        ops = None if var == "hold" else build_ops(var)
        t0 = time.time()
        rows = []
        for i in range(n_test):
            rms_i = np.sqrt(np.mean(np.sum(U_te[i] ** 2, axis=1)))
            c = float(c_te[i])
            u0_rms = float(np.sqrt(np.mean(U_te[i, 0][interior] ** 2)))
            for n0 in STARTS:
                k0 = n0 * RS
                if k0 + Hmax * RS > wc.NUM_STEPS * RS:
                    continue
                zm1 = ZO[(i, max(k0 - 1, 0))][0]
                z0 = ZO[(i, k0)][0]
                if var == "hold":
                    Z = np.repeat(np.asarray(z0)[None], Hmax * RS + 1, axis=0)
                else:
                    tol_abs = wc.GN_TOL * u0_rms * ops["tol_scale"]
                    Sm1 = ops["state_of"](jnp.asarray(zm1))
                    S0 = ops["state_of"](jnp.asarray(z0))

                    def one(carry, _):
                        z, Sn, Snm = carry
                        z2, r_, nJ_, acc_, re_, at_ = ops["step_gen"](
                            z, jnp.stack([Sn, Snm]), c, tol_abs, wc.GN_BUDGET)
                        return (z2, ops["state_of"](z2), Sn), z2

                    carry = (jnp.asarray(z0), S0, Sm1)
                    Zs = [np.asarray(z0)]
                    for _ in range(Hmax * RS):
                        carry, z2 = one(carry, None)
                        Zs.append(np.asarray(z2))
                    Z = np.stack(Zs)
                for h in HS:
                    kh = k0 + h * RS
                    zr = Z[h * RS]
                    if not np.all(np.isfinite(zr)):
                        rows.append(dict(i=i, n0=n0, h=h, rom=float("nan"),
                                         oracle=ZO[(i, kh)][1], blowup=True))
                        continue
                    ur = np.asarray(dec(jnp.asarray(zr), coords))
                    e_fom = float(np.linalg.norm(ur - SUB[i][kh]) / rms_i)
                    e_orc = ZO[(i, kh)][1]
                    zo = ZO[(i, kh)][0]
                    rows.append(dict(i=i, n0=n0, h=h, rom=e_fom, oracle=e_orc,
                                     dz=float(np.linalg.norm(zr - zo)), blowup=False))
        by_h = {}
        for h in HS:
            rr = [r for r in rows if r["h"] == h and not r["blowup"]]
            by_h[h] = dict(
                n=len(rr), n_blowup=sum(1 for r in rows if r["h"] == h and r["blowup"]),
                rom_mean=float(np.mean([r["rom"] for r in rr])) if rr else float("nan"),
                rom_median=float(np.median([r["rom"] for r in rr])) if rr else float("nan"),
                oracle_mean=float(np.mean([r["oracle"] for r in rr])) if rr else float("nan"),
                excess_mean=float(np.mean([r["rom"] - r["oracle"] for r in rr])) if rr else float("nan"),
                dz_mean=float(np.mean([r.get("dz", np.nan) for r in rr])) if rr else float("nan"))
        res[var] = dict(by_h=by_h, secs=time.time() - t0, rows=rows)
        log(f"  {var:24s} " + "  ".join(
            f"H={h}: rom {by_h[h]['rom_mean']:.3e} (oracle {by_h[h]['oracle_mean']:.3e}, "
            f"excess {by_h[h]['excess_mean']:+.3e})" for h in HS) + f"  [{time.time()-t0:.0f}s]")
        report["stepdiag"] = res
        os.makedirs(OUTDIR, exist_ok=True)
        out = os.path.join(OUTDIR, f"wlat_stepdiag_N{N}_K{K}" + (f"_{TAG}" if TAG else "") + ".json")
        with open(out + ".tmp", "w") as f:
            json.dump(report, f, indent=2, default=float)
        os.replace(out + ".tmp", out)
    log("wrote stepdiag json")


if __name__ == "__main__":
    main()
