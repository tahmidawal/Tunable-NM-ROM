"""Follow-up timing for the Burgers-2D latent-stepping ROM, on ONE GPU, all
ladder points measured SEQUENTIALLY IN ONE PROCESS (cross-N and cross-K ratios
measured on different GPUs are not comparable).

Protocol for every reported time: warm-up 2, then median of TIME_REPS (7),
`block_until_ready` on the returned array, same device, same process.  The FOM
baseline is the testbed's own jitted implicit rollout at batch 1 -- the exact
function that generated the truth -- compiled and warmed the same way.

MODE=n : N ladder at fixed (K, M=64, m=256).  The coordinate decoder is
         meshfree, so the SAME N=64 checkpoint is used at every N; the EQ
         quadrature weights are REFIT on each N's grid and the test trajectory
         is regenerated with the FOM at that N (its Newton residual is asserted
         < FOM_RES_TOL, as in blat_common.build_data).  Reports the FOM
         rollout, the ROM rollout (device scan), the cold-start IC fit (the
         Python-loop LM of blat_rom vs the jitted LM of fu_common -- the same
         algorithm, see fu_common), the 51-slice full decode, iteration and
         attempt counts, and the composed end-to-end speedup.
MODE=k : K ladder at N=64: every checkpoint in PKLS + the POD control at
         POD_KS with the same solver -> rollout time, iterations, attempts,
         per-Jacobian time, IC time.

Usage: MODE=n PKL=<pkl> NS=32,64,128,256 [VARIANTS=...] python fu_timing.py <out.json>
       MODE=k PKLS=<pkl,pkl,...> [POD_KS=...] python fu_timing.py <out.json>
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

import fu_common as fu
import blat_common as bc
from blat_common import F64, log

MODE = os.environ.get("MODE", "n")
OUT = sys.argv[1]
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
WARM = int(os.environ.get("TIME_WARM", "2"))
VARIANTS = [v for v in os.environ.get(
    "VARIANTS", "lspg:eq256:weak64,lspg:eq512:weak64,lspg:full:weak64,lspg:eqoff512:weakc64").split(",") if v]
POD_KS = [int(k) for k in os.environ.get("POD_KS", "2,4,6,8,12,16,24,32,64").split(",") if k]
POD_VARIANT = os.environ.get("POD_VARIANT", "lspg:full:fd")
EQ_RNG_SEED = 4321
TEST_IDX = int(os.environ.get("TEST_IDX", "0"))
FOM_RES_TOL = float(os.environ.get("FOM_RES_TOL", "1e-8"))
IC_Z_TOL = float(os.environ.get("IC_Z_TOL", "1e-6"))     # rel. |z_jit - z_python| for a composed e2e number


def load_ck(path, n=None):
    """Load an auto-decoder checkpoint and apply blat_rom.py's configuration
    guards -- a checkpoint trained with a different mesh, BC mode, architecture
    or training draw must never silently enter a timing table."""
    with open(path, "rb") as f:
        ck = pickle.load(f)
    for key, val in (("bc_mode", bc.BC_MODE), ("N", bc.N), ("ad_hidden", bc.AD_HIDDEN),
                     ("ad_layers", bc.AD_LAYERS), ("n_train", bc.N_TRAIN), ("seed", bc.SEED)):
        if ck["config"][key] != val:
            raise SystemExit(f"{os.path.basename(path)}: config mismatch on {key}: "
                             f"{ck['config'][key]} vs {val}")
    dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], ck["k_lat"])
    return ck, dec


def test_traj(n):
    """Test trajectory TEST_IDX of the TEST_SEED draw (the same draw as the
    N_TEST=16 test set of blat_rom.py) regenerated with the FOM at resolution
    n.  Aborts if the FOM's own Newton residual exceeds FOM_RES_TOL -- an
    unconverged 'truth' would silently poison the error column."""
    cxt, cyt, wt, at, nut, zt = bc.bf.sample_params(seed=bc.TEST_SEED, m=bc.N_TEST)
    roll, res = bc.bf.make_rollout(n)
    u0 = bc.bf.blob_ic(n, cxt[TEST_IDX], cyt[TEST_IDX], wt[TEST_IDX], at[TEST_IDX])
    snaps, rr = roll(jnp.asarray(u0)[None], jnp.asarray([nut[TEST_IDX]]))
    U = np.asarray(snaps)[:, 0]                       # (T1, n^2)
    rmax = float(jnp.max(rr))
    if not np.isfinite(rmax) or rmax > FOM_RES_TOL:
        raise SystemExit(f"N={n}: FOM Newton rel residual {rmax:.2e} > {FOM_RES_TOL:.0e} "
                         f"-- the FOM baseline at this N is not converged, refusing to time it")
    return U, float(nut[TEST_IDX]), rmax, roll


def build_ops(dec, n, var, Z_snap, cache):
    solver, colloc_name, objective = var.split(":")
    interior = bc.interior_indices(n)
    if objective.startswith("weak"):
        kind = "weakc" if objective.startswith("weakc") else "weak"
        M = int(objective[len(kind):])
        if colloc_name == "full":
            col = dict(kind="grid", idx=interior, w=None)
        else:
            pool = "off" if colloc_name.startswith("eqoff") else "grid"
            m = int(colloc_name[5:] if pool == "off" else colloc_name[2:])
            key = (dec.kind, dec.k, n, kind, M, m, pool)      # N is part of the key: refit per N
            if key not in cache:
                cache[key] = bc.fit_eq_weights(dec, n, M, m, Z_snap, kind=kind, pool=pool,
                                               rng=np.random.default_rng(EQ_RNG_SEED))
            col = cache[key]
        return bc.make_weak_ops(dec, n, col, kind=kind, M=M, solver=solver)
    col = bc.make_collocation(colloc_name, n, np.random.default_rng(1234))
    return bc.make_step_ops(dec, n, col, objective, solver)


def time_rom(dec, n, ops, z0, nu, u0_rms, U_true=None):
    """Median-of-TIME_REPS device-scan rollout.  Iteration accounting: the scan
    returns accepted Jacobian evaluations per step; LM ATTEMPTS (which each cost
    one residual evaluation) are recovered from one extra UNTIMED Python-loop
    rollout with the identical step kernel, so that 's per Jacobian evaluation'
    can be read next to 's per attempt' rather than being mistaken for the
    isolated cost of one Gauss-Newton iteration."""
    usc = jnp.full((bc.NUM_STEPS,), bc.GN_TOL * u0_rms * ops.get("tol_scale", np.sqrt(ops["m"])))

    def once():
        Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, nu, usc, bc.GN_BUDGET)
        Z_.block_until_ready()

    Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, nu, usc, bc.GN_BUDGET)
    nJ = int(jnp.sum(nj_)); reasons = np.asarray(re_).tolist()
    med, ts = bc.time_fn(once, reps=TIME_REPS, warm=WARM)
    ref = bc.rollout(dec, n, ops, z0, nu, u0_rms)          # UNTIMED: attempts + agreement check
    n_att = int(np.sum(ref["attempts"]))
    out = dict(rollout_s_median=med, all=ts, iters_total=nJ, iters_per_step=nJ / bc.NUM_STEPS,
               attempts_total=n_att, s_per_jacobian_eval=med / max(nJ, 1),
               s_per_attempt=med / max(n_att, 1),
               scan_vs_python_iters=[nJ, int(np.sum(ref["iters"]))],
               reasons={str(r): reasons.count(r) for r in set(reasons)},
               m=int(ops["m"]))
    if U_true is not None:
        coords = jnp.asarray(bc.grid_coords(n))
        Zall = jnp.concatenate([z0[None], Z_], axis=0)
        F = np.asarray(jax.vmap(lambda z: dec(z, coords))(Zall)) if dec.kind == "coord" \
            else np.asarray(jax.vmap(lambda z: dec.V @ z)(Zall))
        per = np.linalg.norm(F - U_true, axis=1) / np.linalg.norm(U_true, axis=1)
        out["traj_rel_vs_fom_at_this_N"] = float(np.mean(per))
        out["per_time"] = per.tolist()
    return out


def ic_inits(dec, ck, n, u0):
    """The two cold starts of blat_rom.py: the mean t=0 training latent, and the
    t=0 latent of the training trajectory whose INITIAL FIELD is nearest to the
    known u0 (field distance at this n -- the same rule as blat_rom.py)."""
    Ztr = ck["Z_train"]
    j0, d0 = fu.nearest_train_ic(n, u0)
    return {"mean_t0": Ztr.mean(axis=0)[0], "nearest_ic": Ztr[j0, 0]}, j0, d0


def main():
    log(f"jax_backend={jax.default_backend()} MODE={MODE} gpu={jax.devices()[0]} "
        f"reps={TIME_REPS} warm={WARM}")
    report = dict(mode=MODE, backend=jax.default_backend(), device=str(jax.devices()[0]),
                  time_reps=TIME_REPS, time_warm=WARM, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
                  ic_budget=bc.IC_BUDGET, test_seed=bc.TEST_SEED, test_idx=TEST_IDX,
                  fom_res_tol=FOM_RES_TOL, rows=[])

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    if MODE == "n":
        ck, dec = load_ck(os.environ["PKL"])
        K = dec.k
        Ztr = ck["Z_train"]; T1 = Ztr.shape[1]
        Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
            Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)]
        report["ckpt"] = dict(path=os.path.basename(os.environ["PKL"]), config=ck["config"], k=K,
                              data_fingerprint=ck["data_fingerprint"])
        NS = [int(v) for v in os.environ.get("NS", "32,64,128,256").split(",")]
        for n in NS:
            t0 = time.time()
            U, nu, fom_res, roll = test_traj(n)
            log(f"== N={n}: FOM test trajectory residual {fom_res:.2e}")
            interior = bc.interior_indices(n)
            u0 = U[0]; u0_rms = float(np.sqrt(np.mean(u0[interior] ** 2)))
            coords = jnp.asarray(bc.grid_coords(n))
            row = dict(N=n, fom_rel_residual=fom_res, nu=nu)
            # ---- FOM: the testbed's own jitted implicit rollout, batch 1
            U0 = jnp.asarray(u0)[None]; nu1 = jnp.asarray([nu])

            def fom_once():
                s, r = roll(U0, nu1); s.block_until_ready()
            med, ts = bc.time_fn(fom_once, reps=TIME_REPS, warm=WARM)
            row["fom_rollout_s"] = med; row["fom_all"] = ts
            # ---- IC cold start: Python-loop LM (blat_rom) vs the jitted port, same inits
            inits, j0, d0 = ic_inits(dec, ck, n, u0)
            row["ic_nearest_train_idx"] = j0; row["ic_nearest_train_dist"] = d0

            def ic_py():
                return bc.fit_ic(dec, n, u0, inits, coords=coords)
            z_py, rel_py, info_py = ic_py()
            ic_py_med, ic_py_all = bc.time_fn(lambda: ic_py(), reps=TIME_REPS, warm=WARM)
            fit_jit = fu.make_fit_ic_jit(dec, n, bc.IC_BUDGET, coords=coords)
            Z0 = jnp.asarray(np.stack([inits["mean_t0"], inits["nearest_ic"]]))
            u0j = jnp.asarray(u0)

            def ic_jit():
                z, rel, nJ, b, att = fit_jit(u0j, Z0); z.block_until_ready(); return z, rel, nJ, b, att
            z_j, rel_j, nJ_j, b_j, att_j = ic_jit()
            ic_jit_med, ic_jit_all = bc.time_fn(lambda: ic_jit(), reps=TIME_REPS, warm=WARM)
            z_rel_diff = float(jnp.linalg.norm(z_j - jnp.asarray(z_py))
                               / (1.0 + jnp.linalg.norm(jnp.asarray(z_py))))
            row["ic_fit"] = dict(python_s=ic_py_med, python_all=ic_py_all, python_rel=float(rel_py),
                                 python_init=info_py.get("init"), python_iters=info_py.get("n_jac"),
                                 jit_s=ic_jit_med, jit_all=ic_jit_all, jit_rel=float(rel_j),
                                 jit_iters=int(nJ_j), jit_attempts=int(att_j),
                                 jit_best_init=["mean_t0", "nearest_ic"][int(b_j)],
                                 z_rel_diff=z_rel_diff, agree=bool(z_rel_diff <= IC_Z_TOL))
            log(f"  IC fit: python {ic_py_med*1e3:.0f} ms (rel {rel_py:.3e})  jit {ic_jit_med*1e3:.1f} ms "
                f"(rel {float(rel_j):.3e}, iters {int(nJ_j)})  |dz|rel {z_rel_diff:.1e}")
            # ---- full-field decode of all 51 slices (the ROM's output stage)
            Zt = jnp.asarray(np.stack([np.asarray(z_j)] * T1))
            dec_all = jax.jit(lambda ZZ: jax.vmap(lambda zz: dec(zz, coords))(ZZ))
            dmed, _ = bc.time_fn(lambda: dec_all(Zt).block_until_ready(), reps=TIME_REPS, warm=WARM)
            row["decode_all_slices_s"] = dmed
            # ---- ROM variants
            cache = {}
            row["rom"] = {}
            for var in VARIANTS:
                ops = build_ops(dec, n, var, Z_snap, cache)
                r = time_rom(dec, n, ops, z_j, nu, u0_rms, U_true=U)
                r["speedup_rollout_only"] = med / r["rollout_s_median"]
                r["speedup_end_to_end_jit_ic"] = med / (r["rollout_s_median"] + ic_jit_med + dmed)
                # composing the PYTHON IC time with a rollout started from the JITTED IC
                # latent is only meaningful when the two solvers land on the same latent
                r["speedup_end_to_end_python_ic_composed"] = (
                    med / (r["rollout_s_median"] + ic_py_med + dmed)
                    if row["ic_fit"]["agree"] else None)
                r["eq_info"] = ops.get("colloc_info")
                row["rom"][var] = r
                log(f"  {var:24s} rollout {r['rollout_s_median']*1e3:.0f} ms  iters {r['iters_total']} "
                    f"(att {r['attempts_total']}, {r['s_per_jacobian_eval']*1e3:.2f} ms/jac) "
                    f"speedup {r['speedup_rollout_only']:.2f}x e2e(jit ic) "
                    f"{r['speedup_end_to_end_jit_ic']:.2f}x  err vs FOM@N "
                    f"{r.get('traj_rel_vs_fom_at_this_N', float('nan')):.3e}")
            row["fom_ms"] = med * 1e3; row["secs"] = time.time() - t0
            log(f"  FOM {med*1e3:.0f} ms  [{row['secs']:.0f}s]")
            report["rows"].append(row); save()
    else:
        n = bc.N
        d = bc.build_data(n)                       # TRAIN (POD snapshots) + the blat_rom test set
        U = d["U_test"][TEST_IDX]; nu = float(d["nu_test"][TEST_IDX])
        U_tr = d["U"][:bc.N_TRAIN]
        fp = bc.data_fingerprint(d["U"])
        roll, _ = bc.bf.make_rollout(n)
        interior = bc.interior_indices(n)
        u0 = U[0]; u0_rms = float(np.sqrt(np.mean(u0[interior] ** 2)))
        coords = jnp.asarray(bc.grid_coords(n))
        U0 = jnp.asarray(u0)[None]; nu1 = jnp.asarray([nu])

        def fom_once():
            s, r = roll(U0, nu1); s.block_until_ready()
        fmed, fall = bc.time_fn(fom_once, reps=TIME_REPS, warm=WARM)
        report["fom_rollout_s"] = fmed; report["fom_all"] = fall
        report["fom_rel_residual"] = d["max_fom_rel_residual"]
        report["data_fingerprint"] = fp
        log(f"FOM N={n}: {fmed*1e3:.0f} ms")
        V = None
        paths = os.environ["PKLS"].split(",")
        for path in paths:
            ck, dec = load_ck(path)
            K = dec.k
            # every checkpoint in the K ladder must have been trained on the SAME data
            for kk in ("sum", "sumsq"):
                a_, b_ = ck["data_fingerprint"][kk], fp[kk]
                if abs(a_ - b_) / abs(b_) > 1e-6:
                    raise SystemExit(f"{os.path.basename(path)}: data fingerprint mismatch on {kk}")
            if V is None:
                V = ck["V"]
            elif float(np.max(np.abs(np.abs(ck["V"][:, :V.shape[1]].T @ V) - np.eye(V.shape[1])))) > 1e-6:
                log(f"  WARNING: POD basis of {os.path.basename(path)} differs from the first checkpoint's")
            Ztr = ck["Z_train"]; T1 = Ztr.shape[1]
            Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
                Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)]
            fit_jit = fu.make_fit_ic_jit(dec, n, bc.IC_BUDGET, coords=coords)
            inits, j0, d0 = ic_inits(dec, ck, n, u0)
            Z0 = jnp.asarray(np.stack([inits["mean_t0"], inits["nearest_ic"]]))
            u0j = jnp.asarray(u0)

            def ic_jit():
                z, rel, nJ, b, att = fit_jit(u0j, Z0); z.block_until_ready(); return z, rel, nJ, b, att
            z_j, rel_j, nJ_j, b_j, att_j = ic_jit()
            ic_med, _ = bc.time_fn(lambda: ic_jit(), reps=TIME_REPS, warm=WARM)
            Zt = jnp.asarray(np.stack([np.asarray(z_j)] * T1))
            dec_all = jax.jit(lambda ZZ: jax.vmap(lambda zz: dec(zz, coords))(ZZ))
            dmed, _ = bc.time_fn(lambda: dec_all(Zt).block_until_ready(), reps=TIME_REPS, warm=WARM)
            row = dict(kind="coord", K=K, ckpt=os.path.basename(path),
                       train_seed=ck["config"].get("train_seed"),
                       ic_fit=dict(jit_s=ic_med, rel=float(rel_j), iters=int(nJ_j),
                                   attempts=int(att_j)),
                       decode_all_slices_s=dmed, rom={})
            cache = {}
            for var in VARIANTS:
                ops = build_ops(dec, n, var, Z_snap, cache)
                r = time_rom(dec, n, ops, z_j, nu, u0_rms, U_true=U)
                r["speedup_rollout_only"] = fmed / r["rollout_s_median"]
                r["speedup_end_to_end_jit_ic"] = fmed / (r["rollout_s_median"] + ic_med + dmed)
                r["eq_info"] = ops.get("colloc_info")
                row["rom"][var] = r
                log(f"  K={K:2d} {var:24s} rollout {r['rollout_s_median']*1e3:.0f} ms iters {r['iters_total']} "
                    f"(att {r['attempts_total']}, {r['s_per_jacobian_eval']*1e3:.2f} ms/jac) "
                    f"speedup {r['speedup_rollout_only']:.2f}x err "
                    f"{r.get('traj_rel_vs_fom_at_this_N', float('nan')):.3e}")
            report["rows"].append(row); save()
        cache = {}
        for k in POD_KS:
            if V is None or k > V.shape[1]:
                continue
            pdec = bc.PODDecoder(V[:, :k])
            z0 = jnp.asarray(pdec.V.T @ jnp.asarray(u0))       # the POD cold start IS a projection
            zs = (U_tr.reshape(-1, n * n) @ np.asarray(pdec.V))[
                np.random.default_rng(EQ_RNG_SEED).choice(U_tr.shape[0] * U_tr.shape[1],
                                                          bc.EQ_SNAPS, replace=False)]
            row = dict(kind="pod", K=k, rom={})
            pod_vars = [POD_VARIANT] + [v for v in VARIANTS
                                        if v.split(":")[2].startswith("weak")
                                        and not v.split(":")[2].startswith("weakc")
                                        and int(v.split(":")[2][4:]) > k]
            for var in pod_vars:
                ops = build_ops(pdec, n, var, zs, cache)
                r = time_rom(pdec, n, ops, z0, nu, u0_rms, U_true=U)
                r["speedup_rollout_only"] = fmed / r["rollout_s_median"]
                row["rom"][var] = r
                log(f"  POD k={k:2d} {var:20s} rollout {r['rollout_s_median']*1e3:.0f} ms iters {r['iters_total']} "
                    f"({r['s_per_jacobian_eval']*1e3:.2f} ms/jac) speedup {r['speedup_rollout_only']:.2f}x "
                    f"err {r.get('traj_rel_vs_fom_at_this_N', float('nan')):.3e}")
            report["rows"].append(row); save()
    report["complete"] = True; save()
    log("DONE")


if __name__ == "__main__":
    main()
