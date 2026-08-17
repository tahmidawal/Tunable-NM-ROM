"""Follow-up timing on ONE GPU, sequentially (cross-N / cross-K ratios are only
valid on one device):

MODE=n : N ladder at fixed (K=8, M=64, m=256).  The N=64-trained coordinate
         decoder is meshfree, so the SAME checkpoint is used at every N (EQ
         weights refit on each N's grid; test trajectory 0 of TEST_SEED
         regenerated with the FOM at each N).  Measures FOM rollout (same
         jitted implicit solver, batch 1), ROM rollout (device scan), the IC
         cold start (python LM as in blat_rom vs the new jitted LM), the
         51-slice full decode, iteration counts -> per-Jacobian-eval time.
         Also reports the (bonus, one-trajectory) ROM error against the FOM
         at that N: the ROM trained at N=64 run against finer FOMs.
MODE=k : K ladder at N=64: every checkpoint in PKLS (comma list) + POD k in
         POD_KS with the same solver -> rollout time, iterations, per-Jacobian
         time, IC time.

Usage: MODE=n PKL=<K8 pkl> NS=32,64,128,256 [VARIANTS=...] python fu_timing.py <out.json>
       MODE=k PKLS=<pkl,pkl,...> [POD_KS=2,4,...] python fu_timing.py <out.json>
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
VARIANTS = [v for v in os.environ.get(
    "VARIANTS", "lspg:eq256:weak64,lspg:eq512:weak64,lspg:full:weak64,lspg:eqoff512:weakc64").split(",") if v]
POD_KS = [int(k) for k in os.environ.get("POD_KS", "2,4,6,8,12,16,24,32,64").split(",") if k]
POD_VARIANT = os.environ.get("POD_VARIANT", "lspg:full:fd")
EQ_RNG_SEED = 4321
TEST_IDX = int(os.environ.get("TEST_IDX", "0"))


def load_ck(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], ck["k_lat"])
    return ck, dec


def test_traj(n):
    """Test trajectory TEST_IDX of the TEST_SEED draw (same draw as blat_rom's
    N_TEST=16 test set) regenerated with the FOM at resolution n."""
    cxt, cyt, wt, at, nut, zt = bc.bf.sample_params(seed=bc.TEST_SEED, m=bc.N_TEST)
    roll, res = bc.bf.make_rollout(n)
    u0 = bc.bf.blob_ic(n, cxt[TEST_IDX], cyt[TEST_IDX], wt[TEST_IDX], at[TEST_IDX])
    snaps, rr = roll(jnp.asarray(u0)[None], jnp.asarray([nut[TEST_IDX]]))
    U = np.asarray(snaps)[:, 0]                       # (T1, n^2)
    return U, float(nut[TEST_IDX]), float(jnp.max(rr)), roll


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
            key = (dec.kind, dec.k, n, kind, M, m, pool)
            if key not in cache:
                cache[key] = bc.fit_eq_weights(dec, n, M, m, Z_snap, kind=kind, pool=pool,
                                               rng=np.random.default_rng(EQ_RNG_SEED))
            col = cache[key]
        return bc.make_weak_ops(dec, n, col, kind=kind, M=M, solver=solver)
    col = bc.make_collocation(colloc_name, n, np.random.default_rng(1234))
    return bc.make_step_ops(dec, n, col, objective, solver)


def time_rom(dec, n, ops, z0, nu, u0_rms, U_true=None):
    usc = jnp.full((bc.NUM_STEPS,), bc.GN_TOL * u0_rms * ops.get("tol_scale", np.sqrt(ops["m"])))
    def once():
        Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, nu, usc, bc.GN_BUDGET)
        Z_.block_until_ready()
    Z_, rn_, nj_, re_ = ops["rollout_jit"](z0, nu, usc, bc.GN_BUDGET)
    nJ = int(jnp.sum(nj_)); reasons = np.asarray(re_).tolist()
    med, ts = bc.time_fn(once, reps=TIME_REPS, warm=2)
    out = dict(rollout_s_median=med, all=ts, iters_total=nJ, iters_per_step=nJ / bc.NUM_STEPS,
               s_per_jacobian_eval=med / max(nJ, 1), reasons={str(r): reasons.count(r) for r in set(reasons)},
               m=int(ops["m"]))
    if U_true is not None:
        coords = jnp.asarray(bc.grid_coords(n))
        Zall = jnp.concatenate([z0[None], Z_], axis=0)
        F = np.asarray(jax.vmap(lambda z: ops["full"](z))(Zall)) if dec.kind == "coord" else None
        if F is not None:
            if F.shape[1] != U_true.shape[1]:          # ops["full"] may return interior only
                F = np.asarray(jax.vmap(lambda z: dec(z, coords))(Zall))
            per = np.linalg.norm(F - U_true, axis=1) / np.linalg.norm(U_true, axis=1)
            out["traj_rel_vs_fom_at_this_N"] = float(np.mean(per))
            out["per_time"] = per.tolist()
    return out


def main():
    log(f"jax_backend={jax.default_backend()} MODE={MODE} gpu={jax.devices()[0]}")
    report = dict(mode=MODE, backend=jax.default_backend(), device=str(jax.devices()[0]),
                  time_reps=TIME_REPS, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
                  ic_budget=bc.IC_BUDGET, test_seed=bc.TEST_SEED, test_idx=TEST_IDX, rows=[])
    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    if MODE == "n":
        ck, dec = load_ck(os.environ["PKL"])
        K = dec.k
        Ztr = ck["Z_train"]; T1 = Ztr.shape[1]
        Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
            Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)]
        zmean0 = Ztr.mean(axis=0)[0]
        report["ckpt"] = dict(path=os.path.basename(os.environ["PKL"]), config=ck["config"], k=K)
        NS = [int(v) for v in os.environ.get("NS", "32,64,128,256").split(",")]
        for n in NS:
            t0 = time.time()
            U, nu, fom_res, roll = test_traj(n)
            log(f"== N={n}: FOM test trajectory residual {fom_res:.2e}")
            interior = bc.interior_indices(n)
            u0 = U[0]; u0_rms = float(np.sqrt(np.mean(u0[interior] ** 2)))
            coords = jnp.asarray(bc.grid_coords(n))
            row = dict(N=n, fom_rel_residual=fom_res, nu=nu)
            # FOM timing
            U0 = jnp.asarray(u0)[None]; nu1 = jnp.asarray([nu])
            def fom_once():
                s, r = roll(U0, nu1); s.block_until_ready()
            med, ts = bc.time_fn(fom_once, reps=TIME_REPS, warm=2)
            row["fom_rollout_s"] = med; row["fom_all"] = ts
            # IC fit: python LM (as in blat_rom) vs jitted LM, same 2 inits, same budget
            # nearest-IC training latent: only meaningful on the training grid (N=64);
            # elsewhere use the mean-t0 latent + the N=64-nearest latent by IC params
            cxt, cyt, wt, at, nut, zt = bc.bf.sample_params(seed=bc.TEST_SEED, m=bc.N_TEST)
            cx, cy, w, a, nu_tr, _ = bc.bf.sample_params()
            j0 = int(np.argmin((cx[:bc.N_TRAIN] - cxt[TEST_IDX]) ** 2 + (cy[:bc.N_TRAIN] - cyt[TEST_IDX]) ** 2
                               + (w[:bc.N_TRAIN] - wt[TEST_IDX]) ** 2 + (a[:bc.N_TRAIN] - at[TEST_IDX]) ** 2))
            inits = {"mean_t0": zmean0, "nearest_ic": Ztr[j0, 0]}
            def ic_py():
                return bc.fit_ic(dec, n, u0, inits, coords=coords)
            z_py, rel_py, info_py = ic_py()
            ic_py_med, _ = bc.time_fn(lambda: ic_py(), reps=3, warm=1)
            fit_jit = fu.make_fit_ic_jit(dec, n, bc.IC_BUDGET)
            Z0 = jnp.asarray(np.stack([inits["mean_t0"], inits["nearest_ic"]]))
            u0j = jnp.asarray(u0)
            def ic_jit():
                z, rel, nJ, b = fit_jit(u0j, Z0); z.block_until_ready(); return z, rel, nJ, b
            z_j, rel_j, nJ_j, b_j = ic_jit()
            ic_jit_med, ic_jit_all = bc.time_fn(lambda: ic_jit(), reps=TIME_REPS, warm=2)
            row["ic_fit"] = dict(python_s=ic_py_med, python_rel=float(rel_py), python_init=info_py.get("init"),
                                 jit_s=ic_jit_med, jit_all=ic_jit_all, jit_rel=float(rel_j),
                                 jit_iters=int(nJ_j), jit_best_init=["mean_t0", "nearest_ic"][int(b_j)],
                                 z_diff=float(jnp.linalg.norm(z_j - jnp.asarray(z_py))))
            log(f"  IC fit: python {ic_py_med*1e3:.0f} ms (rel {rel_py:.3e})  jit {ic_jit_med*1e3:.1f} ms "
                f"(rel {float(rel_j):.3e}, iters {int(nJ_j)})")
            # full decode of 51 slices
            Zt = jnp.asarray(np.stack([np.asarray(z_j)] * T1))
            dec_all = jax.jit(lambda ZZ: jax.vmap(lambda zz: dec(zz, coords))(ZZ))
            dmed, _ = bc.time_fn(lambda: dec_all(Zt).block_until_ready(), reps=TIME_REPS, warm=2)
            row["decode_all_slices_s"] = dmed
            # ROM variants
            cache = {}
            row["rom"] = {}
            for var in VARIANTS:
                ops = build_ops(dec, n, var, Z_snap, cache)
                r = time_rom(dec, n, ops, z_j, nu, u0_rms, U_true=U)
                r["speedup_rollout_only"] = med / r["rollout_s_median"]
                r["speedup_end_to_end_jit_ic"] = med / (r["rollout_s_median"] + ic_jit_med + dmed)
                r["speedup_end_to_end_python_ic"] = med / (r["rollout_s_median"] + ic_py_med + dmed)
                r["eq_info"] = ops.get("colloc_info")
                row["rom"][var] = r
                log(f"  {var:24s} rollout {r['rollout_s_median']*1e3:.0f} ms  iters {r['iters_total']} "
                    f"({r['s_per_jacobian_eval']*1e3:.2f} ms/jac)  speedup {r['speedup_rollout_only']:.2f}x "
                    f"e2e(jit ic) {r['speedup_end_to_end_jit_ic']:.2f}x  err vs FOM@N {r.get('traj_rel_vs_fom_at_this_N', float('nan')):.3e}")
            row["fom_ms"] = med * 1e3; row["secs"] = time.time() - t0
            log(f"  FOM {med*1e3:.0f} ms  [{row['secs']:.0f}s]")
            report["rows"].append(row); save()
    else:
        n = bc.N
        d = bc.build_data(n)                       # TRAIN (POD snapshots) + the blat_rom test set
        U = d["U_test"][TEST_IDX]; nu = float(d["nu_test"][TEST_IDX]); fom_res = d["max_fom_rel_residual"]
        U_tr = d["U"][:bc.N_TRAIN]
        roll, _ = bc.bf.make_rollout(n)
        interior = bc.interior_indices(n)
        u0 = U[0]; u0_rms = float(np.sqrt(np.mean(u0[interior] ** 2)))
        coords = jnp.asarray(bc.grid_coords(n))
        U0 = jnp.asarray(u0)[None]; nu1 = jnp.asarray([nu])
        def fom_once():
            s, r = roll(U0, nu1); s.block_until_ready()
        fmed, _ = bc.time_fn(fom_once, reps=TIME_REPS, warm=2)
        report["fom_rollout_s"] = fmed; report["fom_rel_residual"] = fom_res
        log(f"FOM N={n}: {fmed*1e3:.0f} ms")
        V = None
        for path in os.environ["PKLS"].split(","):
            ck, dec = load_ck(path)
            K = dec.k
            if V is None:
                V = ck["V"]
            Ztr = ck["Z_train"]; T1 = Ztr.shape[1]
            Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
                Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)]
            fit_jit = fu.make_fit_ic_jit(dec, n, bc.IC_BUDGET)
            cxt, cyt, wt, at, nut, zt = bc.bf.sample_params(seed=bc.TEST_SEED, m=bc.N_TEST)
            cx, cy, w, a, nu_tr, _ = bc.bf.sample_params()
            j0 = int(np.argmin((cx[:bc.N_TRAIN] - cxt[TEST_IDX]) ** 2 + (cy[:bc.N_TRAIN] - cyt[TEST_IDX]) ** 2
                               + (w[:bc.N_TRAIN] - wt[TEST_IDX]) ** 2 + (a[:bc.N_TRAIN] - at[TEST_IDX]) ** 2))
            Z0 = jnp.asarray(np.stack([Ztr.mean(axis=0)[0], Ztr[j0, 0]]))
            u0j = jnp.asarray(u0)
            def ic_jit():
                z, rel, nJ, b = fit_jit(u0j, Z0); z.block_until_ready(); return z, rel, nJ, b
            z_j, rel_j, nJ_j, b_j = ic_jit()
            ic_med, _ = bc.time_fn(lambda: ic_jit(), reps=TIME_REPS, warm=2)
            row = dict(kind="coord", K=K, ckpt=os.path.basename(path), train_seed=ck["config"].get("train_seed"),
                       ic_fit=dict(jit_s=ic_med, rel=float(rel_j), iters=int(nJ_j)), rom={})
            cache = {}
            for var in VARIANTS:
                ops = build_ops(dec, n, var, Z_snap, cache)
                r = time_rom(dec, n, ops, z_j, nu, u0_rms, U_true=U)
                r["speedup_rollout_only"] = fmed / r["rollout_s_median"]
                r["speedup_end_to_end_jit_ic"] = fmed / (r["rollout_s_median"] + ic_med)
                row["rom"][var] = r
                log(f"  K={K:2d} {var:24s} rollout {r['rollout_s_median']*1e3:.0f} ms iters {r['iters_total']} "
                    f"({r['s_per_jacobian_eval']*1e3:.2f} ms/jac) speedup {r['speedup_rollout_only']:.2f}x "
                    f"err {r.get('traj_rel_vs_fom_at_this_N', float('nan')):.3e}")
            report["rows"].append(row); save()
        cache = {}
        for k in POD_KS:
            if k > V.shape[1]:
                continue
            pdec = bc.PODDecoder(V[:, :k])
            z0 = jnp.asarray(pdec.V.T @ jnp.asarray(u0))
            row = dict(kind="pod", K=k, rom={})
            for var in [POD_VARIANT] + [v for v in VARIANTS if v.split(":")[2].startswith("weak") and not v.split(":")[2].startswith("weakc")
                                        and int(v.split(":")[2][4:]) > k]:
                zs = (U_tr.reshape(-1, n * n) @ np.asarray(pdec.V))[
                    np.random.default_rng(EQ_RNG_SEED).choice(U_tr.shape[0] * U_tr.shape[1], bc.EQ_SNAPS, replace=False)]
                ops = build_ops(pdec, n, var, zs, cache)
                r = time_rom(pdec, n, ops, z0, nu, u0_rms)
                r["speedup_rollout_only"] = fmed / r["rollout_s_median"]
                row["rom"][var] = r
                log(f"  POD k={k:2d} {var:20s} rollout {r['rollout_s_median']*1e3:.0f} ms iters {r['iters_total']} "
                    f"({r['s_per_jacobian_eval']*1e3:.2f} ms/jac) speedup {r['speedup_rollout_only']:.2f}x")
            report["rows"].append(row); save()
    report["complete"] = True; save()
    log("DONE")


if __name__ == "__main__":
    main()
