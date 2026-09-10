"""Optimized 1D-Burgers ROM rollout driver + parity/timing harness
(2026-08-28 speed study).

Reuses a committed sep_b1d_scale.py run's artifacts — no retraining and no
node refitting, so the arms are bit-identical to the baseline run:

  CKPT_CACHE = runs/b1dqf/b1ds_nX/out/sep_b1d_scale_nX.pkl
  NODES_NPZ  = runs/b1dqf/b1ds_nX/out/sep_b1d_scale_nX_nodes.npz
  BASE_JSON  = runs/b1dqf/b1ds_nX/out/sep_b1d_scale_nX.json   (parity ref)

Modes (MODE env):
  ab    (default): per arm, run the VERBATIM reference implementation and the
        optimized one on the same test set, interleave timing reps, and check
        error parity (vs both the local reference and the committed JSON).
  diag  component timings, marginal LM-iteration cost, HLO kernel census,
        SM-utilization sampling.
  icdiag  per-init IC-fit convergence study (evidence for the ALGORITHMIC
        reduced-init arm; does not change the default path).

Opt flags (defaults = the FINAL winning configuration; see OPTIM-NOTES.md):
  OPT_SOLVER=gj       broadcast Gauss-Jordan normal-equation solve
                      (alternatives: lu = jnp.linalg.solve, spd8 = rejected
                      scalar Cholesky)
  OPT_ONEPASS=1       jax.linearize one-pass r+J
  OPT_NOCOND=1        no lax.cond re-evaluation (unconditional r+J + select)
  OPT_HOIST=1         per-rollout hoisting of wt / DT*nu*lam
  OPT_LEAN=1          fold nu-constants into premultiplied matrices, merge
                      head last layer with h_lin, single stacked projection
  OPT_NODOT=1         matvecs as broadcast-reduce (no cublas calls)
  OPT_SCAN_UNROLL=5   unroll the fixed 50-step outer scan
  OPT_UNROLL/OPT_IC_UNROLL  masked LM-loop unroll (measured, REJECTED; =1)
  ARMS=base_tight,nodes_tight   which arms to run
  IC_ARM=ref|fast|init2   init2 = ALGORITHMIC 2-init arm — REJECTED by the
                          icdiag study (multimodal IC landscape); do not use
                          for production numbers

Timing: TIME_REPS (default 7), BURN (default 2).  Output JSON -> OUT.
Same launch mechanics as every driver here (jaxrun + f64 + highest matmul).
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import b1d_common as b1
import b1d_fast_common as fc

F64 = jnp.float64

N = int(os.environ.get("N", "512"))
MODE = os.environ.get("MODE", "ab")
CKPT_CACHE = os.environ["CKPT_CACHE"]
NODES_NPZ = os.environ["NODES_NPZ"]
BASE_JSON = os.environ.get("BASE_JSON", "")
OUT = os.environ.get("OUT", f"/tmp/sep_b1d_fast_n{N}.json")
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
BURN = int(os.environ.get("BURN", "2"))
ARMS = os.environ.get("ARMS", "base_tight,nodes_tight").split(",")
IC_ARM = os.environ.get("IC_ARM", "fast")
OPT = dict(solver=os.environ.get("OPT_SOLVER", "gj"),
           onepass=os.environ.get("OPT_ONEPASS", "1") == "1",
           hoist=os.environ.get("OPT_HOIST", "1") == "1",
           nocond=os.environ.get("OPT_NOCOND", "1") == "1",
           lean=os.environ.get("OPT_LEAN", "1") == "1",
           nodot=os.environ.get("OPT_NODOT", "1") == "1",
           unroll=int(os.environ.get("OPT_UNROLL", "1")),
           scan_unroll=int(os.environ.get("OPT_SCAN_UNROLL", "5")),
           ic_unroll=int(os.environ.get("OPT_IC_UNROLL", "1")))

log = b1.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def blk(x):
    return jax.block_until_ready(x)


def timeit(fn, reps=TIME_REPS, burn=BURN):
    for _ in range(burn):
        blk(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        blk(fn())
        ts.append(time.perf_counter() - t0)
    return ts


def med_ms(ts):
    return float(np.median(ts) * 1e3)


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B1D-FAST N={N} mode={MODE} "
        f"opt={OPT} ic_arm={IC_ARM} arms={ARMS}")
    log(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
    t_all = time.time()

    su = fc.Setup(CKPT_CACHE, N)
    U_test, nu_test = fc.gen_test(N)
    arms_xw = fc.load_arms(NODES_NPZ)
    interior = su.interior
    T = b1.NUM_STEPS + 1

    base = json.load(open(BASE_JSON)) if BASE_JSON else None

    report = dict(config=dict(
        N=N, mode=MODE, opt=OPT, ic_arm=IC_ARM, arms=ARMS,
        ckpt=CKPT_CACHE, nodes=NODES_NPZ, base_json=BASE_JSON,
        time_reps=TIME_REPS, burn=BURN,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        jax_version=jax.__version__), variants={})

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    def tol_abs_of(ti):
        u0 = U_test[ti, 0]
        u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
        return fc.STEP_TOL * u_scale * float(np.sqrt(su.n_i))

    # ------------------------------------------------------------------ diag
    if MODE == "diag":
        X_v, w_v = arms_xw["base_tight"]
        r_w = su.make_sampled_rw(X_v, w_v)
        nu = float(nu_test[0])
        zprobe = jnp.asarray(su.Z_tr[0])
        prev_c = blk(su.prev_of(zprobe))

        rw_j = jax.jit(lambda z: r_w(z, prev_c, nu))
        rJ_ref = jax.jit(lambda z: (r_w(z, prev_c, nu),
                                    jax.jacfwd(r_w)(z, prev_c, nu)))

        def rJ_one(z):
            r, lin = jax.linearize(lambda zz: r_w(zz, prev_c, nu), z)
            return r, jax.vmap(lin)(jnp.eye(fc.K, dtype=F64))
        rJ_one_j = jax.jit(rJ_one)

        r0, J0 = blk(rJ_ref(zprobe))
        H = blk(jax.jit(lambda J, r: (J.T @ J, J.T @ r))(J0, r0))
        Hm = H[0] + 1e-6 * (jnp.diag(jnp.diag(H[0])) + 1e-30 * jnp.eye(fc.K))
        Hm = blk(Hm)
        g = H[1]
        solve_ref = jax.jit(lambda A, bb: jnp.linalg.solve(A, -bb))
        solve_chol = jax.jit(lambda A, bb: fc.solve_spd8(A, -bb))
        d_par = np.asarray(solve_ref(Hm, g)) - np.asarray(solve_chol(Hm, g))
        log(f"  chol-vs-LU solution abs diff: {np.max(np.abs(d_par)):.2e}")

        comp = dict(chol_vs_lu_diff=float(np.max(np.abs(d_par))))
        for name, fn in (("r_w", lambda: rw_j(zprobe)),
                         ("rJ_ref", lambda: rJ_ref(zprobe)),
                         ("rJ_onepass", lambda: rJ_one_j(zprobe)),
                         ("solve_lu_8x8", lambda: solve_ref(Hm, g)),
                         ("solve_chol_8x8", lambda: solve_chol(Hm, g)),
                         ("head", lambda: su.h_fn(zprobe))):
            ts = timeit(fn, reps=100, burn=10)
            comp[name + "_ms"] = med_ms(ts)
            log(f"  [{name}] {med_ms(ts)*1e3:.1f} us median (100 reps)")

        # marginal LM-iteration cost: fixed-T unconditional LM body loop
        # (DIAGNOSIS ONLY, discarded — not an arm)
        def fixed_iters(Tn):
            def run(z0):
                def body(i, s):
                    z, r, J, rn, lam = s
                    Hh = J.T @ J
                    gg = J.T @ r
                    D = jnp.diag(jnp.diag(Hh)) + 1e-30 * jnp.eye(fc.K,
                                                                 dtype=F64)
                    dz = jnp.linalg.solve(Hh + lam * D, -gg)
                    z_new = z + dz
                    rn_new = jnp.linalg.norm(r_w(z_new, prev_c, nu))
                    accept = jnp.isfinite(rn_new) & (rn_new < rn)
                    r2, J2 = jax.lax.cond(
                        accept,
                        lambda: (r_w(z_new, prev_c, nu),
                                 jax.jacfwd(r_w)(z_new, prev_c, nu)),
                        lambda: (r, J))
                    z = jnp.where(accept, z_new, z)
                    rn = jnp.where(accept, rn_new, rn)
                    lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                                    jnp.minimum(lam * 10.0, 1e12))
                    return (z, r2, J2, rn, lam)
                r0_, J0_ = rJ_ref(z0)
                s = (z0, r0_, J0_, jnp.linalg.norm(r0_),
                     jnp.asarray(1e-6, F64))
                s = jax.lax.fori_loop(0, Tn, body, s)
                return s[0]
            return jax.jit(run)

        marg = {}
        for Tn in (1, 4, 16, 64):
            fn = fixed_iters(Tn)
            ts = timeit(lambda: fn(zprobe), reps=50, burn=5)
            marg[Tn] = med_ms(ts)
            log(f"  [fixed {Tn:3d} LM iters] {med_ms(ts):.3f} ms")
        slope = (marg[64] - marg[16]) / 48.0
        comp["lm_iter_marginal_ms"] = slope
        comp["fixed_iters_ms"] = marg
        log(f"  marginal per-LM-iteration cost: {slope*1e3:.1f} us")

        # HLO census for ref vs fast rollout
        ops_ref = fc.make_device_ref(su, r_w)
        ops_fast = fc.make_device_fast(su, X_v, w_v, OPT)
        z0 = jnp.asarray(su.zbar)
        census = {}
        for name, ops in (("ref", ops_ref), ("fast", ops_fast)):
            lowered = ops["rollout"].lower(z0, nu, tol_abs_of(0),
                                           fc.GN_BUDGET)
            txt = lowered.compile().as_text()
            census[name] = dict(
                fusions=txt.count(" fusion("),
                custom_calls=txt.count("custom-call"),
                whiles=txt.count(" while("),
                conditionals=txt.count(" conditional("),
                instr_lines=sum(1 for ln in txt.splitlines() if " = " in ln))
            log(f"  HLO[{name}]: {census[name]}")
        comp["hlo_census"] = census

        # SM utilization while looping the ref rollout
        smi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=subprocess.PIPE, text=True)
        blk(ops_ref["rollout"](z0, nu, tol_abs_of(0), fc.GN_BUDGET))
        t0 = time.time()
        nrun = 0
        while time.time() - t0 < 6.0:
            blk(ops_ref["rollout"](z0, nu, tol_abs_of(0), fc.GN_BUDGET))
            nrun += 1
        smi.terminate()
        vals = []
        for ln in smi.stdout.read().splitlines():
            ln = ln.strip().replace("%", "").strip()
            if ln and ln != "[N/A]":
                try:
                    vals.append(float(ln))
                except ValueError:
                    pass
        comp["sm_util_samples"] = len(vals)
        comp["sm_util_mean"] = float(np.mean(vals)) if vals else None
        comp["sm_util_p90"] = float(np.quantile(vals, 0.9)) if vals else None
        comp["rollouts_in_6s"] = nrun
        log(f"  SM util during {nrun} rollouts/6s: mean "
            f"{comp['sm_util_mean']} p90 {comp['sm_util_p90']} "
            f"({len(vals)} samples)")
        report["diag"] = comp
        save()
        log(f"DONE diag -> {OUT} [{time.time()-t_all:.0f}s]")
        return

    # ---------------------------------------------------------------- icdiag
    if MODE == "icdiag":
        ic_ref = fc.make_ic_ref(su)
        ic_single = [fc.make_ic_fast(su, OPT, z0s=su.Z0S[i0:i0 + 1])
                     for i0 in range(su.Z0S.shape[0])]
        # per-init results: run each of the 9 inits alone via a 1-row Z0S
        rows = []
        for ti in range(fc.N_TEST):
            u0i = jnp.asarray(U_test[ti, 0][interior])
            z_best, v_best = ic_ref(u0i)
            per = []
            for i0 in range(su.Z0S.shape[0]):
                zi, vi = ic_single[i0](u0i)
                per.append(dict(init=i0, rn=float(vi),
                                dz_best=float(np.max(np.abs(
                                    np.asarray(zi) - np.asarray(z_best))))))
            rows.append(dict(traj=ti, rn_best=float(v_best), per_init=per))
            best_hits = [p["init"] for p in per
                         if p["rn"] <= float(v_best) * (1 + 1e-10)]
            log(f"  traj {ti}: best rn {float(v_best):.6e}; inits matching "
                f"best: {best_hits}; zbar-init dz={per[0]['dz_best']:.2e} "
                f"rn={per[0]['rn']:.6e}")
        report["icdiag"] = rows
        save()
        log(f"DONE icdiag -> {OUT} [{time.time()-t_all:.0f}s]")
        return

    # ------------------------------------------------------------------ thru
    if MODE == "thru":
        # THROUGHPUT arm (clearly labeled): vmap the optimized rollout over
        # all 8 test trajectories.  The vmapped while_loop pays worst-case
        # iterations across the batch, so this is an AMORTIZED number, not a
        # latency claim.  Values remain bit-identical to the sequential fast
        # path (masked lock-step iterations).
        ic_fast = fc.make_ic_fast(su, OPT)
        arm = ARMS[0]
        X_v, w_v = arms_xw[arm]
        ops_fast = fc.make_device_fast(su, X_v, w_v, OPT)
        u0b = jnp.asarray(np.stack([U_test[ti, 0][interior]
                                    for ti in range(fc.N_TEST)]))
        nub = jnp.asarray(nu_test)
        tolb = jnp.asarray([tol_abs_of(ti) for ti in range(fc.N_TEST)])
        ic_b = jax.jit(jax.vmap(ic_fast))
        roll_b = jax.jit(jax.vmap(ops_fast["rollout"],
                                  in_axes=(0, 0, 0, None)),
                         static_argnums=(3,))
        z0b, v0b = ic_b(u0b)
        Zb, rnsb, nJb, reb = roll_b(z0b, nub, tolb, fc.GN_BUDGET)
        blk(Zb)
        # errors must equal the sequential fast path
        errs = []
        for ti in range(fc.N_TEST):
            F = np.asarray(su.decode_all(
                jnp.concatenate([z0b[ti][None], Zb[ti]], axis=0)))
            errs.append(float(np.mean([rel(F[t], U_test[ti, t][interior])
                                       for t in range(T)])))
        tm = dict(ic=[], roll=[])
        for _ in range(BURN):
            blk(ic_b(u0b))
            blk(roll_b(z0b, nub, tolb, fc.GN_BUDGET))
        for _ in range(TIME_REPS):
            t0 = time.perf_counter()
            blk(ic_b(u0b))
            t1 = time.perf_counter()
            blk(roll_b(z0b, nub, tolb, fc.GN_BUDGET))
            t2 = time.perf_counter()
            tm["ic"].append(t1 - t0)
            tm["roll"].append(t2 - t1)
        report["thru"] = dict(
            arm=arm, err_mean=float(np.mean(errs)), errs=errs,
            ic_batch_ms=med_ms(tm["ic"]), roll_batch_ms=med_ms(tm["roll"]),
            ic_per_traj_ms=med_ms(tm["ic"]) / fc.N_TEST,
            roll_per_traj_ms=med_ms(tm["roll"]) / fc.N_TEST,
            times={k_: [float(x) for x in v] for k_, v in tm.items()})
        log(f"  [THRU {arm}] err {np.mean(errs):.6e}  batch ic "
            f"{med_ms(tm['ic']):.2f} ms roll {med_ms(tm['roll']):.2f} ms  "
            f"-> per-traj ic {med_ms(tm['ic'])/fc.N_TEST:.2f} roll "
            f"{med_ms(tm['roll'])/fc.N_TEST:.2f} ms")
        save()
        log(f"DONE thru -> {OUT} [{time.time()-t_all:.0f}s]")
        return

    # -------------------------------------------------------------------- ab
    ic_ref = fc.make_ic_ref(su)
    if IC_ARM == "ref":
        ic_fast = ic_ref
    elif IC_ARM == "init2":
        ic_fast = fc.make_ic_fast(su, OPT, z0s=np.asarray(su.Z0S)[:2])
    else:
        ic_fast = fc.make_ic_fast(su, OPT)

    for arm in ARMS:
        if arm == "oracle":
            r_w = su.make_full_rw()
            X_v = w_v = None
        else:
            X_v, w_v = arms_xw[arm]
            r_w = su.make_sampled_rw(X_v, w_v)
        ops_ref = fc.make_device_ref(su, r_w)
        ops_fast = fc.make_device_fast(su, X_v, w_v, OPT)
        out = dict(rollout=[], parity=dict())
        lat_dev_max = 0.0
        ic_dev_max = 0.0
        reasons_ref_all, reasons_fast_all = {}, {}
        errs_ref_all, errs_fast_all = [], []

        for ti in range(fc.N_TEST):
            nu = float(nu_test[ti])
            tol_abs = tol_abs_of(ti)
            u0i = jnp.asarray(U_test[ti, 0][interior])

            z0r, v0r = ic_ref(u0i)
            z0f, v0f = ic_fast(u0i)
            ic_dev_max = max(ic_dev_max,
                             float(np.max(np.abs(np.asarray(z0r)
                                                 - np.asarray(z0f)))))
            Zr, rnr, nJr, rer = ops_ref["rollout"](z0r, nu, tol_abs,
                                                   fc.GN_BUDGET)
            Zf, rnf, nJf, ref_ = ops_fast["rollout"](z0f, nu, tol_abs,
                                                     fc.GN_BUDGET)
            blk((Zr, Zf))
            lat_dev = float(np.max(np.abs(np.asarray(Zr) - np.asarray(Zf))))
            lat_dev_max = max(lat_dev_max, lat_dev)

            Fr = np.asarray(su.decode_all(
                jnp.concatenate([z0r[None], Zr], axis=0)))
            Ff = np.asarray(su.decode_all(
                jnp.concatenate([z0f[None], Zf], axis=0)))
            er = float(np.mean([rel(Fr[t], U_test[ti, t][interior])
                                for t in range(T)]))
            ef = float(np.mean([rel(Ff[t], U_test[ti, t][interior])
                                for t in range(T)]))
            errs_ref_all.append(er)
            errs_fast_all.append(ef)
            for d_, arr in ((reasons_ref_all, rer), (reasons_fast_all, ref_)):
                for r_ in np.unique(np.asarray(arr)):
                    d_[str(r_)] = d_.get(str(r_), 0) + \
                        int(np.sum(np.asarray(arr) == r_))

            # interleaved timing
            tm = dict(ic_ref=[], ic_fast=[], roll_ref=[], roll_fast=[],
                      dec=[])
            for _ in range(BURN):
                blk(ic_ref(u0i))
                blk(ic_fast(u0i))
                blk(ops_ref["rollout"](z0r, nu, tol_abs, fc.GN_BUDGET))
                blk(ops_fast["rollout"](z0f, nu, tol_abs, fc.GN_BUDGET))
                blk(su.decode_all(jnp.concatenate([z0f[None], Zf], axis=0)))
            for _ in range(TIME_REPS):
                t0 = time.perf_counter()
                blk(ic_ref(u0i))
                t1 = time.perf_counter()
                blk(ic_fast(u0i))
                t2 = time.perf_counter()
                blk(ops_ref["rollout"](z0r, nu, tol_abs, fc.GN_BUDGET))
                t3 = time.perf_counter()
                blk(ops_fast["rollout"](z0f, nu, tol_abs, fc.GN_BUDGET))
                t4 = time.perf_counter()
                blk(su.decode_all(jnp.concatenate([z0f[None], Zf], axis=0)))
                t5 = time.perf_counter()
                tm["ic_ref"].append(t1 - t0)
                tm["ic_fast"].append(t2 - t1)
                tm["roll_ref"].append(t3 - t2)
                tm["roll_fast"].append(t4 - t3)
                tm["dec"].append(t5 - t4)

            out["rollout"].append(dict(
                traj=ti, nu=nu, err_ref=er, err_fast=ef,
                lat_dev_max=lat_dev, ic_dev_max=float(ic_dev_max),
                njac_ref=float(np.mean(np.asarray(nJr))),
                njac_fast=float(np.mean(np.asarray(nJf))),
                times={k_: [float(x) for x in v] for k_, v in tm.items()}))
            log(f"  [{arm}] traj {ti}: err ref {er:.6e} fast {ef:.6e} "
                f"latdev {lat_dev:.2e} | ic {med_ms(tm['ic_ref']):.2f}->"
                f"{med_ms(tm['ic_fast']):.2f} ms  roll "
                f"{med_ms(tm['roll_ref']):.2f}->"
                f"{med_ms(tm['roll_fast']):.2f} ms")

        agg = lambda key: float(np.median(
            [x for r_ in out["rollout"] for x in r_["times"][key]]) * 1e3)
        out["ic_ref_ms"] = agg("ic_ref")
        out["ic_fast_ms"] = agg("ic_fast")
        out["roll_ref_ms"] = agg("roll_ref")
        out["roll_fast_ms"] = agg("roll_fast")
        out["dec_ms"] = agg("dec")
        out["err_ref_mean"] = float(np.mean(errs_ref_all))
        out["err_fast_mean"] = float(np.mean(errs_fast_all))
        out["parity"] = dict(
            err_rel_diff_fast_vs_ref=abs(out["err_fast_mean"]
                                         - out["err_ref_mean"])
            / out["err_ref_mean"],
            lat_dev_max=lat_dev_max, ic_dev_max=ic_dev_max,
            reasons_ref=reasons_ref_all, reasons_fast=reasons_fast_all)
        if base is not None and arm in base.get("variants", {}):
            bv = base["variants"][arm]
            out["parity"]["err_base_json"] = bv["rollout_err_mean"]
            out["parity"]["err_rel_diff_ref_vs_base"] = \
                abs(out["err_ref_mean"] - bv["rollout_err_mean"]) \
                / bv["rollout_err_mean"]
            out["parity"]["err_rel_diff_fast_vs_base"] = \
                abs(out["err_fast_mean"] - bv["rollout_err_mean"]) \
                / bv["rollout_err_mean"]
        report["variants"][arm] = out
        p = out["parity"]
        log(f"  [{arm}] SUMMARY err ref {out['err_ref_mean']:.6e} fast "
            f"{out['err_fast_mean']:.6e} (rel diff "
            f"{p['err_rel_diff_fast_vs_ref']:.2e}; vs base json "
            f"{p.get('err_rel_diff_fast_vs_base', float('nan')):.2e}) "
            f"latdev {lat_dev_max:.2e} | ic {out['ic_ref_ms']:.2f}->"
            f"{out['ic_fast_ms']:.2f} ms  roll {out['roll_ref_ms']:.2f}->"
            f"{out['roll_fast_ms']:.2f} ms  dec {out['dec_ms']:.2f} ms")
        save()

    log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
