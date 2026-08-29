"""E2 — 1D Burgers ROM: the sample-free TENSOR arm vs oracle / NNLS-32 /
learned-32, one invocation per N, from the COMMITTED sep_b1d_scale artifacts
(2026-08-29).  No training, no NNLS refit, no node training.

  CKPT_CACHE = runs/b1dqf/b1ds_nX/out/sep_b1d_scale_nX.pkl
  NODES_NPZ  = runs/b1dqf/b1ds_nX/out/sep_b1d_scale_nX_nodes.npz
  BASE_JSON  = runs/b1dqf/b1ds_nX/out/sep_b1d_scale_nX.json   (parity ref)
  QREF       = runs/b1dtensor/audit/Q_nX.npy  (E1's CPU-built Q, gate TX)

Arms (ARMS env; default oracle,base_tight,nodes_tight,tensor,tensor_nolean):
  oracle        full-grid sign-upwind advection projection (the reference)
  base_tight    NNLS m=32 nodes from the committed npz
  nodes_tight   learned m=32 nodes from the committed npz
  tensor        0.5 h^T Q h with Q built in-job from the frozen bank; fast
                path with the lean fold (production config)
  tensor_nolean same residual, fast path WITHOUT the lean fold (the oracle's
                fast configuration) — the like-for-like timing vs oracle

Per arm BOTH the verbatim reference rollout (jacfwd, lax.cond, LU) and the
optimized rollout run, interleaved, BURN + TIME_REPS repetitions, every
repetition persisted; accuracy is read from the LAST TIMED invocation.

Gates: J, T2, E, F, C, G, V (the sep_b1d_scale set minus D, which is a
node-training FD-vs-AD gate and does not apply — no node training here),
plus TB, TX, TA, T0, TS, TQ for the tensor.
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
import b1d_tensor_common as tc

F64 = jnp.float64

N = int(os.environ.get("N", "512"))
CKPT_CACHE = os.environ["CKPT_CACHE"]
NODES_NPZ = os.environ["NODES_NPZ"]
BASE_JSON = os.environ.get("BASE_JSON", "")
QREF = os.environ.get("QREF", "")
OUT = os.environ.get("OUT", f"/tmp/sep_b1d_tensor_n{N}.json")
TIME_REPS = int(os.environ.get("TIME_REPS", "5"))
BURN = int(os.environ.get("BURN", "2"))
ARMS = os.environ.get(
    "ARMS", "oracle,base_tight,nodes_tight,tensor,tensor_nolean").split(",")
N_TQ = int(os.environ.get("N_TQ", "32"))
DENSE_GATE_MAX_N = int(os.environ.get("DENSE_GATE_MAX_N", "0")) or None
T2_TRAJ = int(os.environ.get("T2_TRAJ", "4"))
# Gate C compares the continuous node machinery (bank evaluated at X +- dx)
# with the grid stencil; its one-sided difference (c - xm)/dx has an f64
# roundoff floor ~ eps/dx (3.6e-12 at N=16384, 1.4e-11 at N=65536), so the
# fixed 1e-12 tripwire set on N <= 4096 must be relaxed at very large N
# (job 3033262, N=16384, failed at 1.99e-12).  Default unchanged.
GATE_C_TOL = float(os.environ.get("GATE_C_TOL", "1e-12"))
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


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def blk(x):
    return jax.block_until_ready(x)


def med_ms(ts):
    return float(np.median(ts) * 1e3)


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                max=float(np.max(v)), min=float(np.min(v)), n=int(v.size))


def hist_of(arr):
    a = np.asarray(arr)
    return {str(int(r_)): int(np.sum(a == r_)) for r_ in np.unique(a)}


def git_commit():
    c = os.environ.get("COMMIT")
    if c:
        return c
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def host_rollout(su, ops, r_w, z0, nu, tol_abs, U_ti, u_of):
    """Host-loop LM rollout (the sep_b1d_scale gate-V loop) for one
    trajectory; returns (mean err, Z (T-1,K))."""
    interior = su.interior
    Tn = b1.NUM_STEPS + 1
    zs = [np.asarray(z0)]
    errs = [rel(np.asarray(u_of(jnp.asarray(z0))), U_ti[0][interior])]
    for t in range(1, Tn):
        z_prev = zs[-1]
        z_init = z_prev if len(zs) < 2 else \
            z_prev + fc.EXTRAP * (zs[-1] - zs[-2])
        prev_c = su.prev_of(jnp.asarray(z_prev, dtype=F64))
        z = jnp.asarray(z_init, dtype=F64)
        r, J = ops["rJ"](z, prev_c, nu)
        rn = float(jnp.linalg.norm(r))
        lam_lm = 1e-6
        if np.isfinite(rn) and rn > tol_abs:
            for attempt in range(1, fc.GN_BUDGET + 1):
                Hm = J.T @ J
                g = J.T @ r
                D = jnp.diag(jnp.diag(Hm)) + 1e-30 * jnp.eye(fc.K, dtype=F64)
                dz = jnp.linalg.solve(Hm + lam_lm * D, -g)
                if not bool(jnp.all(jnp.isfinite(dz))):
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
                    continue
                ndz = float(jnp.linalg.norm(dz))
                if ndz > su.TR_DELTA:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
                    continue
                if ndz <= 1e-12 * (1.0 + float(jnp.linalg.norm(z))):
                    break
                z_new = z + dz
                rn_new = float(ops["rn"](z_new, prev_c, nu))
                if np.isfinite(rn_new) and rn_new < rn:
                    rel_dec = (rn - rn_new) / rn
                    z, rn = z_new, rn_new
                    if rn <= tol_abs:
                        break
                    r, J = ops["rJ"](z, prev_c, nu)
                    lam_lm = max(lam_lm / 3.0, 1e-12)
                    if rel_dec < fc.STALL:
                        break
                else:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
        zs.append(np.asarray(z))
        errs.append(rel(np.asarray(u_of(jnp.asarray(zs[-1]))),
                        U_ti[t][interior]))
    return float(np.mean(errs)), np.stack(zs[1:])


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B1D-TENSOR N={N} "
        f"opt={OPT} arms={ARMS} reps={TIME_REPS} burn={BURN}")
    t_all = time.time()

    su = fc.Setup(CKPT_CACHE, N)
    interior = su.interior
    n_i = su.n_i
    dx = su.dx
    T = b1.NUM_STEPS + 1
    base = json.load(open(BASE_JSON)) if BASE_JSON else None

    report = dict(config=dict(
        pde="burgers1d", kind="b1d_tensor", N=N, K=fc.K, R=fc.R, M=fc.M,
        n_interior=int(n_i), n_test=fc.N_TEST, seed=fc.SEED0,
        test_seed=fc.TEST_SEED, dt=b1.DT, num_steps=b1.NUM_STEPS,
        weak_alpha=b1.WEAK_ALPHA, tr_factor=fc.TR_FACTOR,
        trust_delta=float(su.TR_DELTA), step_tol=fc.STEP_TOL,
        stall=fc.STALL, extrap=fc.EXTRAP, gn_budget=fc.GN_BUDGET,
        ic_budget=fc.IC_BUDGET, time_reps=TIME_REPS, burn=BURN, opt=OPT,
        arms=ARMS, ckpt=CKPT_CACHE, nodes=NODES_NPZ, base_json=BASE_JSON,
        qref=QREF, x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        jax_version=jax.__version__, commit=git_commit(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        gates={}, variants={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- gate J (dense n x n: at min(N, DENSE_GATE_MAX_N)) -----
    N_dense = N if DENSE_GATE_MAX_N is None else min(N, DENSE_GATE_MAX_N)
    n_i_dense = b1.interior_indices_1d(N_dense).size
    jrng = np.random.default_rng(fc.SEED0 + 400)
    gJ = []
    for _ in range(4):
        u = jnp.asarray(jrng.standard_normal(n_i_dense) * 0.5)
        nu = float(np.exp(jrng.uniform(np.log(0.01), np.log(0.1))))
        Jd = jax.jacfwd(lambda v: b1.fom_residual_int(v, u, nu, N_dense))(u)
        dl, d, du = b1.tridiag_jac(u, nu, N_dense)
        Jt = (np.diag(np.asarray(d)) + np.diag(np.asarray(dl)[1:], -1)
              + np.diag(np.asarray(du)[:-1], 1))
        gJ.append(float(np.max(np.abs(np.asarray(Jd) - Jt))
                        / (np.max(np.abs(np.asarray(Jd))) + 1e-300)))
        del Jd, Jt
    report["gates"]["gateJ"] = float(np.max(gJ))
    report["gates"]["gateJ_N"] = int(N_dense)
    log(f"  GATE J (tridiag Jacobian == jacfwd, at N={N_dense}): "
        f"{np.max(gJ):.2e}")
    assert np.max(gJ) < 1e-12

    # ---------------- test data (tri generator) + gate T2 -------------------
    U_test, nu_test = fc.gen_test(N)
    dense_roll = b1.make_rollout_1d(N_dense)
    if N_dense == N:
        U_ref = U_test[:T2_TRAJ]
        sn_d, _ = dense_roll(jnp.asarray(U_test[:T2_TRAJ, 0]),
                             jnp.asarray(nu_test[:T2_TRAJ]))
    else:
        c_, w_, a_, nu_ = b1.sample_params_1d(fc.TEST_SEED, fc.N_TEST)
        U0d = np.stack([b1.blob_ic_1d(N_dense, c_[i], w_[i], a_[i])
                        for i in range(T2_TRAJ)])
        U_ref, _ = b1.make_rollout_1d_tri(N_dense)(
            jnp.asarray(U0d), jnp.asarray(nu_[:T2_TRAJ]))
        U_ref = np.asarray(U_ref)
        sn_d, _ = dense_roll(jnp.asarray(U0d), jnp.asarray(nu_[:T2_TRAJ]))
    gT2 = float(np.max(np.abs(np.asarray(sn_d) - U_ref))
                / (np.max(np.abs(U_ref)) + 1e-300))
    report["gates"]["gateT2"] = gT2
    report["gates"]["gateT2_N"] = int(N_dense)
    report["gates"]["gateT2_traj"] = int(T2_TRAJ)
    log(f"  GATE T2 (tri vs dense truth rollouts, {T2_TRAJ} traj x 50 steps,"
        f" at N={N_dense}): {gT2:.2e}")
    assert gT2 < 1e-8
    del dense_roll, sn_d, U_ref
    report["data"] = dict(test_sum=float(np.sum(U_test)),
                          test_sumsq=float(np.sum(U_test * U_test)),
                          nu_test=[float(v) for v in nu_test])

    Z_tr = np.asarray(su.Z_tr)
    G_int, Phi_j, lam_j, A_j, h_fn = su.G_int, su.Phi_j, su.lam_j, su.A_j, \
        su.h_fn
    Phi_np = np.asarray(Phi_j)
    lam_np = np.asarray(lam_j)
    G_np = np.asarray(G_int)
    u_of = jax.jit(lambda z: G_int @ h_fn(z))

    # ---------------- gate E ------------------------------------------------
    grng = np.random.default_rng(fc.SEED0 + 50)
    gE = []
    for _ in range(4):
        z = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                        + 0.05 * grng.standard_normal(fc.K))
        u = G_int @ h_fn(z)
        Upad = jnp.pad(u, 1)
        lap = (Upad[2:] - 2 * Upad[1:-1] + Upad[:-2]) / dx ** 2
        gE.append(rel(np.asarray(Phi_j.T @ (-lap)),
                      np.asarray(lam_j * (Phi_j.T @ u))))
    report["gates"]["gateE"] = float(np.max(gE))
    log(f"  GATE E (sine modes are exact eigenvectors): {np.max(gE):.2e}")
    assert np.max(gE) < 1e-10

    # ---------------- gate F ------------------------------------------------
    r_or = su.make_full_rw()
    full_rJ = jax.jit(lambda z, p, nu: (r_or(z, p, nu),
                                        jax.jacfwd(r_or)(z, p, nu)))
    gF = []
    for _ in range(4):
        z = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                        + 0.05 * grng.standard_normal(fc.K))
        zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
        nu = float(np.median(nu_test))
        wt = (1.0 + b1.DT * nu * lam_np) ** (-b1.WEAK_ALPHA)
        u = np.asarray(G_int @ h_fn(z))
        up = np.asarray(G_int @ h_fn(zp))
        r_direct = wt * np.asarray(Phi_j.T @ jnp.asarray(np.asarray(
            b1.fom_residual_int(jnp.asarray(u), jnp.asarray(up), nu, N))))
        r_ops, _ = full_rJ(z, su.prev_of(zp), nu)
        gF.append(rel(np.asarray(r_ops), r_direct))
    report["gates"]["gateF"] = float(np.max(gF))
    log(f"  GATE F (weak ops == Phi^T FOM residual): {np.max(gF):.2e}")
    assert np.max(gF) < 1e-10

    # ---------------- gate C (continuous node machinery at grid init) -------
    arms_xw = fc.load_arms(NODES_NPZ)
    X0, w0 = arms_xw["base_tight"]
    pos0 = np.rint(X0 / dx).astype(int) - 1
    assert np.max(np.abs(su.coords_int[pos0, 0] - X0)) < 1e-12
    Hb3 = jnp.asarray(h_fn(jnp.asarray(Z_tr[:3])))
    a_cont = np.asarray(su.adv_nodes(su.g3_of(jnp.asarray(X0)), Hb3))
    Uf3 = np.asarray(Hb3) @ G_np.T
    N_full3 = np.stack([np.asarray(b1.upwind_adv_field_1d(
        jnp.asarray(Uf3[s]), N)) for s in range(3)])
    gateC = float(np.max(np.abs(a_cont - N_full3[:, pos0]))
                  / (np.max(np.abs(N_full3[:, pos0])) + 1e-300))
    report["gates"]["gateC"] = gateC
    log(f"  GATE C (continuous machinery at grid init): {gateC:.2e} "
        f"(tol {GATE_C_TOL:.0e})")
    assert gateC < GATE_C_TOL, (gateC, GATE_C_TOL)

    # ---------------- tensor build + gates TB / TX / TA / T0 / TS -----------
    t0 = time.time()
    T1 = tc.build_T(Phi_np, G_np, dx, chunk=256, reverse=False)
    T2 = tc.build_T(Phi_np, G_np, dx, chunk=97, reverse=True)
    gTB = float(np.max(np.abs(T1 - T2)) / np.max(np.abs(T1)))
    Q = tc.symmetrize(T1)
    report["gates"]["TB_build_order_rel"] = gTB
    report["tensor"] = dict(shape=list(Q.shape), build_secs=time.time() - t0,
                            bytes=int(Q.nbytes),
                            max_abs=float(np.max(np.abs(Q))))
    log(f"  GATE TB (two chunk orders): {gTB:.2e}  [Q {Q.shape}, "
        f"{Q.nbytes/1024:.0f} KiB, {time.time()-t0:.2f}s]")
    if QREF:
        Qr = np.load(QREF)
        gTX = float(np.max(np.abs(Q - Qr)) / np.max(np.abs(Qr)))
        report["gates"]["TX_vs_audit_Q_rel"] = gTX
        log(f"  GATE TX (in-job Q vs E1 CPU-built Q): {gTX:.2e}")
        assert gTX < 1e-12
    Qj = jnp.asarray(Q)
    Hall = np.asarray(h_fn(jnp.asarray(Z_tr)))
    Uall = Hall @ G_np.T
    upw = jax.jit(jax.vmap(lambda u: b1.upwind_adv_field_1d(u, N)))
    Nup = np.asarray(upw(jnp.asarray(Uall)))
    DmU = tc.backward_diff_bank(Uall.T, dx).T
    q_or = Nup @ Phi_np
    q_alg = (Uall * DmU) @ Phi_np
    q_T = np.asarray(jax.jit(lambda Hh: 0.5 * jnp.einsum(
        "ijk,sj,sk->si", Qj, Hh, Hh))(jnp.asarray(Hall)))
    nq = np.linalg.norm(q_or, axis=1) + 1e-300
    gTA = float(np.max(np.linalg.norm(q_T - q_alg, axis=1) / nq))
    report["gates"]["TA_algebraic_identity_max_rel"] = gTA
    log(f"  GATE TA (h^T T h == Phi^T(u D^-u), {len(Hall)} training states):"
        f" {gTA:.2e}")
    assert gTA < 1e-12
    minu = Uall.min(axis=1)
    pos_states = minu > 0
    mis = np.linalg.norm(q_T - q_or, axis=1) / nq
    gT0 = float(np.max(mis[pos_states])) if np.any(pos_states) else None
    report["gates"]["T0_all_positive_states_max_rel"] = gT0
    report["gates"]["T0_n_all_positive_states"] = int(np.sum(pos_states))
    report["TS_train_states"] = dict(
        mismatch_rel=stats(mis), frac_points_u_le_0=float(np.mean(Uall <= 0)),
        frac_states_all_positive=float(np.mean(pos_states)),
        min_u=float(Uall.min()))
    log(f"  GATE T0 (tensor == oracle on the {int(np.sum(pos_states))} "
        f"all-positive states): {gT0:.2e} | all states: mismatch median "
        f"{np.median(mis):.2e} max {np.max(mis):.2e}, frac points u<=0 "
        f"{np.mean(Uall <= 0):.3%}, min u {Uall.min():.2e}")
    if gT0 is not None:
        assert gT0 < 1e-12

    # ---------------- gate TQ: tensor vs oracle residual at 32 states -------
    r_T = su.make_tensor_rw(Q)
    rJ_T = jax.jit(lambda z, p, nu: (r_T(z, p, nu),
                                     jax.jacfwd(r_T)(z, p, nu)))
    qrng = np.random.default_rng(fc.SEED0 + 500)
    tq = []
    for si in range(N_TQ):
        i = qrng.integers(len(Z_tr))
        pert = si >= N_TQ // 2
        z = Z_tr[i] + (0.05 * qrng.standard_normal(fc.K) if pert else 0.0)
        zp = Z_tr[qrng.integers(len(Z_tr))]
        nu = float(np.exp(qrng.uniform(np.log(0.01), np.log(0.1))))
        pv = su.prev_of(jnp.asarray(zp))
        ro, Jo = [np.asarray(v) for v in full_rJ(jnp.asarray(z), pv, nu)]
        rt, Jt = [np.asarray(v) for v in rJ_T(jnp.asarray(z), pv, nu)]
        uu = np.asarray(u_of(jnp.asarray(z)))
        tq.append(dict(perturbed=bool(pert), nu=nu, r_rel=rel(rt, ro),
                       J_rel=rel(Jt, Jo),
                       g_scaled=float(np.linalg.norm(Jt.T @ rt - Jo.T @ ro)
                                      / (np.linalg.norm(Jo)
                                         * np.linalg.norm(ro) + 1e-300)),
                       min_u=float(uu.min()), n_neg=int(np.sum(uu <= 0))))
    report["gates"]["TQ"] = dict(
        r_rel=stats([t_["r_rel"] for t_ in tq]),
        J_rel=stats([t_["J_rel"] for t_ in tq]),
        g_scaled_max=float(max(t_["g_scaled"] for t_ in tq)),
        n_states_with_neg=int(sum(1 for t_ in tq if t_["n_neg"] > 0)),
        min_u=float(min(t_["min_u"] for t_ in tq)), rows=tq)
    g_ = report["gates"]["TQ"]
    log(f"  GATE TQ (tensor vs oracle residual, {N_TQ} states, recorded not "
        f"asserted): r rel median {g_['r_rel']['median']:.2e} max "
        f"{g_['r_rel']['max']:.2e}; J rel max {g_['J_rel']['max']:.2e}; "
        f"states with u<=0: {g_['n_states_with_neg']}/{N_TQ}; min u "
        f"{g_['min_u']:.2e}")
    save()

    # ---------------- gate G (Gram-space IC == banked IC), traj 0 -----------
    ic_ref = fc.make_ic_ref(su)
    ic_fast = fc.make_ic_fast(su, OPT)

    def banked_ic_fit(u0_int):
        tgt = jnp.asarray(u0_int, dtype=F64)
        tn_ = float(np.linalg.norm(u0_int))

        def r_of(z):
            return (G_int @ h_fn(z) - tgt) / tn_
        rJ_f = jax.jit(lambda z: (r_of(z), jax.jacfwd(r_of)(z)))
        best = None
        for z0 in np.asarray(su.Z0S):
            z = jnp.asarray(z0, dtype=F64)
            lam_lm = 1e-6
            r, J = rJ_f(z)
            val = float(jnp.linalg.norm(r))
            for _ in range(fc.IC_BUDGET):
                Hm = J.T @ J
                g = J.T @ r
                dz = jnp.linalg.solve(
                    Hm + lam_lm * jnp.diag(jnp.diag(Hm))
                    + 1e-30 * jnp.eye(fc.K, dtype=F64), -g)
                z_new = z + dz
                v_new = float(jnp.linalg.norm(r_of(z_new)))
                if np.isfinite(v_new) and v_new < val:
                    z, val = z_new, v_new
                    r, J = rJ_f(z)
                    lam_lm = max(lam_lm / 3.0, 1e-12)
                else:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
            if best is None or val < best[1]:
                best = (np.asarray(z), val)
        return best

    u00 = U_test[0, 0][interior]
    zg, _ = ic_ref(jnp.asarray(u00))
    ug = np.asarray(u_of(zg))
    zb, _ = banked_ic_fit(u00)
    ub = np.asarray(u_of(jnp.asarray(zb)))
    tn0 = np.linalg.norm(u00)
    mis_g = float(np.linalg.norm(ug - u00) / tn0)
    mis_b = float(np.linalg.norm(ub - u00) / tn0)
    gateG = abs(mis_g - mis_b) / (mis_b + 1e-300)
    report["gates"]["gateG"] = dict(gram_misfit=mis_g, banked_misfit=mis_b,
                                    rel_diff=float(gateG))
    log(f"  GATE G (Gram-space IC fit == banked fit): gram {mis_g:.6e} "
        f"banked {mis_b:.6e} rel diff {gateG:.2e}")
    assert gateG < 1e-6

    def tol_abs_of(ti):
        u0 = U_test[ti, 0]
        u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
        return fc.STEP_TOL * u_scale * float(np.sqrt(n_i))

    # ---------------- arms -------------------------------------------------
    Z_keep = {}          # arm -> {"ref": [Z per traj], "fast": [...]}
    for arm in ARMS:
        Qarm = None
        opt_arm = dict(OPT)
        if arm == "oracle":
            r_w = r_or
            X_v = w_v = None
        elif arm in ("tensor", "tensor_nolean"):
            r_w = r_T
            X_v = w_v = None
            Qarm = Q
            if arm == "tensor_nolean":
                opt_arm["lean"] = False
        else:
            X_v, w_v = arms_xw[arm]
            r_w = su.make_sampled_rw(X_v, w_v)
        ops_ref = fc.make_device_ref(su, r_w)
        ops_fast = fc.make_device_fast(su, X_v, w_v, opt_arm, Q=Qarm)
        out = dict(m=int(n_i if X_v is None else len(X_v)), opt=opt_arm,
                   rollout=[])
        Z_keep[arm] = dict(ref=[], fast=[])
        reasons_ref_all, reasons_fast_all = {}, {}
        errs_ref_all, errs_fast_all = [], []
        lat_dev_max = ic_dev_max = 0.0

        for ti in range(fc.N_TEST):
            nu = float(nu_test[ti])
            tol_abs = tol_abs_of(ti)
            u0i = jnp.asarray(U_test[ti, 0][interior])

            def one_rep():
                t0 = time.perf_counter()
                z0r, v0r = blk(ic_ref(u0i))
                t1 = time.perf_counter()
                z0f, v0f = blk(ic_fast(u0i))
                t2 = time.perf_counter()
                Rr = blk(ops_ref["rollout"](z0r, nu, tol_abs, fc.GN_BUDGET))
                t3 = time.perf_counter()
                Rf = blk(ops_fast["rollout"](z0f, nu, tol_abs, fc.GN_BUDGET))
                t4 = time.perf_counter()
                Ff = blk(su.decode_all(jnp.concatenate([z0f[None], Rf[0]],
                                                       axis=0)))
                t5 = time.perf_counter()
                return dict(ic_ref=t1 - t0, ic_fast=t2 - t1,
                            roll_ref=t3 - t2, roll_fast=t4 - t3,
                            dec=t5 - t4), (z0r, v0r, z0f, v0f, Rr, Rf, Ff)

            tm = dict(ic_ref=[], ic_fast=[], roll_ref=[], roll_fast=[],
                      dec=[])
            last = None
            for rep_ in range(BURN + TIME_REPS):
                t_, last = one_rep()
                if rep_ >= BURN:
                    for k_ in tm:
                        tm[k_].append(t_[k_])
            # accuracy from the LAST TIMED invocation
            z0r, v0r, z0f, v0f, (Zr, rnr, nJr, rer), \
                (Zf, rnf, nJf, ref_), Ff = last
            Zr, Zf = np.asarray(Zr), np.asarray(Zf)
            Fr = np.asarray(su.decode_all(
                jnp.concatenate([z0r[None], jnp.asarray(Zr)], axis=0)))
            Ff = np.asarray(Ff)
            er_t = [rel(Fr[t], U_test[ti, t][interior]) for t in range(T)]
            ef_t = [rel(Ff[t], U_test[ti, t][interior]) for t in range(T)]
            er, ef = float(np.mean(er_t)), float(np.mean(ef_t))
            lat_dev = float(np.max(np.abs(Zr - Zf)))
            ic_dev = float(np.max(np.abs(np.asarray(z0r) - np.asarray(z0f))))
            lat_dev_max = max(lat_dev_max, lat_dev)
            ic_dev_max = max(ic_dev_max, ic_dev)
            errs_ref_all.append(er)
            errs_fast_all.append(ef)
            hr, hf = hist_of(rer), hist_of(ref_)
            for d_, h_ in ((reasons_ref_all, hr), (reasons_fast_all, hf)):
                for k_, v_ in h_.items():
                    d_[k_] = d_.get(k_, 0) + v_
            Z_keep[arm]["ref"].append(Zr)
            Z_keep[arm]["fast"].append(Zf)
            out["rollout"].append(dict(
                traj=ti, nu=nu, ic_resid_ref=float(v0r),
                ic_resid_fast=float(v0f), err_ref=er, err_fast=ef,
                err_t0=ef_t[0], err_t1=ef_t[1], err_last=ef_t[-1],
                lat_dev_ref_vs_fast=lat_dev, ic_dev=ic_dev,
                mean_njac_ref=float(np.mean(np.asarray(nJr))),
                mean_njac_fast=float(np.mean(np.asarray(nJf))),
                stop_reasons_ref=hr, stop_reasons_fast=hf,
                rn_final_fast=[float(x) for x in np.asarray(rnf)],
                times={k_: [float(x) for x in v] for k_, v in tm.items()}))
            log(f"  [{arm}] traj {ti}: err ref {er:.6e} fast {ef:.6e} "
                f"latdev {lat_dev:.2e} reasons {hf} | ic "
                f"{med_ms(tm['ic_ref']):.2f}->{med_ms(tm['ic_fast']):.2f} ms"
                f"  roll {med_ms(tm['roll_ref']):.2f}->"
                f"{med_ms(tm['roll_fast']):.2f} ms")

        agg = lambda key: float(np.median(
            [x for r_ in out["rollout"] for x in r_["times"][key]]) * 1e3)
        out["ic_ref_ms"] = agg("ic_ref")
        out["ic_fast_ms"] = agg("ic_fast")
        out["roll_ref_ms"] = agg("roll_ref")
        out["roll_fast_ms"] = agg("roll_fast")
        out["dec_ms"] = agg("dec")
        out["e2e_ref_ms"] = float(np.median(
            [a + b_ + c_ for r_ in out["rollout"]
             for a, b_, c_ in zip(r_["times"]["ic_ref"],
                                  r_["times"]["roll_ref"],
                                  r_["times"]["dec"])]) * 1e3)
        out["e2e_fast_ms"] = float(np.median(
            [a + b_ + c_ for r_ in out["rollout"]
             for a, b_, c_ in zip(r_["times"]["ic_fast"],
                                  r_["times"]["roll_fast"],
                                  r_["times"]["dec"])]) * 1e3)
        out["err_ref_mean"] = float(np.mean(errs_ref_all))
        out["err_fast_mean"] = float(np.mean(errs_fast_all))
        out["stop_reasons_ref"] = reasons_ref_all
        out["stop_reasons_fast"] = reasons_fast_all
        out["parity"] = dict(
            err_rel_diff_fast_vs_ref=abs(out["err_fast_mean"]
                                         - out["err_ref_mean"])
            / out["err_ref_mean"],
            lat_dev_max=lat_dev_max, ic_dev_max=ic_dev_max)
        if base is not None and arm in base.get("variants", {}):
            bv = base["variants"][arm]
            per = [abs(r_["err_ref"] - b_["err_mean"]) / b_["err_mean"]
                   for r_, b_ in zip(out["rollout"], bv["rollout"])]
            out["parity"]["err_base_json"] = bv["rollout_err_mean"]
            out["parity"]["err_rel_diff_ref_vs_base"] = \
                abs(out["err_ref_mean"] - bv["rollout_err_mean"]) \
                / bv["rollout_err_mean"]
            out["parity"]["err_rel_diff_fast_vs_base"] = \
                abs(out["err_fast_mean"] - bv["rollout_err_mean"]) \
                / bv["rollout_err_mean"]
            out["parity"]["per_traj_rel_diff_ref_vs_base_max"] = \
                float(max(per))
            out["parity"]["base_e2e_ms_median"] = bv["e2e_ms_median"]
            out["parity"]["base_stop_reasons"] = {}
            for r_ in bv["rollout"]:
                for k_, v_ in r_["stop_reasons"].items():
                    out["parity"]["base_stop_reasons"][k_] = \
                        out["parity"]["base_stop_reasons"].get(k_, 0) + v_
        report["variants"][arm] = out
        p = out["parity"]
        log(f"  [{arm}] SUMMARY err ref {out['err_ref_mean']:.6e} fast "
            f"{out['err_fast_mean']:.6e} (fast vs ref {p['err_rel_diff_fast_vs_ref']:.1e}"
            f"; vs base json {p.get('err_rel_diff_fast_vs_base', float('nan')):.1e})"
            f" reasons {reasons_fast_all} | e2e ref {out['e2e_ref_ms']:.2f} "
            f"fast {out['e2e_fast_ms']:.2f} ms (ic {out['ic_fast_ms']:.2f} + "
            f"roll {out['roll_fast_ms']:.2f} + dec {out['dec_ms']:.2f})")
        save()

    # ---------------- tensor vs oracle comparison --------------------------
    cmp = {}
    if "oracle" in report["variants"]:
        vo = report["variants"]["oracle"]
        for arm in ARMS:
            if arm == "oracle":
                continue
            va = report["variants"][arm]
            rows = []
            for ti in range(fc.N_TEST):
                ro_, ra_ = vo["rollout"][ti], va["rollout"][ti]
                rows.append(dict(
                    traj=ti, err_oracle=ro_["err_fast"], err_arm=ra_["err_fast"],
                    abs_diff=abs(ra_["err_fast"] - ro_["err_fast"]),
                    abs_diff_ref=abs(ra_["err_ref"] - ro_["err_ref"]),
                    lat_dev_fast=float(np.max(np.abs(
                        Z_keep[arm]["fast"][ti] - Z_keep["oracle"]["fast"][ti]))),
                    lat_dev_ref=float(np.max(np.abs(
                        Z_keep[arm]["ref"][ti] - Z_keep["oracle"]["ref"][ti]))),
                    reasons_equal=(ra_["stop_reasons_fast"]
                                   == ro_["stop_reasons_fast"]),
                    njac_equal=(ra_["mean_njac_fast"] == ro_["mean_njac_fast"])))
            cmp[arm] = dict(
                per_traj=rows,
                err_abs_diff_max=float(max(r_["abs_diff"] for r_ in rows)),
                err_abs_diff_max_ref=float(max(r_["abs_diff_ref"]
                                               for r_ in rows)),
                lat_dev_fast_max=float(max(r_["lat_dev_fast"] for r_ in rows)),
                lat_dev_ref_max=float(max(r_["lat_dev_ref"] for r_ in rows)),
                stop_hist_identical=bool(va["stop_reasons_fast"]
                                         == vo["stop_reasons_fast"]),
                stop_hist_identical_per_traj=bool(all(r_["reasons_equal"]
                                                      for r_ in rows)),
                e2e_fast_ms=va["e2e_fast_ms"], e2e_oracle_fast_ms=vo["e2e_fast_ms"],
                e2e_ratio_vs_oracle=va["e2e_fast_ms"] / vo["e2e_fast_ms"],
                roll_ratio_vs_oracle=va["roll_fast_ms"] / vo["roll_fast_ms"])
            if "base_tight" in report["variants"]:
                vb = report["variants"]["base_tight"]
                cmp[arm]["e2e_ratio_vs_base_tight"] = \
                    va["e2e_fast_ms"] / vb["e2e_fast_ms"]
                cmp[arm]["roll_ratio_vs_base_tight"] = \
                    va["roll_fast_ms"] / vb["roll_fast_ms"]
            c = cmp[arm]
            log(f"  [{arm} vs oracle] err |diff| max {c['err_abs_diff_max']:.2e}"
                f" (ref path {c['err_abs_diff_max_ref']:.2e}); latent dev max "
                f"{c['lat_dev_fast_max']:.2e}; stop hist identical: "
                f"{c['stop_hist_identical']} (per traj "
                f"{c['stop_hist_identical_per_traj']}); e2e ratio vs oracle "
                f"{c['e2e_ratio_vs_oracle']:.3f}, vs base_tight "
                f"{c.get('e2e_ratio_vs_base_tight', float('nan')):.3f}")
    report["comparison"] = cmp
    save()

    # ---------------- gate V: device tensor rollout == host-loop ------------
    if "tensor" in report["variants"]:
        ops_T = fc.make_device_ref(su, r_T)
        z0 = ic_ref(jnp.asarray(U_test[0, 0][interior]))[0]
        err_py, Zh = host_rollout(su, ops_T, r_T, z0, float(nu_test[0]),
                                  tol_abs_of(0), U_test[0], u_of)
        err_dev = report["variants"]["tensor"]["rollout"][0]["err_ref"]
        gateV = abs(err_dev - err_py) / (err_py + 1e-300)
        ldev = float(np.max(np.abs(Zh - Z_keep["tensor"]["ref"][0])))
        report["gates"]["gateV"] = dict(err_host=err_py, err_device=err_dev,
                                        rel_diff=float(gateV),
                                        lat_dev=ldev, arm="tensor")
        log(f"  GATE V (tensor arm: device rollout vs host-loop, traj 0): "
            f"host {err_py:.6e} device {err_dev:.6e} rel diff {gateV:.2e} "
            f"latdev {ldev:.1e}")
        assert gateV < 1e-3

    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
