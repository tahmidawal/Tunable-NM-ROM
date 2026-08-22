"""Separable-decoder Burgers-2D cell: train (no POD), weak NM-ROM rollout.

Two arms through the SAME incumbent weak operators and LM stepper:
  meshfree : blat_common.make_weak_ops(dec, ...) -- the network runs in-loop
  cached   : identical formulas with the stencil feature banks G_st (m,5,r)
             cached once; assembled by blat_common._finish_ops so the solver,
             acceptance rule, and rollout are bit-identical code.
GATE 0: weak residual/Jacobian of the two arms agree <= 1e-12 relative.
The FOM-exact sign-upwind stencil is preserved exactly (features are evaluated
at the stencil points; the upwind switch acts on decoded values as in the FOM).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402  (path set by sc)

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "64"))
M_MODES = int(os.environ.get("M", str(4 * K)))
MQ = int(os.environ.get("MQ", str(4 * M_MODES)))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "4"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
OUT = os.environ.get("OUT", "sep_burgers.json")
CKPT = os.environ.get("CKPT", f"sep_burgers_N{N}_K{K}_R{R}.pkl")


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} K={K} R={R} M={M_MODES} m={MQ} steps={STEPS} seed={SEED0}")
    t_all = time.time()
    report = dict(config=dict(
        pde="burgers2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        n_test=N_TEST, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
        num_steps=bc.NUM_STEPS, dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0,
        data_seed=bc.SEED, max_snaps=MAX_SNAPS,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"incumbent weak upwind M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="blat_common lm_step_jit / rollout_jit (incumbent), both arms",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")), rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ data (regenerated from seed) -------------------------
    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)            # (n_traj, T, n^2)
    U_test = np.asarray(d["U_test"], dtype=np.float64)  # (N_TEST_all, T, n^2)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    coords_int = coords[interior]
    report["data"] = dict(n_traj=int(n_traj), T=int(T), n2=int(n2),
                          fingerprint=bc.data_fingerprint(U))

    # training states: every (trajectory, time) state, interior values
    S_all = U.reshape(n_traj * T, n2)[:, interior]
    rng = np.random.default_rng(SEED0)
    if S_all.shape[0] > MAX_SNAPS:
        pick = np.sort(rng.choice(S_all.shape[0], MAX_SNAPS, replace=False))
    else:
        pick = np.arange(S_all.shape[0])
    S_tr = S_all[pick]
    report["data"]["n_states_trained"] = int(S_tr.shape[0])

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, S_tr, K, R,
        steps=STEPS, lr=LR, tag=f"burgers N={N} k={K} r={R}")
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    # trust region exactly as the accepted recipe: factor x training radius
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)

    # ------------------ EQ + incumbent weak ops ------------------------------
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    colloc = bc.fit_eq_weights(dec, N, M_MODES, MQ, Z_tr[eq_pick], kind="weak",
                               pool="grid")
    report["eq"] = colloc["info"]
    ops_ref = bc.make_weak_ops(dec, N, colloc, kind="weak", M=M_MODES,
                               solver="lspg")

    # ------------------ cached fast ops (same formulas, banks cached) --------
    kx, ky, Phi, lam, _lamc = bc.test_modes(N, M_MODES)
    idx = np.asarray(colloc["idx"])
    m = idx.size
    w_q = jnp.asarray(colloc["w"], dtype=F64)
    pos = np.searchsorted(interior, idx)
    assert np.all(interior[pos] == idx)
    Phi_q = jnp.asarray(Phi[pos]) * w_q[:, None]                  # (m, M')
    lam_j = jnp.asarray(lam, dtype=F64)
    st = bc.stencil_indices(idx, N)                               # (m, 5)
    G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)
    G_all = dec.feat_at(coords)                                   # (n^2, r)
    h_fn = dec.head_fn()
    dx = 1.0 / (N - 1)

    def u_and_N_fast(z):
        us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
        c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
        return c, c * (ux + uy)

    def prev_of_fast(z):
        return jnp.einsum("mr,r->m", G_st[:, 0, :], h_fn(z))

    def r_w_fast(z, prev_c, nu):
        wt = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
        u, Nu = u_and_N_fast(z)
        pu = Phi_q.T @ u
        return wt * (Phi_q.T @ (u - prev_c) + bc.DT * ((Phi_q.T @ Nu)
                                                       + nu * lam_j * pu))

    def d_c_fast(z):
        return u_and_N_fast(z)[0]

    def rJ_fast(z, prev_c, nu):
        return (r_w_fast(z, prev_c, nu), jax.jacfwd(r_w_fast)(z, prev_c, nu),
                Phi_q.T @ jax.jacfwd(d_c_fast)(z))

    def full_fast(z):
        return G_all @ h_fn(z)

    ops_fast = bc._finish_ops(rJ_fast, r_w_fast, prev_of_fast, full_fast, m,
                              "lspg")
    ops_fast["M"] = M_MODES
    ops_fast["tol_scale"] = float(np.sqrt(interior.size))
    ops_fast["colloc_used"] = colloc

    # ------------------ GATE 0: identity of the two arms ---------------------
    g0 = []
    for _ in range(5):
        zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))]
                         + 0.05 * rng.standard_normal(K))
        zp = jnp.asarray(Z_tr[rng.integers(len(Z_tr))])
        prev_c = ops_ref["prev_of"](zp)
        nu = float(np.median(nu_test))
        ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
        rb, Jb, _ = ops_fast["rJ"](zt, prev_c, nu)
        pa = ops_ref["prev_of"](zt); pb = ops_fast["prev_of"](zt)
        g0.append(max(
            float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
            float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300)),
            float(jnp.max(jnp.abs(pa - pb)) / (jnp.max(jnp.abs(pa)) + 1e-300))))
    report["gate0_max_rel_dev"] = float(np.max(g0))
    sc.log(f"  GATE 0 (incumbent vs cached weak r/J identity): "
           f"max rel dev {np.max(g0):.2e}")
    assert np.max(g0) < 1e-12, "gate 0 failed"
    save()

    # ------------------ rollouts: error (untimed) + timed jit rollout --------
    t0_mask = (pick % T) == 0            # codes of t=0 training states
    Z0_states = Z_tr[t0_mask] if t0_mask.any() else Z_tr[:8]
    zbar = Z_tr.mean(0)
    n_test = min(N_TEST, U_test.shape[0])
    for arm, ops in (("meshfree", ops_ref), ("cached", ops_fast)):
        rows = []
        for i in range(n_test):
            u0 = U_test[i, 0]
            u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
            inits = {"mean": zbar}
            for j, zc in enumerate(Z0_states[:8]):
                inits[f"tr{j}"] = zc
            z0, ic_rel, ic_info = bc.fit_ic(dec, N, u0, inits)
            ro = bc.rollout(dec, N, ops, z0, float(nu_test[i]), u_scale,
                            U_true=U_test[i])
            # timed: the fully jitted on-device rollout, same tolerance rule
            us = jnp.full((bc.NUM_STEPS,),
                          bc.GN_TOL * u_scale * ops["tol_scale"], dtype=F64)
            med, _ = sc.time_fn(lambda _z=z0, _nu=float(nu_test[i]), _us=us:
                                ops["rollout_jit"](_z, _nu, _us,
                                                   bc.GN_BUDGET)[0]
                                .block_until_ready())
            rows.append(dict(traj=i, ic_rel=float(ic_rel),
                             ic_init=ic_info.get("init"),
                             traj_rel=ro.get("traj_rel"),
                             per_time_mean=float(np.nanmean(ro["per_time"])),
                             per_time_max=float(np.nanmax(ro["per_time"])),
                             n_done=int(ro["n_done"]),
                             jac_total=int(np.sum(ro["iters"])),
                             reasons={r_: ro["reasons"].count(r_)
                                      for r_ in set(ro["reasons"])},
                             rollout_ms=med * 1e3))
            sc.log(f"   {arm:8s} traj {i}: ic {ic_rel:.2e}  err "
                   f"{ro.get('traj_rel', float('nan')):.3e}  jac "
                   f"{int(np.sum(ro['iters']))}  n_done {ro['n_done']}"
                   f"  rollout {med*1e3:8.2f} ms")
        errs = [r_["traj_rel"] for r_ in rows if r_["traj_rel"] is not None
                and np.isfinite(r_["traj_rel"])]
        report["rows"].append(dict(
            pde="burgers2d", method=f"sep_{arm}", N=N, k=K, r=R, M=M_MODES,
            m=int(m), err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            rollout_ms_median=float(np.median([r_["rollout_ms"] for r_ in rows])),
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            n_blowups=int(sum(r_["n_done"] < bc.NUM_STEPS for r_ in rows)),
            per_traj=rows, n_test=n_test))
        save()

    # ------------------ FOM reference rollout timing (same job) --------------
    fom_roll, _ = bc.bf.make_rollout(N)
    per = []
    for i in range(n_test):
        U0 = jnp.asarray(U_test[i:i + 1, 0])
        nu1 = jnp.asarray(nu_test[i:i + 1])
        med, _ = sc.time_fn(lambda _u=U0, _n=nu1:
                            fom_roll(_u, _n)[0].block_until_ready())
        per.append(med)
    report["fom_rollout_ms_median"] = float(np.median(per)) * 1e3
    sc.log(f"   FOM reference rollout: {np.median(per)*1e3:.1f} ms/trajectory")

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
