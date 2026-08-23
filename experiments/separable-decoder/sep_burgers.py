"""Separable-decoder Burgers-2D cell: train (no POD), weak NM-ROM rollout.

Two arms through the SAME incumbent weak operators and LM stepper:
  meshfree : blat_common.make_weak_ops(dec, ...) -- the network runs in-loop
  cached   : identical formulas with the stencil feature banks G_st (m,5,r)
             cached once; assembled by blat_common._finish_ops so the solver,
             acceptance rule, and rollout are bit-identical code.
GATE 0: weak residual/Jacobian of the two arms agree <= 1e-12 relative.
The FOM-exact sign-upwind stencil is preserved exactly.

MEASUREMENT RULES (HANDOFF 2026-08-23; each fixes an N=64 audit FAIL):
  - END-TO-END timing: the timed ROM path is IC-latent-fit (from the known u0
    only) -> jitted 50-step rollout -> FULL-GRID decode of all 51 states, with
    the (ic, rollout, decode) split reported per repetition;
  - the IC fit is a jitted batched LM (data misfit to u0 only) whose inits are
    the nearest training t=0 codes -- an optimized ONLINE cost, charged in full;
  - errors come FROM the timed invocation's decoded fields (last timed rep);
    the deviation vs an untimed incumbent bc.rollout is recorded;
  - ALL raw timing repetitions retained; balanced unit order (reversed on odd
    reps); stop-reason distributions recorded next to every error;
  - classical baselines in the SAME job on the SAME GPU: a tolerance-terminated
    Newton ladder (the strong baseline) AND the truth-generating fixed-8-Newton
    rollout (labelled over-solved -- never the headline comparator).
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
REPS = int(os.environ.get("REPS", "7"))
NIC = int(os.environ.get("NIC_INITS", "8"))
IC_ITERS = int(os.environ.get("IC_ITERS", "60"))
NTOLS = [float(v) for v in os.environ.get("NTOLS", "3e-2,1e-2,1e-3,1e-5").split(",")]
NEWTON_MAX = int(os.environ.get("NEWTON_MAX", "8"))
DATA_CACHE = os.environ.get("DATA_CACHE", "")
OUT = os.environ.get("OUT", "sep_burgers.json")
CKPT = os.environ.get("CKPT", f"sep_burgers_N{N}_K{K}_R{R}.pkl")
ARCH = sc.arch_from_env()

REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}


def burn_in(seconds=1.5, n=1024):
    a = jnp.ones((n, n), F64)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = (a @ a) / jnp.float64(n)
    a.block_until_ready()


def make_newton_tol(n, ntol, max_it):
    """Tolerance-terminated Newton/BiCGStab implicit stepper -- the STRONG
    classical baseline.  Identical residual, linear solver, and NaN guard as
    the truth generator (bf.make_rollout), but the Newton loop stops as soon
    as ||R(u)|| <= ntol * ||u_prev|| instead of always doing 8 iterations.
    Unbatched (one trajectory), 50-step lax.scan; returns
    (snaps (51, n^2), rel_res (50,), newton_iters (50,))."""
    _, residual = bc.bf.make_rollout(n)
    lin_tol, lin_maxiter = bc.bf.LIN_TOL, bc.bf.LIN_MAXITER

    def step(u_prev, nu):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)
        r0 = residual(u_prev, u_prev, nu)

        def cond(s):
            _u, it, _r, rn = s
            return (rn > ntol * u_scale) & (it < max_it)

        def body(s):
            u, it, r, rn = s
            Jv = lambda v: jax.jvp(
                lambda uu: residual(uu, u_prev, nu), (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=lin_maxiter)
            # same guard as the truth generator: skip the update on a machine-
            # eps residual or a non-finite step (BiCGStab breakdown landmine)
            ok = (rn > 1e-12 * u_scale) & jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            r2 = residual(u2, u_prev, nu)
            return u2, it + jnp.int32(1), r2, jnp.linalg.norm(r2)

        u, it, _r, rn = jax.lax.while_loop(
            cond, body, (u_prev, jnp.int32(0), r0, jnp.linalg.norm(r0)))
        return u, rn / u_scale, it

    @jax.jit
    def roll(u0, nu):
        def body(u, _):
            u2, rel, it = step(u, nu)
            return u2, (u2, rel, it)
        _, (snaps, rels, its) = jax.lax.scan(body, u0, None,
                                             length=bc.NUM_STEPS)
        return jnp.concatenate([u0[None], snaps], 0), rels, its

    return roll


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} K={K} R={R} M={M_MODES} m={MQ} steps={STEPS} seed={SEED0} "
           f"arch={ARCH}")
    t_all = time.time()
    report = dict(config=dict(
        pde="burgers2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        n_test=N_TEST, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
        num_steps=bc.NUM_STEPS, dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0,
        data_seed=bc.SEED, max_snaps=MAX_SNAPS, reps=REPS, nic_inits=NIC,
        ic_iters=IC_ITERS, newton_tols=NTOLS, newton_max=NEWTON_MAX,
        arch_cfg=ARCH,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"incumbent weak upwind M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="blat_common lm_step_jit / rollout_jit (incumbent), both arms",
        timing="END-TO-END: ic fit (u0->z0, online) + jitted rollout + "
               "full-grid decode of all 51 states; split + raw reps retained; "
               "balanced unit order; errors from the timed invocation",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")), rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ data (regenerated from seed, in-job cache) -----------
    if DATA_CACHE and os.path.exists(DATA_CACHE):
        d = {k_: v for k_, v in np.load(DATA_CACHE).items()}
        sc.log(f"  data: loaded in-job cache {DATA_CACHE} (regenerated from "
               f"seed earlier in THIS job)")
    else:
        d = bc.build_data(N)
        if DATA_CACHE:
            np.savez(DATA_CACHE, **{k_: np.asarray(v) for k_, v in d.items()
                                    if isinstance(v, np.ndarray) or np.isscalar(v)})
            sc.log(f"  data: cached to {DATA_CACHE} for later cells in this job")
    U = np.asarray(d["U"], dtype=np.float64)            # (n_traj, T, n^2)
    U_test = np.asarray(d["U_test"], dtype=np.float64)  # (N_TEST_all, T, n^2)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    coords_int = coords[interior]
    report["data"] = dict(n_traj=int(n_traj), T=int(T), n2=int(n2),
                          fingerprint=bc.data_fingerprint(U),
                          cache=bool(DATA_CACHE and os.path.exists(DATA_CACHE)))

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
        steps=STEPS, lr=LR, tag=f"burgers N={N} k={K} r={R}", **ARCH)
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

    # ------------------ online IC fit (u0 -> z0), both arms ------------------
    # Inits: the training t=0 codes nearest to u0 (distance computed through
    # the offline QR of the readout bank) + the mean code.  Fit: jitted batched
    # LM on the data misfit to u0.  Uses ONLY u0 (known data) -- no truth.
    t0_mask = (pick % T) == 0
    Z_t0 = Z_tr[t0_mask][:64] if t0_mask.any() else Z_tr[:8]
    n_t0 = Z_t0.shape[0]
    nic = min(NIC, n_t0 + 1)
    report["config"]["n_t0_codes"] = int(n_t0)
    report["config"]["nic_used"] = int(nic)
    zbar = Z_tr.mean(0)
    Z_t0j = jnp.asarray(Z_t0)
    zbar_j = jnp.asarray(zbar)

    Q, Rm = jnp.linalg.qr(G_all)                 # (n^2, r), (r, r) offline
    QT = Q.T
    RH_t0 = h_fn(Z_t0j) @ Rm.T                   # (n_t0, r) offline

    def res_c(z, qt):
        return Rm @ h_fn(z) - qt

    fit_c = sc.make_batched_lm(res_c, IC_ITERS)
    coords_j = jnp.asarray(coords)

    def res_m(z, u0):
        return dec(z, coords_j) - u0

    fit_m = sc.make_batched_lm(res_m, IC_ITERS)

    def select_inits(qt):
        d_ = jnp.linalg.norm(RH_t0 - qt[None, :], axis=1)
        _, sel = jax.lax.top_k(-d_, nic - 1)
        return jnp.concatenate([Z_t0j[sel], zbar_j[None]], 0), sel

    @jax.jit
    def ic_cached(u0):
        qt = QT @ u0
        z0s, sel = select_inits(qt)
        zs, vs = fit_c(z0s, qt)
        j = jnp.argmin(vs)
        const2 = jnp.maximum(jnp.sum(u0 * u0) - jnp.sum(qt * qt), 0.0)
        rel = jnp.sqrt(vs[j] ** 2 + const2) / jnp.linalg.norm(u0)
        return zs[j], rel, j

    @jax.jit
    def ic_meshfree(u0):
        qt = QT @ u0                              # init selection only
        z0s, sel = select_inits(qt)
        zs, vs = fit_m(z0s, u0)
        j = jnp.argmin(vs)
        rel = vs[j] / jnp.linalg.norm(u0)
        return zs[j], rel, j

    # full-grid readouts (51 states), the symmetric-output requirement
    decode_cached = jax.jit(
        lambda z0, Z: h_fn(jnp.concatenate([z0[None], Z], 0)) @ G_all.T)
    decode_meshfree = jax.jit(
        lambda z0, Z: jax.vmap(lambda zz: dec(zz, coords_j))(
            jnp.concatenate([z0[None], Z], 0)))

    arms = dict(
        cached=dict(ic=ic_cached, ops=ops_fast, decode=decode_cached),
        meshfree=dict(ic=ic_meshfree, ops=ops_ref, decode=decode_meshfree))

    # IC-fit equivalence check vs the incumbent bc.fit_ic (untimed, 2 traj)
    ic_check = []
    for i in range(min(2, N_TEST, U_test.shape[0])):
        u0 = U_test[i, 0]
        inits = {"mean": zbar}
        for j, zc in enumerate(Z_t0[:8]):
            inits[f"tr{j}"] = zc
        t0c = time.time()
        _zi, rel_inc, info_inc = bc.fit_ic(dec, N, u0, inits)
        t_inc = time.time() - t0c
        z_new, rel_new, jsel = ic_cached(jnp.asarray(u0))
        ic_check.append(dict(traj=i, incumbent_rel=float(rel_inc),
                             incumbent_secs=t_inc,
                             batched_rel=float(rel_new),
                             init_index=int(jsel)))
        sc.log(f"  IC check traj {i}: incumbent {float(rel_inc):.3e} "
               f"({t_inc:.1f}s host) vs batched {float(rel_new):.3e}")
    report["ic_fit_check"] = ic_check

    # ------------------ classical baselines (same job, same GPU) -------------
    newton_tol = {ntol: make_newton_tol(N, ntol, NEWTON_MAX) for ntol in NTOLS}
    fom_roll, _ = bc.bf.make_rollout(N)          # truth generator (over-solved)

    # ------------------ balanced end-to-end timing ---------------------------
    n_test = min(N_TEST, U_test.shape[0])
    per_traj = {}                                # (method) -> list of dicts

    def run_rom(arm, u0_dev, nu, us_dev):
        a = arms[arm]
        t0 = time.perf_counter()
        z0, ic_rel, jsel = a["ic"](u0_dev)
        z0.block_until_ready()
        t1 = time.perf_counter()
        Z, rns, nJs, reasons = a["ops"]["rollout_jit"](z0, nu, us_dev,
                                                       bc.GN_BUDGET)
        Z.block_until_ready()
        t2 = time.perf_counter()
        F = a["decode"](z0, Z)
        F.block_until_ready()
        t3 = time.perf_counter()
        return dict(F=F, z0=z0, ic_rel=ic_rel, jsel=jsel, Z=Z, rns=rns,
                    nJs=nJs, reasons=reasons,
                    t_ic=t1 - t0, t_roll=t2 - t1, t_dec=t3 - t2,
                    t_e2e=t3 - t0)

    def run_newton(fn, u0_dev, nu):
        t0 = time.perf_counter()
        snaps, rels, its = fn(u0_dev, nu)
        snaps.block_until_ready()
        t1 = time.perf_counter()
        return dict(F=snaps, rels=rels, its=its, t_e2e=t1 - t0)

    def run_fixed8(u0_dev, nu_dev):
        t0 = time.perf_counter()
        snaps, res = fom_roll(u0_dev[None], nu_dev)
        snaps.block_until_ready()
        t1 = time.perf_counter()
        return dict(F=snaps[:, 0, :], rels=res[:, 0], its=None, t_e2e=t1 - t0)

    for i in range(n_test):
        u0 = U_test[i, 0]
        u0_dev = jnp.asarray(u0)
        nu = float(nu_test[i])
        nu_dev = jnp.asarray(nu_test[i:i + 1])
        u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
        Ut = U_test[i]                                          # (51, n^2)
        utn = np.linalg.norm(Ut, axis=1)

        units = []
        for arm in ("cached", "meshfree"):
            us_dev = jnp.full((bc.NUM_STEPS,),
                              bc.GN_TOL * u_scale * arms[arm]["ops"]["tol_scale"],
                              dtype=F64)
            units.append((f"sep_{arm}", None,
                          lambda _a=arm, _u=u0_dev, _n=nu, _s=us_dev:
                          run_rom(_a, _u, _n, _s)))
        for ntol in NTOLS:
            units.append(("fom_newton_tol", ntol,
                          lambda _f=newton_tol[ntol], _u=u0_dev, _n=nu:
                          run_newton(_f, _u, _n)))
        units.append(("fom_newton_fixed8", None,
                      lambda _u=u0_dev, _n=nu_dev: run_fixed8(_u, _n)))

        for _name, _p, fn in units:                              # warm/compile
            fn()
        burn_in(1.5)
        raw = {}
        last = {}
        first_F = {}
        for rep in range(REPS):
            order = units if rep % 2 == 0 else list(reversed(units))
            for name, par, fn in order:
                key = (name, par)
                out = fn()
                st = raw.setdefault(key, dict(e2e=[], ic=[], roll=[], dec=[]))
                st["e2e"].append(out["t_e2e"])
                if "t_ic" in out:
                    st["ic"].append(out["t_ic"])
                    st["roll"].append(out["t_roll"])
                    st["dec"].append(out["t_dec"])
                if rep == 0:
                    first_F[key] = np.asarray(out["F"])
                if rep == REPS - 1:
                    last[key] = out

        for name, par, _fn in units:
            key = (name, par)
            out = last[key]
            F = np.asarray(out["F"])
            per_time = np.linalg.norm(F - Ut, axis=1) / utn
            entry = dict(
                traj=i, nu=nu,
                traj_rel=float(np.mean(per_time)),
                traj_rel_frob=float(np.linalg.norm(F - Ut) / np.linalg.norm(Ut)),
                per_time_max=float(np.max(per_time)),
                n_nonfinite=int(np.sum(~np.isfinite(F.sum(axis=1)))),
                e2e_ms_raw=[t * 1e3 for t in raw[key]["e2e"]],
                e2e_ms=float(np.median(raw[key]["e2e"])) * 1e3,
                dev_first_last_max=float(np.max(np.abs(F - first_F[key]))))
            if name.startswith("sep_"):
                rsn = [int(v) for v in np.asarray(out["reasons"])]
                entry.update(
                    ic_rel=float(out["ic_rel"]),
                    ic_init_index=int(out["jsel"]),
                    jac_total=int(np.sum(np.asarray(out["nJs"]))),
                    stop_reasons={REASON_NAMES.get(r_, str(r_)): rsn.count(r_)
                                  for r_ in set(rsn)},
                    ic_ms_raw=[t * 1e3 for t in raw[key]["ic"]],
                    roll_ms_raw=[t * 1e3 for t in raw[key]["roll"]],
                    dec_ms_raw=[t * 1e3 for t in raw[key]["dec"]],
                    ic_ms=float(np.median(raw[key]["ic"])) * 1e3,
                    roll_ms=float(np.median(raw[key]["roll"])) * 1e3,
                    dec_ms=float(np.median(raw[key]["dec"])) * 1e3)
            else:
                if out["its"] is not None:
                    its = np.asarray(out["its"])
                    rels = np.asarray(out["rels"])
                    entry.update(
                        newton_iters_mean=float(np.mean(its)),
                        newton_iters_max=int(np.max(its)),
                        censored_steps=int(np.sum((its >= NEWTON_MAX)
                                                  & (rels > par))),
                        rel_res_max=float(np.max(rels)))
                else:
                    entry.update(rel_res_max=float(np.max(np.asarray(out["rels"]))))
            per_traj.setdefault((name, par), []).append(entry)
            tag = f"{name}" + (f" ntol={par:.0e}" if par else "")
            sc.log(f"   traj {i} {tag:26s} e2e {entry['e2e_ms']:8.2f} ms  "
                   f"err {entry['traj_rel']:.3e}"
                   + (f"  (ic {entry['ic_ms']:6.2f} roll {entry['roll_ms']:7.2f}"
                      f" dec {entry['dec_ms']:6.2f})  ic_rel {entry['ic_rel']:.2e}"
                      if name.startswith("sep_") else ""))

        # untimed incumbent cross-check for the cached arm (rule 4 analogue)
        z0c, _relc, _jc = ic_cached(u0_dev)
        ro = bc.rollout(dec, N, ops_fast, z0c, nu, u_scale, U_true=Ut)
        F_t = np.asarray(last[("sep_cached", None)]["F"])
        Fu = ro["fields"]
        dev = (float(np.max(np.abs(F_t[:Fu.shape[0]] - Fu)))
               if Fu.shape[0] > 0 else float("nan"))
        per_traj[("sep_cached", None)][-1]["dev_vs_untimed_incumbent"] = dev
        per_traj[("sep_cached", None)][-1]["untimed_traj_rel"] = ro.get("traj_rel")
        per_traj[("sep_cached", None)][-1]["untimed_n_done"] = int(ro["n_done"])
        save()

    # ------------------ aggregate rows ---------------------------------------
    for (name, par), entries in per_traj.items():
        errs = [e["traj_rel"] for e in entries if np.isfinite(e["traj_rel"])]
        row = dict(pde="burgers2d", method=name, N=N, k=K, r=R, M=M_MODES,
                   m=int(m), n_test=n_test,
                   err_traj_rel_mean=float(np.mean(errs)) if errs else None,
                   err_traj_rel_max=float(np.max(errs)) if errs else None,
                   e2e_ms_median=float(np.median([e["e2e_ms"] for e in entries])),
                   n_blowups=int(sum(e["n_nonfinite"] > 0 for e in entries)),
                   per_traj=entries)
        if par is not None:
            row["ntol"] = par
        if name == "fom_newton_fixed8":
            row["label"] = ("truth-generating fixed-8-Newton rollout; "
                            "OVER-SOLVED -- never a headline comparator")
        if name.startswith("sep_"):
            row.update(
                ic_ms_median=float(np.median([e["ic_ms"] for e in entries])),
                roll_ms_median=float(np.median([e["roll_ms"] for e in entries])),
                dec_ms_median=float(np.median([e["dec_ms"] for e in entries])),
                jac_total_mean=float(np.mean([e["jac_total"] for e in entries])),
                ic_rel_mean=float(np.mean([e["ic_rel"] for e in entries])),
                ic_rel_max=float(np.max([e["ic_rel"] for e in entries])))
        report["rows"].append(row)
        save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
