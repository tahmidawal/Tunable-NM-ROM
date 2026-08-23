"""Separable-decoder Burgers-2D cell: train (no POD), weak NM-ROM rollout.

N-scaling round (2026-08-23).  Two arms through the SAME incumbent weak
operators and LM stepper:
  meshfree : blat_common.make_weak_ops(dec, ...) -- the network runs in-loop
  cached   : identical formulas with the stencil feature banks G_st (m,5,r)
             cached once; assembled by blat_common._finish_ops so the solver,
             acceptance rule, and rollout are bit-identical code.
GATE 0: weak residual/Jacobian of the two arms agree <= 1e-12 relative.

MANDATORY MEASUREMENT RULES implemented here (HANDOFF.md; each fixes an audit
FAIL of the N=64 round):
  * END-TO-END timing: the headline timed subject is u0 -> IC latent fit ->
    50-step jitted rollout -> FULL-GRID decode of all 51 states, one jitted
    call; the split (ic / rollout / decode) is timed separately and reported.
    The IC fit is a jitted, vmapped incumbent LM (ctol_tol.lm_tau_generic,
    tau=0 == ms_autodecoder.lm_solve) from the best IC_TOP training-code
    inits, selected by decoder misfit to the KNOWN u0 at the EQ nodes.
  * symmetric outputs: the timed ROM path returns 51 full-grid fields, as the
    timed FOM baselines do.
  * ALL raw timing repetitions retained; balanced AB/BA subject sweeps
    (ROM and FOM subjects interleaved).
  * error comes from the captured outputs of a timed invocation (the same
    jitted call), and the host-loop incumbent bc.rollout is run as an
    in-job cross-check with the recorded max latent deviation.
  * STRONG classical baseline in-job: tolerance-terminated inexact-Newton FOM
    rollouts (ladder of Newton tolerances, BiCGStab forcing lin_tol =
    LIN_FACTOR*ntol), built on the truth generator's own residual; the
    fixed-8-Newton truth generator is also timed but labelled OVER-SOLVED and
    never used for a headline ratio.
  * stop-reason distributions recorded for every ROM row (LM reason codes)
    and every baseline row (converged-vs-cap counts).
CELLS: several (K,R) cells run in one process so the N=256 data generation
(minutes at this mesh) is paid once; each cell re-derives its own rng stream
from SEED0 exactly as the single-cell N=64 script did.
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
import ctol_tol                               # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
CELLS = [(int(p.split(":")[0]), int(p.split(":")[1]))
         for p in os.environ.get("CELLS", "16:64").split(",")]
M_MULT = int(os.environ.get("M_MULT", "4"))          # M = M_MULT*K
MQ_MULT = int(os.environ.get("MQ_MULT", "16"))       # m = MQ_MULT*K
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "4"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "5"))
WARM = int(os.environ.get("WARM", "2"))
IC_TOP = int(os.environ.get("IC_TOP", "12"))         # LM inits kept after scoring
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "1e-2,1e-3,1e-5,1e-8").split(",")]
LIN_FACTOR = float(os.environ.get("LIN_FACTOR", "1e-2"))
MAX_NEWTON = int(os.environ.get("MAX_NEWTON", "20"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}


def make_tol_rollout(n, ntol, lin_tol, max_newton):
    """STRONG classical baseline: the truth generator's own backward-Euler
    residual and BiCGStab linear solver, but Newton TERMINATED at
    ||R(u)|| <= ntol * ||u_prev|| (the truth generator's own convergence
    metric) instead of running a fixed 8 iterations, with inexact-Newton
    linear tolerance lin_tol.  Single trajectory, jitted lax.scan."""
    _, residual = bc.bf.make_rollout(n)

    def step(u_prev, nu):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def cond(s):
            u, it, rn = s
            return (rn > ntol * u_scale) & (it < max_newton)

        def body(s):
            u, it, _ = s
            r = residual(u, u_prev, nu)
            Jv = lambda v: jax.jvp(lambda uu: residual(uu, u_prev, nu),
                                   (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=bc.bf.LIN_MAXITER)
            ok = jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            rn2 = jnp.linalg.norm(residual(u2, u_prev, nu))
            return (u2, it + 1, rn2)

        r0 = jnp.linalg.norm(residual(u_prev, u_prev, nu))
        u, it, rn = jax.lax.while_loop(cond, body,
                                       (u_prev, jnp.int32(0), r0))
        return u, it, rn / u_scale

    def roll(u0, nu):
        def body(u, _):
            u2, it, rel = step(u, nu)
            return u2, (u2, it, rel)
        _, (snaps, its, rels) = jax.lax.scan(body, u0, None,
                                             length=bc.NUM_STEPS)
        return jnp.concatenate([u0[None], snaps], axis=0), its, rels

    return jax.jit(roll)


def run_cell(K, R, d, U, S_flat, interior, coords, report_common):
    M_MODES = M_MULT * K
    MQ = MQ_MULT * K
    dev = jax.devices()[0]
    OUT = f"{OUT_PREFIX}sep_burgers_K{K}_R{R}.json"
    CKPT = f"{OUT_PREFIX}sep_burgers_N{N}_K{K}_R{R}.pkl"
    ARCH = sc.arch_from_env()
    sc.log(f"=== burgers cell N={N} K={K} R={R} M={M_MODES} m={MQ} "
           f"steps={STEPS} arch={ARCH or 'default'} ===")
    t_all = time.time()
    U_test = np.asarray(d["U_test"], dtype=np.float64)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    report = dict(config=dict(
        pde="burgers2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        n_test=N_TEST, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
        num_steps=bc.NUM_STEPS, dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0,
        data_seed=bc.SEED, test_seed=bc.TEST_SEED, max_snaps=MAX_SNAPS,
        reps=REPS, warm=WARM, ic_top=IC_TOP, ic_budget=bc.IC_BUDGET,
        newton_tols=NEWTON_TOLS, lin_factor=LIN_FACTOR, max_newton=MAX_NEWTON,
        arch_overrides=ARCH,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"incumbent weak upwind M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="blat_common lm_step_jit / rollout_jit (incumbent), both arms",
        timing="END-TO-END (ic fit + rollout + full decode) with split; "
               "balanced AB/BA sweeps; raw reps retained; error from the "
               "captured timed invocation; strong tol-Newton baseline in-job",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")), rows=[],
        complete=False)
    report["data"] = dict(report_common)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # per-cell rng stream: identical draw sequence to the N=64 single-cell
    # script (subsample pick, EQ snapshot pick, gate-0 perturbations)
    rng = np.random.default_rng(SEED0)
    n_states = n_traj * T
    if n_states > MAX_SNAPS:
        pick = np.sort(rng.choice(n_states, MAX_SNAPS, replace=False))
    else:
        pick = np.arange(n_states)
    S_tr = S_flat[pick][:, interior]
    report["data"]["n_states_trained"] = int(S_tr.shape[0])
    coords_int = coords[interior]

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

    # ------------------ IC fit (jitted, incumbent LM) ------------------------
    coords_j = jnp.asarray(coords)
    interior_j = jnp.asarray(interior)
    idx_j = jnp.asarray(idx)
    t0_mask = (pick % T) == 0            # codes of t=0 training states
    Z0_states = Z_tr[t0_mask] if t0_mask.any() else Z_tr[:8]
    zbar = Z_tr.mean(0)
    cands = np.concatenate([Z0_states, zbar[None]], axis=0)
    cands_j = jnp.asarray(cands)
    ic_top = min(IC_TOP, cands.shape[0])
    report["ic"] = dict(n_candidates=int(cands.shape[0]), ic_top=int(ic_top),
                        note="inits = all t=0 training codes + mean code, "
                             "scored by decoder misfit to u0 at the EQ nodes, "
                             "top ic_top refined by vmapped incumbent LM")
    G_q0 = G_st[:, 0, :]                                  # (m, r) EQ centres

    def make_ic_fit(dec_eq_fn, dec_full_fn):
        """u0 (n^2,) -> (z0, ic_rn, ic_nJ_total).  All online: candidate
        scoring at the m EQ nodes, then vmapped LM on the full-grid misfit."""
        def fit(u0):
            u0_eq = u0[idx_j]
            scores = jax.vmap(lambda z: jnp.linalg.norm(dec_eq_fn(z) - u0_eq))(
                cands_j)
            _, top = jax.lax.top_k(-scores, ic_top)
            z0s = cands_j[top]

            def f(z):
                return dec_full_fn(z) - u0
            lm = ctol_tol.lm_tau_generic(f, K, bc.IC_BUDGET)
            outs = jax.vmap(lambda z_: lm(z_, 0.0))(z0s)
            zs, rns, nJs = outs[0], outs[1], outs[3]
            b = jnp.argmin(jnp.where(jnp.isfinite(rns), rns, jnp.inf))
            return zs[b], rns[b], jnp.sum(nJs)
        return fit

    ic_fits = dict(
        meshfree=make_ic_fit(lambda z: dec(z, coords_j[idx_j]),
                             lambda z: dec(z, coords_j)),
        cached=make_ic_fit(lambda z: G_q0 @ h_fn(z),
                           lambda z: G_all @ h_fn(z)))
    ic_fit_jit = {a: jax.jit(f) for a, f in ic_fits.items()}

    # IC-solver agreement gate: the jitted incumbent lm_tau_generic (tau=0)
    # must reproduce the host-loop incumbent bc.fit_ic (ms_autodecoder.lm_solve)
    # from the same single init on the same u0.
    u0_gate = jnp.asarray(U_test[0, 0], dtype=F64)

    def f_gate(z):
        return dec(z, coords_j) - u0_gate
    lm_gate = ctol_tol.lm_tau_generic(f_gate, K, bc.IC_BUDGET)
    ic_dev = ctol_tol.check_tau_agreement(
        lm_gate, lambda *a: bc.fit_ic(*a), (jnp.asarray(zbar), 0.0),
        (dec, N, U_test[0, 0], {"mean": zbar}), "ic-jit vs fit_ic", tol=1e-9)
    report["ic"]["jit_vs_incumbent_rel_dev"] = float(ic_dev)
    sc.log(f"  IC solver identity (jit lm_tau_generic vs fit_ic/lm_solve): "
           f"{ic_dev:.2e}")

    # ------------------ end-to-end jitted pipelines --------------------------
    decode_alls = dict(
        meshfree=lambda Zf: jax.vmap(lambda z: dec(z, coords_j))(Zf),
        cached=lambda Zf: jax.vmap(h_fn)(Zf) @ G_all.T)

    def make_e2e(arm, ops):
        ic_fit = ic_fits[arm]
        decode_all = decode_alls[arm]

        def e2e(u0, nu):
            z0, ic_rn, ic_nJ = ic_fit(u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((bc.NUM_STEPS,),
                          bc.GN_TOL * u_scale * ops["tol_scale"], dtype=F64)
            Z, rns, nJs, reasons = ops["rollout_jit"](z0, nu, us, bc.GN_BUDGET)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)     # (T+1, K)
            F = decode_all(Zfull)                              # (T+1, n^2)
            return F, z0, Z, rns, nJs, reasons, ic_rn, ic_nJ
        return jax.jit(e2e)

    arms = dict(meshfree=ops_ref, cached=ops_fast)
    e2e_jit = {a: make_e2e(a, ops) for a, ops in arms.items()}
    decode_jit = {a: jax.jit(decode_alls[a]) for a in arms}

    # ------------------ classical baselines (same job, same GPU) -------------
    fom_roll, _ = bc.bf.make_rollout(N)          # truth generator: OVER-SOLVED
    tol_rolls = {ntol: make_tol_rollout(N, ntol,
                                        max(ntol * LIN_FACTOR, 1e-12),
                                        MAX_NEWTON)
                 for ntol in NEWTON_TOLS}

    # ------------------ per-trajectory balanced timing + metrics -------------
    n_test = min(N_TEST, U_test.shape[0])
    per_arm_rows = {a: [] for a in arms}
    base_rows = {ntol: [] for ntol in NEWTON_TOLS}
    tg_rows = []
    ctol_tol.burn_in(1.5)
    for i in range(n_test):
        u0_np = U_test[i, 0]
        u0 = jnp.asarray(u0_np, dtype=F64)
        nu = float(nu_test[i])
        tnorm = np.linalg.norm(U_test[i], axis=1)
        u_scale = float(np.sqrt(np.mean(u0_np[interior] ** 2)))

        # one untimed invocation per arm: outputs for the splits + host checks
        pre = {a: e2e_jit[a](u0, nu) for a in arms}
        z0_arm = {a: pre[a][1] for a in arms}
        Zfull_arm = {a: jnp.concatenate([pre[a][1][None], pre[a][2]], axis=0)
                     for a in arms}
        us_arm = {a: jnp.full((bc.NUM_STEPS,),
                              bc.GN_TOL * u_scale * arms[a]["tol_scale"],
                              dtype=F64) for a in arms}

        subs = [
            ("rom_cached_e2e",
             lambda _u=u0, _n=nu: (lambda o: (o[0].block_until_ready(), o)[1])(
                 e2e_jit["cached"](_u, _n))),
            (f"fom_ntol_{NEWTON_TOLS[1]:.0e}" if len(NEWTON_TOLS) > 1 else
             f"fom_ntol_{NEWTON_TOLS[0]:.0e}",
             lambda _u=u0, _n=nu,
                    _r=tol_rolls[NEWTON_TOLS[min(1, len(NEWTON_TOLS)-1)]]:
             (lambda o: (o[0].block_until_ready(), o)[1])(_r(_u, _n))),
            ("rom_meshfree_e2e",
             lambda _u=u0, _n=nu: (lambda o: (o[0].block_until_ready(), o)[1])(
                 e2e_jit["meshfree"](_u, _n))),
        ]
        for j, ntol in enumerate(NEWTON_TOLS):
            if j == min(1, len(NEWTON_TOLS) - 1):
                continue                       # already placed second
            subs.append((f"fom_ntol_{ntol:.0e}",
                         lambda _u=u0, _n=nu, _r=tol_rolls[ntol]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_r(_u, _n))))
        subs.append(("fom_newton8_truthgen",
                     lambda _u=jnp.asarray(U_test[i:i + 1, 0]),
                            _n=jnp.asarray(nu_test[i:i + 1]):
                     (lambda o: (o[0].block_until_ready(), o)[1])(
                         fom_roll(_u, _n))))
        for a in arms:
            subs.append((f"rom_{a}_ic",
                         lambda _u=u0, _f=ic_fit_jit[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u))))
            subs.append((f"rom_{a}_roll",
                         lambda _z=z0_arm[a], _n=nu, _us=us_arm[a],
                                _o=arms[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(
                             _o["rollout_jit"](_z, _n, _us, bc.GN_BUDGET))))
            subs.append((f"rom_{a}_decode",
                         lambda _Z=Zfull_arm[a], _d=decode_jit[a]:
                         (lambda o: (o.block_until_ready(), o)[1])(_d(_Z))))
        raw, results = sc.balanced_time(subs, reps=REPS, warm=WARM)

        # metrics from the CAPTURED timed outputs
        for a in arms:
            F, z0_t, Z_t, rns, nJs, reasons, ic_rn, ic_nJ = results[f"rom_{a}_e2e"]
            Fh = np.asarray(F)
            per_time = np.linalg.norm(Fh - U_test[i], axis=1) / tnorm
            det_dev = float(jnp.max(jnp.abs(Z_t - pre[a][2])))
            # host-loop incumbent cross-check with the SAME z0
            ro = bc.rollout(dec, N, arms[a], z0_t, nu, u_scale,
                            U_true=U_test[i])
            lat_dev = float(np.max(np.abs(
                np.asarray(ro["Z"])[1:] - np.asarray(Z_t)[:ro["n_done"]])))
            reasons_np = [int(v) for v in np.asarray(reasons)]
            row = dict(
                traj=i, nu=nu,
                ic_rel=float(ic_rn) / float(np.linalg.norm(u0_np)),
                ic_jac_total=int(ic_nJ),
                traj_rel=float(np.mean(per_time)),
                traj_rel_frob=float(np.linalg.norm(Fh - U_test[i])
                                    / np.linalg.norm(U_test[i])),
                per_time_mean=float(np.mean(per_time)),
                per_time_max=float(np.max(per_time)),
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                jac_total=int(np.sum(np.asarray(nJs))),
                stop_reasons={REASON_NAMES[r_]: reasons_np.count(r_)
                              for r_ in set(reasons_np)},
                e2e_ms=float(np.median(raw[f"rom_{a}_e2e"])) * 1e3,
                e2e_raw_s=[float(t) for t in raw[f"rom_{a}_e2e"]],
                ic_ms=float(np.median(raw[f"rom_{a}_ic"])) * 1e3,
                ic_raw_s=[float(t) for t in raw[f"rom_{a}_ic"]],
                rollout_ms=float(np.median(raw[f"rom_{a}_roll"])) * 1e3,
                rollout_raw_s=[float(t) for t in raw[f"rom_{a}_roll"]],
                decode_ms=float(np.median(raw[f"rom_{a}_decode"])) * 1e3,
                decode_raw_s=[float(t) for t in raw[f"rom_{a}_decode"]],
                timed_vs_untimed_max_latent_dev=det_dev,
                host_rollout_max_latent_dev=lat_dev,
                host_traj_rel=ro.get("traj_rel"),
                host_n_done=int(ro["n_done"]),
                host_reasons={r_: ro["reasons"].count(r_)
                              for r_ in set(ro["reasons"])})
            per_arm_rows[a].append(row)
            sc.log(f"   {a:8s} traj {i}: ic {row['ic_rel']:.2e} "
                   f"({row['ic_ms']:.1f} ms, jac {row['ic_jac_total']})  err "
                   f"{row['traj_rel']:.3e}  jac {row['jac_total']}  e2e "
                   f"{row['e2e_ms']:8.2f} ms (roll {row['rollout_ms']:.2f} + "
                   f"dec {row['decode_ms']:.2f})  latdev {lat_dev:.1e}")
        for ntol in NEWTON_TOLS:
            snaps, its, rels = results[f"fom_ntol_{ntol:.0e}"]
            Sh = np.asarray(snaps)
            per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
            its_np = np.asarray(its)
            rels_np = np.asarray(rels)
            base_rows[ntol].append(dict(
                traj=i, nu=nu,
                traj_rel=float(np.mean(per_time)),
                traj_rel_frob=float(np.linalg.norm(Sh - U_test[i])
                                    / np.linalg.norm(U_test[i])),
                per_time_max=float(np.max(per_time)),
                newton_iters_total=int(np.sum(its_np)),
                newton_iters_max=int(np.max(its_np)),
                steps_converged=int(np.sum(rels_np <= ntol)),
                steps_at_cap=int(np.sum(its_np >= MAX_NEWTON)),
                worst_step_rel_res=float(np.max(rels_np)),
                time_ms=float(np.median(raw[f"fom_ntol_{ntol:.0e}"])) * 1e3,
                time_raw_s=[float(t) for t in raw[f"fom_ntol_{ntol:.0e}"]]))
        snaps, res = results["fom_newton8_truthgen"]
        Sh = np.asarray(snaps)[:, 0, :]
        per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
        tg_rows.append(dict(
            traj=i, nu=nu, traj_rel=float(np.mean(per_time)),
            max_step_rel_res=float(np.max(np.asarray(res))),
            time_ms=float(np.median(raw["fom_newton8_truthgen"])) * 1e3,
            time_raw_s=[float(t) for t in raw["fom_newton8_truthgen"]]))
        sc.log(f"   FOM traj {i}: truthgen {tg_rows[-1]['time_ms']:8.1f} ms "
               f"(OVER-SOLVED); tol-Newton " + " ".join(
                   f"{ntol:.0e}:{base_rows[ntol][-1]['time_ms']:.1f}ms/"
                   f"err{base_rows[ntol][-1]['traj_rel']:.1e}"
                   for ntol in NEWTON_TOLS))
        save()

    # ------------------ aggregate rows ---------------------------------------
    for a in arms:
        rows = per_arm_rows[a]
        errs = [r_["traj_rel"] for r_ in rows if np.isfinite(r_["traj_rel"])]
        agg_reasons = {}
        for r_ in rows:
            for k_, v in r_["stop_reasons"].items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v
        report["rows"].append(dict(
            pde="burgers2d", method=f"sep_{a}", N=N, k=K, r=R, M=M_MODES,
            m=int(m), err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            e2e_ms_median=float(np.median([r_["e2e_ms"] for r_ in rows])),
            ic_ms_median=float(np.median([r_["ic_ms"] for r_ in rows])),
            rollout_ms_median=float(np.median([r_["rollout_ms"] for r_ in rows])),
            decode_ms_median=float(np.median([r_["decode_ms"] for r_ in rows])),
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            ic_jac_mean=float(np.mean([r_["ic_jac_total"] for r_ in rows])),
            stop_reasons=agg_reasons,
            n_blowups=int(sum(r_["n_finite_steps"] < bc.NUM_STEPS for r_ in rows)),
            per_traj=rows, n_test=n_test))
    for ntol in NEWTON_TOLS:
        rows = base_rows[ntol]
        report["rows"].append(dict(
            pde="burgers2d", method="fom_newton_tol", N=N, newton_tol=ntol,
            lin_tol=max(ntol * LIN_FACTOR, 1e-12), max_newton=MAX_NEWTON,
            err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
            time_ms_median=float(np.median([r_["time_ms"] for r_ in rows])),
            newton_iters_mean=float(np.mean([r_["newton_iters_total"]
                                             for r_ in rows])),
            steps_converged_frac=float(np.mean(
                [r_["steps_converged"] / bc.NUM_STEPS for r_ in rows])),
            per_traj=rows, n_test=n_test))
    report["rows"].append(dict(
        pde="burgers2d", method="fom_newton8_truthgen", N=N,
        oversolved=True,
        note="fixed-8-Newton truth generator; NEVER a headline baseline",
        err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in tg_rows])),
        time_ms_median=float(np.median([r_["time_ms"] for r_ in tg_rows])),
        per_traj=tg_rows, n_test=n_test))
    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers cell K={K} R={R} [{time.time()-t_all:.0f}s] -> {OUT}")


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} "
           f"x64={jax.config.jax_enable_x64} N={N} cells={CELLS} "
           f"steps={STEPS} seed={SEED0}")
    # ------------------ data (regenerated from seed, ONCE per job) -----------
    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)            # (n_traj, T, n^2)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    report_common = dict(n_traj=int(n_traj), T=int(T), n2=int(n2),
                         fingerprint=bc.data_fingerprint(U),
                         max_fom_rel_residual=d.get("max_fom_rel_residual"))
    # (n_states, n^2) VIEW; rows are subsampled per cell BEFORE the interior
    # slice so the ~15 GB full interior copy is never materialised at N>=256
    S_flat = U.reshape(n_traj * T, n2)
    for K, R in CELLS:
        run_cell(K, R, d, U, S_flat, interior, coords, report_common)


if __name__ == "__main__":
    main()
