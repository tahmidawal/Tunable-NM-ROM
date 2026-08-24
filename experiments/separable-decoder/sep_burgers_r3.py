"""N=256 push, ROUND 3, Burgers — the span fix, and the same model at N=1024.

Round-2 verdict (runs/push_r2_burgers, commit 24a37ab) plus the round-3
pre-flight (commit 01ad443): every rung below the manifold is tight, and the
sole binding constraint is the TRAINED SPAN, whose numerical rank was capped at
`g_hidden + 1 = 257` because the g-track's last layer is linear.  The fix is
`G_HIDDEN >= 2R`, verified locally to restore a well-conditioned rank-R bank.

This driver runs the SAME cell at two resolutions, which is the point:

  * TRACK A (N=256): validate the span fix cheaply -- drive reconstruction
    toward the R=512 POD floor (7.0e-5 mean / 2.3e-3 worst fresh test state).
  * TRACK B (N=1024): the decisive experiment.  The Burgers crossover already
    exists at N=1024 (report 2026-08-24, VI.5: ROM 59.7 ms vs strong Newton
    310.4 ms) but ONLY for a memory-starved decoder at 1.17e-1 error.  No
    well-trained decoder has ever been run there.

What is new relative to `sep_burgers_r2.py`, and why:

 1. MEMORY-LEAN DATA (the thing that starved N=1024).  The v2 trainer
    point-subsamples, so the full 1.05e6 points per state never need to be
    resident.  Data is generated in trajectory chunks and immediately reduced
    to a fixed random POINT POOL (`POOL`, drawn once from the seed and
    recorded), so training-state coverage is restored instead of cut: host cost
    is MAX_SNAPS x POOL, independent of N.  A small full-interior subset of
    training states is retained to check the pool is not being overfit.
 2. STANDARDISED CLASSICAL BASELINE at both N: tolerance-terminated Newton with
    the SAME exact-Helmholtz-preconditioned BiCGStab inner solve (the n512 /
    n1024 recipe), swept over a LADDER of (newton_tol, lin_tol) so the
    classical cost can be read off AT THE ROM'S ACHIEVED ACCURACY.  The r2
    driver's baseline was unpreconditioned, which is a strawman at large N and
    made the cross-N curve non-comparable.
 3. MATCHED-ACCURACY PAIRING: after the ladder is measured, the cheapest
    classical configuration that is at least as accurate as the ROM is timed
    head-to-head against the ROM champion with the paired AB/BA protocol
    (`sep_common.time_pair`, order 'abbaab').
 4. SPAN DIAGNOSTICS: the f64-Gram spectrum of the trained bank (did the rank
    cap actually lift?) and the unconstrained R-coefficient least-squares floor
    of the learned span on fresh test states -- which separates "the span is
    too small/bad" from "h cannot reach into it".  DIAGNOSTIC ONLY.
 5. Optional per-snapshot loss normalisation (`SNAP_NORM=1`), single-scale
    Fourier features by default (the r2 Poisson control showed multi-scale is
    worse), and chunked bank/oracle evaluation for N=1024 memory.

Unchanged: incumbent discretization, residual and Jacobian definitions; gate 0
(<=1e-12) per EQ set; the rollout identity gate; no test truth in any solve
path; end-to-end timing including the IC fit and the full-grid decode with all
raw repetitions retained; PURE NEURAL -- no POD in the model, SVD is a
diagnostic only.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc
import sep_solvers as ss

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402  (path set by sc)
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
ROUND = int(os.environ.get("ROUND", "3"))
BATCHED = int(os.environ.get("BATCHED", "0"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "512"))
STEPS = int(os.environ.get("STEPS", "300000"))
LR = float(os.environ.get("LR", "1e-3"))
P_SUB = int(os.environ.get("P_SUB", "4096"))
WD = float(os.environ.get("WD", "1e-5"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
FULL_LAST = int(os.environ.get("FULL_LAST", "10000"))
TIME_CAP = float(os.environ.get("TIME_CAP", "0"))
LAM_ORTH = float(os.environ.get("LAM_ORTH", "1e-4"))
SNAP_NORM = int(os.environ.get("SNAP_NORM", "0"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "16384"))
T_EARLY = int(os.environ.get("T_EARLY", "5"))
N_TRAJ = int(os.environ.get("N_TRAJ", "0"))          # 0 = whole canonical draw
POOL = int(os.environ.get("POOL", "0"))              # 0 = all interior points
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "64"))
FULLROWS = int(os.environ.get("FULLROWS", "64"))     # full-grid recon check
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "0"))  # 0 = no chunking
EQ_MS = [int(v) for v in os.environ.get("EQ_MS", "64,256").split(",")]
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))
STEP_TOLS = [float(v) for v in os.environ.get(
    "STEP_TOLS", "1e-9,1e-6").split(",")]
N_TEST = int(os.environ.get("N_TEST", "8"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "5"))
WARM = int(os.environ.get("WARM", "2"))
PAIR_REPS = int(os.environ.get("PAIR_REPS", "3"))
IC_TOP = int(os.environ.get("IC_TOP", "12"))
IC_ENC_BUDGET = int(os.environ.get("IC_ENC_BUDGET", "50"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "150"))
ORACLE_CHUNK = int(os.environ.get("ORACLE_CHUNK", "0"))   # 0 = all 51 at once
SSTEP_TS = [int(v) for v in os.environ.get("SSTEP_TS", "1,2,3,5,10,25,50").split(",")]
SSTEP_BUDGET = int(os.environ.get("SSTEP_BUDGET", "120"))
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4").split(",")]
LIN_FRACS = [float(v) for v in os.environ.get("LIN_FRACS", "0.05,0.5").split(",")]
MAX_NEWTON = int(os.environ.get("MAX_NEWTON", "20"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}


# ===========================================================================
# Classical baseline: tolerance-terminated Newton, exact-Helmholtz-
# preconditioned BiCGStab inner solve.  Identical formulation to the n512 /
# n1024 arms; (ntol, lin_tol) are RUNTIME arguments so one compile serves the
# whole ladder.  Purely classical -- no learned component anywhere.
# ===========================================================================
def make_tol_newton_pc(n):
    _, residual = bc.bf.make_rollout(n)
    lin_maxiter = bc.bf.LIN_MAXITER
    dxl = 1.0 / (n - 1)
    _pp = np.arange(1, n - 1)
    S_pc = jnp.asarray(np.sqrt(2.0 / (n - 1))
                       * np.sin(np.pi * np.outer(_pp, _pp) / (n - 1)))
    _l1 = (4.0 / dxl ** 2) * np.sin(np.pi * _pp / (2 * (n - 1))) ** 2
    lam_pc = jnp.asarray(_l1[:, None] + _l1[None, :])

    def step(u_prev, nu, ntol, lin_tol):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def cond(s):
            _, it, rn = s
            return (rn > ntol * u_scale) & (it < MAX_NEWTON)

        def body(s):
            u, it, rn = s
            r = residual(u, u_prev, nu)

            def Jv(v):
                return jax.jvp(lambda uu: residual(uu, u_prev, nu), (u,), (v,))[1]

            def Minv(v):
                V = v.reshape(n, n)
                C = S_pc.T @ V[1:-1, 1:-1] @ S_pc
                return V.at[1:-1, 1:-1].set(
                    S_pc @ (C / (1.0 + bc.DT * nu * lam_pc)) @ S_pc.T).reshape(-1)

            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=lin_maxiter, M=Minv)
            ok = jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            rn2 = jnp.linalg.norm(residual(u2, u_prev, nu))
            good = jnp.isfinite(rn2)
            u = jnp.where(good, u2, u)
            rn = jnp.where(good, rn2, rn)
            it2 = jnp.where(good & ok, it + 1, jnp.int32(MAX_NEWTON))
            return (u, it2, rn)

        rn0 = jnp.linalg.norm(residual(u_prev, u_prev, nu))
        u, its, rn = jax.lax.while_loop(cond, body, (u_prev, jnp.int32(0), rn0))
        return u, its, rn / u_scale

    def roll(u0, nu, ntol, lin_tol):
        def body(u, _):
            u2, its, rel = step(u, nu, ntol, lin_tol)
            return u2, (u2, its, rel)
        _, (snaps, its, rels) = jax.lax.scan(body, u0, None,
                                             length=bc.NUM_STEPS)
        return jnp.concatenate([u0[None], snaps], axis=0), its, rels

    return jax.jit(roll)


# ===========================================================================
# Memory-lean data build (see module docstring, item 1)
# ===========================================================================
def build_data_lean(n, n_traj, pool_pos, pick, full_rows, gen_chunk, log):
    """Regenerate training trajectories from the CANONICAL seed-0 parameter
    draw and keep only (a) the picked states restricted to the point pool and
    (b) `full_rows` picked states at full interior resolution.  Returns
    (S_tr (n_pick, n_pool), S_full (n_full, n_i2), full_row_pos, worst_res,
     fingerprint).  The FOM residual check runs on the FULL-resolution chunk
    before anything is discarded, so truth convergence is verified exactly as
    `blat_common.build_data` verifies it."""
    interior = bc.interior_indices(n)
    keep_idx = interior[pool_pos]
    T = bc.NUM_STEPS + 1
    cx, cy, w, a, nu, _z = bc.bf.sample_params(seed=bc.SEED)
    assert n_traj <= len(cx), f"n_traj {n_traj} > canonical draw {len(cx)}"
    rollout, _ = bc.bf.make_rollout(n)
    pick_set = {int(v): i for i, v in enumerate(pick)}
    full_set = {int(pick[i]): j for j, i in enumerate(full_rows)}
    S_tr = np.zeros((pick.size, pool_pos.size), dtype=np.float64)
    S_full = np.zeros((len(full_rows), interior.size), dtype=np.float64)
    worst = 0.0
    fp_sum = 0.0
    fp_sumsq = 0.0
    t0 = time.time()
    for s in range(0, n_traj, gen_chunk):
        e = min(s + gen_chunk, n_traj)
        U0 = np.stack([bc.bf.blob_ic(n, cx[i], cy[i], w[i], a[i])
                       for i in range(s, e)])
        snaps, res = rollout(jnp.asarray(U0), jnp.asarray(nu[s:e]))
        snaps_np = np.asarray(snaps)                      # (T, B, n^2)
        cm = float(jnp.max(res))
        if not np.isfinite(cm) or cm > worst:
            worst = cm
        U_chunk = snaps_np.transpose(1, 0, 2)             # (B, T, n^2) view
        wr = bc.max_rel_residual(U_chunk, nu[s:e], n)
        if not np.isfinite(wr) or wr > worst:
            worst = wr
        fp_sum += float(np.sum(snaps_np))
        fp_sumsq += float(np.sum(snaps_np * snaps_np))
        for b in range(e - s):
            base = (s + b) * T
            for t in range(T):
                gid = base + t
                row = pick_set.get(gid)
                if row is None:
                    continue
                st = snaps_np[t, b]
                S_tr[row] = st[keep_idx]
                fr = full_set.get(gid)
                if fr is not None:
                    S_full[fr] = st[interior]
        del snaps, snaps_np, U_chunk
        log(f"   gen: trajectories {e}/{n_traj}  worst FOM rel residual "
            f"{worst:.2e}  [{time.time()-t0:.0f}s]")
    if not np.isfinite(worst) or worst > 1e-8:
        raise SystemExit(f"FOM residual {worst:.2e} > 1e-8: data not converged")
    fp = dict(sum=fp_sum, sumsq=fp_sumsq,
              shape=[int(n_traj), int(T), int(n * n)])
    return S_tr, S_full, worst, fp


def build_test_full(n, n_test, log):
    """Fresh-seed TEST trajectories at FULL resolution (error metrics need the
    whole grid).  Never used by training, EQ fitting, or any solve path."""
    cxt, cyt, wt, at, nut, _ = bc.bf.sample_params(seed=bc.TEST_SEED, m=n_test)
    rollout, _ = bc.bf.make_rollout(n)
    U0 = np.stack([bc.bf.blob_ic(n, cxt[i], cyt[i], wt[i], at[i])
                   for i in range(n_test)])
    snaps, res = rollout(jnp.asarray(U0), jnp.asarray(nut))
    Ut = np.asarray(snaps).transpose(1, 0, 2)
    rt = float(jnp.max(res))
    wr = bc.max_rel_residual(Ut, nut, n)
    worst = max(rt, wr)
    if not np.isfinite(worst) or worst > 1e-8:
        raise SystemExit(f"TEST FOM residual {worst:.2e} > 1e-8")
    log(f"  test data: {n_test} fresh trajectories, worst FOM rel residual "
        f"{worst:.2e}")
    return Ut, np.asarray(nut, dtype=np.float64), worst


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} "
           f"x64={jax.config.jax_enable_x64} ROUND3 N={N} K={K} R={R} "
           f"steps={STEPS} max_snaps={MAX_SNAPS} n_traj={N_TRAJ} pool={POOL} "
           f"t_early={T_EARLY} snap_norm={SNAP_NORM} seed={SEED0}")
    t_all = time.time()
    OUT = f"{OUT_PREFIX}sep_burgers_r3_N{N}_K{K}_R{R}.json"
    CKPT = f"{OUT_PREFIX}sep_burgers_r3_N{N}_K{K}_R{R}.pkl"
    ARCH = sc.arch_from_env()

    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    n_i2 = interior.size
    T = bc.NUM_STEPS + 1
    n_traj = N_TRAJ or (bc.bf.N_TRAIN + bc.bf.N_VAL)
    feat_chunk = FEAT_CHUNK or (0 if N <= 512 else 131072)

    report = dict(config=dict(
        pde="burgers2d", round=ROUND, N=N, k=K, r=R, steps=STEPS, lr=LR,
        p_sub=P_SUB, wd=WD, ema_decay=EMA_DECAY, full_last=FULL_LAST,
        time_cap=TIME_CAP, lam_orth=LAM_ORTH, snap_norm=bool(SNAP_NORM),
        max_snaps=MAX_SNAPS, t_early=T_EARLY, n_traj=n_traj, pool=POOL,
        eq_Ms=EQ_MS, eq_cand_cap=EQ_CAND_CAP, step_tols=STEP_TOLS,
        n_test=N_TEST, gn_budget=bc.GN_BUDGET, num_steps=bc.NUM_STEPS,
        dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0, data_seed=bc.SEED,
        test_seed=bc.TEST_SEED, reps=REPS, warm=WARM, pair_reps=PAIR_REPS,
        ic_top=IC_TOP, ic_budget=bc.IC_BUDGET, ic_enc_budget=IC_ENC_BUDGET,
        enc_steps=ENC_STEPS, extrap=EXTRAP, oracle_budget=ORACLE_BUDGET,
        sstep_ts=SSTEP_TS, sstep_budget=SSTEP_BUDGET,
        newton_tols=NEWTON_TOLS, lin_fracs=LIN_FRACS, max_newton=MAX_NEWTON,
        baseline="tolerance-Newton, exact-Helmholtz-preconditioned BiCGStab "
                 "(n512/n1024 recipe); STANDARDISED across N",
        feat_chunk=feat_chunk, arch_overrides=ARCH, x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        rows=[], gates={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- state pick (early-time weighted) + point pool ----------
    rng = np.random.default_rng(SEED0)
    n_states = n_traj * T
    tidx_of = np.arange(n_states) % T
    early = np.nonzero(tidx_of <= T_EARLY)[0]
    rest = np.nonzero(tidx_of > T_EARLY)[0]
    if early.size >= MAX_SNAPS:
        pick = np.sort(rng.choice(early, MAX_SNAPS, replace=False))
    else:
        extra = rng.choice(rest, min(MAX_SNAPS - early.size, rest.size),
                           replace=False)
        pick = np.sort(np.concatenate([early, extra]))
    pool_pos = (np.arange(n_i2) if (not POOL or POOL >= n_i2)
                else np.sort(rng.choice(n_i2, POOL, replace=False)))
    full_rows = np.sort(rng.choice(pick.size, min(FULLROWS, pick.size),
                                   replace=False))
    n_early_pick = int(np.sum(tidx_of[pick] <= T_EARLY))
    sc.log(f"  pick: {pick.size} states ({n_early_pick} early t<={T_EARLY}) "
           f"from {n_traj} trajectories; point pool {pool_pos.size}/{n_i2}")

    S_tr, S_full, worst_res, fp = build_data_lean(
        N, n_traj, pool_pos, pick, full_rows, GEN_CHUNK, sc.log)
    U_test, nu_test, worst_res_test = build_test_full(N, N_TEST, sc.log)
    report["data"] = dict(
        n_traj=int(n_traj), T=int(T), n2=int(N * N), n_i2=int(n_i2),
        fingerprint=fp, max_fom_rel_residual=worst_res,
        max_fom_rel_residual_test=worst_res_test,
        n_states_total=int(n_states), n_states_trained=int(pick.size),
        n_early_states_in_pick=n_early_pick, n_pool=int(pool_pos.size),
        pool_is_full_interior=bool(pool_pos.size == n_i2),
        n_fullres_check_states=int(len(full_rows)),
        note="training data reduced to a fixed random POINT POOL drawn once "
             "from SEED0; the FOM residual check ran at full resolution")
    save()

    coords_pool = coords[interior[pool_pos]]
    coords_int = coords[interior]

    # ------------------ train (v2, per-snapshot loss optional) ---------------
    params, Z_tr, tinfo = ss.train_autodecoder_v2(
        jax.random.PRNGKey(SEED0), coords_pool, S_tr, K, R,
        steps=STEPS, lr=LR, lam_orth=LAM_ORTH, weight_decay=WD, p_sub=P_SUB,
        ema_decay=EMA_DECAY, full_last=FULL_LAST, time_cap=TIME_CAP,
        snap_norm=bool(SNAP_NORM),
        tag=f"burgers r3 N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    # full-interior reconstruction of a held subset of the SAME training states
    # -- the honest check that fitting on a point pool did not overfit it
    if len(full_rows):
        G_int_chunk = dec.feat_at(coords_int, chunk=feat_chunk)
        Hf = sc.head(params, jnp.asarray(Z_tr[full_rows]))
        per_full = []
        for j in range(len(full_rows)):
            uh = G_int_chunk @ Hf[j]
            per_full.append(float(jnp.linalg.norm(uh - jnp.asarray(S_full[j]))
                                  / jnp.linalg.norm(jnp.asarray(S_full[j]))))
        del G_int_chunk
        report["train"]["recon_fullgrid_subset_mean"] = float(np.mean(per_full))
        report["train"]["recon_fullgrid_subset_max"] = float(np.max(per_full))
        report["train"]["recon_fullgrid_subset_n"] = int(len(full_rows))
        sc.log(f"  recon on FULL interior for {len(full_rows)} training states: "
               f"mean {np.mean(per_full):.3e} max {np.max(per_full):.3e} "
               f"(pool recon mean {tinfo['recon_rel_l2_mean']:.3e})")
    del S_full
    save()

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)
    zbar = Z_tr.mean(0)
    h_fn = dec.head_fn()
    G_all = dec.feat_at(coords, chunk=feat_chunk)
    coords_j = jnp.asarray(coords)
    interior_j = jnp.asarray(interior)

    # ------------------ SPAN DIAGNOSTICS (truth used, labelled) --------------
    # (a) did the rank cap lift?  f64 Gram spectrum of the trained bank.
    G_int = G_all[interior_j]
    Gram = np.asarray(G_int.T @ G_int, dtype=np.float64)
    ev = np.linalg.eigvalsh(Gram)[::-1]
    svr = np.sqrt(np.maximum(ev, 0.0))
    svr = svr / max(svr[0], 1e-300)
    ix = sorted({0, R // 4, R // 2, (3 * R) // 4, R - 1})
    report["span"] = dict(
        note="DIAGNOSTIC ONLY -- SVD/least-squares never enter the model",
        gram_sv_ratio={str(i): float(svr[i]) for i in ix},
        numerical_rank_1e8=int(np.sum(svr > 1e-8)),
        g_hidden=int(ARCH.get("g_hidden", 128)), r=int(R))
    sc.log(f"  SPAN spectrum: " + "  ".join(f"sv[{i}]/sv0={svr[i]:.2e}"
                                            for i in ix)
           + f"   rank(>1e-8) {int(np.sum(svr > 1e-8))}/{R}")
    # (b) unconstrained R-coefficient LS floor of the LEARNED span on fresh
    #     test states: the lower bound the K-dim manifold could ever reach.
    #     Gap(span floor -> oracle) = "h cannot reach into the span";
    #     Gap(POD floor -> span floor) = "the span itself is suboptimal".
    Gram_j = G_int.T @ G_int
    eps = 1e-12 * jnp.trace(Gram_j) / Gram_j.shape[0]
    Lc = jnp.linalg.cholesky(Gram_j + eps * jnp.eye(R, dtype=F64))
    sf = []
    for i in range(min(N_TEST, U_test.shape[0])):
        Ui = jnp.asarray(U_test[i][:, interior], dtype=F64)      # (T, n_i2)
        C = jax.scipy.linalg.cho_solve((Lc, True), G_int.T @ Ui.T)  # (R, T)
        E = (G_int @ C).T - Ui
        rel = np.asarray(jnp.linalg.norm(E, axis=1)
                         / jnp.linalg.norm(Ui, axis=1))
        sf.append(dict(traj=i, mean=float(np.mean(rel)), max=float(np.max(rel)),
                       t0=float(rel[0]), per_time=[float(v) for v in rel]))
    report["span"]["ls_floor_fresh_test_states"] = sf
    report["span"]["ls_floor_mean"] = float(np.mean([r["mean"] for r in sf]))
    sc.log(f"  SPAN least-squares floor on fresh test states: mean "
           f"{report['span']['ls_floor_mean']:.3e}")
    del Gram_j, Lc, G_int
    save()

    # ------------------ EQ sets + cached ops + gate 0 ------------------------
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    dx = 1.0 / (N - 1)
    cand_pos = ctol_eq.candidate_pool(n_i2, cap=EQ_CAND_CAP)
    xy_int_j = jnp.asarray(coords_int)
    u_full_int = jax.jit(lambda z: dec(z, xy_int_j))
    adv_full = jax.jit(lambda uf: bc.upwind_adv_field(uf, N))
    eq_ops = {}
    for Mi in EQ_MS:
        name = "ctrl" if Mi == EQ_MS[0] else f"M{Mi}"
        kx, ky, Phi, lam, _ = bc.test_modes(N, Mi)
        keep, wq_np, eq_info = ctol_eq.eq_fit_burgers(
            u_full_int, adv_full, np.asarray(Phi), cand_pos, Z_eq, K, 4 * Mi,
            f"sep burgers r3 N={N} k={K} M={Mi} m={4*Mi}", bc.nnls_capped)
        cl = dict(kind="grid", idx=interior[cand_pos[keep]], w=wq_np,
                  info=eq_info)
        idx = np.asarray(cl["idx"])
        m = idx.size
        w_q = jnp.asarray(cl["w"], dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx)
        Phi_q = jnp.asarray(np.asarray(Phi)[pos]) * w_q[:, None]
        lam_j = jnp.asarray(lam, dtype=F64)
        st = bc.stencil_indices(idx, N)
        G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)
        del Phi

        def mk(G_st=G_st, Phi_q=Phi_q, lam_j=lam_j):
            def u_and_N_fast(z):
                us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
                c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2], us[:, 3],
                                     us[:, 4])
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
                return c, c * (ux + uy)

            def prev_of_fast(z):
                return jnp.einsum("mr,r->m", G_st[:, 0, :], h_fn(z))

            def r_w_fast(z, prev_c, nu):
                wt = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                u, Nu = u_and_N_fast(z)
                pu = Phi_q.T @ u
                return wt * (Phi_q.T @ (u - prev_c)
                             + bc.DT * ((Phi_q.T @ Nu) + nu * lam_j * pu))

            def d_c_fast(z):
                return u_and_N_fast(z)[0]

            def rJ_fast(z, prev_c, nu):
                return (r_w_fast(z, prev_c, nu),
                        jax.jacfwd(r_w_fast)(z, prev_c, nu),
                        Phi_q.T @ jax.jacfwd(d_c_fast)(z))

            def full_fast(z):
                return G_all @ h_fn(z)
            return r_w_fast, rJ_fast, prev_of_fast, full_fast
        r_w_fast, rJ_fast, prev_of_fast, full_fast_ = mk()
        ops_fast = bc._finish_ops(rJ_fast, r_w_fast, prev_of_fast, full_fast_,
                                  m, "lspg")
        ops_fast["M"] = Mi
        ops_fast["tol_scale"] = float(np.sqrt(n_i2))
        ops_fast["colloc_used"] = cl
        ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=Mi,
                                   solver="lspg")
        g0 = []
        grng = np.random.default_rng(SEED0 + 50)
        for _ in range(5):
            zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                             + 0.05 * grng.standard_normal(K))
            zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
            prev_c = ops_ref["prev_of"](zp)
            nu = float(np.median(nu_test))
            ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
            rb, Jb, _ = ops_fast["rJ"](zt, prev_c, nu)
            pa = ops_ref["prev_of"](zt)
            pb = ops_fast["prev_of"](zt)
            g0.append(max(
                float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300)),
                float(jnp.max(jnp.abs(pa - pb)) / (jnp.max(jnp.abs(pa)) + 1e-300))))
        gate0 = float(np.max(g0))
        sc.log(f"  GATE 0 [{name}]: max rel dev {gate0:.2e}")
        assert gate0 < 1e-12, f"gate 0 failed for EQ set {name}"
        info_rep = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
        info_rep["gate0"] = gate0
        report.setdefault("eq", {})[name] = info_rep
        eq_ops[name] = dict(ops_fast=ops_fast, ops_ref=ops_ref, idx=idx,
                            pos=pos, m=m)
        save()

    e0 = eq_ops["ctrl"]
    ops0 = e0["ops_fast"]
    idx0_j = jnp.asarray(e0["idx"])
    G_q0 = dec.feat_at(coords[e0["idx"]])

    # ------------------ encoder + IC fits ------------------------------------
    pool_of_interior = np.full(n_i2, -1, dtype=np.int64)
    pool_of_interior[pool_pos] = np.arange(pool_pos.size)
    eq_in_pool = pool_of_interior[e0["pos"]]
    if np.all(eq_in_pool >= 0):
        X_tr = S_tr[:, eq_in_pool]
        enc_feat = "u at the ctrl EQ nodes"
    else:
        # the EQ nodes are not all inside the training point pool (N=1024): use
        # the pool positions closest in the interior ordering, a fixed
        # training-data-only feature map.  Never touches test data.
        sub = np.linspace(0, pool_pos.size - 1, e0["m"]).astype(np.int64)
        X_tr = S_tr[:, sub]
        eq_in_pool = sub
        enc_feat = "u at a fixed pool subsample (EQ nodes outside the pool)"
        idx0_enc = interior[pool_pos[sub]]
    enc_idx_j = idx0_j if enc_feat.startswith("u at the ctrl") \
        else jnp.asarray(idx0_enc)
    report["encoder_features"] = enc_feat
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr, steps=ENC_STEPS,
        tag=f"burgers r3 u->z N={N}")
    report["encoder"] = enc_info
    del S_tr

    t0_mask = (pick % T) == 0
    Z0_states = Z_tr[t0_mask] if t0_mask.any() else Z_tr[:8]
    cands_j = jnp.asarray(np.concatenate([Z0_states, zbar[None]], axis=0))
    report["ic"] = dict(n_candidates=int(cands_j.shape[0]))

    def ic_ctrl_fit(u0):
        u0_eq = u0[idx0_j]
        scores = jax.vmap(lambda z: jnp.linalg.norm(G_q0 @ h_fn(z) - u0_eq))(
            cands_j)
        _, top = jax.lax.top_k(-scores, min(IC_TOP, cands_j.shape[0]))
        z0s = cands_j[top]

        def f(z):
            return G_all @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, bc.IC_BUDGET)
        outs = jax.vmap(lambda z_: lm(z_, 0.0))(z0s)
        zs, rns, nJs = outs[0], outs[1], outs[3]
        b = jnp.argmin(jnp.where(jnp.isfinite(rns), rns, jnp.inf))
        return zs[b], rns[b], jnp.sum(nJs)

    def ic_enc_fit(u0):
        z0 = enc_apply(enc_params, u0[enc_idx_j])

        def f(z):
            return G_all @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, rn, nJ

    ic_fits = dict(ctrl=ic_ctrl_fit, enc=ic_enc_fit)
    ic_fit_jit = {a: jax.jit(f) for a, f in ic_fits.items()}

    u0_gate = jnp.asarray(U_test[0, 0], dtype=F64)

    def f_gate(z):
        return dec(z, coords_j) - u0_gate
    lm_gate = ctol_tol.lm_tau_generic(f_gate, K, bc.IC_BUDGET)
    ic_dev = ctol_tol.check_tau_agreement(
        lm_gate, lambda *a: bc.fit_ic(*a), (jnp.asarray(zbar), 0.0),
        (dec, N, U_test[0, 0], {"mean": zbar}), "ic-jit vs fit_ic", tol=1e-9)
    report["ic"]["jit_vs_incumbent_rel_dev"] = float(ic_dev)
    sc.log(f"  IC solver identity: {ic_dev:.2e}")

    # ------------------ rollouts (ctrl + extrap champion) --------------------
    roll_extrap = {name: ss.make_rollout_v2(
        "incumbent", ops=eq_ops[name]["ops_fast"], num_steps=bc.NUM_STEPS,
        extrap=EXTRAP) for name in eq_ops}
    roll_v2_off = ss.make_rollout_v2("incumbent", ops=ops0,
                                     num_steps=bc.NUM_STEPS, extrap=0.0)
    zg = jnp.asarray(Z_tr[3])
    nug = float(np.median(nu_test))
    tolg = STEP_TOLS[0] * float(np.sqrt(np.mean(U_test[0, 0][interior] ** 2))) \
        * ops0["tol_scale"]
    us_g = jnp.full((bc.NUM_STEPS,), tolg, dtype=F64)
    delta0 = jnp.asarray(float(bc.TR_DELTA), dtype=F64)
    Zi = ops0["rollout_jit"](zg, nug, us_g, bc.GN_BUDGET)[0]
    Zv = roll_v2_off(zg, nug, us_g, bc.GN_BUDGET, delta0, delta0, delta0)[0]
    rdev = float(jnp.max(jnp.abs(Zi - Zv)))
    report["gates"]["rollout_v2_off_vs_incumbent"] = rdev
    sc.log(f"  ROLLOUT GATE (v2 no-extrap vs incumbent): max |dZ| {rdev:.2e}")
    assert rdev < 1e-6

    # ------------------ e2e pipelines ----------------------------------------
    def decode_all(Zf):
        return jax.vmap(h_fn)(Zf) @ G_all.T
    decode_jit = jax.jit(decode_all)

    def decode_all_mesh(Zf):
        return jax.vmap(lambda z: dec(z, coords_j))(Zf)

    def make_e2e(ic_name, roll_name, eq_name="ctrl", step_tol=STEP_TOLS[0],
                 mesh=False):
        ops = eq_ops[eq_name]["ops_fast"] if not mesh \
            else eq_ops[eq_name]["ops_ref"]
        if mesh:
            def ic_fit(u0):
                u0_eq = u0[idx0_j]
                scores = jax.vmap(lambda z: jnp.linalg.norm(
                    dec(z, coords_j[idx0_j]) - u0_eq))(cands_j)
                _, top = jax.lax.top_k(-scores, min(IC_TOP, cands_j.shape[0]))
                z0s = cands_j[top]

                def f(z):
                    return dec(z, coords_j) - u0
                lm = ctol_tol.lm_tau_generic(f, K, bc.IC_BUDGET)
                outs = jax.vmap(lambda z_: lm(z_, 0.0))(z0s)
                zs, rns, nJs = outs[0], outs[1], outs[3]
                b = jnp.argmin(jnp.where(jnp.isfinite(rns), rns, jnp.inf))
                return zs[b], rns[b], jnp.sum(nJs)
        else:
            ic_fit = ic_fits[ic_name]
        if roll_name == "ctrl" or mesh:
            def roll_fn(z0, nu, us):
                return ops["rollout_jit"](z0, nu, us, bc.GN_BUDGET)
        else:
            rv = roll_extrap[eq_name]

            def roll_fn(z0, nu, us):
                Z, rns, nJs, reasons, _ = rv(z0, nu, us, bc.GN_BUDGET,
                                             delta0, delta0, delta0)
                return Z, rns, nJs, reasons
        dec_all = decode_all_mesh if mesh else decode_all
        tol_scale = ops["tol_scale"]

        def e2e(u0, nu):
            z0, ic_rn, ic_nJ = ic_fit(u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((bc.NUM_STEPS,),
                          step_tol * u_scale * tol_scale, dtype=F64)
            Z, rns, nJs, reasons = roll_fn(z0, nu, us)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)
            F = dec_all(Zfull)
            return F, z0, Z, rns, nJs, reasons, ic_rn, ic_nJ
        return jax.jit(e2e)

    CHAMP = "cach|ctrl|ic_enc|roll_extrap|t1e-9"
    e2e_arms = {
        "cach|ctrl|ic_ctrl|roll_ctrl|t1e-9": make_e2e("ctrl", "ctrl"),
        "mesh|ctrl|ic_ctrl|roll_ctrl|t1e-9": make_e2e("ctrl", "ctrl",
                                                      mesh=True),
        CHAMP: make_e2e("enc", "extrap"),
    }
    for stol in STEP_TOLS[1:]:
        e2e_arms[f"cach|ctrl|ic_enc|roll_extrap|t{stol:.0e}"] = \
            make_e2e("enc", "extrap", step_tol=stol)
    for name in eq_ops:
        if name != "ctrl":
            e2e_arms[f"cach|{name}|ic_enc|roll_extrap|t1e-9"] = \
                make_e2e("enc", "extrap", eq_name=name)

    fom_roll, _ = bc.bf.make_rollout(N)
    tol_newton = make_tol_newton_pc(N)
    base_cfgs = [(nt, max(nt * lf, 1e-12))
                 for nt in NEWTON_TOLS for lf in LIN_FRACS]

    # ------------------ timing + metrics -------------------------------------
    n_test = min(N_TEST, U_test.shape[0])
    per_arm_rows = {a: [] for a in e2e_arms}
    base_rows = {c: [] for c in base_cfgs}
    tg_rows = []
    ctol_tol.burn_in(1.5)
    for i in range(n_test):
        u0_np = U_test[i, 0]
        u0 = jnp.asarray(u0_np, dtype=F64)
        nu = float(nu_test[i])
        tnorm = np.linalg.norm(U_test[i], axis=1)
        pre = {a: e2e_arms[a](u0, nu) for a in e2e_arms}

        subs = []
        for a in e2e_arms:
            subs.append((f"e2e|{a}",
                         lambda _u=u0, _n=nu, _f=e2e_arms[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u, _n))))
        for (nt, lt) in base_cfgs:
            subs.append((f"fom|nt{nt:.0e}|lt{lt:.0e}",
                         lambda _u=u0, _n=nu, _t=nt, _l=lt:
                         (lambda o: (o[0].block_until_ready(), o)[1])(
                             tol_newton(_u, _n, _t, _l))))
        subs.append(("fom_newton8_truthgen",
                     lambda _u=jnp.asarray(U_test[i:i + 1, 0]),
                            _n=jnp.asarray(nu_test[i:i + 1]):
                     (lambda o: (o[0].block_until_ready(), o)[1])(
                         fom_roll(_u, _n))))
        for a in ("ctrl", "enc"):
            subs.append((f"split_ic_{a}",
                         lambda _u=u0, _f=ic_fit_jit[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u))))
        Zfull_ctrl = jnp.concatenate(
            [pre[CHAMP][1][None], pre[CHAMP][2]], axis=0)
        subs.append(("split_decode",
                     lambda _Z=Zfull_ctrl:
                     (lambda o: (o.block_until_ready(), o)[1])(decode_jit(_Z))))
        raw, results = sc.balanced_time(subs, reps=REPS, warm=WARM)

        for a in e2e_arms:
            F, z0_t, Z_t, rns, nJs, reasons, ic_rn, ic_nJ = results[f"e2e|{a}"]
            Fh = np.asarray(F)
            per_time = np.linalg.norm(Fh - U_test[i], axis=1) / tnorm
            det_dev = float(jnp.max(jnp.abs(Z_t - pre[a][2])))
            reasons_np = [int(v) for v in np.asarray(reasons)]
            per_arm_rows[a].append(dict(
                traj=i, nu=nu,
                ic_rel=float(ic_rn) / float(np.linalg.norm(u0_np)),
                ic_jac_total=int(ic_nJ),
                traj_rel=float(np.mean(per_time)),
                traj_rel_frob=float(np.linalg.norm(Fh - U_test[i])
                                    / np.linalg.norm(U_test[i])),
                per_time=[float(v) for v in per_time],
                per_time_max=float(np.max(per_time)),
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                jac_total=int(np.sum(np.asarray(nJs))),
                stop_reasons={REASON_NAMES[r_]: reasons_np.count(r_)
                              for r_ in set(reasons_np)},
                e2e_ms=float(np.median(raw[f"e2e|{a}"])) * 1e3,
                e2e_raw_s=[float(t) for t in raw[f"e2e|{a}"]],
                timed_vs_untimed_max_latent_dev=det_dev))
            sc.log(f"   {a:42s} traj {i}: err "
                   f"{per_arm_rows[a][-1]['traj_rel']:.3e}  e2e "
                   f"{per_arm_rows[a][-1]['e2e_ms']:8.2f} ms")
        for (nt, lt) in base_cfgs:
            key = f"fom|nt{nt:.0e}|lt{lt:.0e}"
            snaps, its, rels = results[key]
            Sh = np.asarray(snaps)
            per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
            its_np, rels_np = np.asarray(its), np.asarray(rels)
            base_rows[(nt, lt)].append(dict(
                traj=i, nu=nu, traj_rel=float(np.mean(per_time)),
                per_time_max=float(np.max(per_time)),
                newton_iters_total=int(np.sum(its_np)),
                steps_converged=int(np.sum(rels_np <= nt)),
                steps_at_cap=int(np.sum(its_np >= MAX_NEWTON)),
                time_ms=float(np.median(raw[key])) * 1e3,
                time_raw_s=[float(t) for t in raw[key]]))
        snaps, res = results["fom_newton8_truthgen"]
        Sh = np.asarray(snaps)[:, 0, :]
        per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
        tg_rows.append(dict(
            traj=i, nu=nu, traj_rel=float(np.mean(per_time)),
            max_step_rel_res=float(np.max(np.asarray(res))),
            time_ms=float(np.median(raw["fom_newton8_truthgen"])) * 1e3,
            time_raw_s=[float(t) for t in raw["fom_newton8_truthgen"]]))
        splits = dict(traj=i)
        for a in ("ctrl", "enc"):
            z_s, rn_s, nJ_s = results[f"split_ic_{a}"]
            splits[f"ic_{a}_ms"] = float(np.median(raw[f"split_ic_{a}"])) * 1e3
            splits[f"ic_{a}_raw_s"] = [float(t) for t in raw[f"split_ic_{a}"]]
            splits[f"ic_{a}_rel"] = float(rn_s) / float(np.linalg.norm(u0_np))
            splits[f"ic_{a}_jac"] = int(nJ_s)
        splits["decode_ms"] = float(np.median(raw["split_decode"])) * 1e3
        report.setdefault("splits", []).append(splits)
        save()

    for a in e2e_arms:
        rows = per_arm_rows[a]
        errs = [r_["traj_rel"] for r_ in rows if np.isfinite(r_["traj_rel"])]
        agg_reasons = {}
        for r_ in rows:
            for k_, v in r_["stop_reasons"].items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v
        parts = a.split("|")
        report["rows"].append(dict(
            pde="burgers2d", method=a, arm=parts[0], eq_set=parts[1],
            ic=parts[2], roll=parts[3], step_tol=parts[4], N=N, k=K, r=R,
            err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            ic_rel_mean=float(np.mean([r_["ic_rel"] for r_ in rows])),
            ic_rel_max=float(np.max([r_["ic_rel"] for r_ in rows])),
            e2e_ms_median=float(np.median([r_["e2e_ms"] for r_ in rows])),
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            stop_reasons=agg_reasons,
            n_blowups=int(sum(r_["n_finite_steps"] < bc.NUM_STEPS
                              for r_ in rows)),
            per_traj=rows, n_test=n_test))
    for (nt, lt) in base_cfgs:
        rows = base_rows[(nt, lt)]
        report["rows"].append(dict(
            pde="burgers2d", method="fom_newton_tol_pc", N=N, newton_tol=nt,
            lin_tol=lt, max_newton=MAX_NEWTON,
            preconditioner="exact Helmholtz (sine basis)",
            err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
            time_ms_median=float(np.median([r_["time_ms"] for r_ in rows])),
            newton_iters_mean=float(np.mean([r_["newton_iters_total"]
                                             for r_ in rows])),
            steps_converged_frac=float(np.mean(
                [r_["steps_converged"] / bc.NUM_STEPS for r_ in rows])),
            per_traj=rows, n_test=n_test))
    report["rows"].append(dict(
        pde="burgers2d", method="fom_newton8_truthgen", N=N, oversolved=True,
        note="fixed-8-Newton truth generator; NEVER a headline baseline",
        err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in tg_rows])),
        time_ms_median=float(np.median([r_["time_ms"] for r_ in tg_rows])),
        per_traj=tg_rows, n_test=n_test))
    save()

    # ------------------ MATCHED-ACCURACY paired head-to-head -----------------
    champ_err = float(np.mean([r_["traj_rel"] for r_ in per_arm_rows[CHAMP]]))
    champ_ms = float(np.median([r_["e2e_ms"] for r_ in per_arm_rows[CHAMP]]))
    cands_m = [(nt, lt) for (nt, lt) in base_cfgs
               if float(np.mean([r_["traj_rel"] for r_ in base_rows[(nt, lt)]]))
               <= champ_err]
    match = None
    if cands_m:
        match = min(cands_m, key=lambda c: float(np.median(
            [r_["time_ms"] for r_ in base_rows[c]])))
    report["matched_accuracy"] = dict(
        rom_arm=CHAMP, rom_err=champ_err, rom_e2e_ms=champ_ms,
        rule="cheapest (newton_tol, lin_tol) whose mean trajectory error is "
             "<= the ROM champion's; medians over the same test trajectories",
        matched=None if match is None else dict(
            newton_tol=match[0], lin_tol=match[1],
            err=float(np.mean([r_["traj_rel"] for r_ in base_rows[match]])),
            ms=float(np.median([r_["time_ms"] for r_ in base_rows[match]]))),
        cheapest_overall=dict(
            newton_tol=min(base_cfgs, key=lambda c: float(np.median(
                [r_["time_ms"] for r_ in base_rows[c]])))[0]))
    if match is not None:
        pairs = []
        for i in range(n_test):
            u0 = jnp.asarray(U_test[i, 0], dtype=F64)
            nu = float(nu_test[i])
            pr = sc.time_pair(
                lambda _u=u0, _n=nu: e2e_arms[CHAMP](_u, _n)[0].block_until_ready(),
                lambda _u=u0, _n=nu, _t=match[0], _l=match[1]:
                tol_newton(_u, _n, _t, _l)[0].block_until_ready(),
                reps=PAIR_REPS, warm=WARM)
            pairs.append(dict(traj=i, **pr))
        report["matched_accuracy"]["paired"] = dict(
            rom_ms=float(np.median([p["a_ms"] for p in pairs])),
            base_ms=float(np.median([p["b_ms"] for p in pairs])),
            per_traj=pairs)
        sc.log(f"  MATCHED-ACCURACY paired: ROM "
               f"{report['matched_accuracy']['paired']['rom_ms']:.2f} ms vs "
               f"tol-Newton(nt={match[0]:.0e}, lt={match[1]:.0e}) "
               f"{report['matched_accuracy']['paired']['base_ms']:.2f} ms")
    else:
        sc.log(f"  MATCHED-ACCURACY: no classical rung reaches the ROM's "
               f"{champ_err:.3e}; the ROM is more accurate than every "
               f"configuration measured")
    save()

    # ------------------ batched multi-query ----------------------------------
    if BATCHED:
        u0b = jnp.asarray(U_test[:n_test, 0], dtype=F64)
        nub = jnp.asarray(nu_test[:n_test], dtype=F64)
        b_names = [a for a in (CHAMP,
                               f"cach|M{max(EQ_MS)}|ic_enc|roll_extrap|t1e-9")
                   if a in e2e_arms]
        subs = []
        for aname in b_names:
            be = jax.jit(jax.vmap(e2e_arms[aname]))

            def fn(_b=be):
                out = _b(u0b, nub)
                out[0].block_until_ready()
                return out
            subs.append((f"batched|{aname}", fn))
        b_cfgs = [match] if match is not None else []
        for c in ((NEWTON_TOLS[len(NEWTON_TOLS) // 2], None),):
            cc = (c[0], max(c[0] * LIN_FRACS[0], 1e-12))
            if cc not in b_cfgs:
                b_cfgs.append(cc)
        for (nt, lt) in b_cfgs:
            br = jax.jit(jax.vmap(lambda u, n, _t=nt, _l=lt:
                                  tol_newton(u, n, _t, _l)))

            def fn(_b=br):
                out = _b(u0b, nub)
                out[0].block_until_ready()
                return out
            subs.append((f"batched|fom|nt{nt:.0e}|lt{lt:.0e}", fn))
        ctol_tol.burn_in(1.0)
        raw_b, res_b = sc.balanced_time(subs, reps=REPS, warm=WARM)
        report["batched"] = []
        for name in raw_b:
            times = raw_b[name]
            ent = dict(subject=name, n_queries=n_test,
                       total_ms_median=float(np.median(times)) * 1e3,
                       amortized_ms=float(np.median(times)) * 1e3 / n_test,
                       raw_s=[float(t) for t in times])
            Fb = np.asarray(res_b[name][0])
            errs = [float(np.mean(np.linalg.norm(Fb[i] - U_test[i], axis=1)
                                  / np.linalg.norm(U_test[i], axis=1)))
                    for i in range(n_test)]
            ent.update(err_traj_rel_mean=float(np.mean(errs)),
                       err_traj_rel_max=float(np.max(errs)))
            if name.startswith("batched|cach"):
                aname = name.split("batched|")[1]
                dev_ = 0.0
                for i in range(n_test):
                    pt_b = np.linalg.norm(Fb[i] - U_test[i], axis=1) \
                        / np.linalg.norm(U_test[i], axis=1)
                    pt_s = np.asarray(per_arm_rows[aname][i]["per_time"])
                    dev_ = max(dev_, float(np.max(np.abs(pt_b - pt_s))))
                ent["batched_vs_single_max_pertime_dev"] = dev_
            report["batched"].append(ent)
            sc.log(f"   BATCHED {name}: total {ent['total_ms_median']:.2f} ms "
                   f"-> {ent['amortized_ms']:.3f} ms/query  "
                   f"err {ent['err_traj_rel_mean']:.3e}")
        save()

    # ------------------ ladder diagnostics (truth used, labelled) ------------
    full_fast = jax.jit(lambda z: G_all @ h_fn(z))
    oracle_lm = ss.make_oracle_lm(full_fast, K, budget=ORACLE_BUDGET)
    report["oracle"] = []
    z_or_all = {}
    ochunk = ORACLE_CHUNK or T
    for i in range(n_test):
        z_parts, v_parts = [], []
        for s in range(0, T, ochunk):
            e = min(s + ochunk, T)
            targets = jnp.asarray(U_test[i][s:e], dtype=F64)
            enc_in = jax.vmap(lambda u: enc_apply(enc_params, u[enc_idx_j]))(
                targets)
            init_sets = [jnp.tile(jnp.asarray(zbar)[None], (e - s, 1)), enc_in]
            z_or, v_or = ss.oracle_multi_init(oracle_lm, init_sets, targets)
            z_parts.append(np.asarray(z_or))
            v_parts.append(np.asarray(v_or))
        z_or_all[i] = np.concatenate(z_parts, axis=0)
        rel = np.concatenate(v_parts) / np.linalg.norm(U_test[i], axis=1)
        report["oracle"].append(dict(
            traj=i, mean=float(np.mean(rel)), max=float(np.max(rel)),
            t0=float(rel[0]), per_time=[float(v) for v in rel],
            note="per-state representation oracle, DIAGNOSTIC only"))
        sc.log(f"  ORACLE traj {i}: mean {np.mean(rel):.3e} "
               f"max {np.max(rel):.3e} t0 {rel[0]:.3e}")
    save()

    report["single_step_weak_opt"] = {}
    for name in eq_ops:
        ops_n = eq_ops[name]["ops_fast"]
        rows_ss = []
        for i in range(n_test):
            nu = float(nu_test[i])
            u_scale = float(np.sqrt(np.mean(U_test[i, 0][interior] ** 2)))
            tol_abs = STEP_TOLS[0] * u_scale * ops_n["tol_scale"]
            for t in SSTEP_TS:
                if t < 1 or t > bc.NUM_STEPS:
                    continue
                z_prev = jnp.asarray(z_or_all[i][t - 1])
                prev_c = ops_n["prev_of"](z_prev)
                z2, rn, nJ, acc, reason, att = ops_n["step_jit"](
                    z_prev, prev_c, nu, tol_abs, SSTEP_BUDGET)
                u_hat = np.asarray(full_fast(z2))
                err = float(np.linalg.norm(u_hat - U_test[i, t])
                            / np.linalg.norm(U_test[i, t]))
                or_err = float(np.linalg.norm(
                    np.asarray(full_fast(jnp.asarray(z_or_all[i][t])))
                    - U_test[i, t]) / np.linalg.norm(U_test[i, t]))
                rows_ss.append(dict(traj=i, t=t, err=err, oracle_err=or_err,
                                    rn=float(rn), reason=int(reason)))
        report["single_step_weak_opt"][name] = rows_ss
        em = np.mean([r_["err"] for r_ in rows_ss])
        om = np.mean([r_["oracle_err"] for r_ in rows_ss])
        sc.log(f"  single-step weak-opt [{name}]: mean err {em:.3e} vs mean "
               f"oracle {om:.3e} (gap = objective truncation)")
    save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers r3 N={N} [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
