"""2D Burgers ROM: the precomputed quadratic-tensor advection ("exact
projection of the advection term, no hyper-reduction") vs the sampled exlin
rule and the full-grid oracle, one invocation per N (2026-08-29, branch
exp/2026-08-29-b2d-tensor; 2D port of sep_b1d_tensor.py).

Residual paths (all EXACT-LINEAR: A = Phi^T G precomputed, only the advection
term differs; IC fit, LM rule, stall, tolerances, trust region, warm-start
extrapolation and decode are IDENTICAL across arms):

  full        Phi^T N(u) summed over every interior grid point with the FOM's
              sign-upwind stencil (blat_common.upwind_adv_field) -- the oracle
  ex          incumbent exlin: advection sampled at m NNLS grid nodes fit on
              advection rows only (exlin_common.eq_fit_burgers_adv), EQ_MQ=256
  tensor      0.5 h^T Q h with Q = T + T^(jk), T[i,j,k] = sum_x Phi G (D^-_x G
              + D^-_y G) built once in-job from the frozen bank, blocked over x
  ex_learned  (optional, NODES_NPZ) the learned continuous m=M node set of the
              codesign "n" arm (sep_codesign.py mk_var closure)

Arms run INTERLEAVED: repetitions outermost (BURN + TIME_REPS), trajectories
next, arms innermost in forward order on even repetitions and reversed on odd
ones (AB/BA).  Every repetition is persisted; accuracy is read from the last
timed repetition's own fused end-to-end output.  Per repetition each arm is
timed as three separate phases (IC fit / latent solve / decode, each blocked)
AND as one fused end-to-end jit.

Gates (asserted unless noted): bank==meshfree; gate 0 (incumbent-form ops ==
make_weak_ops); L (exlin linear == full-grid linear); A (exlin advection ==
incumbent advection); FOMR (full-grid weak residual == wt * Phi^T R_FOM);
STEP (aux-threaded step == sep_solvers.make_step_lspg_var == incumbent at
1e-12); ROLL (aux-threaded rollout == make_rollout_v2); TB (two chunkings);
TA (algebraic identity on training codes); T0 (tensor == oracle on all-
positive decoded states); TQ (tensor vs oracle at 32 latent states, RECORDED);
IC gram vs full (recorded); positivity of the truth (asserted, tolerance
POS_TOL) and of the decoded states (recorded).

FOM: the standardised classical (newton_tol, lin_tol) ladder of
sep_speed_r5 / sep_burgers_exlin (make_tol_newton_pc: tolerance-terminated
Newton, exact-Helmholtz-preconditioned BiCGStab), balanced timing, plus the
matched-accuracy rules and a paired AB/BA of each ROM arm vs the matched
rung.

TRAIN=1 trains the decoder in-job with the sep_burgers.py recipe (streamed
truth regeneration, same state pick rule, sc.train_autodecoder) when no
checkpoint exists for this N.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import sep_common as sc
import sep_solvers as ss

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402
import exlin_common as xc                     # noqa: E402
import b2d_tensor_common as tc                # noqa: E402
from sep_burgers_r3 import make_tol_newton_pc, build_test_full   # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
CKPT = os.environ.get("CKPT", "")
TRAIN = int(os.environ.get("TRAIN", "0" if CKPT else "1"))
NODES_NPZ = os.environ.get("NODES_NPZ", "")
OUT = os.environ.get("OUT", f"/tmp/sep_b2d_tensor_n{N}.json")
CKPT_OUT = os.environ.get("CKPT_OUT", OUT.replace(".json", "_ckpt.pkl"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "64"))
EQ_M = int(os.environ.get("EQ_M", "64"))            # test modes
EQ_MQ = int(os.environ.get("EQ_MQ", "256"))         # NNLS nodes (incumbent)
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))
N_TEST = int(os.environ.get("N_TEST", "8"))
SEED0 = int(os.environ.get("SEED0", "0"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
STALL = float(os.environ.get("STALL", "1e-3"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
IC_ENC_BUDGET = int(os.environ.get("IC_ENC_BUDGET", "50"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
TIME_REPS = int(os.environ.get("TIME_REPS", "5"))
BURN = int(os.environ.get("BURN", "2"))
FOM_REPS = int(os.environ.get("FOM_REPS", "5"))
PAIR_REPS = int(os.environ.get("PAIR_REPS", "3"))
WARM = int(os.environ.get("WARM", "2"))
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4").split(",")]
LIN_FRACS = [float(v) for v in os.environ.get("LIN_FRACS", "0.05,0.5").split(",")]
ARMS = os.environ.get("ARMS", "full,ex,tensor").split(",")
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "0"))
T_CHUNK = int(os.environ.get("T_CHUNK", "16384"))
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "0"))
N_TQ = int(os.environ.get("N_TQ", "32"))
TA_CHUNK = int(os.environ.get("TA_CHUNK", "64"))
R_CHECK_STATES = int(os.environ.get("R_CHECK_STATES", "64"))
POS_TOL = float(os.environ.get("POS_TOL", "1e-9"))   # truth min(u) >= -POS_TOL
# stretch: head-PCA Tucker compression of the tensor for large R (R=512)
HEAD_PCA = int(os.environ.get("HEAD_PCA", "0"))
PCA_TAIL = float(os.environ.get("PCA_TAIL", "1e-8"))   # keep K' with tail energy <= PCA_TAIL
PCA_T0_TOL = float(os.environ.get("PCA_T0_TOL", "1e-6"))
# skip the streamed training-truth regeneration (only for checkpoints whose
# training draw is too large to regenerate in-job, e.g. the 4608-trajectory
# hfit checkpoint); the positivity assert then covers the test set only
SKIP_TRAIN_TRUTH = int(os.environ.get("SKIP_TRAIN_TRUTH", "0"))
# in-job training recipe (sep_burgers.py defaults for the N=256 cell)
STEPS = int(os.environ.get("STEPS", "60000"))
LR = float(os.environ.get("LR", "1e-3"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}

log = sc.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def blk(x):
    return jax.block_until_ready(x)


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                max=float(np.max(v)), min=float(np.min(v)), n=int(v.size))


def hist_of(arr):
    a = np.asarray(arr)
    return {REASON_NAMES[int(r_)]: int(np.sum(a == r_)) for r_ in np.unique(a)}


def git_commit():
    c = os.environ.get("COMMIT")
    if c:
        return c
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ===========================================================================
# solver pieces with the big arrays threaded as an explicit `aux` argument
# (never captured: at N=1024 the bank and the modes are 0.5 GB each)
# ===========================================================================
def make_step_aux(r_w, K, stall_rel, tr_delta):
    """Verbatim `sep_solvers.make_step_lspg_var` (itself the incumbent
    `blat_common._finish_ops.lm_step_jit` with a parametric stall) with the
    residual signature r_w(z, prev, nu, aux).  Gate STEP asserts bit-identity
    to make_step_lspg_var on the sampled arm."""
    def rJ_lspg(z, p, nu, aux):
        return (r_w(z, p, nu, aux), jax.jacfwd(r_w)(z, p, nu, aux))

    def rn_fn(z, p, nu, aux):
        return jnp.linalg.norm(r_w(z, p, nu, aux))

    def step(z0, prev_c, nu, tol_abs, budget, aux):
        r0, J0 = rJ_lspg(z0, prev_c, nu, aux)
        rn0 = jnp.linalg.norm(r0)
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where(rn0 <= tol_abs, jnp.int32(4),
                                          jnp.int32(0)))
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), jnp.int32(0), init_reason)

        def cond(s):
            return (s[9] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, nJ, _, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            within_trust = jnp.linalg.norm(dz) <= tr_delta
            tiny = finite & (jnp.linalg.norm(dz)
                             <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
            z_new = z + jnp.where(finite & within_trust, dz, 0.0)
            rn_new = rn_fn(z_new, prev_c, nu, aux)
            accept = (finite & within_trust & jnp.isfinite(rn_new)
                      & (rn_new < rn))
            r2, J2 = jax.lax.cond(accept,
                                  lambda: rJ_lspg(z_new, prev_c, nu, aux),
                                  lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(
                accept & (rn <= tol_abs), 1,
                jnp.where((accept & (rel_dec < stall_rel)) | tiny, 2,
                          jnp.where((~accept) & (lam >= 1e12), 3,
                                    0))).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, acc, nJ, jnp.int32(0), reason)

        z, r, J, rn, lam, att, acc, nJ, _, reason = jax.lax.while_loop(
            cond, body, init)
        return z, rn, nJ, acc, reason, att

    return jax.jit(step, static_argnums=(4,)), jax.jit(rn_fn), jax.jit(rJ_lspg)


def make_roll_aux(step_fn, rn_fn, prev_fn, num_steps, extrap):
    """`sep_solvers.make_rollout_v2('incumbent')` with aux threading and the
    LM attempt count emitted per step.  Gate ROLL asserts bit-identity of the
    latent path to make_rollout_v2 on the sampled arm."""
    def roll(z0, nu, us, budget, aux):
        def body(carry, tol_abs):
            z, z_prev, prev_c = carry
            if extrap > 0.0:
                z_ex = z + extrap * (z - z_prev)
                rn_a = rn_fn(z, prev_c, nu, aux)
                rn_b = rn_fn(z_ex, prev_c, nu, aux)
                z_init = jnp.where(jnp.isfinite(rn_b) & (rn_b < rn_a), z_ex, z)
            else:
                z_init = z
            z2, rn, nJ, acc, reason, att = step_fn(z_init, prev_c, nu, tol_abs,
                                                   budget, aux)
            return (z2, z, prev_fn(z2)), (z2, rn, nJ, att, reason)

        _, out = jax.lax.scan(body, (z0, z0, prev_fn(z0)), us)
        return out

    return jax.jit(roll, static_argnums=(3,))


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B2D-TENSOR N={N} K={K} R={R} "
        f"M={EQ_M} m={EQ_MQ} arms={ARMS} train={TRAIN} reps={TIME_REPS} "
        f"burn={BURN} stall={STALL}")
    t_all = time.time()

    interior = bc.interior_indices(N)
    interior_j = jnp.asarray(interior)
    coords = np.asarray(bc.grid_coords(N))
    coords_int = coords[interior]
    n_i2 = interior.size
    n2 = N * N
    T = bc.NUM_STEPS + 1
    dx = 1.0 / (N - 1)
    feat_chunk = FEAT_CHUNK or (0 if N <= 512 else 131072)
    gen_chunk = GEN_CHUNK or (64 if N <= 256 else (16 if N <= 512 else 8))

    report = dict(config=dict(
        pde="burgers2d", kind="b2d_tensor", N=N, k=K, r=R, M=EQ_M, m_nnls=EQ_MQ,
        eq_cand_cap=EQ_CAND_CAP, arms=ARMS, ckpt=CKPT, train_in_job=bool(TRAIN),
        nodes_npz=NODES_NPZ, n_test=N_TEST, seed=SEED0, data_seed=bc.SEED,
        test_seed=bc.TEST_SEED, n_train_traj=bc.bf.N_TRAIN, n_val_traj=bc.bf.N_VAL,
        step_tol=STEP_TOL, stall=STALL, extrap=EXTRAP, tr_factor=TR_FACTOR,
        gn_budget=bc.GN_BUDGET, ic_enc_budget=IC_ENC_BUDGET, enc_steps=ENC_STEPS,
        num_steps=bc.NUM_STEPS, dt=bc.DT, weak_alpha=bc.WEAK_ALPHA,
        time_reps=TIME_REPS, burn=BURN, warm=WARM, fom_reps=FOM_REPS,
        pair_reps=PAIR_REPS, newton_tols=NEWTON_TOLS, lin_fracs=LIN_FRACS,
        t_chunk=T_CHUNK, gen_chunk=gen_chunk, feat_chunk=feat_chunk,
        pos_tol=POS_TOL, head_pca=bool(HEAD_PCA), pca_tail=PCA_TAIL,
        skip_train_truth=bool(SKIP_TRAIN_TRUTH), x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        jax_version=jax.__version__, commit=git_commit(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local"),
        arm_order="reps outermost; trajectories; arms forward on even reps, "
                  "reversed on odd (AB/BA); accuracy from the last timed rep's "
                  "fused e2e output"),
        gates={}, data={}, variants={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- test data (fresh seed) + positivity --------------------
    U_test, nu_test, worst_res_test = build_test_full(N, N_TEST, log)
    Ut_int = U_test[:, :, interior]
    report["data"]["test"] = dict(
        n_test=int(N_TEST), max_fom_rel_residual=worst_res_test,
        min_u=float(Ut_int.min()), max_u=float(Ut_int.max()),
        frac_points_le0=float(np.mean(Ut_int <= 0)),
        frac_points_lt0=float(np.mean(Ut_int < 0)),
        nu=[float(v) for v in nu_test])
    log(f"  TEST positivity: min u {Ut_int.min():.3e} max {Ut_int.max():.3e} "
        f"frac<=0 {np.mean(Ut_int <= 0):.2e} frac<0 {np.mean(Ut_int < 0):.2e}")
    del Ut_int
    save()

    # ---------------- checkpoint: load, or train in-job ----------------------
    n_traj = bc.bf.N_TRAIN + bc.bf.N_VAL
    if TRAIN:
        cfg_ms = MAX_SNAPS
    else:
        params, Z_tr, cfg = sc.load_pkl(CKPT)
        assert int(cfg["N"]) == N and int(cfg["k"]) == K and int(cfg["r"]) == R, cfg
        cfg_ms = int(cfg.get("max_snaps", 8192))
        n_traj = int(cfg.get("n_train_traj", bc.bf.N_TRAIN)
                     + cfg.get("n_val_traj", bc.bf.N_VAL))
        assert n_traj == bc.bf.N_TRAIN + bc.bf.N_VAL, (
            n_traj, bc.bf.N_TRAIN, bc.bf.N_VAL, "set N_TRAIN/N_VAL to the "
            "checkpoint's draw")
    hfit = (not TRAIN) and cfg.get("hfit_pick") is not None
    if hfit:
        # round-5 refined checkpoint: the state ids are recorded, and the
        # trajectory parameters are the canonical draw plus an appended draw
        pick = np.asarray(cfg["hfit_pick"], dtype=np.int64)
        n_traj = int(cfg.get("hfit_n_traj") or n_traj)
        report["config"]["pick_source"] = "cfg[hfit_pick] (round-5 refined)"
    else:
        n_states = n_traj * T
        rng = np.random.default_rng(SEED0)
        if n_states > cfg_ms:
            pick = np.sort(rng.choice(n_states, cfg_ms, replace=False))
        else:
            pick = np.arange(n_states)
        report["config"]["pick_source"] = "sep_burgers.py rule (rng(SEED0).choice)"
    n_states = n_traj * T
    keep_rows = set(int(v) for v in (pick if TRAIN else pick[:R_CHECK_STATES]))

    # ---------------- training truth: streamed regeneration + positivity ----
    # (the same draw the checkpoint was trained on; the states are only kept
    # for training or for the R-lite check)
    cx, cy, w, a, nu_tr, _ = bc.bf.sample_params(seed=bc.SEED)
    ex_seed = int((cfg.get("hfit_extra_seed", 0) if not TRAIN else 0) or 0)
    ex_traj = int((cfg.get("hfit_extra_traj", 0) if not TRAIN else 0) or 0)
    if ex_seed:
        exd = bc.bf.sample_params(seed=ex_seed, m=ex_traj)
        cx = np.concatenate([cx, exd[0]]); cy = np.concatenate([cy, exd[1]])
        w = np.concatenate([w, exd[2]]); a = np.concatenate([a, exd[3]])
        nu_tr = np.concatenate([nu_tr, exd[4]])
        log(f"  trajectory parameters: canonical {len(cx) - ex_traj} + {ex_traj} "
            f"appended from seed {ex_seed} (recorded in the checkpoint)")
    assert len(cx) == n_traj, (len(cx), n_traj)
    rollout_fom, res_fn = bc.bf.make_rollout(N)
    chk = jax.jit(jax.vmap(lambda u1, u0, nu_: jnp.linalg.norm(res_fn(u1, u0, nu_))
                           / jnp.linalg.norm(u0)))
    kept = {}
    tr_min, tr_max, worst_tr = np.inf, -np.inf, 0.0
    n_le0 = n_lt0 = n_pts = 0
    t0 = time.time()
    gen_range = [] if SKIP_TRAIN_TRUTH else range(0, n_traj, gen_chunk)
    if SKIP_TRAIN_TRUTH:
        assert not TRAIN
        log(f"  TRAIN truth regeneration SKIPPED (SKIP_TRAIN_TRUTH=1; {n_traj} "
            f"trajectories) -- positivity asserted on the test set only")
        tr_min, tr_max, worst_tr, n_pts = np.nan, np.nan, np.nan, 1
    for s in gen_range:
        e = min(s + gen_chunk, n_traj)
        U0 = np.stack([bc.bf.blob_ic(N, cx[i], cy[i], w[i], a[i])
                       for i in range(s, e)])
        snaps, res = rollout_fom(jnp.asarray(U0), jnp.asarray(nu_tr[s:e]))
        worst_tr = max(worst_tr, float(jnp.max(res)))
        for k_ in range(bc.NUM_STEPS):
            worst_tr = max(worst_tr, float(jnp.max(chk(
                snaps[k_ + 1], snaps[k_], jnp.asarray(nu_tr[s:e])))))
        si = snaps[:, :, interior_j]
        tr_min = min(tr_min, float(jnp.min(si)))
        tr_max = max(tr_max, float(jnp.max(si)))
        n_le0 += int(jnp.sum(si <= 0))
        n_lt0 += int(jnp.sum(si < 0))
        n_pts += int(si.size)
        del si
        for b_ in range(e - s):
            for t_ in range(T):
                sid = (s + b_) * T + t_
                if sid in keep_rows:
                    kept[sid] = np.asarray(snaps[t_, b_])
        del snaps
    if not SKIP_TRAIN_TRUTH and not (np.isfinite(worst_tr) and worst_tr <= 1e-8):
        raise SystemExit(f"TRAIN FOM residual {worst_tr:.2e} > 1e-8")
    report["data"]["train"] = dict(
        n_traj=int(n_traj), n_states=int(n_states), max_fom_rel_residual=worst_tr,
        min_u=tr_min, max_u=tr_max, frac_points_le0=n_le0 / n_pts,
        frac_points_lt0=n_lt0 / n_pts, n_points_checked=int(n_pts),
        secs=time.time() - t0, pick_size=int(pick.size))
    log(f"  TRAIN truth: {n_traj} traj regenerated [{time.time()-t0:.0f}s], "
        f"max FOM rel residual {worst_tr:.2e}; positivity: min u {tr_min:.3e} "
        f"max {tr_max:.3e} frac<=0 {n_le0/n_pts:.2e} frac<0 {n_lt0/n_pts:.2e}")
    pos_ok = (SKIP_TRAIN_TRUTH or tr_min >= -POS_TOL) and \
        (report["data"]["test"]["min_u"] >= -POS_TOL)
    report["data"]["positivity_assert"] = dict(
        ok=bool(pos_ok), tol=POS_TOL,
        rule="min over all interior points of all training + test states "
             ">= -POS_TOL (the FOM Newton solves to ~1e-12 relative)")
    save()
    assert pos_ok, ("DATA NOT NON-NEGATIVE: tensor arm design stops here",
                    tr_min, report["data"]["test"]["min_u"])

    if TRAIN:
        S_tr = np.stack([kept[int(sid)][interior] for sid in pick])
        del kept
        log(f"  TRAIN in-job: {S_tr.shape[0]} states x {S_tr.shape[1]} points, "
            f"{STEPS} steps, lr {LR}")
        params, Z_tr, tinfo = sc.train_autodecoder(
            jax.random.PRNGKey(SEED0), coords_int, S_tr, K, R, steps=STEPS,
            lr=LR, tag=f"burgers N={N} k={K} r={R}")
        cfg = dict(pde="burgers2d", N=N, k=K, r=R, M=EQ_M, m=EQ_MQ, steps=STEPS,
                   lr=LR, n_test=N_TEST, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
                   num_steps=bc.NUM_STEPS, dt=bc.DT, tr_factor=TR_FACTOR,
                   seed=SEED0, data_seed=bc.SEED, test_seed=bc.TEST_SEED,
                   max_snaps=MAX_SNAPS, n_train_traj=bc.bf.N_TRAIN,
                   n_val_traj=bc.bf.N_VAL, arch_overrides={},
                   arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head "
                        "h(z)->R^r, hard poly BC; NO POD anywhere",
                   recipe="sep_burgers.py N=256 cell (60k steps, 8192 states, "
                          "576 trajectories), trained in-job by sep_b2d_tensor.py",
                   train=tinfo, x64=True,
                   matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                   backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
                   slurm_job=os.environ.get("SLURM_JOB_ID"),
                   node=os.environ.get("SLURMD_NODENAME", "local"))
        sc.save_pkl(CKPT_OUT, params, Z_tr, cfg)
        report["train"] = tinfo
        report["config"]["ckpt"] = CKPT_OUT
        # R-lite on the training states themselves (kept rows)
        r_rows = pick[:R_CHECK_STATES]
        r_fields = S_tr[:R_CHECK_STATES]
        del S_tr
    elif SKIP_TRAIN_TRUTH:
        r_rows, r_fields = pick[:0], None
    else:
        r_rows = pick[:R_CHECK_STATES]
        r_fields = np.stack([kept[int(sid)][interior] for sid in r_rows])
        del kept
    report["config"]["ckpt_cfg"] = {k_: v for k_, v in cfg.items()
                                    if isinstance(v, (int, float, str, bool,
                                                      type(None), dict))}
    dec = sc.SeparableDecoder(params, K, R)
    h_fn = dec.head_fn()
    Z_tr = np.asarray(Z_tr)
    assert pick.size == len(Z_tr), (pick.size, len(Z_tr))
    save()

    # ---------------- bank, modes, exact-linear matrix -----------------------
    G_all = dec.feat_at(coords, chunk=feat_chunk)                 # (n2, R)
    G_int = G_all[interior_j]                                     # (n_i2, R)
    kx, ky, Phi, lam, _ = bc.test_modes(N, EQ_M)
    Phi_np = np.asarray(Phi)
    del Phi
    Phi_j = jnp.asarray(Phi_np)
    lam_j = jnp.asarray(lam, dtype=F64)
    A_j = Phi_j.T @ G_int                                         # (M, R)
    kx_j = jnp.asarray(np.asarray(kx, dtype=np.float64))
    ky_j = jnp.asarray(np.asarray(ky, dtype=np.float64))
    if r_fields is not None:
        Hr = np.asarray(jax.vmap(h_fn)(jnp.asarray(Z_tr[:R_CHECK_STATES])))
        rec = np.asarray(jnp.asarray(Hr) @ G_int.T)
        r_lite = np.linalg.norm(rec - r_fields, axis=1) / np.linalg.norm(r_fields, axis=1)
        report["gates"]["R_lite_recon_on_regenerated_states"] = dict(
            mean=float(np.mean(r_lite)), max=float(np.max(r_lite)),
            n=int(r_lite.size), state_ids=[int(v) for v in r_rows])
        log(f"  GATE R-lite (checkpoint codes vs regenerated picked states, "
            f"{r_lite.size} states): recon rel-L2 mean {np.mean(r_lite):.3e} max "
            f"{np.max(r_lite):.3e}")
        assert np.mean(r_lite) < 0.2, "checkpoint/data/pick lineage broken"
        del rec, r_fields, Hr
    else:
        report["gates"]["R_lite_recon_on_regenerated_states"] = None

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    TRD = float(bc.TR_DELTA)
    report["config"]["trust_delta"] = TRD
    zbar = Z_tr.mean(0)

    bank_apply = jax.jit(lambda Gb, z: Gb @ h_fn(z))
    u_full_int = lambda z: bank_apply(G_int, z)
    _zb = jnp.asarray(Z_tr[0])
    _a, _b = u_full_int(_zb), dec(_zb, jnp.asarray(coords_int))
    dv = float(jnp.max(jnp.abs(_a - _b)) / (jnp.max(jnp.abs(_b)) + 1e-300))
    report["gates"]["bank_vs_meshfree"] = dv
    log(f"  GATE bank==meshfree: {dv:.2e}")
    assert dv < 1e-12
    adv_full = jax.jit(lambda uf: bc.upwind_adv_field(uf, N))

    # ---------------- NNLS m=256 advection-only node set (incumbent exlin) --
    eq_rng = np.random.default_rng(SEED0)
    eq_pick = np.sort(eq_rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    cand_pos = ctol_eq.candidate_pool(n_i2, cap=EQ_CAND_CAP)
    keep, wq_np, eq_info = xc.eq_fit_burgers_adv(
        u_full_int, adv_full, Phi_np, cand_pos, Z_eq, K, EQ_MQ,
        f"exlin N={N} k={K} M={EQ_M} m={EQ_MQ} adv-only", bc.nnls_capped)
    idx = np.asarray(interior[cand_pos[keep]])
    m = idx.size
    w_q = jnp.asarray(wq_np, dtype=F64)
    pos = np.searchsorted(interior, idx)
    assert np.all(interior[pos] == idx)
    Phi_q = jnp.asarray(Phi_np[pos]) * w_q[:, None]
    st = bc.stencil_indices(idx, N)
    G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, R)
    report["eq"] = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    report["eq"]["m"] = int(m)
    cl = dict(kind="grid", idx=idx, w=wq_np, info=eq_info)
    save()

    # ---------------- residual closures: r_w(z, prev_m, nu, aux) -------------
    def wt_of(nu):
        return (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)

    def lin_of(z, prev_m, nu):
        Ah = A_j @ h_fn(z)
        return (Ah - prev_m) + bc.DT * nu * lam_j * Ah

    def prev_of(z):
        return A_j @ h_fn(z)

    def sampled_adv(G5, Pq):
        def adv(z):
            us = jnp.einsum("msr,r->ms", G5, h_fn(z))
            c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
            ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
            uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
            return Pq.T @ (c * (ux + uy))
        return adv

    def mk_parts(adv_fn, uses_aux=False):
        def parts(z, prev_m, nu, aux):
            w_ = wt_of(nu)
            lin = lin_of(z, prev_m, nu)
            adv = adv_fn(z, aux) if uses_aux else adv_fn(z)
            return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

        def r_w(z, prev_m, nu, aux):
            return parts(z, prev_m, nu, aux)[0]
        return r_w, parts

    # ex (incumbent exlin, NNLS grid nodes)
    r_ex, parts_ex = mk_parts(sampled_adv(G_st, Phi_q))
    # full (oracle): aux = (G_int, Phi_j), explicit
    def adv_full_fn(z, aux):
        Gb, Ph = aux
        return Ph.T @ bc.upwind_adv_field(Gb @ h_fn(z), N)
    r_full, parts_full = mk_parts(adv_full_fn, uses_aux=True)
    aux_full = (G_int, Phi_j)

    # ---------------- gate 0 (incumbent-form ops == make_weak_ops) ----------
    def r_inc(z, prev_c, nu):
        us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
        c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
        Nu = c * (ux + uy)
        pu = Phi_q.T @ c
        return wt_of(nu) * (Phi_q.T @ (c - prev_c)
                            + bc.DT * ((Phi_q.T @ Nu) + nu * lam_j * pu))
    prev_inc = jax.jit(lambda z: G_st[:, 0, :] @ h_fn(z))
    rJ_inc = jax.jit(lambda z, p, nu: (r_inc(z, p, nu), jax.jacfwd(r_inc)(z, p, nu)))

    def adv_inc(z, nu):
        us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
        c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
        return wt_of(nu) * bc.DT * (Phi_q.T @ (c * (ux + uy)))
    adv_inc_j = jax.jit(adv_inc)
    ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=EQ_M, solver="lspg")
    grng = np.random.default_rng(SEED0 + 50)
    nu_med = float(np.median(nu_test))
    g0, gL, gA, gF = [], [], [], []
    parts_ex_j = jax.jit(parts_ex)
    parts_full_j = jax.jit(parts_full)
    prev_j = jax.jit(prev_of)
    fom_res_j = jax.jit(lambda u, up, nu: res_fn(u, up, nu))
    for _ in range(5):
        zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                         + 0.05 * grng.standard_normal(K))
        zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
        pc = ops_ref["prev_of"](zp)
        ra, Ja, _ = ops_ref["rJ"](zt, pc, nu_med)
        rb, Jb = rJ_inc(zt, prev_inc(zp), nu_med)
        g0.append(max(float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                      float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
        pm = prev_j(zp)
        _, lin_x, adv_x = parts_ex_j(zt, pm, nu_med, ())
        rf, lin_f, adv_f = parts_full_j(zt, pm, nu_med, aux_full)
        gL.append(float(jnp.max(jnp.abs(lin_x - lin_f)) / (jnp.max(jnp.abs(lin_f)) + 1e-300)))
        # incumbent advection part, computed DIRECTLY from the incumbent
        # stencil closure (sep_burgers_exlin.parts_inc), not by subtracting the
        # linear part from the residual: the subtraction cancels ~|r| eps and
        # tripped the 1e-12 tripwire at 2.3e-12 for R=512 (job 3039205)
        adv_i = adv_inc_j(zt, nu_med)
        gA.append(float(jnp.max(jnp.abs(adv_x - adv_i)) / (jnp.max(jnp.abs(adv_i)) + 1e-300)))
        # FOMR: full-grid weak residual == wt * Phi^T R_FOM[interior]
        uf = G_all @ h_fn(zt)
        upf = G_all @ h_fn(zp)
        Rf = fom_res_j(uf, upf, nu_med)[interior_j]
        r_direct = wt_of(nu_med) * (Phi_j.T @ Rf)
        gF.append(float(jnp.max(jnp.abs(rf - r_direct)) / (jnp.max(jnp.abs(r_direct)) + 1e-300)))
    report["gates"]["gate0"] = float(np.max(g0))
    report["gates"]["gateL"] = float(np.max(gL))
    report["gates"]["gateA"] = float(np.max(gA))
    report["gates"]["gateFOMR"] = float(np.max(gF))
    log(f"  GATE 0 (incumbent ops == make_weak_ops): {np.max(g0):.2e}")
    log(f"  GATE L (exlin linear == full-grid linear): {np.max(gL):.2e}")
    log(f"  GATE A (exlin advection == incumbent advection): {np.max(gA):.2e}")
    log(f"  GATE FOMR (full-grid weak residual == wt*Phi^T R_FOM): {np.max(gF):.2e}")
    assert np.max(g0) < 1e-12 and np.max(gL) < 1e-12 and np.max(gA) < 1e-12
    assert np.max(gF) < 1e-10
    del ops_ref
    save()

    # ---------------- tensor build + gates TB / TA / T0 / TS ----------------
    H_all = np.asarray(jnp.concatenate(
        [jax.vmap(h_fn)(jnp.asarray(Z_tr[s:s + 4096])) for s in range(0, len(Z_tr), 4096)]))
    if HEAD_PCA:
        # head-PCA Tucker compression: SVD of the head outputs over the training
        # codes, keep K' directions with tail energy <= PCA_TAIL, project the
        # bank (G P) and build the (M, K', K') tensor from it.  h^T T h ==
        # (P^T h)^T T' (P^T h) exactly for h in span(P); the projection tail is
        # the only new error (gate T0 becomes RECORDED with tripwire PCA_T0_TOL).
        HtH = np.asarray(jnp.asarray(H_all).T @ jnp.asarray(H_all))
        ev, V = np.linalg.eigh(HtH)
        order = np.argsort(ev)[::-1]
        ev, V = np.maximum(ev[order], 0.0), V[:, order]
        sv = np.sqrt(ev)
        energy = np.cumsum(ev) / np.sum(ev)
        Kp = int(np.searchsorted(1.0 - energy <= PCA_TAIL, True) + 1)
        Kp = min(max(Kp, 1), R)
        P_np = V[:, :Kp]
        tail_rel = float(np.sqrt(max(1.0 - energy[Kp - 1], 0.0)))
        Hp = H_all @ P_np @ P_np.T
        proj_rel = np.linalg.norm(H_all - Hp, axis=1) / (np.linalg.norm(H_all, axis=1) + 1e-300)
        report["head_pca"] = dict(
            K_prime=Kp, R=int(R), tail_energy=float(1.0 - energy[Kp - 1]),
            tail_rel_norm=tail_rel, singular_values=[float(v) for v in sv],
            energy_cumulative=[float(v) for v in energy],
            code_projection_rel_l2=stats(proj_rel), n_codes=int(len(H_all)))
        log(f"  HEAD-PCA: R={R} -> K'={Kp} (tail energy {1.0 - energy[Kp-1]:.2e}, "
            f"sv[0]={sv[0]:.3e} sv[K'-1]={sv[Kp-1]:.3e} sv[-1]={sv[-1]:.3e}; "
            f"per-code projection rel-L2 median {np.median(proj_rel):.2e} max "
            f"{np.max(proj_rel):.2e})")
        P_j = jnp.asarray(P_np)
        G_T = G_int @ P_j                                    # (n_i2, K')
        R_T = Kp
    else:
        P_j, G_T, R_T = None, G_int, R
    t0 = time.time()
    T1 = tc.build_T(Phi_j, G_T, N, chunk=T_CHUNK, reverse=False)
    t_build = time.time() - t0
    T2 = tc.build_T(Phi_j, G_T, N, chunk=max(1000, T_CHUNK // 3 + 7),
                    reverse=True)
    gTB = float(np.max(np.abs(T1 - T2)) / np.max(np.abs(T1)))
    Q = tc.symmetrize(T1)
    del T2
    report["gates"]["TB_build_order_rel"] = gTB
    report["tensor"] = dict(shape=list(Q.shape), build_secs=t_build,
                            bytes=int(Q.nbytes), max_abs=float(np.max(np.abs(Q))),
                            T_asym_rel=float(np.max(np.abs(T1 - T1.swapaxes(1, 2)))
                                             / np.max(np.abs(T1))))
    log(f"  GATE TB (two chunkings): {gTB:.2e}  [Q {Q.shape}, {Q.nbytes/2**20:.1f} "
        f"MiB, build {t_build:.2f}s, T asymmetry {report['tensor']['T_asym_rel']:.2f}]")
    assert gTB < 1e-14
    Qj = jnp.asarray(Q)
    DG_j = tc.backward_diff_bank_2d(G_int, N)

    def to_T(h):
        """head output -> tensor coordinates (identity, or P^T h under HEAD_PCA)"""
        return h if P_j is None else h @ P_j

    @jax.jit
    def ta_chunk(Hc, Gb, Db, Ph, Qq):
        U = Hc @ Gb.T                                          # (c, n_i2)
        q_alg = (U * (Hc @ Db.T)) @ Ph
        q_or = jax.vmap(lambda u: bc.upwind_adv_field(u, N))(U) @ Ph
        Ht = to_T(Hc)
        q_T = 0.5 * jnp.einsum("ijk,sj,sk->si", Qq, Ht, Ht)
        return (q_alg, q_or, q_T, jnp.min(U, axis=1),
                jnp.sum(U <= 0, axis=1), jnp.sum(U < 0, axis=1))
    qa, qo, qt, mn, nle, nlt = [], [], [], [], [], []
    for s in range(0, len(H_all), TA_CHUNK):
        o = ta_chunk(jnp.asarray(H_all[s:s + TA_CHUNK]), G_int, DG_j, Phi_j, Qj)
        for L_, v in zip((qa, qo, qt, mn, nle, nlt), o):
            L_.append(np.asarray(v))
    q_alg, q_or, q_T = np.concatenate(qa), np.concatenate(qo), np.concatenate(qt)
    minu, n_le, n_lt = np.concatenate(mn), np.concatenate(nle), np.concatenate(nlt)
    nq = np.linalg.norm(q_or, axis=1) + 1e-300
    gTA = float(np.max(np.linalg.norm(q_T - q_alg, axis=1) / nq))
    report["gates"]["TA_algebraic_identity_max_rel"] = gTA
    log(f"  GATE TA (h^T T h == Phi^T(u (D-x u + D-y u)), {len(H_all)} training "
        f"states): {gTA:.2e}" + ("  [HEAD_PCA: includes the projection tail; "
                                 "recorded]" if HEAD_PCA else ""))
    if HEAD_PCA:
        # the exact identity holds on the PROJECTED codes: check it there
        qa2 = []
        for s in range(0, len(H_all), TA_CHUNK):
            Hc = jnp.asarray(H_all[s:s + TA_CHUNK] @ P_np @ P_np.T)
            qa2.append(np.asarray(ta_chunk(Hc, G_int, DG_j, Phi_j, Qj)[0]))
        gTAp = float(np.max(np.linalg.norm(q_T - np.concatenate(qa2), axis=1) / nq))
        report["gates"]["TA_on_projected_codes_max_rel"] = gTAp
        log(f"  GATE TA' (identity on P P^T h): {gTAp:.2e}")
        assert gTAp < 1e-12
    else:
        assert gTA < 1e-13
    pos_states = minu > 0
    mis = np.linalg.norm(q_T - q_or, axis=1) / nq
    gT0 = float(np.max(mis[pos_states])) if np.any(pos_states) else None
    report["gates"]["T0_all_positive_states_max_rel"] = gT0
    report["gates"]["T0_n_all_positive_states"] = int(np.sum(pos_states))
    report["TS_train_states"] = dict(
        mismatch_rel=stats(mis), frac_points_u_le0=float(n_le.sum() / (len(H_all) * n_i2)),
        frac_points_u_lt0=float(n_lt.sum() / (len(H_all) * n_i2)),
        frac_states_all_positive=float(np.mean(pos_states)), min_u=float(minu.min()),
        mismatch_rel_nonpositive_states=stats(mis[~pos_states]) if np.any(~pos_states) else None)
    log(f"  GATE T0 (tensor == oracle on the {int(np.sum(pos_states))} all-positive "
        f"decoded training states): {gT0}; all states: mismatch median "
        f"{np.median(mis):.2e} max {np.max(mis):.2e}; frac points u<=0 "
        f"{report['TS_train_states']['frac_points_u_le0']:.3%}, min u {minu.min():.2e}")
    if gT0 is not None:
        if HEAD_PCA:
            report["gates"]["T0_tripwire"] = PCA_T0_TOL
            assert gT0 < PCA_T0_TOL, ("HEAD-PCA projection tail too large", gT0)
        else:
            assert gT0 < 1e-12
    del q_alg, q_or, q_T, DG_j
    save()

    # tensor arm closure
    Qm = jnp.asarray(Q.reshape(EQ_M * R_T, R_T))  # (M*R', R'): one matvec, then (M,R')@h

    def adv_tensor(z):
        h = to_T(h_fn(z))
        return 0.5 * ((Qm @ h).reshape(EQ_M, R_T) @ h)
    r_T, parts_T = mk_parts(adv_tensor)

    # ---------------- gate TQ: tensor vs oracle residual/Jacobian ----------
    rJ_full = jax.jit(lambda z, p, nu, aux: (r_full(z, p, nu, aux),
                                             jax.jacfwd(r_full)(z, p, nu, aux)))
    rJ_T = jax.jit(lambda z, p, nu: (r_T(z, p, nu, ()),
                                     jax.jacfwd(r_T)(z, p, nu, ())))
    qrng = np.random.default_rng(SEED0 + 500)
    tq = []
    for si in range(N_TQ):
        i = qrng.integers(len(Z_tr))
        pert = si >= N_TQ // 2
        z = Z_tr[i] + (0.05 * qrng.standard_normal(K) if pert else 0.0)
        zp = Z_tr[qrng.integers(len(Z_tr))]
        nu = float(np.exp(qrng.uniform(np.log(0.01), np.log(0.1))))
        pv = prev_j(jnp.asarray(zp))
        ro, Jo = [np.asarray(v) for v in rJ_full(jnp.asarray(z), pv, nu, aux_full)]
        rt, Jt = [np.asarray(v) for v in rJ_T(jnp.asarray(z), pv, nu)]
        uu = np.asarray(u_full_int(jnp.asarray(z)))
        tq.append(dict(perturbed=bool(pert), nu=nu, r_rel=rel(rt, ro),
                       J_rel=rel(Jt, Jo),
                       g_scaled=float(np.linalg.norm(Jt.T @ rt - Jo.T @ ro)
                                      / (np.linalg.norm(Jo) * np.linalg.norm(ro) + 1e-300)),
                       min_u=float(uu.min()), n_neg=int(np.sum(uu <= 0))))
    report["gates"]["TQ"] = dict(
        r_rel=stats([t_["r_rel"] for t_ in tq]), J_rel=stats([t_["J_rel"] for t_ in tq]),
        g_scaled_max=float(max(t_["g_scaled"] for t_ in tq)),
        n_states_with_neg=int(sum(1 for t_ in tq if t_["n_neg"] > 0)),
        min_u=float(min(t_["min_u"] for t_ in tq)), rows=tq)
    g_ = report["gates"]["TQ"]
    log(f"  GATE TQ (tensor vs oracle, {N_TQ} states, recorded): r rel median "
        f"{g_['r_rel']['median']:.2e} max {g_['r_rel']['max']:.2e}; J rel max "
        f"{g_['J_rel']['max']:.2e}; states with u<=0 {g_['n_states_with_neg']}/{N_TQ}")
    save()

    # ---------------- learned continuous nodes (optional) -------------------
    r_ln = None
    if NODES_NPZ:
        nz = np.load(NODES_NPZ)
        X_l, w_l = np.asarray(nz["X"], np.float64), np.asarray(nz["w"], np.float64)
        OFF = np.array([[0.0, 0.0], [dx, 0.0], [-dx, 0.0], [0.0, dx], [0.0, -dx]])
        Xs = (X_l[:, None, :] + OFF[None, :, :]).reshape(-1, 2)
        G5_l = dec.feat_at(Xs).reshape(len(X_l), 5, R)
        sx = np.sin(np.pi * np.asarray(kx)[None, :] * X_l[:, 0:1])
        sy = np.sin(np.pi * np.asarray(ky)[None, :] * X_l[:, 1:2])
        Phi_l = jnp.asarray((sx * sy) / ((N - 1) / 2.0) * w_l[:, None])
        r_ln, _ = mk_parts(sampled_adv(G5_l, Phi_l))
        report["eq_learned"] = dict(npz=NODES_NPZ, m=int(len(X_l)),
                                    w_nonzero=int(np.sum(w_l > 0)),
                                    min_x=float(X_l.min()), max_x=float(X_l.max()))
        log(f"  learned nodes: {len(X_l)} continuous nodes from {NODES_NPZ}")

    # ---------------- IC encoder (t=0 pairs, training data only) ------------
    t0_rows = np.nonzero((pick % T) == 0)[0]
    U0_tr = np.stack([bc.bf.blob_ic(N, cx[i], cy[i], w[i], a[i])
                      for i in (pick[t0_rows] // T)])
    X_tr = U0_tr[:, idx]
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr[t0_rows], steps=ENC_STEPS,
        tag=f"b2d tensor u0->z0 N={N}")
    report["encoder"] = enc_info
    del U0_tr, X_tr
    idx_j = jnp.asarray(idx)

    Gram_all = G_all.T @ G_all
    eps_g = 1e-13 * jnp.trace(Gram_all) / R
    L_all = jnp.linalg.cholesky(Gram_all + eps_g * jnp.eye(R, dtype=F64))

    def ic_enc_full(Gb, u0):
        z0 = enc_apply(enc_params, u0[idx_j])

        def f(z):
            return Gb @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, rn, nJ

    def ic_enc_gram(Gb, u0):
        z0 = enc_apply(enc_params, u0[idx_j])
        b = Gb.T @ u0
        y = jax.scipy.linalg.solve_triangular(L_all, b, lower=True)
        c2 = jnp.maximum(u0 @ u0 - y @ y, 0.0)

        def f(z):
            return L_all.T @ h_fn(z) - y
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, jnp.sqrt(rn * rn + c2), nJ

    ic_full_j, ic_gram_j = jax.jit(ic_enc_full), jax.jit(ic_enc_gram)
    zf, rf_, _ = ic_full_j(G_all, jnp.asarray(U_test[0, 0], dtype=F64))
    zg, rg, _ = ic_gram_j(G_all, jnp.asarray(U_test[0, 0], dtype=F64))
    report["gates"]["ic_gram_vs_full_latent_dev"] = float(
        jnp.linalg.norm(zg - zf) / (1.0 + jnp.linalg.norm(zf)))
    report["gates"]["ic_gram_vs_full_resnorm_rel"] = float(abs(rg - rf_) / (abs(rf_) + 1e-300))
    log(f"  IC gram vs full: |dz|/(1+|z|) {report['gates']['ic_gram_vs_full_latent_dev']:.2e}, "
        f"||r|| rel diff {report['gates']['ic_gram_vs_full_resnorm_rel']:.2e}")
    decode_j = jax.jit(lambda Gb, Zf: jax.vmap(h_fn)(Zf) @ Gb.T)
    tol_scale = float(np.sqrt(n_i2))

    # ---------------- arms: step / rollout / e2e per residual path ----------
    prev_fn = prev_of
    arm_defs = {"full": (r_full, aux_full), "ex": (r_ex, ()), "tensor": (r_T, ())}
    if r_ln is not None:
        arm_defs["ex_learned"] = (r_ln, ())
    arms = [a_ for a_ in ARMS if a_ in arm_defs]
    if r_ln is not None and "ex_learned" not in arms:
        arms.append("ex_learned")
    report["config"]["arms_run"] = arms

    def make_arm(r_w, aux):
        step, rn_j, rJ_j = make_step_aux(r_w, K, STALL, TRD)
        roll = make_roll_aux(step, rn_j, prev_fn, bc.NUM_STEPS, EXTRAP)

        def e2e(Gb, u0, nu, aux_):
            z0, ic_rn, ic_nJ = ic_enc_gram(Gb, u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((bc.NUM_STEPS,), STEP_TOL * u_scale * tol_scale, dtype=F64)
            Z, rns, nJs, atts, reasons = roll(z0, nu, us, bc.GN_BUDGET, aux_)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)
            return (decode_j(Gb, Zfull), z0, Z, rns, nJs, atts, reasons, ic_rn, ic_nJ)
        return dict(step=step, rn=rn_j, rJ=rJ_j, roll=roll, e2e=jax.jit(e2e), aux=aux)

    A_ = {a_: make_arm(*arm_defs[a_]) for a_ in arms}

    # gate STEP: aux-threaded step == make_step_lspg_var (same stall) ==
    # incumbent _finish_ops step at stall 1e-12, on the sampled arm
    r_ex3 = lambda z, p, nu: r_ex(z, p, nu, ())
    step_ss = ss.make_step_lspg_var(r_ex3, K, stall_rel=STALL)
    step_mine12, _, _ = make_step_aux(r_ex, K, 1e-12, TRD)
    ops_inc = bc._finish_ops(lambda z, p, nu: (r_ex3(z, p, nu), jax.jacfwd(r_ex3)(z, p, nu), None),
                             r_ex3, prev_of, lambda z: A_j @ h_fn(z), m, "lspg")
    zt = jnp.asarray(Z_tr[3])
    pc = prev_j(jnp.asarray(Z_tr[5]))
    tolg = STEP_TOL * float(np.sqrt(np.mean(U_test[0, 0][interior] ** 2))) * tol_scale
    arm_ex = A_["ex"] if "ex" in A_ else make_arm(*arm_defs["ex"])
    a_ = arm_ex["step"](zt, pc, nu_med, tolg, bc.GN_BUDGET, ())
    b_ = step_ss(zt, pc, nu_med, tolg, bc.GN_BUDGET)
    c_ = step_mine12(zt, pc, nu_med, tolg, bc.GN_BUDGET, ())
    d_ = ops_inc["step_jit"](zt, pc, nu_med, tolg, bc.GN_BUDGET)
    sdev = float(jnp.max(jnp.abs(a_[0] - b_[0])))
    sdev12 = float(jnp.max(jnp.abs(c_[0] - d_[0])))
    report["gates"]["STEP_aux_vs_make_step_lspg_var"] = sdev
    report["gates"]["STEP_aux_1e-12_vs_incumbent_finish_ops"] = sdev12
    log(f"  GATE STEP: aux step vs make_step_lspg_var(stall={STALL:g}) max|dz| "
        f"{sdev:.2e}; at 1e-12 vs incumbent _finish_ops {sdev12:.2e}")
    assert sdev == 0.0 and sdev12 == 0.0
    # gate ROLL: aux rollout == make_rollout_v2 (same step) on traj 0
    ops_v = dict(ops_inc)
    ops_v["step_jit"] = step_ss
    roll_ref = ss.make_rollout_v2("incumbent", ops=ops_v, num_steps=bc.NUM_STEPS,
                                  extrap=EXTRAP)
    u0j = jnp.asarray(U_test[0, 0], dtype=F64)
    z0g, _, _ = ic_gram_j(G_all, u0j)
    us0 = jnp.full((bc.NUM_STEPS,), tolg, dtype=F64)
    d0 = jnp.asarray(TRD, dtype=F64)
    Zr, _, nJr, _, _ = roll_ref(z0g, float(nu_test[0]), us0, bc.GN_BUDGET, d0, d0, d0)
    Zm, _, nJm, attm, _ = arm_ex["roll"](z0g, float(nu_test[0]), us0, bc.GN_BUDGET, ())
    rdev = float(jnp.max(jnp.abs(Zr - Zm)))
    report["gates"]["ROLL_aux_vs_make_rollout_v2"] = rdev
    log(f"  GATE ROLL: aux rollout vs make_rollout_v2 max|dZ| {rdev:.2e} "
        f"(nJ equal {bool(jnp.all(nJr == nJm))}, attempts/traj {int(jnp.sum(attm))})")
    assert rdev == 0.0
    del ops_inc, ops_v, roll_ref
    save()

    # ---------------- interleaved timing: reps > trajectories > arms --------
    n_test = min(N_TEST, U_test.shape[0])
    u0_dev = [jnp.asarray(U_test[i, 0], dtype=F64) for i in range(n_test)]
    us_dev = []
    for i in range(n_test):
        u_scale = float(np.sqrt(np.mean(U_test[i, 0][interior] ** 2)))
        us_dev.append(jnp.full((bc.NUM_STEPS,), STEP_TOL * u_scale * tol_scale, dtype=F64))
    times = {a_: [dict(ic=[], roll=[], dec=[], e2e=[]) for _ in range(n_test)]
             for a_ in arms}
    last = {a_: [None] * n_test for a_ in arms}
    Z_reps = {a_: [[] for _ in range(n_test)] for a_ in arms}

    def one(arm, i):
        ar = A_[arm]
        nu = float(nu_test[i])
        t0 = time.perf_counter()
        z0, ic_rn, ic_nJ = blk(ic_gram_j(G_all, u0_dev[i]))
        t1 = time.perf_counter()
        Rl = blk(ar["roll"](z0, nu, us_dev[i], bc.GN_BUDGET, ar["aux"]))
        t2 = time.perf_counter()
        F = blk(decode_j(G_all, jnp.concatenate([z0[None], Rl[0]], axis=0)))
        t3 = time.perf_counter()
        E = blk(ar["e2e"](G_all, u0_dev[i], nu, ar["aux"]))
        t4 = time.perf_counter()
        return dict(ic=t1 - t0, roll=t2 - t1, dec=t3 - t2, e2e=t4 - t3), (z0, Rl, F, E)

    ctol_tol.burn_in(1.5)
    for rep_ in range(BURN + TIME_REPS):
        order = arms if rep_ % 2 == 0 else list(reversed(arms))
        for i in range(n_test):
            for arm in order:
                tm, outs = one(arm, i)
                if rep_ >= BURN:
                    for k_ in tm:
                        times[arm][i][k_].append(tm[k_])
                    Z_reps[arm][i].append(np.asarray(outs[3][2]))
                    last[arm][i] = outs
        log(f"  rep {rep_} ({'timed' if rep_ >= BURN else 'burn'}, order {order}) done "
            f"[{time.time()-t_all:.0f}s]")
    for arm in arms:
        rows = []
        agg_reasons = {}
        for i in range(n_test):
            z0, Rl, F, E = last[arm][i]
            Fh = np.asarray(E[0])
            tnorm = np.linalg.norm(U_test[i], axis=1)
            pt = np.linalg.norm(Fh - U_test[i], axis=1) / tnorm
            Zs = np.asarray(E[2])
            reasons = np.asarray(E[6])
            hr = hist_of(reasons)
            for k_, v_ in hr.items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v_
            Fint = Fh[:, interior]
            rep_dev = float(max(np.max(np.abs(Zk - Zs)) for Zk in Z_reps[arm][i]))
            tm = times[arm][i]
            rows.append(dict(
                traj=i, nu=float(nu_test[i]), traj_rel=float(np.mean(pt)),
                per_time=[float(v) for v in pt], per_time_max=float(np.max(pt)),
                ic_rel=float(E[7]) / float(np.linalg.norm(U_test[i, 0])),
                ic_jac=int(E[8]), jac_total=int(np.sum(np.asarray(E[4]))),
                attempts_total=int(np.sum(np.asarray(E[5]))),
                attempts_per_step=[int(v) for v in np.asarray(E[5])],
                stop_reasons=hr, rn_final=[float(v) for v in np.asarray(E[3])],
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                split_vs_fused_latent_dev=float(np.max(np.abs(np.asarray(Rl[0]) - Zs))),
                timed_reps_latent_dev_max=rep_dev,
                decoded_min_u=float(Fint.min()),
                decoded_min_u_per_state=[float(v) for v in Fint.min(axis=1)],
                decoded_frac_states_with_u_le0=float(np.mean(Fint.min(axis=1) <= 0)),
                decoded_frac_points_le0=float(np.mean(Fint <= 0)),
                ic_ms=float(np.median(tm["ic"])) * 1e3,
                roll_ms=float(np.median(tm["roll"])) * 1e3,
                dec_ms=float(np.median(tm["dec"])) * 1e3,
                e2e_ms=float(np.median(tm["e2e"])) * 1e3,
                split_sum_ms=float(np.median(np.asarray(tm["ic"]) + np.asarray(tm["roll"])
                                             + np.asarray(tm["dec"]))) * 1e3,
                raw_s={k_: [float(v) for v in tm[k_]] for k_ in tm}))
        v = dict(name=arm, n_test=n_test,
                 err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
                 err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
                 ic_rel_mean=float(np.mean([r_["ic_rel"] for r_ in rows])),
                 jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
                 attempts_total_mean=float(np.mean([r_["attempts_total"] for r_ in rows])),
                 stop_reasons=agg_reasons,
                 n_blowups=int(sum(r_["n_finite_steps"] < bc.NUM_STEPS for r_ in rows)),
                 decoded_min_u=float(min(r_["decoded_min_u"] for r_ in rows)),
                 decoded_frac_states_with_u_le0=float(np.mean(
                     [r_["decoded_frac_states_with_u_le0"] for r_ in rows])),
                 decoded_frac_points_le0=float(np.mean(
                     [r_["decoded_frac_points_le0"] for r_ in rows])),
                 per_traj=rows)
        for k_ in ("ic_ms", "roll_ms", "dec_ms", "e2e_ms", "split_sum_ms"):
            allv = [x for r_ in rows for x in
                    (np.asarray(r_["raw_s"][k_.replace("_ms", "")]) * 1e3
                     if k_ != "split_sum_ms" else
                     (np.asarray(r_["raw_s"]["ic"]) + np.asarray(r_["raw_s"]["roll"])
                      + np.asarray(r_["raw_s"]["dec"])) * 1e3)]
            v[k_ + "_median"] = float(np.median(allv))
        report["variants"][arm] = v
        log(f"   ARM {arm:11s} err {v['err_traj_rel_mean']:.6e} (max {v['err_traj_rel_max']:.3e})"
            f"  e2e {v['e2e_ms_median']:.2f} ms = ic {v['ic_ms_median']:.2f} + solve "
            f"{v['roll_ms_median']:.2f} + dec {v['dec_ms_median']:.2f} (split sum "
            f"{v['split_sum_ms_median']:.2f}); attempts {v['attempts_total_mean']:.0f} "
            f"jac {v['jac_total_mean']:.0f}; {v['stop_reasons']}; decoded min u "
            f"{v['decoded_min_u']:.2e}, states with u<=0 {v['decoded_frac_states_with_u_le0']:.1%}")
    save()

    # ---------------- comparisons vs full and vs ex --------------------------
    cmp = {}
    for ref_arm in ("full", "ex"):
        if ref_arm not in report["variants"]:
            continue
        vo = report["variants"][ref_arm]
        for arm in arms:
            if arm == ref_arm:
                continue
            va = report["variants"][arm]
            rows = []
            for i in range(n_test):
                ro_, ra_ = vo["per_traj"][i], va["per_traj"][i]
                rows.append(dict(
                    traj=i, err_ref=ro_["traj_rel"], err_arm=ra_["traj_rel"],
                    abs_diff=abs(ra_["traj_rel"] - ro_["traj_rel"]),
                    lat_dev=float(np.max(np.abs(np.asarray(last[arm][i][3][2])
                                                - np.asarray(last[ref_arm][i][3][2])))),
                    reasons_equal=(ra_["stop_reasons"] == ro_["stop_reasons"]),
                    attempts_equal=(ra_["attempts_total"] == ro_["attempts_total"]),
                    e2e_ratio=ra_["e2e_ms"] / ro_["e2e_ms"],
                    roll_ratio=ra_["roll_ms"] / ro_["roll_ms"]))
            cmp[f"{arm}_vs_{ref_arm}"] = dict(
                per_traj=rows,
                err_abs_diff_max=float(max(r_["abs_diff"] for r_ in rows)),
                lat_dev_max=float(max(r_["lat_dev"] for r_ in rows)),
                err_ratio=va["err_traj_rel_mean"] / vo["err_traj_rel_mean"],
                stop_hist_identical=bool(va["stop_reasons"] == vo["stop_reasons"]),
                stop_hist_identical_per_traj=bool(all(r_["reasons_equal"] for r_ in rows)),
                attempts_identical_per_traj=bool(all(r_["attempts_equal"] for r_ in rows)),
                e2e_ratio=va["e2e_ms_median"] / vo["e2e_ms_median"],
                roll_ratio=va["roll_ms_median"] / vo["roll_ms_median"],
                ic_ratio=va["ic_ms_median"] / vo["ic_ms_median"],
                dec_ratio=va["dec_ms_median"] / vo["dec_ms_median"])
            c = cmp[f"{arm}_vs_{ref_arm}"]
            log(f"  [{arm} vs {ref_arm}] err |diff| max {c['err_abs_diff_max']:.2e}; err ratio "
                f"{c['err_ratio']:.4f}; latent dev max {c['lat_dev_max']:.2e}; stop hist "
                f"identical {c['stop_hist_identical']} (per traj "
                f"{c['stop_hist_identical_per_traj']}, attempts {c['attempts_identical_per_traj']}); "
                f"e2e ratio {c['e2e_ratio']:.3f} solve ratio {c['roll_ratio']:.3f}")
    report["comparison"] = cmp
    save()

    # ---------------- FOM ladder (standardised tol-Newton, same GPU) --------
    tol_newton = make_tol_newton_pc(N)
    base_cfgs = [(nt, max(nt * lf, 1e-12)) for nt in NEWTON_TOLS for lf in LIN_FRACS]
    base_rows = {c_: [] for c_ in base_cfgs}
    ctol_tol.burn_in(1.5)
    for i in range(n_test):
        u0 = u0_dev[i]
        nu = float(nu_test[i])
        tnorm = np.linalg.norm(U_test[i], axis=1)
        subs = [(f"fom|nt{nt:.0e}|lt{lt:.0e}",
                 lambda _u=u0, _n=nu, _t=nt, _l=lt:
                 (lambda o: (o[0].block_until_ready(), o)[1])(tol_newton(_u, _n, _t, _l)))
                for (nt, lt) in base_cfgs]
        raw, res = sc.balanced_time(subs, reps=FOM_REPS, warm=WARM)
        for (nt, lt) in base_cfgs:
            key = f"fom|nt{nt:.0e}|lt{lt:.0e}"
            snaps, its, rels = res[key]
            pt = np.linalg.norm(np.asarray(snaps) - U_test[i], axis=1) / tnorm
            base_rows[(nt, lt)].append(dict(
                traj=i, traj_rel=float(np.mean(pt)),
                newton_iters_total=int(np.sum(np.asarray(its))),
                time_ms=float(np.median(raw[key])) * 1e3,
                time_raw_s=[float(t) for t in raw[key]]))
        log(f"  FOM ladder traj {i} done [{time.time()-t_all:.0f}s]")
    report["fom"] = []
    for (nt, lt) in base_cfgs:
        rows = base_rows[(nt, lt)]
        report["fom"].append(dict(
            method="fom_newton_tol_pc", N=N, newton_tol=nt, lin_tol=lt,
            preconditioner="exact Helmholtz (sine basis)",
            err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
            time_ms_median=float(np.median([r_["time_ms"] for r_ in rows])),
            newton_iters_mean=float(np.mean([r_["newton_iters_total"] for r_ in rows])),
            per_traj=rows, n_test=n_test))
    for f_ in report["fom"]:
        log(f"   FOM nt={f_['newton_tol']:.0e} lt={f_['lin_tol']:.0e}: err "
            f"{f_['err_traj_rel_mean']:.3e}  {f_['time_ms_median']:.2f} ms  "
            f"newton {f_['newton_iters_mean']:.0f}")
    save()

    # matched rules + paired AB/BA per ROM arm
    fom_err = {c_: float(np.mean([r_["traj_rel"] for r_ in base_rows[c_]])) for c_ in base_cfgs}
    fom_ms = {c_: float(np.median([r_["time_ms"] for r_ in base_rows[c_]])) for c_ in base_cfgs}
    tight = min(base_cfgs, key=lambda c_: fom_err[c_])
    report["matched"] = dict(tightest=dict(newton_tol=tight[0], lin_tol=tight[1],
                                           err=fom_err[tight], ms=fom_ms[tight]), arms={})
    for arm in arms:
        cerr = report["variants"][arm]["err_traj_rel_mean"]
        cands = [c_ for c_ in base_cfgs if fom_err[c_] <= cerr]
        match = min(cands, key=lambda c_: fom_ms[c_]) if cands else None
        closest = min(base_cfgs, key=lambda c_: abs(np.log(fom_err[c_] + 1e-300) - np.log(cerr)))
        ent = dict(rom_err=cerr, rom_e2e_ms=report["variants"][arm]["e2e_ms_median"],
                   rule_matched="cheapest (newton_tol, lin_tol) at least as accurate as the arm",
                   matched=None if match is None else dict(
                       newton_tol=match[0], lin_tol=match[1], err=fom_err[match], ms=fom_ms[match]),
                   closest=dict(newton_tol=closest[0], lin_tol=closest[1],
                                err=fom_err[closest], ms=fom_ms[closest]))
        if match is not None:
            pairs = []
            for i in range(n_test):
                u0 = u0_dev[i]
                nu = float(nu_test[i])
                ar = A_[arm]
                pairs.append(dict(traj=i, **sc.time_pair(
                    lambda _u=u0, _n=nu, _a=ar: _a["e2e"](G_all, _u, _n, _a["aux"])[0].block_until_ready(),
                    lambda _u=u0, _n=nu, _t=match[0], _l=match[1]:
                    tol_newton(_u, _n, _t, _l)[0].block_until_ready(),
                    reps=PAIR_REPS, warm=WARM)))
            ent["paired"] = dict(rom_ms=float(np.median([p["a_ms"] for p in pairs])),
                                 fom_ms=float(np.median([p["b_ms"] for p in pairs])),
                                 per_traj=pairs)
            ent["paired"]["speedup"] = ent["paired"]["fom_ms"] / ent["paired"]["rom_ms"]
            log(f"  MATCHED paired [{arm}]: ROM {ent['paired']['rom_ms']:.2f} ms vs tol-Newton"
                f"(nt={match[0]:.0e},lt={match[1]:.0e}, err {fom_err[match]:.2e}) "
                f"{ent['paired']['fom_ms']:.2f} ms -> {ent['paired']['speedup']:.2f}x")
        report["matched"]["arms"][arm] = ent
    save()

    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
