"""Burgers 3D ROM: the precomputed quadratic-tensor advection vs the sampled
exact-linear rule and the full-grid oracle, one invocation per N, with the
two classical ladders (newton, defect) on the same GPU (2026-09-03, design
B3D-DESIGN.md r3, branch exp/2026-09-03-burgers3d-tensor; 3D port of
sep_b2d_tensor.py on the self-contained b3d_common.py).

Modes (env):
  PILOT=1   train + bank + D1..D4 + M-stability on the VALIDATION rows only,
            write the JSON, stop (the capacity pilot; the test table is never
            opened).
  MICRO=1   the M1 micro-pilot: real shapes, short training, one trajectory,
            one step of each classical arm; device/host peak memory recorded.
  default   phases 1-3: gates D1..D4, L/A/FOMR, STEP/ROLL, TB/TA/T0/TQ, the
            interleaved arms (full, ex, tensor), per-step optimality (P1), E1
            comparisons incl. along-path operator fidelity, TR candidate-path
            audit, the newton + defect ladders with matched/bracketed paired
            AB/BA and the trajectory-clustered bootstrap, kernel export.

Residual paths (all EXACT-LINEAR, A = Phi^T G precomputed; only advection
differs; IC fit, LM rule, stall, tolerances, trust region, warm-start
extrapolation and decode are IDENTICAL across arms):
  full    Phi^T N(u) on the full interior grid with the FOM's sign-upwind
          stencil (the oracle; bank + modes threaded as explicit aux)
  ex      advection sampled at m NNLS grid nodes fit on advection rows only
  tensor  0.5 h^T Q h with Q built in-job from the frozen bank (blocked)
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import b3d_common as b3
import b3d_tensor_common as tc

F64 = jnp.float64
HERE = os.path.dirname(os.path.abspath(__file__))

N = int(os.environ.get("N", "33"))
K = int(os.environ.get("K", "32"))
R = int(os.environ.get("R", "128"))
EQ_M = int(os.environ.get("EQ_M", "256"))
EQ_MQ = int(os.environ.get("EQ_MQ", str(4 * EQ_M)))
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))
G_HIDDEN = int(os.environ.get("G_HIDDEN", str(2 * R)))
CKPT = os.environ.get("CKPT", "")
TRAIN = int(os.environ.get("TRAIN", "0" if CKPT else "1"))
OUT = os.environ.get("OUT", f"/tmp/sep_b3d_tensor_n{N}.json")
CKPT_OUT = os.environ.get("CKPT_OUT", OUT.replace(".json", "_ckpt.pkl"))
KERNEL_OUT = os.environ.get("KERNEL_OUT", OUT.replace(".json", "_kernel.npz"))
TABLE_DIR = os.environ.get("TABLE_DIR", os.path.join(HERE, "runs", "b3dtensor", "tables"))
N_TRAIN_TABLE = 576
N_TRAIN = int(os.environ.get("N_TRAIN", "512"))         # training prefix rows 0..N_TRAIN-1
VAL_ROWS = np.arange(512, 576)                          # validation rows at every N
N_TEST = int(os.environ.get("N_TEST", "8"))
SEED0 = int(os.environ.get("SEED0", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", "1"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
STEPS = int(os.environ.get("STEPS", "60000"))
LR = float(os.environ.get("LR", "1e-3"))
P_SUB = int(os.environ.get("P_SUB", "16384"))
POOL_CAP = int(os.environ.get("POOL_CAP", str(63 ** 3)))
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "0"))
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "262144"))
T_CHUNK = int(os.environ.get("T_CHUNK", str(max(256, (512 * 2 ** 20) // (8 * R * R)))))
TA_CHUNK = int(os.environ.get("TA_CHUNK", "32"))
DST = os.environ.get("DST", "mm")
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
STALL = float(os.environ.get("STALL", "1e-3"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
GN_BUDGET = int(os.environ.get("GN_BUDGET", "30"))
IC_ENC_BUDGET = int(os.environ.get("IC_ENC_BUDGET", "50"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
ORACLE_STARTS = int(os.environ.get("ORACLE_STARTS", "8"))
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "200"))
ORACLE_TIMES = [int(v) for v in os.environ.get("ORACLE_TIMES", "0,10,25,50").split(",")]
ORACLE_VAL_TRAJ = int(os.environ.get("ORACLE_VAL_TRAJ", "64"))
TIME_REPS = int(os.environ.get("TIME_REPS", "5"))
BURN = int(os.environ.get("BURN", "2"))
FOM_REPS = int(os.environ.get("FOM_REPS", "5"))
PAIR_REPS = int(os.environ.get("PAIR_REPS", "5"))
WARM = int(os.environ.get("WARM", "2"))
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4").split(",")]
LIN_FRACS = [float(v) for v in os.environ.get("LIN_FRACS", "0.05,0.5").split(",")]
DEFECT_FIXED = [int(v) for v in os.environ.get("DEFECT_FIXED", "0,1,2").split(",")]
ARMS = os.environ.get("ARMS", "full,ex,tensor").split(",")
N_TQ = int(os.environ.get("N_TQ", "32"))
R_CHECK_STATES = int(os.environ.get("R_CHECK_STATES", "64"))
POS_TOL = float(os.environ.get("POS_TOL", "1e-9"))
OPT_TOL = float(os.environ.get("OPT_TOL", "1e-4"))
PILOT = int(os.environ.get("PILOT", "0"))
MICRO = int(os.environ.get("MICRO", "0"))
SKIP_FOM = int(os.environ.get("SKIP_FOM", "0"))
TR_TRAJ = int(os.environ.get("TR_TRAJ", str(N_TEST)))
BOOT = int(os.environ.get("BOOT", "2000"))
GATES_SOFT = int(os.environ.get("GATES_SOFT", "0"))     # smoke only: record gate failures, do not abort
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max", 4: "tol_at_init", 5: "nan_at_init"}

log = b3.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def blk(x):
    return jax.block_until_ready(x)


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)), max=float(np.max(v)),
                min=float(np.min(v)), n=int(v.size))


def hist_of(arr):
    a = np.asarray(arr)
    return {REASON_NAMES[int(r_)]: int(np.sum(a == r_)) for r_ in np.unique(a)}


def git_commit():
    c = os.environ.get("COMMIT")
    if c:
        return c
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL, cwd=HERE).strip()
    except Exception:
        return None


def mem_snapshot(tag, store):
    """device peak (jax memory stats) and host RSS, recorded per phase [A25, A46]."""
    try:
        ms = jax.devices()[0].memory_stats() or {}
        dev_peak = float(ms.get("peak_bytes_in_use", 0)) / 2 ** 30
        dev_now = float(ms.get("bytes_in_use", 0)) / 2 ** 30
        dev_lim = float(ms.get("bytes_limit", 0)) / 2 ** 30
    except Exception:
        dev_peak = dev_now = dev_lim = float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    store[tag] = dict(device_peak_gb=dev_peak, device_now_gb=dev_now, device_limit_gb=dev_lim,
                      host_maxrss_gb=rss, t=time.time())
    log(f"  [mem {tag}] device peak {dev_peak:.1f} GB (now {dev_now:.1f}, limit {dev_lim:.1f}); "
        f"host maxrss {rss:.1f} GB")


# ===========================================================================
# solver pieces with the big arrays threaded as an explicit `aux` argument
# (verbatim sep_b2d_tensor.make_step_aux / make_roll_aux)
# ===========================================================================
def make_step_aux(r_w, K_, stall_rel, tr_delta):
    def rJ_lspg(z, p, nu, aux):
        return (r_w(z, p, nu, aux), jax.jacfwd(r_w)(z, p, nu, aux))

    def rn_fn(z, p, nu, aux):
        return jnp.linalg.norm(r_w(z, p, nu, aux))

    def step(z0, prev_c, nu, tol_abs, budget, aux):
        r0, J0 = rJ_lspg(z0, prev_c, nu, aux)
        rn0 = jnp.linalg.norm(r0)
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where(rn0 <= tol_abs, jnp.int32(4), jnp.int32(0)))
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(1), jnp.int32(0), init_reason)

        def cond(s):
            return (s[9] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, nJ, _, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K_, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            within_trust = jnp.linalg.norm(dz) <= tr_delta
            tiny = finite & (jnp.linalg.norm(dz) <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
            z_new = z + jnp.where(finite & within_trust, dz, 0.0)
            rn_new = rn_fn(z_new, prev_c, nu, aux)
            accept = (finite & within_trust & jnp.isfinite(rn_new) & (rn_new < rn))
            r2, J2 = jax.lax.cond(accept, lambda: rJ_lspg(z_new, prev_c, nu, aux), lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (rn <= tol_abs), 1,
                               jnp.where((accept & (rel_dec < stall_rel)) | tiny, 2,
                                         jnp.where((~accept) & (lam >= 1e12), 3, 0))).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, acc, nJ, jnp.int32(0), reason)

        z, r, J, rn, lam, att, acc, nJ, _, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, nJ, acc, reason, att

    return jax.jit(step, static_argnums=(4,)), jax.jit(rn_fn), jax.jit(rJ_lspg)


def make_roll_aux(step_fn, rn_fn, prev_fn, num_steps, extrap):
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
            z2, rn, nJ, acc, reason, att = step_fn(z_init, prev_c, nu, tol_abs, budget, aux)
            return (z2, z, prev_fn(z2)), (z2, rn, nJ, att, reason)

        _, out = jax.lax.scan(body, (z0, z0, prev_fn(z0)), us)
        return out

    return jax.jit(roll, static_argnums=(3,))


def eager_lm_step(rJ, rn, z0, prev_c, nu, tol_abs, budget, aux, stall_rel, tr_delta):
    """INDEPENDENT eager (host-loop) implementation of the same LM rule, for
    the STEP/ROLL gates (r3: sep_solvers is not importable without the 2D
    FiLM stack)."""
    z = np.asarray(z0, dtype=np.float64)
    r, J = [np.asarray(v) for v in rJ(jnp.asarray(z), prev_c, nu, aux)]
    rn_ = float(np.linalg.norm(r))
    if not np.isfinite(rn_):
        return z, 5, 0
    if rn_ <= tol_abs:
        return z, 4, 0
    lam = 1e-6
    reason, att = 0, 0
    while reason == 0 and att < budget:
        H = J.T @ J
        g = J.T @ r
        D = np.diag(np.diag(H)) + 1e-30 * np.eye(len(z))
        dz = np.linalg.solve(H + lam * D, -g)
        finite = bool(np.all(np.isfinite(dz)))
        within = float(np.linalg.norm(dz)) <= tr_delta
        tiny = finite and float(np.linalg.norm(dz)) <= 1e-12 * (1.0 + float(np.linalg.norm(z)))
        z_new = z + (dz if (finite and within) else 0.0)
        rn_new = float(rn(jnp.asarray(z_new), prev_c, nu, aux))
        accept = finite and within and np.isfinite(rn_new) and rn_new < rn_
        if accept:
            rel_dec = (rn_ - rn_new) / rn_
            z, rn_ = z_new, rn_new
            r, J = [np.asarray(v) for v in rJ(jnp.asarray(z), prev_c, nu, aux)]
            lam = max(lam / 3.0, 1e-12)
            if rn_ <= tol_abs:
                reason = 1
            elif rel_dec < stall_rel or tiny:
                reason = 2
        else:
            lam = min(lam * 10.0, 1e12)
            if tiny:
                reason = 2
            elif lam >= 1e12:
                reason = 3
        att += 1
    return z, reason, att


def bootstrap_lower(pairs, rng, n_boot, q=0.05):
    """HIERARCHICAL bootstrap of the median paired speedup [A59]: resample
    trajectories with replacement, and within each drawn trajectory resample
    its raw ROM and FOM repetition times with replacement, so both the
    trajectory-to-trajectory and the timing-repetition variability enter."""
    A = [np.asarray(p["a_raw_ms"], dtype=np.float64) for p in pairs]
    B = [np.asarray(p["b_raw_ms"], dtype=np.float64) for p in pairs]
    nt = len(pairs)
    meds = []
    for _ in range(n_boot):
        ti = rng.integers(0, nt, nt)
        sp = []
        for i in ti:
            a = A[i][rng.integers(0, len(A[i]), len(A[i]))]
            b = B[i][rng.integers(0, len(B[i]), len(B[i]))]
            sp.append(np.median(b) / np.median(a))
        meds.append(np.median(sp))
    return float(np.quantile(meds, q))


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B3D-TENSOR N={N} K={K} R={R} M={EQ_M} m={EQ_MQ} "
        f"arms={ARMS} train={TRAIN} pilot={PILOT} micro={MICRO} reps={TIME_REPS} burn={BURN} stall={STALL}")
    t_all = time.time()
    n = N
    ni = n - 2
    n_i = ni ** 3
    dx = 1.0 / (n - 1)
    T = b3.NUM_STEPS + 1
    interior = b3.interior_indices_3d(n)
    interior_j = jnp.asarray(interior)
    coords = b3.grid_coords_3d(n)
    coords_int = coords[interior]
    gen_chunk = GEN_CHUNK or (64 if n <= 33 else (16 if n <= 65 else 4))
    mem = {}
    mem_snapshot("start", mem)

    tabs = b3.get_tables(TABLE_DIR, N_TRAIN_TABLE, N_TEST, SEED0, TEST_SEED)
    tt, tr_tab = tabs["test"], tabs["train"]
    train_rows = np.arange(N_TRAIN)
    cohort_rows = np.concatenate([train_rows, VAL_ROWS])
    n_traj = cohort_rows.size

    report = dict(config=dict(
        pde="burgers3d", kind="b3d_tensor", design="B3D-DESIGN.md r3", N=N, k=K, r=R, M=EQ_M,
        m_nnls=EQ_MQ, g_hidden=G_HIDDEN, eq_cand_cap=EQ_CAND_CAP, arms=ARMS, ckpt=CKPT,
        train_in_job=bool(TRAIN), pilot=bool(PILOT), micro=bool(MICRO), n_test=N_TEST, seed=SEED0,
        test_seed=TEST_SEED, n_train_traj=int(N_TRAIN), val_rows=[int(VAL_ROWS[0]), int(VAL_ROWS[-1])],
        table_train=dict(path=tr_tab["path"], sha256=tr_tab["sha256"]),
        table_test=dict(path=tt["path"], sha256=tt["sha256"]) if not PILOT else None,
        step_tol=STEP_TOL, stall=STALL, extrap=EXTRAP, tr_factor=TR_FACTOR, gn_budget=GN_BUDGET,
        ic_enc_budget=IC_ENC_BUDGET, enc_steps=ENC_STEPS, num_steps=b3.NUM_STEPS, dt=b3.DT,
        weak_alpha=b3.WEAK_ALPHA, time_reps=TIME_REPS, burn=BURN, warm=WARM, fom_reps=FOM_REPS,
        pair_reps=PAIR_REPS, newton_tols=NEWTON_TOLS, lin_fracs=LIN_FRACS, defect_fixed=DEFECT_FIXED,
        t_chunk=T_CHUNK, gen_chunk=gen_chunk, feat_chunk=FEAT_CHUNK, pool_cap=POOL_CAP, p_sub=P_SUB,
        max_snaps=MAX_SNAPS, steps=STEPS, lr=LR, pos_tol=POS_TOL, opt_tol=OPT_TOL, dst=DST,
        oracle=dict(starts=ORACLE_STARTS, budget=ORACLE_BUDGET, times=ORACLE_TIMES,
                    val_traj=ORACLE_VAL_TRAJ), x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"), backend=dev.platform,
        gpu=getattr(dev, "device_kind", str(dev)), jax_version=jax.__version__, commit=git_commit(),
        slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME", "local"),
        arm_order="reps outermost; trajectories; arms forward on even reps, reversed on odd "
                  "(AB/BA); accuracy from the last timed rep's fused e2e output"),
        gates={}, data={}, variants={}, memory=mem, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    def gate(name, value, passed, control=None, control_fired=None, **extra):
        d = dict(value=value, passed=bool(passed), **extra)
        if control is not None:
            d["control"] = control
            d["control_fired"] = bool(control_fired)
        report["gates"][name] = d
        ctl = "" if control is None else f"  | control {control:.3e} fired={bool(control_fired)}"
        vs = value if isinstance(value, str) else f"{value:.3e}"
        log(f"  GATE {name}: {vs} {'PASS' if passed else 'FAIL'}{ctl}")
        save()
        if GATES_SOFT:
            report["config"]["gates_soft"] = True
            return
        assert passed, (name, d)
        if control is not None:
            assert control_fired, ("CONTROL DID NOT FIRE", name, d)

    # ---------------- training pool (the same sampling measure at every N) ---
    if n_i <= POOL_CAP:
        pool = np.arange(n_i)
    else:
        pool = np.sort(np.random.default_rng(SEED0 + 11).choice(n_i, POOL_CAP, replace=False))
    coords_pool = coords_int[pool]
    report["config"]["pool_size"] = int(pool.size)

    # ---------------- test truth (NOT in pilot mode) -------------------------
    roll_fom = b3.make_truth_rollout(n, DST)
    if not PILOT:
        n_test = 1 if MICRO else N_TEST
        U_test, worst_te, umin_te, umax_te, fle_te, secs_te = b3.build_truth(
            n, tt, np.arange(n_test), gen_chunk, roll_fom, coords)        # (n_test, T, n_i)
        report["data"]["test"] = dict(n_test=n_test, max_fom_rel_residual=worst_te, min_u=umin_te,
                                      max_u=umax_te, frac_points_le0=fle_te, gen_secs=secs_te,
                                      nu=[float(v) for v in tt["nu"][:n_test]],
                                      B=[int(v) for v in tt["B"][:n_test]])
        assert np.isfinite(worst_te) and worst_te <= 1e-8, worst_te
        assert umin_te >= -POS_TOL, ("TEST DATA NOT NON-NEGATIVE", umin_te)
        log(f"  TEST truth: {n_test} traj, worst res {worst_te:.2e}, min u {umin_te:.2e} [{secs_te:.0f}s]")
    else:
        U_test, n_test = None, 0
    save()

    # ---------------- training truth: streamed regeneration -------------------
    if TRAIN:
        cfg_ms = MAX_SNAPS
    else:
        params, Z_tr, cfg = b3.load_pkl(CKPT)
        assert int(cfg["N"]) == N and int(cfg["k"]) == K and int(cfg["r"]) == R, cfg
        cfg_ms = int(cfg["max_snaps"])
        assert int(cfg["n_train_traj"]) == N_TRAIN, (cfg["n_train_traj"], N_TRAIN)
    n_states = n_traj * T
    rng = np.random.default_rng(SEED0)
    pick = np.sort(rng.choice(n_states, cfg_ms, replace=False)) if n_states > cfg_ms else np.arange(n_states)
    report["config"]["pick_source"] = "sep_burgers.py rule (rng(SEED0).choice over cohort x 51 states)"
    keep_rows = set(int(v) for v in (pick if TRAIN else pick[:max(R_CHECK_STATES, min(len(pick), 4096))]))
    # D4 states: validation trajectories x ORACLE_TIMES, full interior
    val_local = np.arange(N_TRAIN, N_TRAIN + min(ORACLE_VAL_TRAJ, VAL_ROWS.size))
    d4_ids = set(int(li * T + k_) for li in val_local for k_ in ORACLE_TIMES)
    kept, d4_states = {}, {}
    tr_min, tr_max, worst_tr, n_le0, n_pts = np.inf, -np.inf, 0.0, 0, 0
    t0 = time.time()
    for s in range(0, n_traj, gen_chunk):
        e = min(s + gen_chunk, n_traj)
        rr = cohort_rows[s:e]
        U0 = np.stack([b3.blob_ic_3d(n, tr_tab, j, coords)[interior] for j in rr])
        snaps, res = roll_fom(jnp.asarray(U0), jnp.asarray(tr_tab["nu"][rr]))
        worst_tr = max(worst_tr, float(res))
        si = snaps[:, 1:]
        tr_min = min(tr_min, float(jnp.min(si))); tr_max = max(tr_max, float(jnp.max(snaps)))
        n_le0 += int(jnp.sum(si <= 0)); n_pts += int(si.size)
        sn = np.asarray(snaps)
        for b_ in range(e - s):
            for t_ in range(T):
                sid = (s + b_) * T + t_
                if sid in keep_rows:
                    kept[sid] = sn[b_, t_][pool]
                if sid in d4_ids:
                    d4_states[sid] = sn[b_, t_]
        del snaps, sn, si
    assert np.isfinite(worst_tr) and worst_tr <= 1e-8, ("TRAIN FOM residual", worst_tr)
    report["data"]["train"] = dict(n_traj=int(n_traj), n_states=int(n_states), max_fom_rel_residual=worst_tr,
                                   min_u=tr_min, max_u=tr_max, frac_points_le0=n_le0 / n_pts,
                                   secs=time.time() - t0, pick_size=int(pick.size),
                                   n_d4_states=len(d4_states))
    log(f"  TRAIN truth: {n_traj} traj [{time.time()-t0:.0f}s], worst res {worst_tr:.2e}, min u "
        f"{tr_min:.2e}, frac<=0 {n_le0/n_pts:.2e}; kept {len(kept)} pool states, {len(d4_states)} D4 states")
    gate("F5_nonnegativity_train_val", tr_min, tr_min >= -POS_TOL)
    mem_snapshot("after_truth", mem)

    # ---------------- train / load -------------------------------------------
    if TRAIN:
        S_tr = np.stack([kept[int(sid)] for sid in pick])
        S_pod = S_tr[:min(len(S_tr), 4096)]                      # POD comparator snapshots (pool points) [A56]
        del kept
        log(f"  TRAIN in-job: {S_tr.shape[0]} states x {S_tr.shape[1]} pool points, {STEPS} steps, "
            f"lr {LR}, p_sub {P_SUB}")
        params, Z_tr, tinfo = b3.train_autodecoder_3d(
            jax.random.PRNGKey(SEED0), coords_pool, S_tr, K, R, steps=STEPS, lr=LR, p_sub=P_SUB,
            tag=f"b3d N={N} k={K} r={R}", g_hidden=G_HIDDEN)
        cfg = dict(pde="burgers3d", N=N, k=K, r=R, M=EQ_M, m=EQ_MQ, g_hidden=G_HIDDEN, steps=STEPS,
                   lr=LR, p_sub=P_SUB, pool_size=int(pool.size), max_snaps=MAX_SNAPS, seed=SEED0,
                   n_train_traj=int(N_TRAIN), val_rows=[512, 575], table_sha256=tr_tab["sha256"],
                   train=tinfo, x64=True, backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
                   slurm_job=os.environ.get("SLURM_JOB_ID"), commit=git_commit())
        b3.save_pkl(CKPT_OUT, params, Z_tr, cfg)
        report["train"] = tinfo
        report["config"]["ckpt"] = CKPT_OUT
        r_fields = S_tr[:R_CHECK_STATES]
        r_rows = pick[:R_CHECK_STATES]
        del S_tr
    else:
        r_rows = pick[:R_CHECK_STATES]
        r_fields = np.stack([kept[int(sid)] for sid in r_rows])
        S_pod = np.stack([kept[int(sid)] for sid in pick[:min(len(pick), 4096)]]) if len(kept) >= min(len(pick), 4096) else r_fields
        del kept
    report["config"]["ckpt_cfg"] = {k_: v for k_, v in cfg.items()
                                    if isinstance(v, (int, float, str, bool, type(None), list))}
    dec = b3.SeparableDecoder3D(params, K, R)
    h_fn = dec.head_fn()
    Z_tr = np.asarray(Z_tr)
    assert pick.size == len(Z_tr)
    mem_snapshot("after_train", mem)
    save()

    # ---------------- bank, modes, exact-linear matrix -----------------------
    G_all = dec.feat_at(coords, chunk=FEAT_CHUNK)                    # (n^3, R)
    G_int = G_all[interior_j]                                        # (n_i, R)
    kx, ky, kz, Phi_np, lam = b3.test_modes_3d(n, EQ_M)
    if Phi_np.shape[1] != EQ_M:
        log(f"  M extended from {EQ_M} to {Phi_np.shape[1]} to complete the degenerate eigenshell [A70]")
    M_ACT = int(Phi_np.shape[1])
    report["config"]["M_requested"] = EQ_M
    report["config"]["M"] = M_ACT
    Phi_j = jnp.asarray(Phi_np)
    lam_j = jnp.asarray(lam, dtype=F64)
    A_j = Phi_j.T @ G_int                                            # (M, R)
    A_np = np.asarray(A_j)
    G_pool = G_int[jnp.asarray(pool)]
    mem_snapshot("after_bank", mem)

    # D1: bank == meshfree, against an INDEPENDENT numpy evaluation
    zb = Z_tr[0]
    u_bank = np.asarray(G_pool @ h_fn(jnp.asarray(zb)))
    u_np = b3.features_np(params, coords_pool[:4096]) @ b3.head_np(params, zb)
    d1 = float(np.max(np.abs(u_bank[:4096] - u_np)) / (np.max(np.abs(u_np)) + 1e-300))
    u_shift = np.asarray(dec.feat_at(coords_pool[:4096] + 0.5 * dx) @ h_fn(jnp.asarray(zb)))
    d1c = float(np.max(np.abs(u_shift - u_np)) / (np.max(np.abs(u_np)) + 1e-300))
    gate("D1_bank_vs_meshfree_numpy", d1, d1 < 1e-12, control=d1c, control_fired=d1c > 1e-3)

    # D2: lineage (R-lite) with shuffled-code control
    Hr = np.asarray(jax.vmap(h_fn)(jnp.asarray(Z_tr[:R_CHECK_STATES])))
    rec = Hr @ np.asarray(G_pool).T
    r_lite = np.linalg.norm(rec - r_fields, axis=1) / np.linalg.norm(r_fields, axis=1)
    perm = np.random.default_rng(SEED0 + 3).permutation(len(Z_tr))[:R_CHECK_STATES]
    Hs = np.asarray(jax.vmap(h_fn)(jnp.asarray(Z_tr[perm])))
    r_shuf = np.linalg.norm(Hs @ np.asarray(G_pool).T - r_fields, axis=1) / np.linalg.norm(r_fields, axis=1)
    gate("D2_lineage_rlite", float(np.mean(r_lite)), np.mean(r_lite) < 0.2, control=float(np.mean(r_shuf)),
         control_fired=np.mean(r_shuf) > 0.5, max=float(np.max(r_lite)), n=int(r_lite.size))
    del rec, r_fields, Hr, Hs

    # D3: rank of A (+ deterministic control) and M-stability
    sv = np.linalg.svd(A_np, compute_uv=False)
    d3 = float(sv[-1] / sv[0])
    A_bad = A_np.copy(); A_bad[:, R - 1] = A_bad[:, 0]           # duplicated COLUMN: rank loss guaranteed [A68]
    svb = np.linalg.svd(A_bad, compute_uv=False)
    d3c = float(svb[-1] / svb[0])
    # M-stability: energy of the exlin residual in modes M+1..2M vs 1..M at 256 training codes
    _, _, _, Phi2_np, lam2 = b3.test_modes_3d(n, 2 * M_ACT)
    Phi2_j = jnp.asarray(Phi2_np); lam2_j = jnp.asarray(lam2)
    A2_j = Phi2_j.T @ G_int
    mrng = np.random.default_rng(SEED0 + 21)
    ids = mrng.choice(len(Z_tr), min(256, len(Z_tr)), replace=False)
    nu_med = float(np.exp(np.mean(np.log(tr_tab["nu"][cohort_rows]))))
    head_e, tail_e = [], []
    for i in ids:
        z = jnp.asarray(Z_tr[i]); zp = jnp.asarray(Z_tr[mrng.integers(len(Z_tr))])
        h, hp = h_fn(z), h_fn(zp)
        u = G_int @ h
        q2 = Phi2_j.T @ b3.upwind_adv_field_3d(u, n)
        w2 = (1.0 + b3.DT * nu_med * lam2_j) ** (-b3.WEAK_ALPHA)
        r2 = np.asarray(w2 * (A2_j @ (h - hp) + b3.DT * (q2 + nu_med * lam2_j * (A2_j @ h))))
        head_e.append(float(np.sum(r2[:M_ACT] ** 2))); tail_e.append(float(np.sum(r2[M_ACT:] ** 2)))
    mstab = float(np.sum(tail_e) / np.sum(head_e))
    gate("D3_rank_of_A", d3, d3 > 1e-8, control=d3c, control_fired=d3c < 1e-12,
         M_stability_tail_over_head=mstab, M_stability_pass=bool(mstab <= 0.05))
    del Phi2_j, A2_j
    mem_snapshot("after_D3", mem)

    # ---------------- D4: held-out representation oracle (validation) --------
    Gram_int = G_int.T @ G_int
    eps_g = 1e-13 * jnp.trace(Gram_int) / R
    L_int = jnp.linalg.cholesky(Gram_int + eps_g * jnp.eye(R, dtype=F64))
    Gram_pool = G_pool.T @ G_pool
    L_pool = jnp.linalg.cholesky(Gram_pool + 1e-13 * jnp.trace(Gram_pool) / R * jnp.eye(R, dtype=F64))
    lm_full = b3.make_lm(lambda z: None, K, ORACLE_BUDGET)  # placeholder (rebuilt per target below)

    def make_oracle(Gb, Lc, budget):
        def solve(u, z0s):
            b = Gb.T @ u
            y = jax.scipy.linalg.solve_triangular(Lc, b, lower=True)
            c2 = jnp.maximum(u @ u - y @ y, 0.0)

            def f(z):
                return Lc.T @ h_fn(z) - y
            lm = b3.make_lm(f, K, budget)

            def one(z0):
                z, rn, rn0, nJ, att, reason = lm(z0)
                r_ = f(z); J_ = jax.jacfwd(f)(z)
                opt = jnp.linalg.norm(J_.T @ r_) / (jnp.linalg.norm(J_) * jnp.linalg.norm(r_) + 1e-300)
                return z, jnp.sqrt(rn * rn + c2), opt, att
            zs, rns, opts, atts = jax.vmap(one)(z0s)
            i = jnp.argmin(rns)
            return zs[i], rns[i] / jnp.linalg.norm(u), opts[i], atts[i]
        return jax.jit(solve)

    orc_full = make_oracle(G_int, L_int, ORACLE_BUDGET)
    orc_full2 = make_oracle(G_int, L_int, 2 * ORACLE_BUDGET)
    orc_pool = make_oracle(G_pool, L_pool, ORACLE_BUDGET)
    z0s = jnp.asarray(np.concatenate([np.zeros((1, K)), Z_tr[np.random.default_rng(SEED0 + 5).choice(
        len(Z_tr), ORACLE_STARTS - 1, replace=False)]]))
    d4_rows = []
    d4_keys = sorted(d4_states.keys())
    for sid in d4_keys:
        uf = jnp.asarray(d4_states[sid])
        z, e_full, opt, att = orc_full(uf, z0s)
        zp_, e_poolfit, _, _ = orc_pool(uf[jnp.asarray(pool)], z0s)
        e_pool_on_full = float(jnp.linalg.norm(G_int @ h_fn(zp_) - uf) / jnp.linalg.norm(uf))
        d4_rows.append(dict(sid=int(sid), k=int(sid % T), oracle_full=float(e_full), optimality=float(opt),
                            attempts=int(att), oracle_pool_fit=float(e_poolfit),
                            pool_fit_on_full=e_pool_on_full))
    ef = np.array([r_["oracle_full"] for r_ in d4_rows])
    kk = np.array([r_["k"] for r_ in d4_rows])
    ratio = np.array([r_["pool_fit_on_full"] / max(r_["oracle_pool_fit"], 1e-300) for r_ in d4_rows])
    # budget doubling on 4 states
    dbl = []
    for sid in d4_keys[:4]:
        uf = jnp.asarray(d4_states[sid])
        _, e2, _, _ = orc_full2(uf, z0s)
        e1 = [r_["oracle_full"] for r_ in d4_rows if r_["sid"] == sid][0]
        dbl.append(abs(float(e2) - e1) / max(e1, 1e-300))
    # POD-K comparator [A56]: POD of the TRAINING snapshots on the pool (S_pod, kept before training),
    # held-out validation states restricted to the pool, projected on the leading K modes; the oracle
    # restricted to the pool is compared with it (both on the pool, so N=129 is well defined)
    S_val_pool = np.stack([d4_states[s_][pool] for s_ in d4_keys])
    Xs = S_pod - S_pod.mean(0, keepdims=True) if False else S_pod          # uncentred, as the 2D cells
    Gm = Xs @ Xs.T
    ev, V = np.linalg.eigh(Gm)
    order = np.argsort(ev)[::-1][:K]
    Upod = (Xs.T @ V[:, order]) / np.sqrt(np.maximum(ev[order], 1e-300))[None, :]      # (P, K) orthonormal
    proj = S_val_pool - (S_val_pool @ Upod) @ Upod.T
    pod_floor = np.linalg.norm(proj, axis=1) / np.linalg.norm(S_val_pool, axis=1)
    # control: shuffled bank rows
    perm_rows = np.random.default_rng(SEED0 + 9).permutation(n_i)
    G_shuf = G_int[jnp.asarray(perm_rows)]
    L_shuf = jnp.linalg.cholesky(G_shuf.T @ G_shuf + eps_g * jnp.eye(R, dtype=F64))
    orc_shuf = make_oracle(G_shuf, L_shuf, ORACLE_BUDGET)
    e_sh = [float(orc_shuf(jnp.asarray(d4_states[s]), z0s)[1]) for s in d4_keys[:8]]
    ep = np.array([r_["oracle_pool_fit"] for r_ in d4_rows])
    d4 = dict(mean=float(np.mean(ef)), worst=float(np.max(ef)), mean_k_gt0=float(np.mean(ef[kk > 0])),
              worst_k_gt0=float(np.max(ef[kk > 0])), n=int(ef.size), optimality_max=float(max(r_["optimality"] for r_ in d4_rows)),
              budget_doubling_rel_change_max=float(np.max(dbl)), pool_to_full_ratio_max=float(np.max(ratio)),
              pool_to_full_ratio_median=float(np.median(ratio)), pod_K_floor_mean=float(np.mean(pod_floor)),
              oracle_pool_mean=float(np.mean(ep)), oracle_over_podK=float(np.mean(ep) / max(np.mean(pod_floor), 1e-300)),
              rows=d4_rows, control_shuffled_bank_mean=float(np.mean(e_sh)))
    passed = (d4["mean"] <= 5e-2 and d4["worst"] <= 1.5e-1 and d4["pool_to_full_ratio_max"] <= 1.5
              and d4["budget_doubling_rel_change_max"] < 1e-2 and d4["optimality_max"] <= 1e-6
              and d4["oracle_over_podK"] <= 0.5)
    gate("D4_heldout_oracle_validation", d4["mean"], passed, control=d4["control_shuffled_bank_mean"],
         control_fired=d4["control_shuffled_bank_mean"] > 0.5, **{k_: v for k_, v in d4.items() if k_ != "rows"})
    report["D4_rows"] = d4_rows
    del d4_states
    mem_snapshot("after_D4", mem)
    if PILOT:
        report["complete"] = True
        report["secs_total"] = time.time() - t_all
        save()
        log(f"PILOT DONE -> {OUT} [{time.time()-t_all:.0f}s]")
        return

    # ---------------- test-state oracle (for A1; recorded) --------------------
    te_or = []
    for i in range(n_test):
        for k_ in ORACLE_TIMES:
            _, e_, _, _ = orc_full(jnp.asarray(U_test[i, k_]), z0s)
            te_or.append(dict(traj=i, k=k_, oracle=float(e_)))
    report["test_oracle"] = dict(mean=float(np.mean([r_["oracle"] for r_ in te_or])), rows=te_or)
    save()

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    TRD = float(TR_FACTOR * train_radius) if TR_FACTOR > 0 else float("inf")
    report["config"]["trust_delta"] = TRD
    u_full_int = jax.jit(lambda z: G_int @ h_fn(z))
    adv_full = jax.jit(lambda uf: b3.upwind_adv_field_3d(uf, n))

    # ---------------- NNLS advection-only node set (ex arm) --------------------
    eq_rng = np.random.default_rng(SEED0)
    eq_pick = np.sort(eq_rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    cand_pos = b3.candidate_pool(n_i, EQ_CAND_CAP)
    keep, wq_np, eq_info = b3.eq_fit_adv_3d(u_full_int, adv_full, Phi_np, cand_pos, Z_tr[eq_pick], EQ_MQ,
                                            f"exlin3d N={N} k={K} M={EQ_M} m={EQ_MQ}")
    idx_int = cand_pos[keep]                                          # interior indices of the nodes
    idx = interior[idx_int]                                           # full-grid flat indices
    m = idx.size
    w_q = jnp.asarray(wq_np, dtype=F64)
    Phi_q = jnp.asarray(Phi_np[idx_int]) * w_q[:, None]
    # 7-point stencil in FULL-grid flat indices: [c, x+, x-, y+, y-, z+, z-]
    st = np.stack([idx, idx + n * n, idx - n * n, idx + n, idx - n, idx + 1, idx - 1], axis=1)
    G_st = dec.feat_at(coords[st.reshape(-1)], chunk=FEAT_CHUNK).reshape(m, 7, R)
    report["eq"] = dict(eq_info, m=int(m))
    save()

    # ---------------- residual closures: r_w(z, prev_m, nu, aux) -------------
    def wt_of(nu):
        return (1.0 + b3.DT * nu * lam_j) ** (-b3.WEAK_ALPHA)

    def lin_of(z, prev_m, nu):
        Ah = A_j @ h_fn(z)
        return (Ah - prev_m) + b3.DT * nu * lam_j * Ah

    def prev_of(z):
        return A_j @ h_fn(z)

    def stencil_adv(us, mode="upwind"):
        c, xp, xm, yp, ym, zp, zm = [us[:, i] for i in range(7)]
        if mode == "central":
            return c * ((xp - xm) + (yp - ym) + (zp - zm)) / (2.0 * dx)
        pos = c > 0.0
        ux = jnp.where(pos, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(pos, (c - ym) / dx, (yp - c) / dx)
        uz = jnp.where(pos, (c - zm) / dx, (zp - c) / dx)
        return c * (ux + uy + uz)

    def sampled_adv(G7, Pq, mode="upwind"):
        def adv(z):
            us = jnp.einsum("msr,r->ms", G7, h_fn(z))
            return Pq.T @ stencil_adv(us, mode)
        return adv

    def mk_parts(adv_fn, uses_aux=False):
        def parts(z, prev_m, nu, aux):
            w_ = wt_of(nu)
            lin = lin_of(z, prev_m, nu)
            adv = adv_fn(z, aux) if uses_aux else adv_fn(z)
            return w_ * (lin + b3.DT * adv), w_ * lin, w_ * b3.DT * adv

        def r_w(z, prev_m, nu, aux):
            return parts(z, prev_m, nu, aux)[0]
        return r_w, parts

    r_ex, parts_ex = mk_parts(sampled_adv(G_st, Phi_q))

    def adv_full_fn(z, aux):
        Gb, Ph = aux
        return Ph.T @ b3.upwind_adv_field_3d(Gb @ h_fn(z), n)
    r_full, parts_full = mk_parts(adv_full_fn, uses_aux=True)
    aux_full = (G_int, Phi_j)

    # ---------------- gates L / A / FOMR (with controls) ----------------------
    L_sp = b3.assemble_L_3d(n)
    Lnorm = float(np.abs(L_sp).sum(1).max())
    grng = np.random.default_rng(SEED0 + 50)
    nu_t = float(np.median(tt["nu"][:n_test]))
    parts_ex_j = jax.jit(parts_ex)
    parts_full_j = jax.jit(parts_full)
    prev_j = jax.jit(prev_of)
    gL, gLc, gA, gAc, gF, gFc = [], [], [], [], [], []
    adv_inc_j = jax.jit(lambda z: wt_of(nu_t) * b3.DT * (Phi_q.T @ stencil_adv(jnp.einsum("msr,r->ms", G_st, h_fn(z)))))
    adv_cen_j = jax.jit(lambda z: wt_of(nu_t) * b3.DT * (Phi_q.T @ stencil_adv(jnp.einsum("msr,r->ms", G_st, h_fn(z)), "central")))
    for _ in range(5):
        zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))] + 0.05 * grng.standard_normal(K))
        zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
        pm = prev_j(zp)
        _, lin_x, adv_x = parts_ex_j(zt, pm, nu_t, ())
        rf, lin_f, adv_f = parts_full_j(zt, pm, nu_t, aux_full)
        # L vs the scipy-assembled linear weak terms
        u = np.asarray(u_full_int(zt)); up = np.asarray(u_full_int(zp))
        wt = np.asarray(wt_of(nu_t))
        lin_as = wt * (Phi_np.T @ (u - up + b3.DT * nu_t * (-(L_sp @ u))))
        opn = 1.0 + b3.DT * nu_t * Lnorm
        gL.append(float(np.linalg.norm(np.asarray(lin_x) - lin_as) / (opn * np.linalg.norm(u) * np.max(wt) + np.linalg.norm(lin_as) + 1e-300)))
        lin_flip = wt * (Phi_np.T @ (u - up - b3.DT * nu_t * (-(L_sp @ u))))     # diffusion sign flipped
        gLc.append(float(np.linalg.norm(np.asarray(lin_x) - lin_flip) / (opn * np.linalg.norm(u) * np.max(wt) + np.linalg.norm(lin_as) + 1e-300)))
        adv_i = adv_inc_j(zt)
        gA.append(float(jnp.max(jnp.abs(adv_x - adv_i)) / (jnp.max(jnp.abs(adv_i)) + 1e-300)))
        gAc.append(float(jnp.max(jnp.abs(adv_cen_j(zt) - adv_i)) / (jnp.max(jnp.abs(adv_i)) + 1e-300)))
        Rf = np.asarray(b3.fom_residual_int(jnp.asarray(u), jnp.asarray(up), nu_t, n))
        r_direct = wt * (Phi_np.T @ Rf)
        gF.append(rel(np.asarray(rf), r_direct))
        gFc.append(rel(np.asarray(rf), Phi_np.T @ Rf))
    gate("L_exlin_linear_vs_assembled", float(np.max(gL)), np.max(gL) <= 1e-12, control=float(np.min(gLc)),
         control_fired=np.min(gLc) > 1e-5, control_note="diffusion sign flipped in the assembled reference")
    gate("A_exlin_advection_direct", float(np.max(gA)), np.max(gA) <= 1e-12, control=float(np.min(gAc)),
         control_fired=np.min(gAc) > 1e-6)
    gate("FOMR_fullgrid_weak_vs_fom_residual", float(np.max(gF)), np.max(gF) <= 1e-10, control=float(np.min(gFc)),
         control_fired=np.min(gFc) > 1e-3)

    # ---------------- tensor build + gates TB / TA / T0 --------------------------
    H_all = np.asarray(jnp.concatenate([jax.vmap(h_fn)(jnp.asarray(Z_tr[s:s + 2048])) for s in range(0, len(Z_tr), 2048)]))
    t0 = time.time()
    T1 = tc.build_T(Phi_j, G_int, n, chunk=T_CHUNK, reverse=False)
    t_build = time.time() - t0
    T2 = tc.build_T(Phi_j, G_int, n, chunk=max(256, T_CHUNK // 3 + 7), reverse=True)
    n_chunks = int(np.ceil(n_i / T_CHUNK))
    gTB = float(np.max(np.abs(T1 - T2)) / np.max(np.abs(T1)))
    # control: the last chunk dropped
    T_drop = tc.build_T(Phi_j[:n_i - (n_i % T_CHUNK or T_CHUNK)], G_int[:n_i - (n_i % T_CHUNK or T_CHUNK)], n, chunk=T_CHUNK) \
        if False else None
    Tc = T1 - np.asarray(tc._chunk_T(Phi_j[-T_CHUNK:], G_int[-T_CHUNK:], tc.backward_diff_bank_3d(G_int, n)[-T_CHUNK:]))
    gTBc = float(np.max(np.abs(T1 - Tc)) / np.max(np.abs(T1)))
    Q = tc.symmetrize(T1)
    del T2, Tc
    report["tensor"] = dict(shape=list(Q.shape), build_secs=t_build, bytes=int(Q.nbytes), n_chunks=n_chunks,
                            max_abs=float(np.max(np.abs(Q))),
                            T_asym_rel=float(np.max(np.abs(T1 - T1.swapaxes(1, 2))) / np.max(np.abs(T1))))
    gate("TB_build_order", gTB, gTB <= n_chunks * 1e-15, control=gTBc, control_fired=gTBc > 1e-6,
         threshold=n_chunks * 1e-15, n_chunks=n_chunks)
    Qj = jnp.asarray(Q)
    Tj = jnp.asarray(T1)
    DG_j = tc.backward_diff_bank_3d(G_int, n)

    @jax.jit
    def ta_chunk(Hc, Gb, Db, Ph, Qq, Tq):
        U = Hc @ Gb.T
        q_alg = (U * (Hc @ Db.T)) @ Ph
        q_or = jax.vmap(lambda u_: b3.upwind_adv_field_3d(u_, n))(U) @ Ph
        q_T = 0.5 * jnp.einsum("ijk,sj,sk->si", Qq, Hc, Hc)
        q_Tc = 0.5 * jnp.einsum("ijk,sj,sk->si", Tq, Hc, Hc)
        return q_alg, q_or, q_T, q_Tc, jnp.min(U, axis=1), jnp.sum(U <= 0, axis=1)
    qa, qo, qt, qtc, mn, nle = [], [], [], [], [], []
    for s in range(0, len(H_all), TA_CHUNK):
        o = ta_chunk(jnp.asarray(H_all[s:s + TA_CHUNK]), G_int, DG_j, Phi_j, Qj, Tj)
        for L_, v in zip((qa, qo, qt, qtc, mn, nle), o):
            L_.append(np.asarray(v))
    q_alg, q_or, q_T, q_Tc = [np.concatenate(v) for v in (qa, qo, qt, qtc)]
    minu, n_le = np.concatenate(mn), np.concatenate(nle)
    nq = np.linalg.norm(q_or, axis=1) + 1e-300
    gTA = float(np.max(np.linalg.norm(q_T - q_alg, axis=1) / nq))
    gTAc = float(np.min(np.linalg.norm(q_Tc - q_alg, axis=1) / nq))
    gate("TA_algebraic_identity", gTA, gTA < 1e-13, control=gTAc, control_fired=gTAc > 1e-1,
         n_states=int(len(H_all)))
    pos_states = minu > 0
    mis = np.linalg.norm(q_T - q_or, axis=1) / nq
    report["gates"]["T0_decoded"] = dict(
        max_rel_on_all_positive_states=float(np.max(mis[pos_states])) if np.any(pos_states) else None,
        n_all_positive_states=int(np.sum(pos_states)), mismatch_all_states=stats(mis),
        frac_points_u_le0=float(n_le.sum() / (len(H_all) * n_i)), frac_states_all_positive=float(np.mean(pos_states)),
        min_u=float(minu.min()), recorded=True)
    log(f"  T0-decoded (recorded): {int(np.sum(pos_states))} all-positive states; mismatch median "
        f"{np.median(mis):.2e} max {np.max(mis):.2e}; frac points u<=0 {n_le.sum()/(len(H_all)*n_i):.3%}")
    # T0-scope: on truth snapshots (FOM scope check, not a tensor precondition)
    ut = jnp.asarray(U_test[0, 25])
    q_up = np.asarray(Phi_j.T @ b3.upwind_adv_field_3d(ut, n)); q_bw = np.asarray(Phi_j.T @ b3.backward_adv_field_3d(ut, n))
    t0s = rel(q_up, q_bw)
    us_ = jnp.asarray(U_test[0, 0] - np.roll(U_test[0, 0].reshape(ni, ni, ni), 3, axis=0).reshape(-1))
    t0c = rel(np.asarray(Phi_j.T @ b3.upwind_adv_field_3d(us_, n)), np.asarray(Phi_j.T @ b3.backward_adv_field_3d(us_, n)))
    gate("T0_scope_truth_upwind_eq_backward", t0s, t0s < 1e-13, control=t0c, control_fired=t0c > 1e-3,
         note="FOM scope check only [A45]")
    del q_alg, q_or, q_T, q_Tc, DG_j, Tj
    mem_snapshot("after_tensor", mem)
    save()

    Qm = jnp.asarray(Q.reshape(M_ACT * R, R))

    def adv_tensor(z):
        h = h_fn(z)
        return 0.5 * ((Qm @ h).reshape(M_ACT, R) @ h)
    r_T, parts_T = mk_parts(adv_tensor)

    # ---------------- TQ: tensor vs oracle at 32 latent states (recorded) ----
    rJ_full = jax.jit(lambda z, p, nu, aux: (r_full(z, p, nu, aux), jax.jacfwd(r_full)(z, p, nu, aux)))
    rJ_T = jax.jit(lambda z, p, nu: (r_T(z, p, nu, ()), jax.jacfwd(r_T)(z, p, nu, ())))
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
        tq.append(dict(perturbed=bool(pert), nu=nu, r_rel=rel(rt, ro), J_rel=rel(Jt, Jo),
                       g_scaled=float(np.linalg.norm(Jt.T @ rt - Jo.T @ ro) / (np.linalg.norm(Jo) * np.linalg.norm(ro) + 1e-300)),
                       min_u=float(uu.min()), n_neg=int(np.sum(uu <= 0))))
    report["gates"]["TQ"] = dict(r_rel=stats([t_["r_rel"] for t_ in tq]), J_rel=stats([t_["J_rel"] for t_ in tq]),
                                 g_scaled_max=float(max(t_["g_scaled"] for t_ in tq)),
                                 n_states_with_neg=int(sum(1 for t_ in tq if t_["n_neg"] > 0)), rows=tq, recorded=True)
    g_ = report["gates"]["TQ"]
    log(f"  TQ (recorded): r rel median {g_['r_rel']['median']:.2e} max {g_['r_rel']['max']:.2e}; J rel max "
        f"{g_['J_rel']['max']:.2e}; states with u<=0 {g_['n_states_with_neg']}/{N_TQ}")
    save()

    # ---------------- IC encoder + Gram-space IC fit ---------------------------
    t0_rows = np.nonzero((pick % T) == 0)[0]
    U0_tr = np.stack([b3.blob_ic_3d(n, tr_tab, cohort_rows[i], coords)[idx] for i in (pick[t0_rows] // T)])
    enc_params, enc_apply, enc_info = fit_encoder(jax.random.PRNGKey(SEED0 + 7), U0_tr, Z_tr[t0_rows], ENC_STEPS)
    report["encoder"] = enc_info
    idx_j = jnp.asarray(idx)
    Gram_all = G_all.T @ G_all
    L_all = jnp.linalg.cholesky(Gram_all + 1e-13 * jnp.trace(Gram_all) / R * jnp.eye(R, dtype=F64))

    def ic_enc_gram(Gb, u0):
        z0 = enc_apply(enc_params, u0[idx_j])
        b = Gb.T @ u0
        y = jax.scipy.linalg.solve_triangular(L_all, b, lower=True)
        c2 = jnp.maximum(u0 @ u0 - y @ y, 0.0)

        def f(z):
            return L_all.T @ h_fn(z) - y
        lm = b3.make_lm(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0)
        return z, jnp.sqrt(rn * rn + c2), nJ
    ic_gram_j = jax.jit(ic_enc_gram)
    decode_j = jax.jit(lambda Gb, Zf: jax.vmap(h_fn)(Zf) @ Gb.T)
    tol_scale = float(np.sqrt(n_i))

    # ---------------- arms ------------------------------------------------------
    arm_defs = {"full": (r_full, aux_full), "ex": (r_ex, ()), "tensor": (r_T, ())}
    arms = [a_ for a_ in ARMS if a_ in arm_defs]
    report["config"]["arms_run"] = arms

    def make_arm(r_w, aux):
        step, rn_j, rJ_j = make_step_aux(r_w, K, STALL, TRD)
        roll = make_roll_aux(step, rn_j, prev_of, b3.NUM_STEPS, EXTRAP)

        def e2e(Gb, u0, nu, aux_):
            z0, ic_rn, ic_nJ = ic_enc_gram(Gb, u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((b3.NUM_STEPS,), STEP_TOL * u_scale * tol_scale, dtype=F64)
            Z, rns, nJs, atts, reasons = roll(z0, nu, us, GN_BUDGET, aux_)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)
            return (decode_j(Gb, Zfull), z0, Z, rns, nJs, atts, reasons, ic_rn, ic_nJ)

        # per-step first-order optimality at the accepted latents (P1)
        def optimality(Zfull, nu, aux_):
            def one(z, zp):
                r_, J_ = rJ_j(z, prev_of(zp), nu, aux_)
                return jnp.linalg.norm(J_.T @ r_) / (jnp.linalg.norm(J_) * jnp.linalg.norm(r_) + 1e-300)
            return jax.vmap(one)(Zfull[1:], Zfull[:-1])
        return dict(step=step, rn=rn_j, rJ=rJ_j, roll=roll, e2e=jax.jit(e2e), opt=jax.jit(optimality), aux=aux)

    A_ = {a_: make_arm(*arm_defs[a_]) for a_ in arms}
    arm_ex = A_["ex"] if "ex" in A_ else make_arm(*arm_defs["ex"])

    # STEP / ROLL: device vs the independent eager LM on a preselected witness (>= 2 accepted steps)
    u00 = jnp.asarray(np.zeros(n ** 3)); u00 = u00.at[interior_j].set(jnp.asarray(U_test[0, 0]))
    tolg = STEP_TOL * float(np.sqrt(np.mean(U_test[0, 0] ** 2))) * tol_scale
    witness = None
    wrng = np.random.default_rng(SEED0 + 77)
    for _ in range(50):
        zt = Z_tr[wrng.integers(len(Z_tr))] + 0.1 * wrng.standard_normal(K)
        zp = Z_tr[wrng.integers(len(Z_tr))]
        pc = prev_j(jnp.asarray(zp))
        out = arm_ex["step"](jnp.asarray(zt), pc, nu_t, tolg, GN_BUDGET, ())
        if int(out[3]) >= 2:
            witness = (zt, zp, pc, out)
            break
    assert witness is not None, "no STEP witness with >= 2 accepted LM steps found"
    zt, zp, pc, out_dev = witness
    z_e, reason_e, att_e = eager_lm_step(arm_ex["rJ"], arm_ex["rn"], zt, pc, nu_t, tolg, GN_BUDGET, (), STALL, TRD)
    zn = 1.0 + float(np.linalg.norm(z_e))
    sdev = float(np.max(np.abs(np.asarray(out_dev[0]) - z_e))) / zn
    fdev = rel(np.asarray(u_full_int(jnp.asarray(out_dev[0]))), np.asarray(u_full_int(jnp.asarray(z_e))))
    z_c, _, _ = eager_lm_step(arm_ex["rJ"], arm_ex["rn"], zt, pc, nu_t, tolg, GN_BUDGET, (), 1e-1, TRD)
    sdevc = float(np.max(np.abs(np.asarray(out_dev[0]) - z_c))) / zn
    gate("STEP_device_vs_eager", max(sdev, fdev), max(sdev, fdev) <= 1e-10, control=sdevc, control_fired=sdevc > 1e-10,
         latent_rel=sdev, field_rel=fdev, witness_accepted=int(out_dev[3]), reasons=[int(out_dev[4]), reason_e],
         note="normalised latent and decoded-field discrepancy; deviation from the 2D bit-identity gate (r3/r4)")
    z0g, _, _ = ic_gram_j(G_all, u00)
    us0 = jnp.full((b3.NUM_STEPS,), tolg, dtype=F64)
    Zd, _, _, _, _ = arm_ex["roll"](z0g, float(tt["nu"][0]), us0, GN_BUDGET, ())
    z_prev2, z_cur = np.asarray(z0g), np.asarray(z0g)
    Zh = []
    for t_ in range(b3.NUM_STEPS):
        pc = prev_j(jnp.asarray(z_cur))
        z_ex = z_cur + EXTRAP * (z_cur - z_prev2)
        ra = float(arm_ex["rn"](jnp.asarray(z_cur), pc, float(tt["nu"][0]), ()))
        rb = float(arm_ex["rn"](jnp.asarray(z_ex), pc, float(tt["nu"][0]), ()))
        z_init = z_ex if (np.isfinite(rb) and rb < ra) else z_cur
        z_new, _, _ = eager_lm_step(arm_ex["rJ"], arm_ex["rn"], z_init, pc, float(tt["nu"][0]), tolg, GN_BUDGET, (), STALL, TRD)
        Zh.append(z_new); z_prev2, z_cur = z_cur, z_new
    Zh = np.stack(Zh)
    rdev = float(np.max(np.abs(np.asarray(Zd) - Zh)) / (1.0 + np.max(np.linalg.norm(Zh, axis=1))))
    rfd = float(np.max([rel(np.asarray(u_full_int(jnp.asarray(np.asarray(Zd)[t_]))), np.asarray(u_full_int(jnp.asarray(Zh[t_])))) for t_ in range(0, b3.NUM_STEPS, 7)]))
    gate("ROLL_device_vs_eager", max(rdev, rfd), max(rdev, rfd) <= 1e-8, latent_rel=rdev, field_rel=rfd,
         note="eager host loop of the same rule, traj 0; normalised latent and decoded-field discrepancy")
    mem_snapshot("after_gates", mem)

    if MICRO:
        # M1 micro-pilot [A60]: everything above ran at the REAL shapes (streamed truth, a trainer with the
        # real snapshot count for STEPS steps, bank, D4 multistart oracle batch, NNLS fit, tensor build,
        # STEP/ROLL); here one e2e per arm and one full 50-step rollout of each classical arm, then stop.
        for a_ in arms:
            E = blk(A_[a_]["e2e"](G_all, u00, float(tt["nu"][0]), A_[a_]["aux"]))
            report["variants"][a_] = dict(err_first=rel(np.asarray(E[0])[-1][interior], U_test[0, -1]))
        mem_snapshot("after_e2e", mem)
        newt = b3.make_newton_tol_rollout(n, DST); dfc = b3.make_defect_tol_rollout(n, DST)
        t0 = time.time(); blk(newt(jnp.asarray(U_test[0, 0]), float(tt["nu"][0]), 1e-3, 5e-5)); report["micro_newton_s"] = time.time() - t0
        t0 = time.time(); blk(dfc(jnp.asarray(U_test[0, 0]), float(tt["nu"][0]), 1e-3, 60)); report["micro_defect_s"] = time.time() - t0
        mem_snapshot("after_fom", mem)
        report["complete"] = True; report["secs_total"] = time.time() - t_all; save()
        log(f"MICRO DONE -> {OUT} [{time.time()-t_all:.0f}s]")
        return

    # ---------------- interleaved timing: reps > trajectories > arms ----------
    u0_dev, us_dev = [], []
    for i in range(n_test):
        u0f = np.zeros(n ** 3); u0f[interior] = U_test[i, 0]
        u0_dev.append(jnp.asarray(u0f))
        u_scale = float(np.sqrt(np.mean(U_test[i, 0] ** 2)))
        us_dev.append(jnp.full((b3.NUM_STEPS,), STEP_TOL * u_scale * tol_scale, dtype=F64))
    times = {a_: [dict(ic=[], roll=[], dec=[], e2e=[]) for _ in range(n_test)] for a_ in arms}
    last = {a_: [None] * n_test for a_ in arms}
    Z_reps = {a_: [[] for _ in range(n_test)] for a_ in arms}

    def one(arm, i):
        ar = A_[arm]
        nu = float(tt["nu"][i])
        t0 = time.perf_counter()
        z0, ic_rn, ic_nJ = blk(ic_gram_j(G_all, u0_dev[i]))
        t1 = time.perf_counter()
        Rl = blk(ar["roll"](z0, nu, us_dev[i], GN_BUDGET, ar["aux"]))
        t2 = time.perf_counter()
        F = blk(decode_j(G_all, jnp.concatenate([z0[None], Rl[0]], axis=0)))
        t3 = time.perf_counter()
        E = blk(ar["e2e"](G_all, u0_dev[i], nu, ar["aux"]))
        t4 = time.perf_counter()
        return dict(ic=t1 - t0, roll=t2 - t1, dec=t3 - t2, e2e=t4 - t3), (z0, Rl, F, E)

    b3.burn_in(1.5)
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
        log(f"  rep {rep_} ({'timed' if rep_ >= BURN else 'burn'}, order {order}) done [{time.time()-t_all:.0f}s]")
    mem_snapshot("after_arms", mem)
    Fields = {}
    for arm in arms:
        rows, agg_reasons = [], {}
        for i in range(n_test):
            z0, Rl, F, E = last[arm][i]
            Fh = np.asarray(E[0])[:, interior]                       # (T, n_i)
            Fields[(arm, i)] = Fh
            pt = np.linalg.norm(Fh - U_test[i], axis=1) / np.linalg.norm(U_test[i], axis=1)
            Zs = np.asarray(E[2]); Zfull = np.concatenate([np.asarray(E[1])[None], Zs])
            reasons = np.asarray(E[6]); hr = hist_of(reasons)
            for k_, v_ in hr.items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v_
            opt = np.asarray(A_[arm]["opt"](jnp.asarray(Zfull), float(tt["nu"][i]), A_[arm]["aux"]))
            tm = times[arm][i]
            rows.append(dict(
                traj=i, nu=float(tt["nu"][i]), traj_rel=float(np.mean(pt)), per_time=[float(v) for v in pt],
                per_time_max=float(np.max(pt)), ic_rel=float(E[7]) / float(np.linalg.norm(U_test[i, 0])), ic_jac=int(E[8]),
                jac_total=int(np.sum(np.asarray(E[4]))), attempts_total=int(np.sum(np.asarray(E[5]))),
                attempts_per_step=[int(v) for v in np.asarray(E[5])], stop_reasons=hr,
                rn_final=[float(v) for v in np.asarray(E[3])], optimality=[float(v) for v in opt],
                optimality_max=float(np.max(opt)), censored_steps=int(np.sum(~(opt <= OPT_TOL))),
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                split_vs_fused_latent_dev=float(np.max(np.abs(np.asarray(Rl[0]) - Zs))),
                timed_reps_latent_dev_max=float(max(np.max(np.abs(Zk - Zs)) for Zk in Z_reps[arm][i])),
                decoded_min_u=float(Fh.min()), decoded_frac_states_with_u_le0=float(np.mean(Fh.min(axis=1) <= 0)),
                decoded_frac_points_le0=float(np.mean(Fh <= 0)),
                ic_ms=float(np.median(tm["ic"])) * 1e3, roll_ms=float(np.median(tm["roll"])) * 1e3,
                dec_ms=float(np.median(tm["dec"])) * 1e3, e2e_ms=float(np.median(tm["e2e"])) * 1e3,
                raw_s={k_: [float(v) for v in tm[k_]] for k_ in tm}))
        v = dict(name=arm, n_test=n_test, err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
                 err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
                 ic_rel_mean=float(np.mean([r_["ic_rel"] for r_ in rows])),
                 jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
                 attempts_total_mean=float(np.mean([r_["attempts_total"] for r_ in rows])), stop_reasons=agg_reasons,
                 censored_steps_total=int(sum(r_["censored_steps"] for r_ in rows)),
                 optimality_max=float(max(r_["optimality_max"] for r_ in rows)),
                 n_blowups=int(sum(r_["n_finite_steps"] < b3.NUM_STEPS for r_ in rows)),
                 decoded_min_u=float(min(r_["decoded_min_u"] for r_ in rows)),
                 decoded_frac_states_with_u_le0=float(np.mean([r_["decoded_frac_states_with_u_le0"] for r_ in rows])),
                 decoded_frac_points_le0=float(np.mean([r_["decoded_frac_points_le0"] for r_ in rows])), per_traj=rows)
        for k_ in ("ic", "roll", "dec", "e2e"):
            v[k_ + "_ms_median"] = float(np.median([x * 1e3 for r_ in rows for x in r_["raw_s"][k_]]))
        report["variants"][arm] = v
        log(f"   ARM {arm:7s} err {v['err_traj_rel_mean']:.6e} (max {v['err_traj_rel_max']:.3e})  e2e "
            f"{v['e2e_ms_median']:.2f} ms = ic {v['ic_ms_median']:.2f} + solve {v['roll_ms_median']:.2f} + dec "
            f"{v['dec_ms_median']:.2f}; attempts {v['attempts_total_mean']:.0f}; {v['stop_reasons']}; censored "
            f"{v['censored_steps_total']} (opt max {v['optimality_max']:.1e}); decoded states with u<=0 "
            f"{v['decoded_frac_states_with_u_le0']:.1%}")
    save()

    # ---------------- E1 / E2 comparisons incl. along-path operator fidelity ----
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
                Fa, Fo = Fields[(arm, i)], Fields[(ref_arm, i)]
                fdiff = float(np.max(np.linalg.norm(Fa - Fo, axis=1) / (np.linalg.norm(Fo, axis=1) + 1e-300)))
                Za_, Zo_ = np.asarray(last[arm][i][3][2]), np.asarray(last[ref_arm][i][3][2])
                rows.append(dict(traj=i, err_ref=ro_["traj_rel"], err_arm=ra_["traj_rel"],
                                 abs_diff=abs(ra_["traj_rel"] - ro_["traj_rel"]), field_rel_diff_max=fdiff,
                                 lat_dev=float(np.max(np.abs(Za_ - Zo_)) / (1.0 + np.max(np.linalg.norm(Zo_, axis=1)))),
                                 reasons_equal=(ra_["stop_reasons"] == ro_["stop_reasons"]),
                                 attempts_equal=(ra_["attempts_total"] == ro_["attempts_total"]),
                                 e2e_ratio=ra_["e2e_ms"] / ro_["e2e_ms"], roll_ratio=ra_["roll_ms"] / ro_["roll_ms"]))
            c = dict(per_traj=rows, err_abs_diff_max=float(max(r_["abs_diff"] for r_ in rows)),
                     field_rel_diff_max=float(max(r_["field_rel_diff_max"] for r_ in rows)),
                     lat_dev_max=float(max(r_["lat_dev"] for r_ in rows)),
                     err_ratio=va["err_traj_rel_mean"] / vo["err_traj_rel_mean"],
                     stop_hist_identical=bool(va["stop_reasons"] == vo["stop_reasons"]),
                     stop_hist_identical_per_traj=bool(all(r_["reasons_equal"] for r_ in rows)),
                     attempts_identical_per_traj=bool(all(r_["attempts_equal"] for r_ in rows)),
                     e2e_ratio=va["e2e_ms_median"] / vo["e2e_ms_median"], roll_ratio=va["roll_ms_median"] / vo["roll_ms_median"])
            if arm == "tensor" and ref_arm == "full":
                # along the tensor's accepted path: oracle-vs-tensor r, J, J^T r at every step
                pr = dict(r_rel=[], J_rel=[], g_scaled=[])
                for i in range(n_test):
                    Zs = np.asarray(last["tensor"][i][3][2]); z0 = np.asarray(last["tensor"][i][3][1])
                    Zf = np.concatenate([z0[None], Zs]); nu = float(tt["nu"][i])
                    for t_ in range(b3.NUM_STEPS):
                        pv = prev_j(jnp.asarray(Zf[t_]))
                        ro, Jo = [np.asarray(v_) for v_ in rJ_full(jnp.asarray(Zf[t_ + 1]), pv, nu, aux_full)]
                        rt, Jt = [np.asarray(v_) for v_ in rJ_T(jnp.asarray(Zf[t_ + 1]), pv, nu)]
                        pr["r_rel"].append(rel(rt, ro)); pr["J_rel"].append(rel(Jt, Jo))
                        pr["g_scaled"].append(float(np.linalg.norm(Jt.T @ rt - Jo.T @ ro) / (np.linalg.norm(Jo) * np.linalg.norm(ro) + 1e-300)))
                c["path_fidelity"] = {k_: stats(v_) for k_, v_ in pr.items()}
                c["E1_pass"] = bool(c["field_rel_diff_max"] <= 1e-3 and c["lat_dev_max"] <= 1e-4 and abs(c["err_ratio"] - 1) <= 1e-2
                                    and c["stop_hist_identical_per_traj"] and c["attempts_identical_per_traj"]
                                    and max(pr["r_rel"]) <= 1e-3 and max(pr["J_rel"]) <= 1e-2 and max(pr["g_scaled"]) <= 1e-3)
            cmp[f"{arm}_vs_{ref_arm}"] = c
            log(f"  [{arm} vs {ref_arm}] err ratio {c['err_ratio']:.4f}; field rel diff max {c['field_rel_diff_max']:.2e}; "
                f"latent dev max {c['lat_dev_max']:.2e}; stop hist identical {c['stop_hist_identical_per_traj']}; "
                f"attempts identical {c['attempts_identical_per_traj']}" + (f"; E1 {c['E1_pass']}" if "E1_pass" in c else ""))
    report["comparison"] = cmp
    save()

    # ---------------- TR: candidate-path audit of the tensor arm (host loop) -----
    if "tensor" in arms:
        tr_rows = []
        for i in range(min(TR_TRAJ, n_test)):
            nu = float(tt["nu"][i]); tol_abs = float(us_dev[i][0])
            z0 = np.asarray(last["tensor"][i][3][1]); z_prev, z = z0.copy(), z0.copy()
            for t_ in range(b3.NUM_STEPS):
                pc = prev_j(jnp.asarray(z))
                z_ex = z + EXTRAP * (z - z_prev)
                ra = float(rJ_T(jnp.asarray(z), pc, nu)[0] @ rJ_T(jnp.asarray(z), pc, nu)[0]) ** 0.5
                rb = float(jnp.linalg.norm(rJ_T(jnp.asarray(z_ex), pc, nu)[0]))
                zc = z_ex if (np.isfinite(rb) and rb < ra) else z
                rt, Jt = [np.asarray(v_) for v_ in rJ_T(jnp.asarray(zc), pc, nu)]
                ro, Jo = [np.asarray(v_) for v_ in rJ_full(jnp.asarray(zc), pc, nu, aux_full)]
                rn_t, rn_o = np.linalg.norm(rt), np.linalg.norm(ro)
                lam = 1e-6; reason = 0; att = 0
                cands = [dict(kind="init", r_rel=rel(rt, ro), J_rel=rel(Jt, Jo))]
                while reason == 0 and att < GN_BUDGET and rn_t > tol_abs:
                    H = Jt.T @ Jt; g = Jt.T @ rt
                    dz = np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-30 * np.eye(K)), -g)
                    finite = bool(np.all(np.isfinite(dz))); within = float(np.linalg.norm(dz)) <= TRD
                    tiny = finite and float(np.linalg.norm(dz)) <= 1e-12 * (1 + np.linalg.norm(zc))
                    z_new = zc + (dz if (finite and within) else 0.0)
                    rt2, Jt2 = [np.asarray(v_) for v_ in rJ_T(jnp.asarray(z_new), pc, nu)]
                    ro2, Jo2 = [np.asarray(v_) for v_ in rJ_full(jnp.asarray(z_new), pc, nu, aux_full)]
                    rn_t2, rn_o2 = np.linalg.norm(rt2), np.linalg.norm(ro2)
                    acc_t = finite and within and np.isfinite(rn_t2) and rn_t2 < rn_t
                    acc_o = finite and within and np.isfinite(rn_o2) and rn_o2 < rn_o
                    uu = np.asarray(u_full_int(jnp.asarray(z_new)))
                    cands.append(dict(kind="accept" if acc_t else "reject", r_rel=rel(rt2, ro2), J_rel=rel(Jt2, Jo2),
                                      g_scaled=float(np.linalg.norm(Jt2.T @ rt2 - Jo2.T @ ro2) / (np.linalg.norm(Jo2) * np.linalg.norm(ro2) + 1e-300)),
                                      decision_agrees=bool(acc_t == acc_o), n_neg=int(np.sum(uu <= 0))))
                    if acc_t:
                        rel_dec = (rn_t - rn_t2) / rn_t
                        zc, rt, Jt, ro, Jo, rn_t, rn_o = z_new, rt2, Jt2, ro2, Jo2, rn_t2, rn_o2
                        lam = max(lam / 3, 1e-12)
                        if rn_t <= tol_abs:
                            reason = 1
                        elif rel_dec < STALL or tiny:
                            reason = 2
                    else:
                        lam = min(lam * 10, 1e12)
                        if tiny:
                            reason = 2
                        elif lam >= 1e12:
                            reason = 3
                    att += 1
                tr_rows.append(dict(traj=i, step=t_, n_candidates=len(cands), reason=reason,
                                    r_rel_max=float(max(c_["r_rel"] for c_ in cands)),
                                    J_rel_max=float(max(c_["J_rel"] for c_ in cands)),
                                    decisions_agree=int(sum(c_.get("decision_agrees", True) for c_ in cands[1:])),
                                    n_decisions=len(cands) - 1, candidates=cands))
                z_prev, z = z, zc
        n_dec = sum(r_["n_decisions"] for r_ in tr_rows); n_agree = sum(r_["decisions_agree"] for r_ in tr_rows)
        report["gates"]["TR_candidate_path"] = dict(
            recorded=True, n_traj=min(TR_TRAJ, n_test), n_candidates=int(sum(r_["n_candidates"] for r_ in tr_rows)),
            r_rel_max=float(max(r_["r_rel_max"] for r_ in tr_rows)), J_rel_max=float(max(r_["J_rel_max"] for r_ in tr_rows)),
            decision_agreement=float(n_agree / max(n_dec, 1)),
            concern=bool(max(r_["r_rel_max"] for r_ in tr_rows) > 1e-3 or (n_agree / max(n_dec, 1)) < 0.99), rows=tr_rows)
        g_ = report["gates"]["TR_candidate_path"]
        log(f"  TR (recorded): {g_['n_candidates']} candidates; r rel max {g_['r_rel_max']:.2e}; J rel max "
            f"{g_['J_rel_max']:.2e}; decision agreement {g_['decision_agreement']:.4f}; concern {g_['concern']}")
    save()

    # ---------------- kernel export for the same-GPU job (C1) -------------------
    np.savez(KERNEL_OUT, N=N, K=K, R=R, M=M_ACT, A=A_np, lam=np.asarray(lam), Q=Q, TRD=TRD, stall=STALL,
             extrap=EXTRAP, gn_budget=GN_BUDGET,
             z0=np.stack([np.asarray(last["tensor"][i][3][1]) for i in range(n_test)]) if "tensor" in arms else np.zeros((0, K)),
             nu=np.asarray(tt["nu"][:n_test]), tol_abs=np.asarray([float(u[0]) for u in us_dev]),
             params_B=np.asarray(params["B"]), params_h_lin=np.asarray(params["h_lin"]),
             **{f"h{i}_w": np.asarray(w) for i, (w, b) in enumerate(params["h"])},
             **{f"h{i}_b": np.asarray(b) for i, (w, b) in enumerate(params["h"])})
    report["config"]["kernel_npz"] = KERNEL_OUT

    if SKIP_FOM:
        report["complete"] = True; report["secs_total"] = time.time() - t_all; save()
        log(f"DONE (SKIP_FOM) -> {OUT} [{time.time()-t_all:.0f}s]")
        return

    # ---------------- classical ladders: newton + defect, same GPU -------------
    newt = b3.make_newton_tol_rollout(n, DST)
    dfc = b3.make_defect_tol_rollout(n, DST)
    cfgs = [("newton", nt, max(nt * lf, 1e-12)) for nt in NEWTON_TOLS for lf in LIN_FRACS]
    cfgs += [("defect", nt, b3.MAX_PICARD) for nt in NEWTON_TOLS] + [("defect", 0.0, k_) for k_ in DEFECT_FIXED]
    base_rows = {c_: [] for c_ in cfgs}
    b3.burn_in(1.5)
    for i in range(n_test):
        u0 = jnp.asarray(U_test[i, 0]); nu = float(tt["nu"][i]); tnorm = np.linalg.norm(U_test[i], axis=1)
        subs = []
        for (arm, nt, p2) in cfgs:
            if arm == "newton":
                subs.append((f"newton|{nt:.0e}|{p2:.0e}", lambda _u=u0, _n=nu, _t=nt, _l=p2: (lambda o: (o[0].block_until_ready(), o)[1])(newt(_u, _n, _t, _l))))
            else:
                subs.append((f"defect|{nt:.0e}|{p2}", lambda _u=u0, _n=nu, _t=nt, _k=int(p2): (lambda o: (o[0].block_until_ready(), o)[1])(dfc(_u, _n, _t, _k))))
        raw, res = b3.balanced_time(subs, reps=FOM_REPS, warm=WARM)
        for (arm, nt, p2), (key, _) in zip(cfgs, subs):
            o = res[key]
            snaps = np.asarray(o[0]); its = np.asarray(o[1])
            pt = np.linalg.norm(snaps - U_test[i], axis=1) / tnorm
            row = dict(traj=i, traj_rel=float(np.mean(pt)), iters_total=int(np.sum(its)),
                       time_ms=float(np.median(raw[key])) * 1e3, time_raw_s=[float(t) for t in raw[key]])
            if arm == "defect":
                row["stalled_steps"] = int(np.sum(np.asarray(o[3])))
            base_rows[(arm, nt, p2)].append(row)
        log(f"  FOM ladders traj {i} done [{time.time()-t_all:.0f}s]")
    report["fom"] = []
    for c_ in cfgs:
        rows = base_rows[c_]
        report["fom"].append(dict(arm=c_[0], N=N, ntol=c_[1], lin_tol=c_[2] if c_[0] == "newton" else None,
                                  max_iter=c_[2] if c_[0] == "defect" else None,
                                  preconditioner="exact Helmholtz (3D DST, %s)" % DST,
                                  err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
                                  err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
                                  time_ms_median=float(np.median([r_["time_ms"] for r_ in rows])),
                                  iters_mean=float(np.mean([r_["iters_total"] for r_ in rows])),
                                  stalled_steps_total=int(sum(r_.get("stalled_steps", 0) for r_ in rows)), per_traj=rows, n_test=n_test))
    for f_ in report["fom"]:
        log(f"   FOM {f_['arm']:6s} ntol={f_['ntol']:.0e} {('lt=%.0e' % f_['lin_tol']) if f_['lin_tol'] is not None else ('k=%d' % f_['max_iter'])}: "
            f"err {f_['err_traj_rel_mean']:.3e}  {f_['time_ms_median']:.2f} ms  iters {f_['iters_mean']:.0f}")
    save()
    mem_snapshot("after_fom", mem)

    # matched (cheaper arm), bracket, paired AB/BA, clustered bootstrap
    fom_err = {c_: float(np.mean([r_["traj_rel"] for r_ in base_rows[c_]])) for c_ in cfgs}
    fom_ms = {c_: float(np.median([r_["time_ms"] for r_ in base_rows[c_]])) for c_ in cfgs}
    tight = min(cfgs, key=lambda c_: fom_err[c_])
    report["matched"] = dict(tightest=dict(arm=tight[0], ntol=tight[1], p2=tight[2], err=fom_err[tight], ms=fom_ms[tight]), arms={})
    brng = np.random.default_rng(SEED0 + 123)
    for arm in arms:
        cerr = report["variants"][arm]["err_traj_rel_mean"]
        cands = [c_ for c_ in cfgs if fom_err[c_] <= cerr]
        looser = [c_ for c_ in cfgs if fom_err[c_] > cerr]
        bracket = bool(cands and looser)
        match = min(cands, key=lambda c_: fom_ms[c_]) if cands else None
        ent = dict(rom_err=cerr, rom_e2e_ms=report["variants"][arm]["e2e_ms_median"], bracket=bracket,
                   rule_matched="cheapest rung of EITHER classical arm at least as accurate as the ROM arm",
                   matched=None if match is None else dict(arm=match[0], ntol=match[1], p2=match[2], err=fom_err[match], ms=fom_ms[match]),
                   cheapest_per_arm={a_: (lambda cc: None if not cc else dict(ntol=min(cc, key=lambda c_: fom_ms[c_])[1], ms=fom_ms[min(cc, key=lambda c_: fom_ms[c_])]))(
                       [c_ for c_ in cands if c_[0] == a_]) for a_ in ("newton", "defect")})
        if match is not None:
            pairs = []
            for i in range(n_test):
                u0 = jnp.asarray(U_test[i, 0]); nu = float(tt["nu"][i]); ar = A_[arm]
                fom_fn = (lambda _u=u0, _n=nu, _t=match[1], _l=match[2]: newt(_u, _n, _t, _l)[0].block_until_ready()) if match[0] == "newton" \
                    else (lambda _u=u0, _n=nu, _t=match[1], _k=int(match[2]): dfc(_u, _n, _t, _k)[0].block_until_ready())
                pairs.append(dict(traj=i, **b3.time_pair(
                    lambda _u=u0_dev[i], _n=nu, _a=ar: _a["e2e"](G_all, _u, _n, _a["aux"])[0].block_until_ready(), fom_fn,
                    reps=PAIR_REPS, warm=WARM)))
            sp = np.array([p["b_ms"] / p["a_ms"] for p in pairs])
            ent["paired"] = dict(rom_ms=float(np.median([p["a_ms"] for p in pairs])), fom_ms=float(np.median([p["b_ms"] for p in pairs])),
                                 per_traj=pairs, speedup_per_traj=[float(v) for v in sp], speedup=float(np.median(sp)),
                                 speedup_min=float(np.min(sp)), all_gt1=bool(np.all(sp > 1)),
                                 boot_lower95=bootstrap_lower(pairs, brng, BOOT), n_boot=BOOT,
                                 outliers=int(sum(np.sum(np.abs(np.array(p["a_raw_ms"]) - np.median(p["a_raw_ms"])) > 0.5 * np.median(p["a_raw_ms"])) for p in pairs)))
            ent["speed_win"] = bool(bracket and ent["paired"]["all_gt1"] and ent["paired"]["speedup"] > 1.1 and ent["paired"]["boot_lower95"] > 1.0)
            log(f"  MATCHED paired [{arm}]: ROM {ent['paired']['rom_ms']:.2f} ms vs {match[0]}(ntol={match[1]:.0e}, err {fom_err[match]:.2e}) "
                f"{ent['paired']['fom_ms']:.2f} ms -> median {ent['paired']['speedup']:.2f}x, min {ent['paired']['speedup_min']:.2f}x, "
                f"boot lower95 {ent['paired']['boot_lower95']:.2f}; bracket {bracket}; speed win {ent['speed_win']}")
        report["matched"]["arms"][arm] = ent
    save()

    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


def fit_encoder(key, X_np, Z_np, steps, hidden=128, layers=2, lr=1e-3):
    """sep_solvers.fit_code_encoder, copied: small MLP standardised X -> code,
    trained on TRAINING pairs only (initial guesses for the IC fit)."""
    import optax
    X = jnp.asarray(np.asarray(X_np), dtype=F64)
    Z = jnp.asarray(np.asarray(Z_np), dtype=F64)
    mu = jnp.mean(X, axis=0); sd = jnp.std(X, axis=0) + 1e-8
    sizes = [X.shape[1]] + [hidden] * layers + [Z.shape[1]]
    params = {"mu": mu, "sd": sd, "w": [], "b": []}
    for i in range(len(sizes) - 1):
        key, k1 = jax.random.split(key)
        params["w"].append(jax.random.normal(k1, (sizes[i], sizes[i + 1]), dtype=F64) * jnp.sqrt(2.0 / sizes[i]))
        params["b"].append(jnp.zeros((sizes[i + 1],), dtype=F64))

    def apply_fn(p, x):
        h = (x - p["mu"]) / p["sd"]
        for w, b in zip(p["w"][:-1], p["b"][:-1]):
            h = jax.nn.silu(h @ w + b)
        return h @ p["w"][-1] + p["b"][-1]

    z_ms = jnp.mean(Z * Z)
    loss = lambda p: jnp.mean((apply_fn(p, X) - Z) ** 2) / z_ms
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    masked = lambda p: {"w": p["w"], "b": p["b"]}
    state = opt.init(masked(params))

    @jax.jit
    def step(p, s):
        v, g = jax.value_and_grad(lambda q: loss({**p, **q}))(masked(p))
        upd, s = opt.update(g, s)
        return {**p, **optax.apply_updates(masked(p), upd)}, s, v

    t0 = time.time()
    v = jnp.inf
    for _ in range(steps):
        params, state, v = step(params, state)
    info = dict(final_rel_mse=float(v), steps=steps, hidden=hidden, layers=layers, seconds=time.time() - t0,
                n_pairs=int(X.shape[0]), n_features=int(X.shape[1]))
    log(f"  encoder: rel-MSE {float(v):.3e} on {info['n_pairs']} training pairs [{info['seconds']:.0f}s]")
    return params, apply_fn, info


if __name__ == "__main__":
    main()
