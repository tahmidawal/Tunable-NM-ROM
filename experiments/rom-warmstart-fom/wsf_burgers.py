"""BURGERS-2D arm of the ROM-WARM-STARTED-FOM cell.

Use the ROM's state at step n as the FOM's Newton INITIAL GUESS instead of
u_{n-1}, and finish every step to a full-accuracy Newton tolerance.  The answer
is FOM-exact; the question is cost.

  hybrid total = t_rom_ic + t_rom_rollout + t_decode + t_fom(from the ROM guesses)
  baseline     = the SAME implicit chain warm-started from the PREVIOUS STEP.

RISK, STATED UP FRONT: the FOM already warm-starts from u_{n-1}, which at
dt = 0.005 is a very good guess.  The ROM must beat THAT bar, not a cold start.
A third arm -- linear extrapolation 2 u_{n-1} - u_{n-2}, the classical trick --
is measured too, so the ROM is compared against the best cheap alternative and
not only against the weakest one.

Reused verbatim from the reference harnesses:
  * the discrete operator: `burgers2d_film.make_rollout(n)`'s `residual`, i.e.
    the exact backward-Euler upwind/centred operator that generated the data;
  * the ROM: `blat_common.make_weak_ops` / `fit_eq_weights` / `rollout_jit`
    (variant lspg:eq256:weak64) and `fu_common.make_fit_ic_jit` (EQ-node cold
    start), exactly as in the reference `followup/fu_timing.py`.

NEW here, and shared bit-for-bit by all three arms:
  * `make_bicgstab` -- a counting BiCGStab with an explicit breakdown/NaN guard.
    The testbed calls `jax.scipy.sparse.linalg.bicgstab`, which cannot report an
    iteration count, and the testbed's Newton loop is a FIXED-LENGTH scan of 8
    steps, so neither can answer "how many iterations did the warm start save".
    The counting solver keeps the testbed's `LIN_TOL` / `LIN_MAXITER` and is
    VERIFIED against the testbed's own rollout (`fom_reference_check`).
  * `make_chain`   -- one jitted 50-step implicit chain whose only difference
    between arms is a traced `guess_mode` integer, so the Newton stopping test,
    the linear tolerance, the operator, the compilation and the warm-up are
    identical across arms by construction.

BiCGStab NaN landmine (seen earlier in this project): once the Newton residual
reaches machine epsilon, BiCGStab's rho/omega inner products can underflow and
return a NaN step.  Here the Newton loop EXITS on its tolerance test before that
can happen, and any non-finite step or linear breakdown is still detected,
counted and reported (`lin_breakdowns`, `newton_flags`) rather than dropped.

Timing protocol: all ladder points SEQUENTIALLY IN ONE PROCESS ON ONE GPU,
warm-up 2, median of 7, `block_until_ready`, f64, matmul precision `highest`.

Usage: PKL=<blat_ad_N64_K8.pkl> [NS=32,64,128,256] [FOM_TAUS=1e-6,1e-8,1e-10]
       [VARIANT=lspg:eq256:weak64] python wsf_burgers.py <out.json>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax

HERE = os.path.dirname(os.path.abspath(__file__))
_EXPS = os.path.dirname(HERE)
for d in (HERE,
          os.path.join(_EXPS, "burgers2d-rom-latent-stepping"),
          os.path.join(_EXPS, "burgers2d-rom-latent-stepping", "followup")):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import blat_common as bc                       # noqa: E402  (enables x64)
from blat_common import F64, log               # noqa: E402
import fu_common as fu                         # noqa: E402
import jax.numpy as jnp                        # noqa: E402
import wsf_util as wu                          # noqa: E402

OUT = sys.argv[1]
PKL = os.environ["PKL"]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256").split(",") if v]
FOM_TAUS = [float(v) for v in os.environ.get("FOM_TAUS", "1e-6,1e-8,1e-10").split(",")]
VARIANT = os.environ.get("VARIANT", "lspg:eq256:weak64")
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
TEST_IDX = int(os.environ.get("TEST_IDX", "0"))
N_TEST_TRAJ = int(os.environ.get("N_TEST_TRAJ", "4"))    # trajectories for iteration stats
MAX_NEWTON = int(os.environ.get("MAX_NEWTON", "25"))
# time slices decoded at once.  vmapping the FiLM decoder over all 51 slices at
# N=256 asks for a single 12.5 GB activation buffer and OOMs a 40 GB A100, which
# would make the decode cost depend on which card the job landed on.  Chunking
# bounds the working set and makes the measurement hardware-independent.
DEC_CHUNK = int(os.environ.get("DEC_CHUNK", "8"))
BURN_S = float(os.environ.get("BURN_S", "3"))   # GPU clock burn-in before timing
LIN_TOL = float(os.environ.get("LIN_TOL", str(bc.bf.LIN_TOL)))
LIN_MAXITER = int(os.environ.get("LIN_MAXITER", str(bc.bf.LIN_MAXITER)))
FOM_RES_TOL = float(os.environ.get("FOM_RES_TOL", "1e-8"))
EQ_RNG_SEED = 4321
# see wsf_poisson.py for the panel / consolidated contract
RUN_ROLE = os.environ.get("RUN_ROLE", "consolidated")
ARMS = ("prev", "extrap", "rom")
ARM_ID = {"prev": 0, "extrap": 1, "rom": 2}
T = bc.NUM_STEPS


# --------------------------------------------------------------- counting BiCGStab
def make_bicgstab(tol, maxiter):
    """BiCGStab for A x = b from x0 = 0 (the FOM solves for a Newton CORRECTION,
    so a zero start is the right and the testbed's choice), with an iteration
    count and an explicit breakdown / non-finite guard.

    Stopping test ||r_k|| <= tol * ||b||, matching
    `jax.scipy.sparse.linalg.bicgstab(..., tol=LIN_TOL)` (whose atol default is 0).

    The ALPHA HALF-STEP convergence test is included: if ||s|| = ||r - alpha A p||
    already meets the threshold, x + alpha p is the answer and the omega half-step
    is skipped (omega is forced to 0, which makes x_new = x + alpha p and r_new = s
    exactly).  Omitting that test is a real correctness defect: when s is exactly
    zero, t^T t = 0 and a naive implementation declares a breakdown and DISCARDS a
    converged iterate.  The A(s) product is still evaluated in that final sweep --
    the recursion is branchless so that all three arms execute one identical
    kernel -- so `matvecs` (not `k`) is the honest work count.

    flag: 0 converged, 1 maxiter, 2 breakdown (rho, rhat^T v or t^T t underflowed,
    or a non-finite quantity appeared).  All three are returned and are propagated
    into the Newton status by the caller; none is silently dropped."""
    def bicgstab(A, b):
        bn = jnp.linalg.norm(b)
        thr = tol * bn
        x = jnp.zeros_like(b)
        r = b
        rhat = b
        rho = jnp.asarray(1.0, F64); alpha = jnp.asarray(1.0, F64)
        omega = jnp.asarray(1.0, F64)
        v = jnp.zeros_like(b); p = jnp.zeros_like(b)
        tiny = 1e-300
        bad0 = ~(jnp.all(jnp.isfinite(b)) & jnp.isfinite(bn))

        def cond(s):
            x, r, p, v, rho, alpha, omega, k, mv, bad = s
            return (jnp.linalg.norm(r) > thr) & (k < maxiter) & (~bad)

        def body(s):
            x, r, p, v, rho, alpha, omega, k, mv, bad = s
            rho_new = jnp.sum(rhat * r)
            brk = (jnp.abs(rho_new) < tiny) | (jnp.abs(omega) < tiny)
            beta = (rho_new / jnp.where(brk, 1.0, rho)) * \
                   (alpha / jnp.where(brk, 1.0, omega))
            p_new = r + beta * (p - omega * v)
            v_new = A(p_new)
            rv = jnp.sum(rhat * v_new)
            brk = brk | (jnp.abs(rv) < tiny)
            alpha_new = rho_new / jnp.where(brk, 1.0, rv)
            s_ = r - alpha_new * v_new
            # ALPHA HALF-STEP CONVERGENCE: accept x + alpha p and skip the omega stage
            s_conv = jnp.linalg.norm(s_) <= thr
            t_ = A(s_)
            tt = jnp.sum(t_ * t_)
            brk = brk | ((tt < tiny) & (~s_conv))       # tt == 0 with s == 0 is SUCCESS
            omega_new = jnp.where(s_conv, 0.0,
                                  jnp.sum(t_ * s_) / jnp.where(brk | s_conv, 1.0, tt))
            x_new = x + alpha_new * p_new + omega_new * s_
            r_new = s_ - omega_new * t_
            nonfinite = ~(jnp.all(jnp.isfinite(x_new)) & jnp.all(jnp.isfinite(r_new)))
            brk = brk | nonfinite
            # on a breakdown freeze the state (the caller sees flag 2 and the last
            # good iterate) rather than propagating a NaN into the Newton step
            keep = lambda new, old: jnp.where(brk, old, new)
            return (keep(x_new, x), keep(r_new, r), keep(p_new, p), keep(v_new, v),
                    keep(rho_new, rho), keep(alpha_new, alpha), keep(omega_new, omega),
                    k + 1, mv + 2, brk)

        x, r, p, v, rho, alpha, omega, k, mv, bad = jax.lax.while_loop(
            cond, body, (x, r, p, v, rho, alpha, omega, jnp.int32(0), jnp.int32(0),
                         bad0))
        flag = jnp.where(bad, jnp.int32(2),
                         jnp.where(jnp.linalg.norm(r) > thr, jnp.int32(1), jnp.int32(0)))
        return x, k, mv, flag
    return bicgstab


# --------------------------------------------------------------- counting Newton chain
def make_chain(n, tol_rel):
    """One jitted implicit chain of T backward-Euler steps.

    chain(u0, nu, guesses, guess_mode) -> dict of per-step arrays.
      guess_mode 0 = previous step u_{n-1}   (the FOM's own warm start)
                 1 = linear extrapolation 2 u_{n-1} - u_{n-2}  (u_0 at step 1)
                 2 = the supplied ROM guesses
    `guess_mode` is a TRACED argument, so all three arms run the identical
    compiled kernel and therefore the identical Newton stopping test
    ||R(u, u_prev, nu)|| <= tol_rel * ||u_prev||  -- the testbed's own convergence
    metric (`burgers2d_film.newton_step` reports exactly this ratio).

    newton_flag: 0 converged, 1 max_newton reached without meeting the tolerance,
    2 non-finite Newton step, 3 non-finite residual, 4 the linear solve broke down
    or hit its own iteration cap.  A NON-FINITE INITIAL RESIDUAL is flagged before
    the loop, so a NaN can never be reported as a zero-iteration success."""
    _, residual = bc.bf.make_rollout(n)
    bicg = make_bicgstab(LIN_TOL, LIN_MAXITER)

    def step(u_prev, u_prev2, g, mode, nu):
        u_start = jnp.where(mode == 0, u_prev,
                            jnp.where(mode == 1, 2.0 * u_prev - u_prev2, g))
        tol_abs = tol_rel * jnp.linalg.norm(u_prev)
        r0 = residual(u_start, u_prev, nu)
        rn0 = jnp.linalg.norm(r0)
        # a NaN start must NOT slip through the `rn > tol_abs` test as "converged"
        init0 = jnp.where(jnp.isfinite(rn0), jnp.int32(0), jnp.int32(3))
        init = (u_start, r0, rn0, jnp.int32(0), jnp.int32(0), jnp.int32(0),
                jnp.int32(0), init0)

        def cond(s):
            u, r, rn, k, nl, nbrk, nlmax, flag = s
            return (rn > tol_abs) & (k < MAX_NEWTON) & (flag == 0)

        def body(s):
            u, r, rn, k, nl, nbrk, nlmax, flag = s
            Jv = lambda vv: jax.jvp(lambda uu: residual(uu, u_prev, nu), (u,), (vv,))[1]
            du, li, lmv, lflag = bicg(Jv, -r)
            ok = jnp.all(jnp.isfinite(du))
            u2 = jnp.where(ok, u + du, u)
            r2 = residual(u2, u_prev, nu)
            rn2 = jnp.linalg.norm(r2)
            fin = jnp.isfinite(rn2)
            # the linear solver's own status is PROPAGATED, not dropped: an inexact
            # correction is allowed to continue (Newton may still converge) but the
            # occurrence is both counted and raised into the Newton flag.
            new_flag = jnp.where(~ok, jnp.int32(2),
                          jnp.where(~fin, jnp.int32(3),
                            jnp.where(lflag != 0, jnp.int32(4), jnp.int32(0))))
            # flag 4 must not abort the Newton loop -- it records that a linear solve
            # was inexact.  Convergence is still decided by the residual test below.
            cont = jnp.where(new_flag == 4, jnp.int32(0), new_flag)
            return (jnp.where(fin, u2, u), jnp.where(fin, r2, r),
                    jnp.where(fin, rn2, rn), k + 1, nl + li,
                    nbrk + (lflag == 2).astype(jnp.int32),
                    nlmax + (lflag == 1).astype(jnp.int32), cont)

        u, r, rn, k, nl, nbrk, nlmax, flag = jax.lax.while_loop(cond, body, init)
        flag = jnp.where((flag == 0) & (rn > tol_abs), jnp.int32(1),
                         jnp.where((flag == 0) & ((nbrk > 0) | (nlmax > 0)),
                                   jnp.int32(4), flag))
        return (u, k, nl, nbrk + nlmax, flag,
                rn / jnp.maximum(jnp.linalg.norm(u_prev), 1e-300))

    def chain(u0, nu, guesses, mode):
        def body(carry, g):
            u_prev, u_prev2 = carry
            u, k, nl, nbrk, flag, rr = step(u_prev, u_prev2, g, mode, nu)
            return (u, u_prev), (u, k, nl, nbrk, flag, rr)
        _, (U, KK, NL, NB, FL, RR) = jax.lax.scan(body, (u0, u0), guesses)
        return U, KK, NL, NB, FL, RR

    return jax.jit(chain), residual


def fom_reference_check(n, trajs, chain):
    """The counting Newton chain (previous-step arm) must reproduce the testbed's
    own fixed-8-iteration rollout -- otherwise the "FOM" being timed here is not the
    FOM that produced the data.  Checked on EVERY test trajectory and reported as
    the PER-STEP maximum as well as the global trajectory norm (a global norm can
    hide one bad step).  The linear solver is checked separately against
    `jax.scipy.sparse.linalg.bicgstab` on a representative Newton correction."""
    roll, residual = bc.bf.make_rollout(n)
    bicg = make_bicgstab(LIN_TOL, LIN_MAXITER)
    dummy = jnp.zeros((T, n * n), F64)
    per, worst_step, worst_glob = [], 0.0, 0.0
    for U, nu, _ in trajs:
        u0 = U[0]
        snaps, rr = roll(jnp.asarray(u0)[None], jnp.asarray([nu]))
        U_ref = np.asarray(snaps)[1:, 0]
        Uc, KK, NL, NB, FL, RR = chain(jnp.asarray(u0), nu, dummy, jnp.int32(0))
        Uc = np.asarray(Uc)
        glob = float(np.linalg.norm(Uc - U_ref) / np.linalg.norm(U_ref))
        step = float(np.max(np.linalg.norm(Uc - U_ref, axis=1)
                            / np.linalg.norm(U_ref, axis=1)))
        worst_glob = max(worst_glob, glob); worst_step = max(worst_step, step)
        per.append(dict(rel_diff_trajectory_norm=glob, rel_diff_max_over_steps=step,
                        testbed_max_rel_newton_residual=float(jnp.max(rr)),
                        counting_chain_max_rel_newton_residual=float(jnp.max(RR)),
                        newton_iters=int(np.sum(np.asarray(KK))),
                        lin_iters=int(np.sum(np.asarray(NL))),
                        lin_failures=int(np.sum(np.asarray(NB))),
                        flags=sorted(set(np.asarray(FL).tolist()))))
    # linear-solver cross-check on one representative Newton correction
    U, nu, _ = trajs[0]
    u_prev = jnp.asarray(U[0]); u = jnp.asarray(U[1])
    r = residual(u_prev, u_prev, nu)
    Jv = lambda vv: jax.jvp(lambda uu: residual(uu, u_prev, nu), (u_prev,), (vv,))[1]
    du_ours, k_, mv_, fl_ = bicg(Jv, -r)
    du_ref, _ = jax.scipy.sparse.linalg.bicgstab(Jv, -r, tol=LIN_TOL,
                                                 maxiter=LIN_MAXITER)
    lin = dict(rel_diff_vs_jax_scipy_bicgstab=float(
                   jnp.linalg.norm(du_ours - du_ref)
                   / jnp.maximum(jnp.linalg.norm(du_ref), 1e-300)),
               ours_rel_lin_residual=float(jnp.linalg.norm(Jv(du_ours) + r)
                                           / jnp.maximum(jnp.linalg.norm(r), 1e-300)),
               jax_rel_lin_residual=float(jnp.linalg.norm(Jv(du_ref) + r)
                                          / jnp.maximum(jnp.linalg.norm(r), 1e-300)),
               ours_iters=int(k_), ours_matvecs=int(mv_), ours_flag=int(fl_))
    return dict(per_trajectory=per, linear_solver=lin,
                rel_diff_vs_testbed_rollout=worst_glob,
                rel_diff_max_over_steps=worst_step)


# --------------------------------------------------------------- ROM
def build_ops(dec, n, var, Z_snap, cache):
    """Exactly `followup/fu_timing.build_ops` (EQ weights refit per N)."""
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


def test_traj(n, idx):
    """Test trajectory `idx` of the TEST_SEED draw regenerated with the FOM at
    resolution n (identical to followup/fu_timing.test_traj)."""
    cxt, cyt, wt, at, nut, zt = bc.bf.sample_params(seed=bc.TEST_SEED, m=bc.N_TEST)
    roll, _ = bc.bf.make_rollout(n)
    u0 = bc.bf.blob_ic(n, cxt[idx], cyt[idx], wt[idx], at[idx])
    snaps, rr = roll(jnp.asarray(u0)[None], jnp.asarray([nut[idx]]))
    U = np.asarray(snaps)[:, 0]
    rmax = float(jnp.max(rr))
    if not np.isfinite(rmax) or rmax > FOM_RES_TOL:
        raise SystemExit(f"N={n} traj {idx}: FOM Newton rel residual {rmax:.2e} > "
                         f"{FOM_RES_TOL:.0e} -- refusing to time an unconverged baseline")
    return U, float(nut[idx]), rmax


def main():
    dev = jax.devices()[0]
    prov = wu.provenance(HERE)
    log(f"jax_backend={jax.default_backend()} device={dev} commit={prov['commit_short']} "
        f"NS={NS} fom_taus={FOM_TAUS} variant={VARIANT} n_traj={N_TEST_TRAJ} "
        f"reps={TIME_REPS} warm={TIME_WARM} lin_tol={LIN_TOL} max_newton={MAX_NEWTON}")
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu -- refusing to produce timings")

    with open(PKL, "rb") as f:
        ck = pickle.load(f)
    for key, val in (("bc_mode", bc.BC_MODE), ("N", bc.N), ("ad_hidden", bc.AD_HIDDEN),
                     ("ad_layers", bc.AD_LAYERS), ("n_train", bc.N_TRAIN), ("seed", bc.SEED)):
        if ck["config"][key] != val:
            raise SystemExit(f"checkpoint config mismatch on {key}: {ck['config'][key]} vs {val}")
    dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                          ck["n_freq"], ck["eps"], ck["k_lat"])
    K = dec.k
    Ztr = ck["Z_train"]; T1 = Ztr.shape[1]
    Z_snap = Ztr.reshape(-1, K)[np.random.default_rng(EQ_RNG_SEED).choice(
        Ztr.shape[0] * T1, bc.EQ_SNAPS, replace=False)]

    report = dict(config=dict(bc.CONFIG, pde="burgers2d", pkl=os.path.basename(PKL),
                              ad_config=ck["config"], ns=NS, fom_taus=FOM_TAUS,
                              variant=VARIANT, arms=list(ARMS), n_test_traj=N_TEST_TRAJ,
                              time_reps=TIME_REPS, time_warm=TIME_WARM,
                              max_newton=MAX_NEWTON, lin_tol=LIN_TOL,
                              lin_maxiter=LIN_MAXITER, test_idx=TEST_IDX,
                              test_seed=bc.TEST_SEED, num_steps=T, dt=bc.DT,
                              run_role=RUN_ROLE,
                              newton_tau_definition="||R(u,u_prev,nu)||_2 <= tau * ||u_prev||_2, "
                                                    "the testbed's own Newton convergence metric"),
                  provenance=prov, rows=[], checks=[], per_step=[])

    def save():
        # allow_nan=False: an invalid number must fail loudly here rather than be
        # serialised as a bare NaN and silently averaged into a headline later.
        json.dump(report, open(OUT, "w"), indent=1, default=float, allow_nan=False)

    for n in NS:
        t_n0 = time.time()
        n2 = n * n
        coords = jnp.asarray(bc.grid_coords(n))
        interior = bc.interior_indices(n)
        trajs = []
        for j in range(N_TEST_TRAJ):
            U, nu, rmax = test_traj(n, TEST_IDX + j)
            trajs.append((U, nu, rmax))
        U0_ref, nu0, rmax0 = trajs[0]
        log(f"== N={n}: FOM test trajectories regenerated, max Newton rel residual "
            f"{max(t[2] for t in trajs):.2e}")

        # ---- ROM (variant + EQ weights refit at this N, as in fu_timing)
        cache = {}
        ops = build_ops(dec, n, VARIANT, Z_snap, cache)
        col = ops.get("colloc_used")
        if col is None or col.get("kind") != "grid":
            raise SystemExit(f"variant {VARIANT} has no grid EQ node set -- the "
                             f"hyper-reduced cold start needs one")
        fit_eq = fu.make_fit_ic_jit(dec, n, bc.IC_BUDGET, coords=coords,
                                    idx=col["idx"], w=col.get("w"))
        def _dec_all(ZZ):
            k = ZZ.shape[1]
            pad = (-ZZ.shape[0]) % DEC_CHUNK
            Zp = jnp.concatenate([ZZ, jnp.zeros((pad, k), F64)]) if pad else ZZ
            out = jax.lax.map(lambda zc: jax.vmap(lambda z: dec(z, coords))(zc),
                              Zp.reshape(-1, DEC_CHUNK, k))
            return out.reshape(-1, coords.shape[0])[:ZZ.shape[0]]
        dec_all = jax.jit(_dec_all)

        # ---- ONLINE PREPROCESSING, charged to the hybrid total.
        # blat_rom.py picks the cold start as the best of {mean t=0 latent, the t=0
        # latent of the training trajectory whose INITIAL FIELD is nearest to the
        # query u0}.  That nearest-neighbour search is QUERY-DEPENDENT O(n_train n^2)
        # work, so it cannot be treated as free.  The bank of training initial fields
        # is genuinely offline (it is training data, and blob_ic is analytic at any
        # mesh), so it is built and timed separately; the SEARCH, the latent gather,
        # the tolerance scale and the per-step tolerance vector are the online part
        # and are timed into t_pre_ms.
        t_bank0 = time.time()
        cxb, cyb, wb, ab, nub, _zb = bc.bf.sample_params()
        U0_bank = jnp.asarray(np.stack([np.asarray(bc.bf.blob_ic(n, cxb[i], cyb[i],
                                                                 wb[i], ab[i])).reshape(-1)
                                        for i in range(bc.N_TRAIN)]))
        Zt0_bank = jnp.asarray(Ztr[:, 0, :])
        zmean0 = jnp.asarray(Ztr.mean(axis=0)[0])
        bank_build_s = time.time() - t_bank0
        interior_j = jnp.asarray(interior)
        tol_scale = float(ops.get("tol_scale", np.sqrt(ops["m"])))

        @jax.jit
        def prep(u0):
            j = jnp.argmin(jnp.sum((U0_bank - u0[None, :]) ** 2, axis=1))
            Z0 = jnp.stack([zmean0, Zt0_bank[j]])
            u0_rms = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            usc = jnp.full((T,), bc.GN_TOL * u0_rms * tol_scale)
            return Z0, usc, j

        rom_per_traj = []
        for j, (U, nu, _) in enumerate(trajs):
            u0 = U[0]
            u0j_ = jnp.asarray(u0)
            Z0, usc, jj = prep(u0j_)
            u0_rms = float(np.sqrt(np.mean(u0[interior] ** 2)))
            z_e, rel_e, nJ_e, b_e, att_e = fit_eq(u0j_, Z0)
            Zr, rn_, nj_, re_ = ops["rollout_jit"](z_e, nu, usc, bc.GN_BUDGET)
            G = np.asarray(dec_all(Zr))                       # (T, n^2) guesses, t=1..T
            err = np.linalg.norm(G - U[1:], axis=1) / np.linalg.norm(U[1:], axis=1)
            rom_per_traj.append(dict(z_e=z_e, Z0=Z0, usc=usc, u0=u0, u0_rms=u0_rms, nu=nu,
                                     G=G, U=U, rom_err=err,
                                     ic_rel_on_eq=float(rel_e), ic_iters=int(nJ_e),
                                     rom_iters=int(jnp.sum(nj_))))
        rom_err_mean = float(np.mean([np.mean(r["rom_err"]) for r in rom_per_traj]))
        rom_iters_mean = float(np.mean([r["rom_iters"] for r in rom_per_traj]))
        log(f"   ROM ({VARIANT}) m={ops['m']}: traj rel-L2 vs FOM@N {rom_err_mean:.3e}, "
            f"IC misfit on EQ nodes {np.mean([r['ic_rel_on_eq'] for r in rom_per_traj]):.2e}, "
            f"latent-solve iters {rom_iters_mean:.0f} over {T} steps")

        # ---- ROM online cost (same protocol)
        r0 = rom_per_traj[0]
        u0j = jnp.asarray(r0["u0"])
        pre_med, _ = wu.time_fn(lambda: prep(u0j)[0].block_until_ready(),
                                TIME_REPS, TIME_WARM)
        ic_med, _ = wu.time_fn(lambda: fit_eq(u0j, r0["Z0"])[0].block_until_ready(),
                               TIME_REPS, TIME_WARM)
        usc0 = r0["usc"]
        roll_med, _ = wu.time_fn(
            lambda: ops["rollout_jit"](r0["z_e"], r0["nu"], usc0, bc.GN_BUDGET)[0].block_until_ready(),
            TIME_REPS, TIME_WARM)
        Zr0 = ops["rollout_jit"](r0["z_e"], r0["nu"], usc0, bc.GN_BUDGET)[0]
        dec_med, _ = wu.time_fn(lambda: dec_all(Zr0).block_until_ready(), TIME_REPS, TIME_WARM)
        # SENSITIVITY BASELINE: the testbed's OWN jitted rollout at batch 1 (a fixed
        # 8-Newton-iteration scan, no tolerance test and no dummy guess stream), i.e.
        # the production FOM as the reference cell timed it.
        roll_tb, _ = bc.bf.make_rollout(n)
        U0b = jnp.asarray(r0["u0"])[None]; nu1b = jnp.asarray([r0["nu"]])
        tb_med, _ = wu.time_fn(lambda: roll_tb(U0b, nu1b)[0].block_until_ready(),
                               TIME_REPS, TIME_WARM)
        log(f"   ROM cost: pre {pre_med*1e3:.1f} ms + IC {ic_med*1e3:.1f} ms "
            f"+ rollout {roll_med*1e3:.1f} ms + decode {dec_med*1e3:.1f} ms   "
            f"(training-IC bank built offline in {bank_build_s:.1f} s; "
            f"testbed rollout {tb_med*1e3:.1f} ms)")

        for tau in FOM_TAUS:
            chain, _res = make_chain(n, tau)
            chk = fom_reference_check(n, trajs, chain)
            chk.update(N=n, fom_tau=tau)
            report["checks"].append(chk)
            log(f"   tau={tau:.0e}: counting chain vs testbed rollout: traj-norm "
                f"{chk['rel_diff_vs_testbed_rollout']:.2e}, worst step "
                f"{chk['rel_diff_max_over_steps']:.2e}; BiCGStab vs jax.scipy "
                f"{chk['linear_solver']['rel_diff_vs_jax_scipy_bicgstab']:.2e} "
                f"(lin resid ours {chk['linear_solver']['ours_rel_lin_residual']:.1e} "
                f"vs jax {chk['linear_solver']['jax_rel_lin_residual']:.1e})")
            # The testbed's fixed-8-Newton scan converges to ~1e-12 per step, while this
            # chain stops at `tau`, so the two trajectories legitimately differ by about
            # the per-step tolerance accumulated over T steps.  The check must therefore
            # scale with tau; it is a test that the DISCRETE OPERATOR is the same one, and
            # at the tightest tau in the ladder it is a stringent one.
            op_tol = max(1e-8, 20.0 * tau)
            chk["operator_agreement_threshold"] = op_tol
            if chk["rel_diff_max_over_steps"] > op_tol:
                raise SystemExit(f"N={n} tau={tau}: the counting Newton chain does not "
                                 f"reproduce the testbed rollout (worst step "
                                 f"{chk['rel_diff_max_over_steps']:.2e} > {op_tol:.1e}) "
                                 f"-- refusing to time it")
            if chk["linear_solver"]["ours_rel_lin_residual"] > 1e3 * LIN_TOL:
                raise SystemExit(f"N={n}: the counting BiCGStab left a relative linear "
                                 f"residual {chk['linear_solver']['ours_rel_lin_residual']:.2e}")

            # GPU burn-in before the arm loop: the three arms are already timed back
            # to back, but the preceding NNLS-EQ fit is a long CPU-bound phase during
            # which the device idles and drops clocks (see wsf_util.gpu_burn).
            wu.gpu_burn(lambda: chain(u0j, r0["nu"], jnp.zeros((T, n2), F64),
                                      jnp.int32(0))[0].block_until_ready(), BURN_S)
            arm_out = {}
            for arm in ARMS:
                mode = jnp.int32(ARM_ID[arm])
                per_traj = []
                for j, r in enumerate(rom_per_traj):
                    Gj = jnp.asarray(r["G"]) if arm == "rom" else jnp.zeros((T, n2), F64)
                    U_, KK, NL, NB, FL, RR = chain(jnp.asarray(r["u0"]), r["nu"], Gj, mode)
                    U_ = np.asarray(U_)
                    per_traj.append(dict(
                        newton=np.asarray(KK).tolist(), lin=np.asarray(NL).tolist(),
                        brk=np.asarray(NB).tolist(), flags=np.asarray(FL).tolist(),
                        rel_res=np.asarray(RR).tolist(),
                        err_vs_fom=float(np.linalg.norm(U_ - r["U"][1:])
                                         / np.linalg.norm(r["U"][1:]))))
                G0 = jnp.asarray(r0["G"]) if arm == "rom" else jnp.zeros((T, n2), F64)
                med, all_ = wu.time_fn(
                    lambda: chain(u0j, r0["nu"], G0, mode)[0].block_until_ready(),
                    TIME_REPS, TIME_WARM)
                newton = np.array([p["newton"] for p in per_traj], float)
                lin = np.array([p["lin"] for p in per_traj], float)
                arm_out[arm] = dict(
                    t_ms=med * 1e3, t_all=all_,
                    # the wall clock is trajectory 0 only, so its own counts are kept
                    # beside the mean over all N_TEST_TRAJ trajectories
                    newton_total_timed_traj=float(newton[0].sum()),
                    lin_total_timed_traj=float(lin[0].sum()),
                    newton_total=float(newton.sum(1).mean()),
                    lin_total=float(lin.sum(1).mean()),
                    newton_per_step=newton.mean(0).tolist(),
                    lin_per_step=lin.mean(0).tolist(),
                    newton_step1=float(newton[:, 0].mean()),
                    lin_step1=float(lin[:, 0].mean()),
                    breakdowns=int(np.sum([np.sum(p["brk"]) for p in per_traj])),
                    flags_nonzero=int(np.sum([np.sum(np.asarray(p["flags"]) != 0)
                                              for p in per_traj])),
                    max_rel_newton_residual=float(np.max([np.max(p["rel_res"])
                                                          for p in per_traj])),
                    all_finite=bool(np.all(np.isfinite(np.concatenate(
                        [np.asarray(p["rel_res"]) for p in per_traj])))),
                    err_vs_fom=float(np.mean([p["err_vs_fom"] for p in per_traj])))
                a = arm_out[arm]
                log(f"     arm {arm:7s}: {a['newton_total']:.1f} Newton, "
                    f"{a['lin_total']:.0f} BiCGStab iters, "
                    f"{med*1e3:.1f} ms, err vs FOM {a['err_vs_fom']:.2e}, "
                    f"max rel Newton resid {a['max_rel_newton_residual']:.2e}, "
                    f"lin failures {a['breakdowns']}, "
                    f"nonzero Newton flags {a['flags_nonzero']}")
                # SOLVER HEALTH GATE: a configuration is publishable only if EVERY
                # step of EVERY arm actually met the Newton tolerance with finite
                # arithmetic.  A cheap failed warm solve must never be able to
                # contribute a headline speedup (Codex MUST FIX).
                if not a["all_finite"]:
                    raise SystemExit(f"N={n} tau={tau} arm {arm}: non-finite Newton residual")
                if a["max_rel_newton_residual"] > tau:
                    raise SystemExit(
                        f"N={n} tau={tau} arm {arm}: max relative Newton residual "
                        f"{a['max_rel_newton_residual']:.3e} exceeds the tolerance -- "
                        f"the FOM finish did not converge, refusing to publish the row")
                if a["flags_nonzero"] or a["breakdowns"]:
                    # The two conditions above (non-finite, or the Newton tolerance not
                    # met) are the ones that make a row invalid, and they abort.  A
                    # BiCGStab breakdown or linear max-iteration that the outer Newton
                    # still recovered from is NOT silently dropped: it is counted, logged
                    # loudly here, and carried into the row as health_warning so it lands
                    # in the README.
                    a["health_warning"] = (f"{a['flags_nonzero']} non-zero Newton flags, "
                                           f"{a['breakdowns']} linear-solver failures "
                                           f"(BiCGStab breakdown or max-iteration)")
                    log(f"     !! N={n} tau={tau} arm {arm}: {a['health_warning']} -- "
                        f"the Newton tolerance was still met at every step, so the row is "
                        f"kept, but the occurrence is recorded")

            t_rom_ms = (ic_med + roll_med) * 1e3
            t_dec_ms = dec_med * 1e3
            t_pre_ms = pre_med * 1e3
            t_total = t_pre_ms + t_rom_ms + t_dec_ms + arm_out["rom"]["t_ms"]
            row = dict(
                pde="burgers2d", N=n, n_dof=n2, rom_tau=bc.GN_TOL, fom_tau=tau,
                t_rom_ms=t_rom_ms, t_pre_ms=t_pre_ms, t_rom_ic_ms=ic_med * 1e3,
                t_fom_testbed_ms=tb_med * 1e3,
                offline_train_ic_bank_s=bank_build_s,
                t_rom_rollout_ms=roll_med * 1e3, t_decode_ms=t_dec_ms,
                t_fom_ms=arm_out["rom"]["t_ms"], t_total_ms=t_total,
                t_fom_baseline_ms=arm_out["prev"]["t_ms"],
                t_fom_extrap_ms=arm_out["extrap"]["t_ms"],
                iters_from_rom=arm_out["rom"]["newton_total"],
                iters_from_rom_timed=arm_out["rom"]["newton_total_timed_traj"],
                iters_from_baseline_timed=arm_out["prev"]["newton_total_timed_traj"],
                iters_from_extrap_timed=arm_out["extrap"]["newton_total_timed_traj"],
                lin_iters_from_rom_timed=arm_out["rom"]["lin_total_timed_traj"],
                lin_iters_from_baseline_timed=arm_out["prev"]["lin_total_timed_traj"],
                iters_from_baseline=arm_out["prev"]["newton_total"],
                iters_from_extrap=arm_out["extrap"]["newton_total"],
                lin_iters_from_rom=arm_out["rom"]["lin_total"],
                lin_iters_from_baseline=arm_out["prev"]["lin_total"],
                lin_iters_from_extrap=arm_out["extrap"]["lin_total"],
                err_rel_l2_rom=rom_err_mean, rom_latent_iters=rom_iters_mean,
                rom_ic_rel_on_eq=float(np.mean([r["ic_rel_on_eq"]
                                                for r in rom_per_traj])),
                err_final=arm_out["rom"]["err_vs_fom"],
                err_final_baseline=arm_out["prev"]["err_vs_fom"],
                speedup_vs_fom=arm_out["prev"]["t_ms"] / t_total,
                speedup_fom_stage_only=arm_out["prev"]["t_ms"] / arm_out["rom"]["t_ms"],
                bicgstab_breakdowns=sum(arm_out[a]["breakdowns"] for a in ARMS),
                newton_flags_nonzero=sum(arm_out[a]["flags_nonzero"] for a in ARMS),
                max_rel_newton_residual={a: arm_out[a]["max_rel_newton_residual"]
                                         for a in ARMS},
                arm_all_finite={a: arm_out[a]["all_finite"] for a in ARMS},
                health_warning={a: arm_out[a].get("health_warning") for a in ARMS
                                if arm_out[a].get("health_warning")} or None,
                m=int(ops["m"]), variant=VARIANT, n_traj=N_TEST_TRAJ,
                run_role=RUN_ROLE,
                seed=bc.SEED, gpu=prov["gpu"], gpu_kind=prov["gpu_kind"],
                jax_backend=prov["jax_backend"], commit=prov["commit"],
                slurm_job_id=prov["slurm_job_id"])
            report["rows"].append(row)
            report["per_step"].append(dict(
                N=n, fom_tau=tau, run_role=RUN_ROLE,
                newton_per_step={a: arm_out[a]["newton_per_step"] for a in ARMS},
                lin_per_step={a: arm_out[a]["lin_per_step"] for a in ARMS},
                rom_err_per_step=np.mean([r["rom_err"] for r in rom_per_traj],
                                         axis=0).tolist()))
            save()
            log(f"   tau={tau:.0e}: total {t_total:.0f} ms vs FOM {arm_out['prev']['t_ms']:.0f} ms "
                f"({row['speedup_vs_fom']:.2f}x)  Newton {row['iters_from_rom']:.1f} vs "
                f"{row['iters_from_baseline']:.1f} (extrap {row['iters_from_extrap']:.1f})")
        log(f"== N={n} done [{time.time()-t_n0:.0f}s]")

    report["complete"] = True
    save()
    log("DONE")


if __name__ == "__main__":
    main()
