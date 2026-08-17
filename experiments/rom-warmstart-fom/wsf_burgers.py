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
    flag: 0 converged, 1 maxiter, 2 breakdown (rho, rhat^T v or t^T t underflowed,
    or a non-finite quantity appeared)."""
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

        def cond(s):
            x, r, p, v, rho, alpha, omega, k, bad = s
            return (jnp.linalg.norm(r) > thr) & (k < maxiter) & (~bad)

        def body(s):
            x, r, p, v, rho, alpha, omega, k, bad = s
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
            t_ = A(s_)
            tt = jnp.sum(t_ * t_)
            brk = brk | (tt < tiny)
            omega_new = jnp.sum(t_ * s_) / jnp.where(brk, 1.0, tt)
            x_new = x + alpha_new * p_new + omega_new * s_
            r_new = s_ - omega_new * t_
            nonfinite = ~(jnp.all(jnp.isfinite(x_new)) & jnp.all(jnp.isfinite(r_new)))
            brk = brk | nonfinite
            # on a breakdown freeze the state (the caller sees flag 2 and the last
            # good iterate) rather than propagating a NaN into the Newton step
            keep = lambda new, old: jnp.where(brk, old, new)
            return (keep(x_new, x), keep(r_new, r), keep(p_new, p), keep(v_new, v),
                    keep(rho_new, rho), keep(alpha_new, alpha), keep(omega_new, omega),
                    k + 1, brk)

        x, r, p, v, rho, alpha, omega, k, bad = jax.lax.while_loop(
            cond, body, (x, r, p, v, rho, alpha, omega, jnp.int32(0), jnp.bool_(False)))
        flag = jnp.where(bad, jnp.int32(2),
                         jnp.where(k >= maxiter, jnp.int32(1), jnp.int32(0)))
        return x, k, flag
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

    newton_flag: 0 converged, 1 max_newton reached, 2 non-finite Newton step,
    3 non-finite residual."""
    _, residual = bc.bf.make_rollout(n)
    bicg = make_bicgstab(LIN_TOL, LIN_MAXITER)

    def step(u_prev, u_prev2, g, mode, nu):
        u_start = jnp.where(mode == 0, u_prev,
                            jnp.where(mode == 1, 2.0 * u_prev - u_prev2, g))
        tol_abs = tol_rel * jnp.linalg.norm(u_prev)
        r0 = residual(u_start, u_prev, nu)
        init = (u_start, r0, jnp.linalg.norm(r0), jnp.int32(0), jnp.int32(0),
                jnp.int32(0), jnp.int32(0))

        def cond(s):
            u, r, rn, k, nl, nbrk, flag = s
            return (rn > tol_abs) & (k < MAX_NEWTON) & (flag == 0)

        def body(s):
            u, r, rn, k, nl, nbrk, flag = s
            Jv = lambda vv: jax.jvp(lambda uu: residual(uu, u_prev, nu), (u,), (vv,))[1]
            du, li, lflag = bicg(Jv, -r)
            ok = jnp.all(jnp.isfinite(du))
            u2 = jnp.where(ok, u + du, u)
            r2 = residual(u2, u_prev, nu)
            rn2 = jnp.linalg.norm(r2)
            fin = jnp.isfinite(rn2)
            new_flag = jnp.where(~ok, jnp.int32(2), jnp.where(~fin, jnp.int32(3), jnp.int32(0)))
            return (jnp.where(fin, u2, u), jnp.where(fin, r2, r),
                    jnp.where(fin, rn2, rn), k + 1, nl + li,
                    nbrk + (lflag == 2).astype(jnp.int32), new_flag)

        u, r, rn, k, nl, nbrk, flag = jax.lax.while_loop(cond, body, init)
        flag = jnp.where((flag == 0) & (rn > tol_abs), jnp.int32(1), flag)
        return u, k, nl, nbrk, flag, rn / jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

    def chain(u0, nu, guesses, mode):
        def body(carry, g):
            u_prev, u_prev2 = carry
            u, k, nl, nbrk, flag, rr = step(u_prev, u_prev2, g, mode, nu)
            return (u, u_prev), (u, k, nl, nbrk, flag, rr)
        _, (U, KK, NL, NB, FL, RR) = jax.lax.scan(body, (u0, u0), guesses)
        return U, KK, NL, NB, FL, RR

    return jax.jit(chain), residual


def fom_reference_check(n, u0, nu, chain, residual):
    """The counting Newton chain (previous-step arm) must reproduce the testbed's
    own fixed-8-iteration rollout -- otherwise the 'FOM' being timed here is not
    the FOM that produced the data."""
    roll, _ = bc.bf.make_rollout(n)
    snaps, rr = roll(jnp.asarray(u0)[None], jnp.asarray([nu]))
    U_ref = np.asarray(snaps)[1:, 0]                       # (T, n^2), steps 1..T
    dummy = jnp.zeros((T, n * n), F64)
    U, KK, NL, NB, FL, RR = chain(jnp.asarray(u0), nu, dummy, jnp.int32(0))
    U = np.asarray(U)
    rel = float(np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref))
    return U_ref, dict(rel_diff_vs_testbed_rollout=rel,
                       testbed_max_rel_newton_residual=float(jnp.max(rr)),
                       counting_chain_max_rel_newton_residual=float(jnp.max(RR)),
                       counting_chain_newton_iters=int(np.sum(np.asarray(KK))),
                       counting_chain_lin_iters=int(np.sum(np.asarray(NL))),
                       counting_chain_lin_breakdowns=int(np.sum(np.asarray(NB))),
                       counting_chain_flags=sorted(set(np.asarray(FL).tolist())))


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
        json.dump(report, open(OUT, "w"), indent=1, default=float)

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
        dec_all = jax.jit(lambda ZZ: jax.vmap(lambda zz: dec(zz, coords))(ZZ))

        rom_per_traj = []
        for j, (U, nu, _) in enumerate(trajs):
            u0 = U[0]
            u0_rms = float(np.sqrt(np.mean(u0[interior] ** 2)))
            jj, _d = fu.nearest_train_ic(n, u0)
            Z0 = jnp.asarray(np.stack([Ztr.mean(axis=0)[0], Ztr[jj, 0]]))
            z_e, rel_e, nJ_e, b_e, att_e = fit_eq(jnp.asarray(u0), Z0)
            usc = jnp.full((T,), bc.GN_TOL * u0_rms * ops.get("tol_scale", np.sqrt(ops["m"])))
            Zr, rn_, nj_, re_ = ops["rollout_jit"](z_e, nu, usc, bc.GN_BUDGET)
            G = np.asarray(dec_all(Zr))                       # (T, n^2) guesses, t=1..T
            err = np.linalg.norm(G - U[1:], axis=1) / np.linalg.norm(U[1:], axis=1)
            rom_per_traj.append(dict(z_e=z_e, Z0=Z0, u0=u0, u0_rms=u0_rms, nu=nu,
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
        ic_med, _ = wu.time_fn(lambda: fit_eq(u0j, r0["Z0"])[0].block_until_ready(),
                               TIME_REPS, TIME_WARM)
        usc0 = jnp.full((T,), bc.GN_TOL * r0["u0_rms"] * ops.get("tol_scale", np.sqrt(ops["m"])))
        roll_med, _ = wu.time_fn(
            lambda: ops["rollout_jit"](r0["z_e"], r0["nu"], usc0, bc.GN_BUDGET)[0].block_until_ready(),
            TIME_REPS, TIME_WARM)
        Zr0 = ops["rollout_jit"](r0["z_e"], r0["nu"], usc0, bc.GN_BUDGET)[0]
        dec_med, _ = wu.time_fn(lambda: dec_all(Zr0).block_until_ready(), TIME_REPS, TIME_WARM)
        log(f"   ROM cost: IC {ic_med*1e3:.1f} ms + rollout {roll_med*1e3:.1f} ms "
            f"+ decode {dec_med*1e3:.1f} ms")

        for tau in FOM_TAUS:
            chain, residual = make_chain(n, tau)
            U_ref_chk, chk = fom_reference_check(n, r0["u0"], r0["nu"], chain, residual)
            chk.update(N=n, fom_tau=tau)
            report["checks"].append(chk)
            log(f"   tau={tau:.0e}: counting chain vs testbed rollout rel diff "
                f"{chk['rel_diff_vs_testbed_rollout']:.2e} "
                f"({chk['counting_chain_newton_iters']} Newton, "
                f"{chk['counting_chain_lin_iters']} BiCGStab iters, "
                f"{chk['counting_chain_lin_breakdowns']} breakdowns)")
            if chk["rel_diff_vs_testbed_rollout"] > 1e-8:
                raise SystemExit(f"N={n} tau={tau}: the counting Newton chain does not "
                                 f"reproduce the testbed rollout -- refusing to time it")

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
                    err_vs_fom=float(np.mean([p["err_vs_fom"] for p in per_traj])))
                log(f"     arm {arm:7s}: {arm_out[arm]['newton_total']:.1f} Newton, "
                    f"{arm_out[arm]['lin_total']:.0f} BiCGStab iters, "
                    f"{med*1e3:.1f} ms, err vs FOM {arm_out[arm]['err_vs_fom']:.2e}, "
                    f"breakdowns {arm_out[arm]['breakdowns']}, "
                    f"nonzero flags {arm_out[arm]['flags_nonzero']}")

            t_rom_ms = (ic_med + roll_med) * 1e3
            t_dec_ms = dec_med * 1e3
            t_total = t_rom_ms + t_dec_ms + arm_out["rom"]["t_ms"]
            row = dict(
                pde="burgers2d", N=n, n_dof=n2, rom_tau=bc.GN_TOL, fom_tau=tau,
                t_rom_ms=t_rom_ms, t_rom_ic_ms=ic_med * 1e3,
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
