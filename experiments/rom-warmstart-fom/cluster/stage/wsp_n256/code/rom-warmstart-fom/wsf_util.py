"""Shared machinery for the ROM-WARM-STARTED-FOM cell (2026-08-17).

Question: hand the ROM's decoded field to the FULL-ORDER solver as its INITIAL
GUESS and finish to full accuracy.  The answer is then FOM-exact by construction,
so the only open question is COST.

This module holds the pieces both arms need:
  * `time_fn`      -- the project's timing protocol (warm-ups, median of reps).
  * `provenance`   -- git commit, a CONTENT HASH of the harness sources (a dirty
                      tree can otherwise publish different code under one commit),
                      GPU, backend, precision, Slurm job id.
  * `make_cg`      -- a jitted, iteration-counting conjugate-gradient solver with
                      TRUE-RESIDUAL RESTARTS.  ONE function object serves BOTH the
                      warm-started and the zero-start arm; only `x0` differs, so
                      the stopping test, the operator, the compilation and the
                      warm-up are identical by construction.  The returned iterate
                      is guaranteed to satisfy ||b - A x||/||b|| <= tau on the TRUE
                      residual, not merely on the recursively updated one -- the
                      recursive residual drifts differently along different
                      trajectories, so accepting it would make "FOM-exact to
                      tolerance" initial-guess dependent.
  * `cg_error_curve` -- a post-hoc diagnostic that grades SAVED iterates against a
                      reference solution.  It never stops a solver on the
                      reference, so the reference solution stays out of every
                      tolerance and every solve path.

Nothing here re-implements a PDE operator: the operator is always passed in from
the reference harness (`ms_parametric.neg_lap_interior` for Poisson, the
`burgers2d_film` backward-Euler residual for Burgers).
"""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

F64 = jnp.float64


# ------------------------------------------------------------------ timing
def time_fn(fn, reps=7, warm=2):
    """Median wall time of fn() (which MUST block on the device) after `warm`
    warm-ups.  Returns (median_seconds, [all seconds])."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


def gpu_burn(fn, seconds=3.0):
    """Run `fn` (which must block) for `seconds` to bring the GPU to a steady
    clock before any measurement.

    Why this exists: on an A100 at N=512 the FIRST timed arm measured 0.0468 ms
    per CG iteration and a later arm measured 0.0399 ms for identical work -- a
    17% systematic bias, in the direction that flatters whichever arm is timed
    LAST.  Both the instrumented and the library solver showed it, so it is the
    device ramping up after a long CPU-bound phase (the NNLS-EQ fit), not a code
    difference.  A bias of that size is larger than the entire effect this
    experiment is trying to measure, so every mesh burns in first AND times the
    two arms back to back (see wsf_poisson)."""
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        fn(); n += 1
    return n


# ------------------------------------------------------------------ provenance
def source_hashes(here=None, patterns=("wsf_*.py",)):
    """sha256 of every harness source actually present, so a DIRTY tree cannot
    publish two different codes under one commit hash."""
    here = here or os.path.dirname(os.path.abspath(__file__))
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(here, pat))):
            with open(p, "rb") as f:
                out[os.path.basename(p)] = hashlib.sha256(f.read()).hexdigest()[:16]
    return out


def provenance(here=None):
    here = here or os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", here, *a],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"
    dev = jax.devices()[0]
    return dict(commit=git("rev-parse", "HEAD"),
                commit_short=git("rev-parse", "--short", "HEAD"),
                dirty=git("status", "--porcelain"),
                source_sha256=source_hashes(here),
                jax_backend=jax.default_backend(),
                gpu=str(dev),
                gpu_kind=getattr(dev, "device_kind", "unknown"),
                jax_version=jax.__version__,
                x64=bool(jax.config.jax_enable_x64),
                matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
                slurm_job_id=os.environ.get("SLURM_JOB_ID", "local"),
                host=os.environ.get("HOSTNAME", os.uname().nodename))


# ------------------------------------------------------------------ counting CG
def make_cg(op, maxiter=20000, max_restarts=4):
    """Jitted conjugate gradient for the SPD operator `op`, counting iterations,
    guarding NaNs, and enforcing the tolerance on the TRUE residual.

        cg(b, x0, tau) -> (x, iters, true_rel_res, flag)

    Structure: an OUTER loop whose condition is the freshly computed true residual
    ||b - A x|| > tau ||b||, and an INNER textbook CG recursion that runs until the
    recursively updated residual meets the same threshold.  Recomputing the true
    residual (one extra matvec per restart) is what makes the delivered iterate
    genuinely accurate to `tau`; without it, "FOM-exact to tolerance" would depend
    on how far the recursive residual had drifted, which differs between a warm and
    a cold trajectory.

    `x0` and `tau` are RUNTIME ARGUMENTS, so the warm-started arm and the zero-start
    arm execute the same compiled kernel with the same stopping test.  Giving the
    two arms different tests is the single easiest way to fake this experiment, and
    this construction makes it impossible.

    flag: 0 converged, 1 iteration/restart budget exhausted, 2 breakdown or a
    non-finite quantity (including a non-finite input).
    """
    def cg(b, x0, tau):
        bn = jnp.linalg.norm(b)
        tol = tau * bn
        bad0 = ~(jnp.all(jnp.isfinite(b)) & jnp.all(jnp.isfinite(x0))
                 & jnp.isfinite(tau) & jnp.isfinite(bn))

        def inner(x):
            r = b - op(x)
            p = r
            rs = jnp.sum(r * r)

            def cond(s):
                x, r, p, rs, k, bad = s
                return (jnp.sqrt(rs) > tol) & (k < maxiter) & (~bad)

            def body(s):
                x, r, p, rs, k, bad = s
                Ap = op(p)
                pAp = jnp.sum(p * Ap)
                alpha = rs / pAp
                bad2 = bad | (~jnp.isfinite(alpha)) | (pAp <= 0.0)
                step = jnp.where(bad2, 0.0, alpha)
                x = x + step * p
                r = r - step * Ap
                rs_new = jnp.sum(r * r)
                beta = jnp.where(bad2, 0.0, rs_new / rs)
                p = r + beta * p
                return (x, r, p, rs_new, k + 1, bad2)

            return jax.lax.while_loop(cond, body, (x, r, p, rs, jnp.int32(0),
                                                   jnp.bool_(False)))

        def ocond(s):
            x, it, nr, bad = s
            tr = jnp.linalg.norm(b - op(x))
            return (tr > tol) & (nr <= max_restarts) & (~bad)

        def obody(s):
            x, it, nr, bad = s
            x2, r2, p2, rs2, k2, bad2 = inner(x)
            return (x2, it + k2, nr + 1, bad | bad2 | (k2 >= maxiter))

        x, it, nr, bad = jax.lax.while_loop(
            ocond, obody, (x0, jnp.int32(0), jnp.int32(0), bad0))
        true_res = jnp.linalg.norm(b - op(x)) / jnp.maximum(bn, 1e-300)
        ok = true_res <= tau
        flag = jnp.where(bad | ~jnp.isfinite(true_res), jnp.int32(2),
                         jnp.where(ok, jnp.int32(0), jnp.int32(1)))
        return x, it, true_res, flag

    return jax.jit(cg)


def cg_error_curve(op, b, u_ref, n_iters):
    """POST-HOC DIAGNOSTIC.  Run plain CG from a zero start for exactly `n_iters`
    iterations and GRADE the saved iterates against `u_ref`; the reference solution
    is never used to stop a solver, never enters a tolerance, and never touches the
    hybrid's critical path.  Answers: *how many plain CG iterations is the ROM's
    answer actually worth?*

    Returns the (n_iters+1,) array of relative L2 errors, starting at k = 0."""
    un = jnp.linalg.norm(u_ref)

    def body(s, _):
        x, r, p, rs = s
        Ap = op(p)
        alpha = rs / jnp.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.sum(r * r)
        p = r + (rs_new / rs) * p
        return (x, r, p, rs_new), jnp.linalg.norm(x - u_ref) / jnp.maximum(un, 1e-300)

    x0 = jnp.zeros_like(b)
    r0 = b - op(x0)
    _, errs = jax.lax.scan(body, (x0, r0, r0, jnp.sum(r0 * r0)), None, length=n_iters)
    return jnp.concatenate([jnp.array([1.0], F64), errs])


def cg_reference_check(op, b, tau, cg_jit, ref_maxiter=100_000):
    """Cross-check the counting CG against the testbed's own
    `jax.scipy.sparse.linalg.cg` (the function that produced the reference data) AT
    THE SAME TOLERANCE, and record the TRUE residual of both.  Also times a shared
    `jax.scipy` CG wrapper that takes `x0` at runtime, so the native library solver
    can be reported next to the counting one as a baseline sensitivity check."""
    ref_fn = jax.jit(lambda bb, xx: jax.scipy.sparse.linalg.cg(
        op, bb, x0=xx, tol=tau, maxiter=ref_maxiter)[0])
    ref = ref_fn(b, jnp.zeros_like(b))
    x, k, tr, flag = cg_jit(b, jnp.zeros_like(b), tau)
    bn = jnp.linalg.norm(b)
    return ref_fn, dict(
        tau=float(tau),
        rel_diff_vs_jax_scipy_cg=float(jnp.linalg.norm(x - ref)
                                       / jnp.maximum(jnp.linalg.norm(ref), 1e-300)),
        counting_cg_iters=int(k), counting_cg_true_rel_res=float(tr),
        counting_cg_flag=int(flag),
        jax_scipy_cg_true_rel_res=float(jnp.linalg.norm(op(ref) - b) / bn))
