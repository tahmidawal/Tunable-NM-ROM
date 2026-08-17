"""Shared machinery for the ROM-WARM-STARTED-FOM cell (2026-08-17).

Question: hand the ROM's decoded field to the FULL-ORDER solver as its INITIAL
GUESS and finish to full accuracy.  The answer is then FOM-exact by construction,
so the only open question is COST.

This module holds the pieces both arms need:
  * `time_fn`      -- the project's timing protocol (warm-ups, median of reps).
  * `provenance`   -- git commit / GPU / backend / precision stamped on every row.
  * `make_cg`      -- a jitted, ITERATION-COUNTING conjugate-gradient solver for
                      the Poisson FOM.  ONE function object is used for BOTH the
                      warm-started and the zero-start arm; only `x0` differs, so
                      the stopping test, the operator, the compilation and the
                      warm-up are identical by construction.  (The testbed's own
                      `jax.scipy.sparse.linalg.cg` cannot report iterations, so it
                      is used as the CORRECTNESS REFERENCE -- `cg_reference_check`
                      asserts the two agree -- not as the timed baseline.)

Nothing here re-implements a PDE operator: the operator is always passed in from
the reference harness (`ms_parametric.neg_lap_interior` for Poisson, the
`burgers2d_film` backward-Euler residual for Burgers).
"""
from __future__ import annotations

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


# ------------------------------------------------------------------ provenance
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
                jax_backend=jax.default_backend(),
                gpu=str(dev),
                gpu_kind=getattr(dev, "device_kind", "unknown"),
                jax_version=jax.__version__,
                x64=bool(jax.config.jax_enable_x64),
                matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
                slurm_job_id=os.environ.get("SLURM_JOB_ID", "local"),
                host=os.environ.get("HOSTNAME", os.uname().nodename))


# ------------------------------------------------------------------ counting CG
def make_cg(op, maxiter=20000):
    """Jitted conjugate gradient for the SPD operator `op` with an ITERATION
    COUNT and a NaN/breakdown guard.

    cg(b, x0, tau) -> (x, iters, rel_res_recursive, flag)

    Stopping test: ||r_k||_2 <= tau * ||b||_2 on the RECURSIVELY updated residual
    (the textbook test; the TRUE residual of the returned iterate is measured
    separately by the caller and reported alongside).  `tau` and `x0` are runtime
    ARGUMENTS, so the warm-started arm and the zero-start arm run the SAME
    compiled kernel with the SAME stopping test -- the single easiest way to fake
    this experiment is to give the two arms different tests, and this makes that
    impossible.

    flag: 0 converged, 1 maxiter, 2 breakdown (non-finite alpha or p^T A p <= 0).
    """
    def cg(b, x0, tau):
        bn = jnp.linalg.norm(b)
        tol = tau * bn
        r = b - op(x0)
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

        x, r, p, rs, k, bad = jax.lax.while_loop(
            cond, body, (x0, r, p, rs, jnp.int32(0), jnp.bool_(False)))
        flag = jnp.where(bad, jnp.int32(2),
                         jnp.where(k >= maxiter, jnp.int32(1), jnp.int32(0)))
        return x, k, jnp.sqrt(rs) / jnp.maximum(bn, 1e-300), flag

    return jax.jit(cg)


def cg_reference_check(op, b, tau, cg_jit, ref_tol=1e-13, ref_maxiter=100_000):
    """Cross-check the counting CG against the testbed's own
    `jax.scipy.sparse.linalg.cg` (the function that produced the reference data).
    Returns a dict of agreement diagnostics; the caller asserts on it."""
    ref = jax.jit(lambda bb: jax.scipy.sparse.linalg.cg(
        op, bb, tol=ref_tol, maxiter=ref_maxiter)[0])(b)
    x, k, rr, flag = cg_jit(b, jnp.zeros_like(b), tau)
    true_res = float(jnp.linalg.norm(op(x) - b) / jnp.linalg.norm(b))
    return dict(rel_diff_vs_jax_scipy_cg=float(jnp.linalg.norm(x - ref)
                                               / jnp.linalg.norm(ref)),
                counting_cg_iters=int(k), counting_cg_true_rel_res=true_res,
                counting_cg_recursive_rel_res=float(rr), counting_cg_flag=int(flag),
                jax_scipy_cg_true_rel_res=float(jnp.linalg.norm(op(ref) - b)
                                                / jnp.linalg.norm(b)))
