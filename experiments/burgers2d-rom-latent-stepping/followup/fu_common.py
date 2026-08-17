"""Follow-up helpers for the Burgers-2D latent-stepping ROM.

`lm_jit_solver` is an EXACT `lax.while_loop` port of `ms_autodecoder.lm_solve`
-- the Python-loop LM that `blat_common.fit_ic` uses for the online cold start
(0.6-1.3 s, which dominated end-to-end timing).  "Exact" means: same initial
Jacobian evaluation, same damping schedule (lam0=1e-6, /3 on accept, x10 on
reject, clamped to [1e-12, 1e12]), same acceptance test (finite and strictly
decreasing ||r||), same TWO stopping tests applied ONLY after an accepted step
(relative decrease < 1e-12, or ||dz||/(1+||z_old||) < 1e-13), same
lambda-saturation aborts, and the same accounting (attempts, accepted,
rejected, residual evaluations, Jacobian evaluations).  There is no absolute
residual tolerance in the reference solver and there is none here.

Reason codes: 0 budget, 1 converged, 2 lambda_max, 3 nan_step_lambda_max,
4 nan_at_init.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for d in (PARENT,):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import jax
import jax.numpy as jnp

import blat_common as bc                       # noqa: E402  (x64 enabled there)
from blat_common import F64                    # noqa: E402


def lm_jit_solver(f, K, budget):
    """Jitted lm(z0) -> (z, rn, n_jac, n_res, accepted, rejected, attempts,
    reason) minimising ||f(z)||, algorithmically identical to
    ms_autodecoder.lm_solve (see the module docstring)."""
    rJ = lambda z: (f(z), jax.jacfwd(f)(z))
    rn_fn = lambda z: jnp.linalg.norm(f(z))

    def lm(z0):
        r0, J0 = rJ(z0)
        rn0 = jnp.linalg.norm(r0)
        # state: z, r, J, rn, lam, attempts, accepted, rejected, n_res, n_jac, reason
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), jnp.int32(1),
                jnp.where(jnp.isfinite(rn0), jnp.int32(0), jnp.int32(4)))

        def cond(s):
            return (s[10] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, rej, n_r, n_J, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            # --- non-finite step: damp, count a rejection, do NOT evaluate the residual
            z_new = z + jnp.where(finite, dz, 0.0)
            rn_new = jnp.where(finite, rn_fn(z_new), jnp.inf)
            accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))   # ||z_old||, as in the reference
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new), lambda: (r, J))
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            rej = rej + (~accept).astype(jnp.int32)
            n_J = n_J + accept.astype(jnp.int32)
            # residual evaluations: +1 for the trial (only when the step was finite),
            # +1 more for the rJ re-evaluation after an accepted step
            n_r = n_r + finite.astype(jnp.int32) + accept.astype(jnp.int32)
            reason = jnp.where(
                accept & ((rel_dec < 1e-12) | (step < 1e-13)), jnp.int32(1),
                jnp.where((~accept) & (lam >= 1e12),
                          jnp.where(finite, jnp.int32(2), jnp.int32(3)), jnp.int32(0)))
            return (z, r2, J2, rn, lam, att + 1, acc, rej, n_r, n_J, reason)

        z, r, J, rn, lam, att, acc, rej, n_r, n_J, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, n_J, n_r, acc, rej, att, reason

    return jax.jit(lm)


def make_fit_ic_jit(dec, n, budget, coords=None, idx=None, w=None):
    """Jitted cold start: LM on the data misfit to the KNOWN u0 from S inits at
    once (vmap), best-of by the smallest final ||r|| -- the same selection rule
    as blat_common.fit_ic.

    idx=None (default): the misfit is taken over the FULL grid, exactly as
    blat_common.fit_ic does.  That costs O(n) per iteration, so the cold start is
    the one piece of the online path that is NOT n-free.

    idx=(m,) interior flat indices with quadrature weights w=(m,): the SAME
    hyper-reduction as the rollout -- the misfit is the weighted least squares
    sum_q w_q (u(z, x_q) - u0(x_q))^2 over the m EQ nodes, so the cold start
    becomes n-free too.  u0 is still supplied on the grid (the ROM is given the
    initial condition); only the m sampled values are used.

    Returns fit(u0 (n^2,), Z0 (S,K)) ->
    (z, rel_misfit_on_the_fitted_points, n_jac_total, best_index, attempts_total)."""
    coords = jnp.asarray(bc.grid_coords(n)) if coords is None else coords
    K = dec.k
    if idx is None:
        pts = coords
        sel = None
        sw = None
    else:
        sel = jnp.asarray(np.asarray(idx))
        pts = coords[sel]
        sw = jnp.sqrt(jnp.asarray(np.asarray(w) if w is not None else np.ones(len(idx)), F64))

    def one(u0q, z0):
        if sw is None:
            f = lambda z: dec(z, pts) - u0q
        else:
            f = lambda z: sw * (dec(z, pts) - u0q)
        return lm_jit_solver(f, K, budget)(z0)

    def fit(u0, Z0):
        u0q = u0 if sel is None else u0[sel]
        if sw is not None:
            u0n = jnp.linalg.norm(sw * u0q)
        else:
            u0n = jnp.linalg.norm(u0q)
        zs, rns, nJs, nrs, accs, rejs, atts, reasons = jax.vmap(lambda z0: one(u0q, z0))(Z0)
        b = jnp.argmin(rns)
        return zs[b], rns[b] / u0n, jnp.sum(nJs), b, jnp.sum(atts)

    return jax.jit(fit)


def nearest_train_ic(n, u0_test, chunk=64):
    """Index of the TRAIN trajectory whose INITIAL FIELD is closest to u0_test in
    L2 at resolution n -- the identical rule to blat_rom.py (which uses the
    N=64 fields).  The ROM knows u0, so this is legitimate online information."""
    cx, cy, w, a, nu, _ = bc.bf.sample_params()
    best_j, best_d = -1, np.inf
    u0_test = np.asarray(u0_test).reshape(-1)
    for s in range(0, bc.N_TRAIN, chunk):
        e = min(s + chunk, bc.N_TRAIN)
        U0 = np.stack([np.asarray(bc.bf.blob_ic(n, cx[i], cy[i], w[i], a[i])).reshape(-1)
                       for i in range(s, e)])
        d = np.linalg.norm(U0 - u0_test, axis=1)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d, best_j = float(d[j]), s + j
    return best_j, best_d


def median_time(fn, reps=7, warm=2):
    return bc.time_fn(fn, reps=reps, warm=warm)
