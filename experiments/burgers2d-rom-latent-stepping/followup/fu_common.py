"""Follow-up helpers for the Burgers-2D latent-stepping ROM: a fully JITTED
LM data-misfit solver (used for the cold-start IC fit; the original
`blat_common.fit_ic` is a Python-loop LM that dominated end-to-end timing)."""
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
    """Returns jitted lm(z0, tol_abs) -> (z, rn, n_jac, accepted, reason, attempts)
    minimising ||f(z)|| with the SAME LM acceptance/damping rule as
    blat_common._finish_ops.lm_step_jit (lax.while_loop, budget static)."""
    rJ = lambda z: (f(z), jax.jacfwd(f)(z))
    rn_fn = lambda z: jnp.linalg.norm(f(z))

    def lm(z0, tol_abs):
        r0, J0 = rJ(z0)
        rn0 = jnp.linalg.norm(r0)
        init_reason = jnp.where(~jnp.isfinite(rn0), jnp.int32(5),
                                jnp.where(rn0 <= tol_abs, jnp.int32(4), jnp.int32(0)))
        init = (z0, r0, J0, rn0, jnp.asarray(1e-6, F64), jnp.int32(0),
                jnp.int32(0), jnp.int32(1), init_reason)

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, r, J, rn, lam, att, acc, nJ, _ = s
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            tiny = finite & (jnp.linalg.norm(dz) <= 1e-12 * (1.0 + jnp.linalg.norm(z)))
            z_new = z + jnp.where(finite, dz, 0.0)
            rn_new = rn_fn(z_new)
            accept = finite & jnp.isfinite(rn_new) & (rn_new < rn)
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new), lambda: (r, J))
            rel_dec = jnp.where(accept, (rn - rn_new) / rn, 1.0)
            z = jnp.where(accept, z_new, z)
            rn = jnp.where(accept, rn_new, rn)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12),
                            jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32)
            nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (rn <= tol_abs), 1,
                      jnp.where((accept & (rel_dec < 1e-12)) | tiny, 2,
                       jnp.where((~accept) & (lam >= 1e12), 3, 0))).astype(jnp.int32)
            return (z, r2, J2, rn, lam, att + 1, acc, nJ, reason)

        z, r, J, rn, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, rn, nJ, acc, reason, att

    return jax.jit(lm)


def make_fit_ic_jit(dec, n, budget):
    """Jitted cold start: LM on the data misfit to the KNOWN u0 from S inits at
    once (vmap), best-of.  Returns fit(u0 (n^2,), Z0 (S,K)) -> (z, rel, nJ_total)."""
    coords = jnp.asarray(bc.grid_coords(n))
    K = dec.k

    def one(u0, z0):
        f = lambda z: dec(z, coords) - u0
        lm = lm_jit_solver(f, K, budget)
        return lm(z0, jnp.asarray(0.0, F64))

    def fit(u0, Z0):
        zs, rns, nJs, accs, reasons, atts = jax.vmap(lambda z0: one(u0, z0))(Z0)
        b = jnp.argmin(rns)
        return zs[b], rns[b] / jnp.linalg.norm(u0), jnp.sum(nJs), b

    return jax.jit(fit)


def median_time(fn, reps=7, warm=2):
    return bc.time_fn(fn, reps=reps, warm=warm)
