"""burgers-eqcert: hires-burgers' `hfast.make_query` with an optional exact-first-steps phase.

Source: `experiments/hires-burgers/hfast.py` @ 0ab60014 (imported, unmodified). With `exact_steps = j > 0` the
first j backward-Euler steps are solved with the exact (all-node) weak residual through the separable
projection (`hfast.make_dense_eval`), the remaining 50 - j with the EQ residual (`hfast.make_eq_eval`). Both
phases use the same fused LM, stopping rule, damping carry-over and predictor; the damping and the predictor
history are carried across the phase boundary. With j = 0 this is `hfast.make_query(..., 'eq', ...)` and with
quadrature='dense' it is the exact-residual query; both paths are delegated to hfast unchanged.

`data` for j > 0 carries the EQ arrays (A, lam, G, G5, Pq) AND the separable tables (sx, sy); `tab` must be
built from the EQ data (it then has AC, C, G5C, Qr, Rr, RC, which is a superset of what the dense path reads).
Output tuple layout identical to `hfast.make_query`.
"""
from __future__ import annotations

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import hfast as HF
import hops as H


def make_query(params, C, K, q, L, dt, trust, quadrature, exact_steps=0, ic_budget=400, step_budget=600, gtol=1e-6,
               ic_gtol=1e-6, ridge=1e-10, solver='lu', tangent_chunks=1, parts=False, clip=False, lam_carry=False,
               predictor='lin'):
    if exact_steps == 0 or quadrature != 'eq':
        return HF.make_query(params, C, K, q, L, dt, trust, quadrature, ic_budget=ic_budget, step_budget=step_budget,
                             gtol=gtol, ic_gtol=ic_gtol, ridge=ridge, solver=solver, tangent_chunks=tangent_chunks,
                             parts=parts, clip=clip, lam_carry=lam_carry, predictor=predictor)
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    assert 0 < exact_steps < steps
    res_q, evalJ_q, hz = HF.make_eq_eval(params, K, q, L, dt)
    res_d, evalJ_d, _ = HF.make_dense_eval(params, C, K, q, L, dt, tangent_chunks)
    lm_q = HF.make_fused_lm(evalJ_q, K, q, step_budget, trust, gtol, ridge, solver, clip)
    lm_d = HF.make_fused_lm(evalJ_d, K, q, step_budget, trust, gtol, ridge, solver, clip)
    # initializer and decoder: exactly hfast's (they do not depend on the quadrature)
    _, hp = HF.make_query(params, C, K, q, L, dt, trust, 'eq', ic_budget=ic_budget, step_budget=step_budget,
                          gtol=gtol, ic_gtol=ic_gtol, ridge=ridge, solver=solver, parts=True)
    initialize, decode_fields = hp['initialize'], hp['decode']

    def ahead(w, data, tab):
        return data['A'] @ hz(w[:K]) + tab['AC'] @ w[K:]

    def make_step(res, lm, nu, scale, data, tab):
        def step(carry, k):
            wv, wprev, wprev2, lam0 = carry
            p = ahead(wv, data, tab)
            we = wv + (wv - wprev)
            if predictor == 'quad':
                wq = jnp.where(k >= 2, 3. * wv - 3. * wprev + wprev2, we)
                cand = jnp.stack((wv, we, wq))
                rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, p, nu, data, tab)))(cand)
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                wi = cand[jnp.argmin(rs)]
            else:
                r0 = jnp.linalg.norm(res(wv, p, nu, data, tab))
                re = jnp.linalg.norm(res(we, p, nu, data, tab))
                wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            w2, rn, it, reason, gn, rej, lam = lm(wi, (p, nu, data, tab), 1e-9 * scale, lam0)
            return (w2, wv, wprev, lam if lam_carry else lam0), (w2, rn, it, reason, gn, rej)
        return step

    def evolve(w0, nu, scale, data, tab):
        carry = (w0, w0, w0, jnp.asarray(1e-6, dtype=jnp.float64))
        carry, o1 = jax.lax.scan(make_step(res_d, lm_d, nu, scale, data, tab), carry, jnp.arange(exact_steps))
        _, o2 = jax.lax.scan(make_step(res_q, lm_q, nu, scale, data, tab), carry, jnp.arange(exact_steps, steps))
        return tuple(jnp.concatenate((a, b)) for a, b in zip(o1, o2))

    def query(u0, nu, data, cold, tab):
        w0, scale, icit, icreason, icgn, icrn, uin, icgj = initialize(u0, data, cold, tab)
        ws, rn, it, reason, gn, rej = evolve(w0, nu, scale, data, tab)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = decode_fields(W, data, tab)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn, icrn, uin, gn, icgj, gn, rej)

    if parts:
        return jax.jit(query), dict(initialize=initialize, evolve=jax.jit(evolve), decode=decode_fields,
                                    evalJ=jax.jit(evalJ_q), res=jax.jit(res_q))
    return jax.jit(query)
