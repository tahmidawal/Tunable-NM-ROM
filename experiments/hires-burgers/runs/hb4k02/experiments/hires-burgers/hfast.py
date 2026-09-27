"""hires-burgers: the corrected (q > 0) Burgers query with a structured Jacobian, and the
Phi-free dense truth query.

Same reduced state, weak residual, block-damped Levenberg-Marquardt iteration, initializer
policy, stopping rule and output contract as `topfix.make_query(..., arm='base')`:

    u(w) = G ( h_theta(z) + C y ),    w = (z, y) in R^{K+q},
    r(w) = S ( A h - p + dt ( Pq^T a(G5 h) + nu Lambda A h ) ),   S = 1 / (1 + dt nu Lambda).

What changes is only how the arithmetic is emitted (DESIGN.md section 6):

  fold    h is affine in y, so A C and G5 C are formed once offline; the y columns of the
          stencil values and of A h never touch the R = 512 bank dimension online.
  ajac    the Jacobian is assembled analytically,
              J = [A Jh | A C] + dt S Pq^T ( D . [G5 Jh | G5 C] ),
          with Jh = d h_theta / dz (16 tangents through the head only) and D = d a / d us the
          (m, 5) pointwise derivative of the upwind flux; forward mode through the whole
          residual pushes K + q tangents through the R-wide bank instead.
  fuse    residual and Jacobian are evaluated together at the candidate, once per iteration;
          the audited path evaluates the residual, then re-evaluates residual and Jacobian
          under a `lax.cond` when the step is accepted (b-speed's `fuse`, 1.25x at q = 0).
  hoist   the QR of R C used by the exact y-elimination of the initial fit is formed offline.
  nodiag  the per-step `blocks()` stationarity diagnostic (two extra forward-mode Jacobians
          per time step) is dropped; the LM's own exit gradient is the same quantity,
          ||J^T r|| / (||J||_F ||r||), and is returned in its place.
  decfuse the six output fields are one G [h(w_1) .. h(w_6)] product.
  solver  'lu' (the audited `jnp.linalg.solve`) or 'chol' (Cholesky of the damped SPD normal
          matrix; a different factorisation, so a labelled arm).

Reassociation-class parity is the most this module can have: fields are gated at a stated
relative tolerance and the integer iteration / exit-reason vectors are compared and REPORTED;
every arm's error is in any case measured directly against the same-job full-order solve.

The dense truth query uses the same fused LM with the exact (all-node) advection projected by
`hops.sep_project` and the Jacobian by chunked linearisation, so no (n, M) matrix and no
(n, K+q) tangent block is ever resident.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import sep_common as sc
import hops as H


def make_fused_lm(evalJ, K, q, budget, trust, gtol, ridge, solver='lu'):
    """`varpro.make_block_lm` with one (r, J) evaluation per iteration."""
    mask_z = jnp.concatenate((jnp.ones(K), jnp.zeros(q)))
    mask_y = 1. - mask_z

    def solve(Hm, g):
        if solver == 'chol':
            c = jax.scipy.linalg.cho_factor(Hm, lower=True)
            return jax.scipy.linalg.cho_solve(c, -g)
        return jnp.linalg.solve(Hm, -g)

    def ratio(g, J, rn):
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * rn + 1e-300)

    def lm(w0, args, tol):
        r, J = evalJ(w0, args)
        rn = jnp.linalg.norm(r)
        g = J.T @ r
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(ratio(g, J, rn) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, g, rn, lam, it, reason, rej = s
            Hm = J.T @ J
            d = jnp.diag(Hm) + 1e-30
            step = solve(Hm + jnp.diag(lam * d * mask_z + ridge * d * mask_y), g)
            ok = jnp.all(jnp.isfinite(step)) & (jnp.linalg.norm(step[:K]) <= trust)
            wn = w + jnp.where(ok, step, 0.)
            r2, J2 = evalJ(wn, args)
            rn2 = jnp.linalg.norm(r2)
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            w3 = jnp.where(accept, wn, w)
            r3 = jnp.where(accept, r2, r)
            J3 = jnp.where(accept, J2, J)
            rn3 = jnp.where(accept, rn2, rn)
            g3 = J3.T @ r3
            gn = ratio(g3, J3, rn3)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn3 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            return (w3, r3, J3, g3, rn3, lam2, it + 1, reason, rej + (~accept).astype(jnp.int32))

        w, r, J, g, rn, lam, it, reason, rej = jax.lax.while_loop(
            lambda s: (s[6] < budget) & (s[7] == 0), body,
            (w0, r, J, g, rn, jnp.asarray(1e-6), jnp.int32(0), reason, jnp.int32(0)))
        return w, rn, it, reason, ratio(g, J, rn), rej
    return lm


def build_tables(params, C, K, data, cold):
    """Offline folds for the EQ kernel. `data` carries A, lam, G, G5, Pq."""
    Cq = jnp.asarray(C)
    Rc = cold[3]
    tab = dict(AC=data['A'] @ Cq, C=Cq)
    if 'G5' in data:
        tab['G5C'] = jnp.einsum('msr,rq->msq', data['G5'], Cq)
    if Cq.shape[1]:
        Qr, Rr = jnp.linalg.qr(Rc @ Cq, mode='reduced')
        tab.update(Qr=Qr, Rr=Rr, RC=Rc @ Cq)
    return jax.block_until_ready(tab)


def _head_parts(params, K):
    hz = lambda z: sc.head(params, z)
    return hz, jax.jacfwd(hz)


def make_eq_eval(params, K, q, L, dt):
    hz, jh = _head_parts(params, K)
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))

    def parts(w, data, tab):
        z, y = w[:K], w[K:]
        h = hz(z)
        us = jnp.einsum('msr,r->ms', data['G5'], h) + tab['G5C'] @ y
        ah = data['A'] @ h + tab['AC'] @ y
        return z, h, us, ah

    def res(w, prev, nu, data, tab):
        _, _, us, ah = parts(w, data, tab)
        lam = data['lam']
        return (ah - prev + dt * (data['Pq'].T @ H.advect_points(us, L) + nu * lam * ah)) / (1 + dt * nu * lam)

    def evalJ(w, args):
        prev, nu, data, tab = args
        z, h, us, ah = parts(w, data, tab)
        lam = data['lam']
        S = 1. / (1 + dt * nu * lam)
        r = (ah - prev + dt * (data['Pq'].T @ H.advect_points(us, L) + nu * lam * ah)) * S
        Jh = jh(z)                                                   # (R, K)
        Us = jnp.concatenate((jnp.einsum('msr,rk->msk', data['G5'], Jh), tab['G5C']), axis=2)
        dA = jnp.einsum('ms,msk->mk', dadv(us), Us)                  # (m, K+q)
        J = jnp.concatenate((data['A'] @ Jh, tab['AC']), axis=1) + (dt * S)[:, None] * (data['Pq'].T @ dA)
        return r, J
    return res, evalJ, hz


def make_dense_eval(params, C, K, q, L, dt, tangent_chunks):
    """Exact (all-node) advection; `data` carries A, lam, G, sx, sy. No Phi."""
    hz = lambda z: sc.head(params, z)
    d = K + q
    assert d % tangent_chunks == 0, (d, tangent_chunks)

    def res(w, prev, nu, data, tab):
        h = hz(w[:K]) + tab['C'] @ w[K:]
        adv = e.spatial(data['G'] @ h, L)[0].reshape(L - 1, L - 1)
        ah = data['A'] @ h
        lam = data['lam']
        return (ah - prev + dt * (H.sep_project(adv, data['sx'], data['sy'], L) + nu * lam * ah)) / (1 + dt * nu * lam)

    def evalJ(w, args):
        prev, nu, data, tab = args
        r, lin = jax.linearize(lambda ww: res(ww, prev, nu, data, tab), w)
        basis = jnp.eye(d).reshape(tangent_chunks, d // tangent_chunks, d)
        J = jax.lax.map(lambda B: jax.vmap(lin)(B), basis).reshape(d, -1).T
        return r, J
    return res, evalJ, hz


def make_query(params, C, K, q, L, dt, trust, quadrature, ic_budget=400, step_budget=600, gtol=1e-6,
               ic_gtol=1e-6, ridge=1e-10, solver='lu', tangent_chunks=1, decode='fused', parts=False):
    """Supplied dense field on GPU -> six dense GPU fields; `topfix.make_query`'s tuple layout,
    with slot 8/12/14 all carrying the LM's joint exit gradient and slot 15 the rejected-step
    count per time step (the damping retries)."""
    if quadrature == 'eq':
        res, evalJ, hz = make_eq_eval(params, K, q, L, dt)
    else:
        res, evalJ, hz = make_dense_eval(params, C, K, q, L, dt, tangent_chunks)
    lm = make_fused_lm(evalJ, K, q, step_budget, trust, gtol, ridge, solver)
    if q:
        ic_lm = A.make_stationary_lm(lambda z, tgt, Rm, Qr: (lambda v: v - Qr @ (Qr.T @ v))(Rm @ hz(z) - tgt),
                                     ic_budget, gtol=ic_gtol, linear='gj')
    else:
        ic_lm = A.make_stationary_lm(lambda z, tgt, Rm: Rm @ hz(z) - tgt, ic_budget, gtol=ic_gtol, linear='gj')
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    head = lambda w, tab: hz(w[:K]) + tab['C'] @ w[K:]

    def initialize(u0, data, cold, tab):
        xy, wq, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ yv)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        if q:
            z, icrn, icit, icreason, icgn = ic_lm(Zcand[idx][:K], (yv, R, tab['Qr']), 0.)
            ycor = jax.scipy.linalg.solve_triangular(tab['Rr'], tab['Qr'].T @ (yv - R @ hz(z)), lower=False)
            w0 = jnp.concatenate((z, ycor))
            icres = R @ head(w0, tab) - yv
            icJz = jax.jacfwd(lambda zz: R @ hz(zz))(z)
            icgj = jnp.linalg.norm(jnp.concatenate((icJz.T @ icres, tab['RC'].T @ icres))) / (
                jnp.sqrt(jnp.linalg.norm(icJz) ** 2 + jnp.linalg.norm(tab['RC']) ** 2) * jnp.linalg.norm(icres) + 1e-300)
        else:
            w0, icrn, icit, icreason, icgn = ic_lm(Zcand[idx][:K], (yv, R), 0.)
            icgj = icgn
        return w0, scale, icit, icreason, icgn, icrn, jnp.linalg.norm(ui), icgj

    def ahead(w, data, tab):
        return data['A'] @ hz(w[:K]) + tab['AC'] @ w[K:] if quadrature == 'eq' else data['A'] @ head(w, tab)

    def evolve(w0, nu, scale, data, tab):
        def step(carry, _):
            wv, wprev = carry
            p = ahead(wv, data, tab)
            we = wv + (wv - wprev)
            r0 = jnp.linalg.norm(res(wv, p, nu, data, tab))
            re = jnp.linalg.norm(res(we, p, nu, data, tab))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            w2, rn, it, reason, gn, rej = lm(wi, (p, nu, data, tab), 1e-9 * scale)
            return (w2, wv), (w2, rn, it, reason, gn, rej)
        _, out = jax.lax.scan(step, (w0, w0), None, length=steps)
        return out

    def decode_fields(W, data, tab):
        if decode == 'fused':
            Hm = jax.vmap(lambda v: head(v, tab))(W)                 # (6, R)
            U = (data['G'] @ Hm.T).T.reshape(len(W), L - 1, L - 1)
            return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))
        return jax.vmap(lambda v: e.output_field(data['G'] @ head(v, tab), L, L))(W)

    def query(u0, nu, data, cold, tab):
        w0, scale, icit, icreason, icgn, icrn, uin, icgj = initialize(u0, data, cold, tab)
        ws, rn, it, reason, gn, rej = evolve(w0, nu, scale, data, tab)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = decode_fields(W, data, tab)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn, icrn, uin, gn, icgj, gn, rej)

    if parts:
        return jax.jit(query), dict(initialize=jax.jit(initialize), evolve=jax.jit(evolve),
                                    decode=jax.jit(decode_fields), evalJ=jax.jit(evalJ), res=jax.jit(res))
    return jax.jit(query)
