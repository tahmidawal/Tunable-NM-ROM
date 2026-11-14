"""b2d-query-speed: lagged-operator solves of the linear-rung step problem (PROTOCOL.md section 4).

Same reduced model as burgers2d-speed/b2fast.make_linear_query (bank prefix, tests, rule, initial fit, decoder,
dt, 50 backward-Euler steps); only the way each step's nonlinear least-squares problem

    min_w || r(w) ||,   r(w) = A w - S.(A w_prev) + dt S.Pq^T adv(G5 w),   S = 1 / (1 + dt nu lam)

is solved differs. (This r is b2fast's res_q: (A w - prev + dt (Pq^T adv + nu lam A w)) S, since
S (1 + dt nu lam) = 1.)

Per step: an extrapolation predictor (no residual guard), then a FIXED number k of corrections with operators
built from a reference Jacobian J_ref that is refreshed only every s steps:

  mode 'chord'   w <- w - P r(w),  P = (J_ref^T J_ref + ridge diag)^-1 J_ref^T.
                 Fixed point: J_ref^T r(w) = 0 (Petrov-Galerkin with a LAGGED test space; not the exact
                 least-squares stationary point J(w)^T r(w) = 0 unless J_ref = J(w)).
  mode 'grad'    w <- w - H_ref^-1 J(w)^T r(w)  (exact gradient by reverse mode, lagged Gauss-Newton matrix).
                 Fixed point: the exact least-squares stationary point.
  jref 'J'       J_ref = J(w) at the refresh state (analytic, as b2fast.evalJ_q);
  jref 'A'       J_ref = A (the linear part; a Picard iteration; needs no Jacobian of the advection).

  form 'P'       explicit pseudo-inverse (one gemv per correction after the residual);
  form 'chol'    the Cholesky factor is kept and applied per correction;
  form 'fused'   (chord only) the correction is emitted as  w <- Kw w + Cp w_prev - Qt adv(G5 w)  with
                 Kw = I - P A, Cp = P (S.A), Qt = dt (P S) Pq^T built at each refresh (same arithmetic up to
                 reassociation).

`stencil32`: the stencil contraction G5 w and the pointwise advection are evaluated in float32 from
pre-differenced stencil columns (c, c - xm, xp - c, c - ym, yp - c); everything else stays float64.

Outputs follow b2fast's query tuple so the driver is unchanged: `it` is the fixed correction count per step,
`rn` is the relative size of the last correction of each step (a convergence indicator, not a residual norm),
`reason` is 1 everywhere.
"""
from __future__ import annotations

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import hops as H
import b2fast as B2


def stencil_diff32(G5):
    """(m, 5, R) stencil columns (c, xp, xm, yp, ym) -> float32 (c, c - xm, xp - c, c - ym, yp - c)."""
    c, xp, xm, yp, ym = (G5[:, i] for i in range(5))
    return jnp.stack((c, c - xm, xp - c, c - ym, yp - c), 1).astype(jnp.float32)


def make_query(Rp, L, dt, k=2, s=50, mode='chord', jref='J', form='P', pred='quad', kfirst=None, nfirst=2,
               stencil32=False, ridge=1e-10, emit='scan', batch=False, compiler_options=None):
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    assert steps % s == 0 and mode in ('chord', 'grad') and jref in ('J', 'A') and form in ('P', 'chol', 'fused')
    assert not (form == 'fused' and mode != 'chord') and pred in ('quad', 'lin', 'none') and emit in ('scan', 'unroll')
    if kfirst is None or kfirst == k:
        nfirst, kfirst = 0, k
    assert nfirst <= s
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))

    def adv_of(w, data):
        if stencil32:
            d = jnp.einsum('msr,r->ms', data['G5d32'], w.astype(jnp.float32))
            c = d[:, 0]
            a = c * L * (jnp.where(c > 0, d[:, 1], d[:, 2]) + jnp.where(c > 0, d[:, 3], d[:, 4]))
            return a.astype(jnp.float64)
        return H.advect_points(jnp.einsum('msr,r->ms', data['G5'], w), L)

    def PqT_vec(data, v):
        return B2.proj_vec(v, data['sepq']) if 'sepq' in data else data['Pq'].T @ v

    def PqT_cols(data, X):
        return B2.proj_cols(X, data['sepq']) if 'sepq' in data else data['Pq'].T @ X

    def Pq_cols(data, C):
        """Pq C for C (M, R) -> (m, R): the transpose of proj_cols through the distinct wave numbers."""
        if 'sepq' not in data:
            return data['Pq'] @ C
        T = data['sepq']
        Z = jnp.zeros((T['SX'].shape[1], T['SY'].shape[1], C.shape[1])).at[T['ix'], T['iy']].add(C)
        F = jnp.einsum('au,uvr,bv->abr', T['SX'], Z, T['SY'])
        return T['coef'] * F.reshape(-1, C.shape[1])

    def jac(w, S, data):
        if jref == 'A':
            return data['A']
        us = jnp.einsum('msr,r->ms', data['G5'], w)
        dA = jnp.einsum('ms,msr->mr', dadv(us), data['G5'])
        return data['A'] + (dt * S)[:, None] * PqT_cols(data, dA)

    def refresh(w, S, data):
        J = jac(w, S, data)
        Hm = J.T @ J
        c = jax.scipy.linalg.cho_factor(Hm + jnp.diag(ridge * (jnp.diag(Hm) + 1e-30)), lower=True)[0]
        if form == 'chol':
            return (c, J)
        if mode == 'grad':
            return (jax.scipy.linalg.cho_solve((c, True), jnp.eye(Rp)),)
        P = jax.scipy.linalg.cho_solve((c, True), J.T)                       # (R', M)
        if form == 'P':
            return (P,)
        PS = P * S[None, :]
        return (jnp.eye(Rp) - P @ data['A'], PS @ data['A'], dt * Pq_cols(data, PS.T).T)

    def res(w, Sp, S, data):
        return data['A'] @ w - Sp + (dt * S) * PqT_vec(data, adv_of(w, data))

    def correct(w, pre, ops, S, data):
        """One correction; returns the new iterate."""
        if form == 'fused':
            Kw, _, Qt = ops
            return Kw @ w + pre - Qt @ adv_of(w, data)
        if mode == 'grad':
            r, back = jax.vjp(lambda ww: res(ww, pre, S, data), w)
            g = back(r)[0]
            if form == 'chol':
                return w - jax.scipy.linalg.cho_solve((ops[0], True), g)
            return w - ops[0] @ g
        r = res(w, pre, S, data)
        if form == 'chol':
            return w - jax.scipy.linalg.cho_solve((ops[0], True), ops[1].T @ r)
        return w - ops[0] @ r

    def make_step(niter, ops, S, data):
        def step(carry, j):
            wv, w1, w2 = carry
            if pred == 'none':
                w = wv
            else:
                w = 2. * wv - w1
                if pred == 'quad':
                    w = jnp.where(j >= 2, 3. * wv - 3. * w1 + w2, w)
            pre = ops[1] @ wv if form == 'fused' else S * (data['A'] @ wv)
            wl = w
            for _ in range(niter):
                wl, w = w, correct(w, pre, ops, S, data)
            dn = jnp.linalg.norm(w - wl) / (jnp.linalg.norm(w) + 1e-300)
            return (w, wv, w1), (w, dn)
        return step

    def evolve(w0, nu, data):
        S = 1. / (1 + dt * nu * data['lam'])
        carry = (w0, w0, w0)
        if emit == 'unroll':
            ws, dns, ops = [], [], None
            for j in range(steps):
                if j % s == 0:
                    ops = refresh(carry[0], S, data)
                carry, (w, dn) = make_step(kfirst if j < nfirst else k, ops, S, data)(carry, j)
                ws.append(w)
                dns.append(dn)
            return jnp.stack(ws), jnp.stack(dns)
        ops = refresh(w0, S, data)
        ws, dns = [], []
        for j in range(nfirst):
            carry, (w, dn) = make_step(kfirst, ops, S, data)(carry, j)
            ws.append(w[None])
            dns.append(dn[None])
        if nfirst < s:
            carry, (w, dn) = jax.lax.scan(make_step(k, ops, S, data), carry, jnp.arange(nfirst, s))
            ws.append(w)
            dns.append(dn)
        if s < steps:
            def block(carry, b):
                ops = refresh(carry[0], S, data)
                return jax.lax.scan(make_step(k, ops, S, data), carry, b * s + jnp.arange(s))
            _, (w, dn) = jax.lax.scan(block, carry, jnp.arange(1, steps // s))
            ws.append(w.reshape(-1, Rp))
            dns.append(dn.reshape(-1))
        return jnp.concatenate(ws), jnp.concatenate(dns)

    def initialize(u0, cold):
        """b2fast.make_linear_query.initialize (unchanged initial fit)."""
        xy, wq, Q, R, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Q.T @ ui
        w0 = jax.scipy.linalg.solve_triangular(R, yv, lower=False)
        return w0, jnp.linalg.norm(R @ w0 - yv), jnp.linalg.norm(ui)

    def states(u0, nu, data, cold):
        w0, icrn, uin = initialize(u0, cold)
        ws, dn = evolve(w0, nu, data)
        return jnp.concatenate((w0[None], ws)), dn, icrn, uin

    def decode_fields(W, data):
        U = H.bank_apply(data['G'], W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold, tab):
        internal, dn, icrn, uin = states(u0, nu, data, cold)
        W = internal[::keep]
        fields = decode_fields(W, data)
        z = jnp.zeros(())
        zs = jnp.zeros(steps)
        return (fields, jnp.where(jnp.arange(steps) < nfirst, kfirst, k).astype(jnp.int32), dn, jnp.ones(steps, jnp.int32), W, jnp.int32(0), jnp.int32(1),
                internal, zs, z, icrn, uin, zs, z, zs, jnp.zeros(steps, jnp.int32))

    def query_batch(u0, nu, data, cold, tab):
        """B queries at once: u0 (B, L+1, L+1), nu (B,). Returns (fields (B, 6, L+1, L+1), dn (B, steps))."""
        internal, dn, _, _ = jax.vmap(lambda u, v: states(u, v, data, cold))(u0, nu)
        W = internal[:, ::keep]                                               # (B, 6, R')
        B = W.shape[0]
        U = H.bank_apply(data['G'], W.reshape(-1, Rp).T).T.reshape(B, W.shape[1], L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (0, 0), (1, 1), (1, 1))), dn

    fn = query_batch if batch else query
    return jax.jit(fn, compiler_options=compiler_options) if compiler_options else jax.jit(fn)


def make_probe(L, dt):
    """Diagnosis only: J at a state (analytic, b2fast.evalJ_q text) and the initial fit."""
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))

    @jax.jit
    def jac(w, nu, data):
        S = 1. / (1 + dt * nu * data['lam'])
        us = jnp.einsum('msr,r->ms', data['G5'], w)
        dA = jnp.einsum('ms,msr->mr', dadv(us), data['G5'])
        return data['A'] + (dt * S)[:, None] * B2.proj_cols(dA, data['sepq'])

    @jax.jit
    def init(u0, cold):
        xy, wq, Q, R, _, _, _ = cold
        return jax.scipy.linalg.solve_triangular(R, Q.T @ (e.sample_field(u0, xy, L) * wq), lower=False)
    return jac, init
