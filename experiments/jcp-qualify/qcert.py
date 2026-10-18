"""jcp-qualify certificates beyond P1 (DESIGN 4.1b, 4.3): the Galerkin (GAL) per-step certificate and the Galerkin step
Jacobian for conditioning. The LSPG certificate and Jacobian are P1's (`p1core.make_certifier`, `p1core.make_jac`),
used unchanged.

GAL step n solves Q^T R_n(c) = 0 (A = Q Rm, thin QR), R_n the two-step LMM residual of p1core.make_evolve. From the saved
internal trajectory c_0..c_S the certifier rebuilds every step's Galerkin residual (scheme coefficients incl. start-up
steps, history A c_n, A c_{n-1}, F(c_n)) and evaluates, at the returned state c_{n+1}:
    eta_n   = ||Q^T R_n(c_{n+1})|| / ||A c_n||          (the solver's own tolerance normalisation)
    gamma_n = ||J_G^T r|| / (||J_G||_F ||r|| + 1e-300)  (the literal least-squares ratio; reported, no verdict)
    cond_n  = cond(J_G),  J_G = d(Q^T R_n)/dc by forward-mode autodiff (not the solver's analytic Jacobian),
plus the raw residual norm and the scale ||A c_n|| (A2.5: a zero-scale step is certified iff its raw residual is 0).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import p1core as T2

ETA_BAR = 1e-8


def gal_residual(nl, co, dt, nu, data, hist):
    a0, a1, a2, b0, b1 = co[0], co[1], co[2], co[3], co[4]
    Acn, Acm, Fn = hist
    lam = data['lam']

    def r(c):
        Aw = data['A'] @ c
        R = a0 * Aw + a1 * Acn + a2 * Acm + dt * (b0 * (nl(c, data) + nu * lam * Aw) + b1 * Fn)
        return data['Qm'].T @ R
    return r


def make_gal_certifier(nl, Rp):
    be = jnp.asarray(T2.BE_CO, jnp.float64)

    def one(k, states, nu, data, sch):
        dt = sch['dt']
        co = jnp.where(k < sch['nstart'], be, sch['co'])
        ck, cm, cn = states[k], states[jnp.maximum(k - 1, 0)], states[k + 1]
        A, lam = data['A'], data['lam']
        Fn = jnp.where(co[4] != 0., nl(ck, data) + nu * lam * (A @ ck), 0.)
        Acn = A @ ck
        r = gal_residual(nl, co, dt, nu, data, (Acn, A @ cm, Fn))
        rv = r(cn)
        J = jax.jacfwd(r)(cn)
        rn = jnp.linalg.norm(rv)
        s = jnp.linalg.svd(J, compute_uv=False)
        eta = rn / (jnp.linalg.norm(Acn) + 1e-300)
        gam = jnp.linalg.norm(J.T @ rv) / (jnp.linalg.norm(J) * rn + 1e-300)
        return eta, gam, s[0] / s[-1], rn, jnp.linalg.norm(Acn)

    def cert(states, nu, data, sch):
        S = states.shape[0] - 1
        return jax.lax.map(lambda k: one(k, states, nu, data, sch), jnp.arange(S))
    return jax.jit(cert)


def make_gal_jac(nl):
    """History-free Galerkin step Jacobian J_G = a0 Rm + dt b0 (Q^T dN + nu Q^T Lambda A) (autodiff), for Q3."""
    def jac(c, co, dt, nu, data):
        M = data['A'].shape[0]
        z = jnp.zeros(M, jnp.float64)
        return jax.jacfwd(gal_residual(nl, co, dt, nu, data, (z, z, z)))(c)
    return jax.jit(jac)


def recorded_eta(tolratio, scale, tolf):
    """The solver's own eta: tolratio = rn / (tolf ||A c_n|| + 1e-300) -> rn / ||A c_n|| = tolratio * tolf (scale > 0).
    Zero-scale (exceptional) steps: rn = tolratio * 1e-300, reported as +inf unless rn == 0 (A2.5)."""
    import numpy as np
    tr, sc = np.asarray(tolratio, float), np.asarray(scale, float)
    raw = tr * 1e-300
    return np.where(sc > 0, tr * tolf, np.where(raw == 0, 0., np.inf))


def certified_gal(eta):
    import numpy as np
    e = np.asarray(eta, float)
    return np.isfinite(e) & (e <= ETA_BAR)    # NaN / +-inf never certified


def gcert_gal(eta, raw, scale, tolratio, tolf):
    """G-cert for GAL (A1.3, A2.5): eta agreement for positive scale; raw-norm agreement for zero scale."""
    import numpy as np
    eta, raw, sc, tr = (np.asarray(x, float) for x in (eta, raw, scale, tolratio))
    if not eta.size:
        return False
    pos = sc > 0
    rec = tr * tolf
    ok_pos = np.isfinite(eta) & np.isfinite(rec) & (np.abs(eta - rec) <= 1e-6 * np.abs(rec) + 1e-13)
    raw_rec = tr * 1e-300
    ok_zero = np.isfinite(tr) & np.isfinite(raw) & (np.abs(raw - raw_rec) <= 1e-6 * np.abs(raw_rec) + 1e-15)
    return bool(np.all(np.where(pos, ok_pos, ok_zero)))
