"""jcp-dims certificate helpers (DESIGN A2.1, A3.3) shared by the 2D and 3D drivers.

  * square-system LSPG (kappa = 1, M = r): tolerance levels with a residual alternative (A2.1) and the weighted
    residual certificate eta_w = ||D R_n(c_{n+1})|| / scale <= 1e-7 (scale = 0: certified iff the raw weighted residual
    is exactly 0);
  * G-cert by exact inversion of the solver's record tolratio = r_n / (tol + 1e-300): r_rec = tolratio * (tol + 1e-300),
    finite required, |r_cert - r_rec| <= 1e-6 r_rec + 1e-15 (A3.3), used for square LSPG and for GAL.
"""
from __future__ import annotations

import numpy as np

ETA_W_BAR = 1e-7          # 100 x the production tolf 1e-9 (A2.1)
TOL_SQUARE_LSPG = {'prod': (1e-3, 1e-9), 'tight': (1e-7, 1e-13), 'tighter': (1e-8, 1e-15)}


def tol_of(form, level, square=False):
    if form == 'LSPG' and square:
        return TOL_SQUARE_LSPG[level]
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


def eta_w(raw, scale):
    raw, sc = np.asarray(raw, float), float(scale)
    if sc > 0:
        return raw / sc
    return np.where(raw == 0, 0., np.inf)


def certified_square(raw, scale):
    e = eta_w(raw, scale)
    return np.isfinite(e) & (e <= ETA_W_BAR)


def recovered(tolratio, tol):
    return np.asarray(tolratio, float) * (np.asarray(tol, float) + 1e-300)


def gcert(raw_cert, tolratio, tol):
    """Per-step G-cert (A3.3); tol may be a scalar (square LSPG: tolf * scale) or per-step (GAL: tolf * ||A c_n||)."""
    rc = np.asarray(raw_cert, float)
    rr = recovered(tolratio, tol)
    if not rc.size:
        return False
    return bool(np.all(np.isfinite(rc) & np.isfinite(rr) & (np.abs(rc - rr) <= 1e-6 * np.abs(rr) + 1e-15)))


def make_square_diag(nl):
    """A2.1 diagnostics for square LSPG (M = r), per step from the saved states: cond of the weighted step Jacobian
    (autodiff) and the unweighted relative residual ||R_n(c_{n+1})|| / ||A c_n|| (reported, no verdict)."""
    import jax
    import jax.numpy as jnp
    import p1core as T2
    be = jnp.asarray(T2.BE_CO, jnp.float64)

    def one(k, states, nu, data, sch):
        dt = sch['dt']
        co = jnp.where(k < sch['nstart'], be, sch['co'])
        ck, cm, cn = states[k], states[jnp.maximum(k - 1, 0)], states[k + 1]
        A, lam = data['A'], data['lam']
        Fn = jnp.where(co[4] != 0., nl(ck, data) + nu * lam * (A @ ck), 0.)
        r = T2.lspg_residual(nl, co, dt, nu, data, (A @ ck, A @ cm, Fn))
        J = jax.jacfwd(r)(cn)
        s = jnp.linalg.svd(J, compute_uv=False)
        D = co[0] + dt * co[3] * nu * lam
        return s[0] / s[-1], jnp.linalg.norm(r(cn) * D) / (jnp.linalg.norm(A @ ck) + 1e-300)

    def diag(states, nu, data, sch):
        S = states.shape[0] - 1
        return jax.lax.map(lambda k: one(k, states, nu, data, sch), jnp.arange(S))
    return jax.jit(diag)
