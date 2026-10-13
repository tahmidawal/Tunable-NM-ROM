"""jcp-references: finite-difference reference solver (DESIGN §3.1, §3.2, A1.3, A1.13, A1.15, A2.1).

One streaming solver for d = 2, 3 on L intervals per axis (interior state (L-1,)*d, zero Dirichlet walls):

    stage:  u - u_n + theta*tau*F(u) + (1-theta)*tau*F(u_n) - tau*(theta*g(t_n+tau) + (1-theta)*g(t_n)) = 0,
    F(u) = N_h(u) - nu * Lap_h(u),

  N_h central  u_c * sum_a (u_{+a} - u_{-a}) / (2h)                  (the references)
      upwind   the vendor sign-upwind stencil, arithmetic identical to engines.spatial (2D) / b3d_common (3D)
  Lap_h        the vendor 5-/7-point Laplacian, arithmetic identical to the vendors
  g            manufactured forcing (gate G2 only; None in every production call)

Schedule (traced): the first `nstart` stages are BE (theta=1) of length dt/2 (Rannacher start: nstart=4 replaces the
first two CN steps), then stages of length dt with implicitness theta_main (1/2 CN, 1 BE). Time is counted in units
of dt/2; outputs at t_k = 0.05 k, k = 0..5.
Each stage: Newton from the previous state, BiCGStab on the JVP, preconditioned by the exact DST-I inverse of
(1 + theta*tau*nu*Lambda_h); stop when ||res|| <= ntol ||u_n|| or max_newton or non-finite (the vendor rule).
Per stage the final relative residual and Newton count are recorded; the running max |u| over all stages too.
Only the six output states are kept on the device (streaming). f64; every large array is a traced argument.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import cohorts as CH  # noqa: E402

E2, C3 = CH.E2, CH.C3
NST = 1300                      # per-stage evidence buffer (longest schedule: 4 + 1198 stages, 2D L=12288)
LIN_MAXITER = {2: 200, 3: 2000}  # the vendors' BiCGStab limits
TIMES = np.arange(6) * 0.05


# ------------------------------------------------------------------------------------------------- operators --

def _pad(u):
    return jnp.pad(u, 1)


def adv_upwind(u, L):
    d = u.ndim
    p = _pad(u)
    if d == 2:          # engines.spatial, verbatim arithmetic
        c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        return c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))
    dx = 1.0 / L        # b3d_common.upwind_adv_field_3d, verbatim arithmetic
    c = p[1:-1, 1:-1, 1:-1]
    pos = c > 0
    ux = jnp.where(pos, (c - p[:-2, 1:-1, 1:-1]) / dx, (p[2:, 1:-1, 1:-1] - c) / dx)
    uy = jnp.where(pos, (c - p[1:-1, :-2, 1:-1]) / dx, (p[1:-1, 2:, 1:-1] - c) / dx)
    uz = jnp.where(pos, (c - p[1:-1, 1:-1, :-2]) / dx, (p[1:-1, 1:-1, 2:] - c) / dx)
    return c * (ux + uy + uz)


def adv_central(u, L):
    d = u.ndim
    p = _pad(u)
    if d == 2:
        c = p[1:-1, 1:-1]
        s = (p[2:, 1:-1] - p[:-2, 1:-1]) + (p[1:-1, 2:] - p[1:-1, :-2])
    else:
        c = p[1:-1, 1:-1, 1:-1]
        s = ((p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) + (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1])
             + (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]))
    return c * s * (L / 2.0)


def lap(u, L):
    d = u.ndim
    p = _pad(u)
    if d == 2:          # engines.spatial
        c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        return L ** 2 * (xm + xp + ym + yp - 4 * c)
    dx = 1.0 / L        # b3d_common.lap_3d (zscale = 1)
    c = p[1:-1, 1:-1, 1:-1]
    lxy = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] - 4.0 * c)
    lz = (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] - 2.0 * c)
    return (lxy + 1.0 * lz) / dx ** 2


def dstn(u):
    """Orthonormal self-inverse DST-I on every axis (2D: engines.dst2; 3D: common.dst3)."""
    return E2.dst2(u) if u.ndim == 2 else C3.dst3(u)


def node_coords(L, d):
    x = jnp.arange(1, L, dtype=jnp.float64) / L
    return jnp.meshgrid(*([x] * d), indexing='ij')


# ---------------------------------------------------------------------------------------------------- solver --

def make_solver(d, L, adv='central', forcing=None, max_newton=20, lin_maxiter=None):
    """Return run(u0, nu, dt, scheme, ntol, ltol) -> dict(out (6, n), res (NST,), nwt (NST,), umax, stages, units).

    The state is FLAT (n = (L-1)^d, x slowest), as in both vendor FOMs. sched (traced): dt, nstart (int), theta (main
    implicitness), steps (total time = steps*dt; the Rannacher start replaces the first two), keep_units (= 2*0.05/dt),
    ntol, ltol. forcing: None or g(t, X) -> grid array, X the list of interior coordinate arrays (G2 only)."""
    shape = (L - 1,) * d
    advf = adv_central if adv == 'central' else adv_upwind
    lin_maxiter = LIN_MAXITER[d] if lin_maxiter is None else lin_maxiter

    def F(u, nu):
        g = u.reshape(shape)
        return (advf(g, L) - nu * lap(g, L)).reshape(-1)

    def query(u0, nu, sched, lam):
        dt, nstart, theta_m = sched['dt'], sched['nstart'], sched['theta']
        total_units = 2 * sched['steps']
        if forcing is not None:
            X = node_coords(L, d)
            g = lambda t: forcing(t, X).reshape(-1)
        else:
            g = lambda t: jnp.zeros(u0.shape, jnp.float64)

        def stage(k, units, un):
            start = k < nstart
            tau = jnp.where(start, dt / 2, dt)
            th = jnp.where(start, 1.0, theta_m)
            tn = units * dt / 2
            Fn = F(un, nu)
            const = (1 - th) * tau * Fn - tau * (th * g(tn + tau) + (1 - th) * g(tn))
            res = lambda u: u - un + th * tau * F(u, nu) + const
            den = 1 + th * tau * nu * lam

            def pre(v):
                return dstn(dstn(v.reshape(shape)) / den).reshape(-1)
            scale = jnp.maximum(jnp.linalg.norm(un), 1e-300)

            def body(s):
                u, it, _ = s
                r = res(u)
                jv = lambda v: jax.jvp(res, (u,), (v,))[1]
                du, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=sched['ltol'], maxiter=lin_maxiter, M=pre)
                u = u + du
                return u, it + 1, jnp.linalg.norm(res(u))
            u, it, rn = jax.lax.while_loop(
                lambda s: (s[2] > sched['ntol'] * scale) & (s[1] < max_newton) & jnp.isfinite(s[2]), body,
                (un, jnp.int32(0), jnp.linalg.norm(res(un))))
            return u, it, rn / scale, jnp.where(start, 1, 2)

        def body(c):
            k, units, u, out, res_, nwt, umax = c
            u, it, rel, du = stage(k, units, u)
            units = units + du
            ks = jnp.minimum(k, NST - 1)
            res_ = res_.at[ks].set(rel)
            nwt = nwt.at[ks].set(it)
            umax = jnp.maximum(umax, jnp.max(jnp.abs(u)))
            store = (units % sched['keep_units']) == 0
            idx = jnp.clip(units // sched['keep_units'], 0, 5)
            out = jax.lax.cond(store, lambda o: o.at[idx].set(u), lambda o: o, out)   # in place
            return k + 1, units, u, out, res_, nwt, umax

        out0 = jnp.zeros((6,) + u0.shape, jnp.float64).at[0].set(u0)
        c0 = (jnp.int32(0), jnp.int32(0), u0, out0, jnp.full(NST, jnp.nan), jnp.full(NST, -1, jnp.int32),
              jnp.max(jnp.abs(u0)))
        k, units, _, out, res_, nwt, umax = jax.lax.while_loop(lambda c: c[1] < total_units, body, c0)
        return dict(out=out, res=res_, nwt=nwt, umax=umax, stages=k, units=units)

    if d == 2:          # engines.helmholtz's eigenvalues, same (jnp) arithmetic
        k1 = jnp.arange(1, L, dtype=jnp.float64)
        lam1 = 4 * L ** 2 * jnp.sin(jnp.pi * k1 / (2 * L)) ** 2
        lam = lam1[:, None] + lam1[None, :]
    else:               # common.lam1 / make_fom's spec, same (numpy) arithmetic
        l1 = jnp.asarray(C3.lam1(L + 1))
        lam = l1[:, None, None] + l1[None, :, None] + l1[None, None, :]
    jq = jax.jit(query)
    cache = {}

    def run(u0, nu, dt, scheme, ntol, ltol):
        """Ahead-of-time compiled on first use (compile time recorded in run.compile_seconds, 0 afterwards)."""
        sched = schedule(dt, scheme, ntol, ltol)
        args = (jnp.asarray(u0, jnp.float64).reshape(-1), jnp.float64(nu), sched, lam)
        run.compile_seconds = 0.0
        if 'exe' not in cache:
            t = time.perf_counter()
            cache['exe'] = jq.lower(*args).compile()
            run.compile_seconds = time.perf_counter() - t
        return cache['exe'](*args)
    run.compile_seconds = 0.0
    return run


def peak_bytes():
    """Device peak bytes in use so far (None where the backend does not report it)."""
    try:
        return (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use')
    except Exception:
        return None


SCHEMES = {'cnr': (4, 0.5), 'cn': (0, 0.5), 'be': (0, 1.0)}


def schedule(dt, scheme, ntol, ltol, horizon=0.25, spacing=0.05):
    nstart, theta = SCHEMES[scheme]
    steps, keep = int(round(horizon / dt)), int(round(spacing / dt))
    assert abs(steps * dt - horizon) < 1e-12 and abs(keep * dt - spacing) < 1e-12, dt
    assert n_stages(dt, scheme) <= NST
    return dict(dt=jnp.float64(dt), nstart=jnp.int32(nstart), theta=jnp.float64(theta), steps=jnp.int32(steps),
                keep_units=jnp.int32(2 * keep), ntol=jnp.float64(ntol), ltol=jnp.float64(ltol))


def n_stages(dt, scheme):
    nstart, _ = SCHEMES[scheme]
    steps = int(round(0.25 / dt))
    return nstart + steps - nstart // 2


# ------------------------------------------------------------------------------------------------ lattices ----

LATTICE = {2: 256, 3: 64}       # scoring lattice: intervals per axis (2D 257^2 incl. walls; 3D 63^3 interior)


def to_lattice(out, L, d):
    """(6, (L-1)^d) flat interior outputs -> scoring lattice: 2D (6, 257, 257) with zero walls; 3D (6, 63^3) flat."""
    m = LATTICE[d]
    s = L // m
    assert s * m == L, (L, m)
    out = out.reshape((6,) + (L - 1,) * d)
    if d == 2:
        a = jnp.pad(out, ((0, 0), (1, 1), (1, 1)))
        return np.asarray(a[:, ::s, ::s])
    idx = jnp.arange(1, m) * s - 1
    a = out[:, idx][:, :, idx][:, :, :, idx]
    return np.asarray(a.reshape(6, -1))


def ic_lattice(cohort, j):
    """Canonical t0 field on the scoring lattice (nodal samples at the lattice nodes)."""
    d = cohort.dim
    m = LATTICE[d]
    if d == 2:
        return np.asarray(CH.ic_2d(m, cohort.D[j]))
    return CH.ic_3d(m, cohort.T, j)
