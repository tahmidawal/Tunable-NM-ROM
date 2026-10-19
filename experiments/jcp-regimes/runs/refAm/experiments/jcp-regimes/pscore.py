"""jcp-references: independent pseudo-spectral cross-check solver (DESIGN §3.3, A1.6, A2.3, A2.6, A2.7, A3.4, A3.6).

State: continuous sine-series coefficients b_k, u(x) = sum_k b_k prod_a sin(k_a pi x_a), 1 <= k_a <= N-1.
Linear part exact: L_k = -nu pi^2 |k|^2. Nonlinear part -u sum_a d_a u: u and d_a u are synthesised on the 3/2-padded
grid (M = 3N/2 intervals, interior nodes j = 1..M-1) by sine (DST-I type) and, along a, cosine (DCT-I type) sums,
multiplied pointwise and projected back with the trapezoid rule b_k = (2/M)^d sum_j f_j prod sin(pi k_a j_a / M),
then truncated to k_a < N. This is oversampled collocation (a quadrature of the sine projection, O(M^-4) error for
these integrands), not exact de-aliasing (A1.6). Time: ETDRK4 (Cox-Matthews), coefficients by the Kassam-Trefethen
contour mean (32 points; checked against 64). Forcing (gate G4 only): coefficient arrays at the stage times.

IC: sine coefficients of the CONTINUOUS u0 by composite Gauss-Legendre quadrature of its separable 1D factors
(A1.6/A2.6; see amendment A4 for the composite rule). Outputs: lattice values by direct synthesis of the series at the
scoring-lattice coordinates. All large arrays are traced jit arguments.
"""
from __future__ import annotations

import math

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ------------------------------------------------------------------------------------------------ transforms --

def _ext_fft(o):
    return jnp.fft.rfft(o, axis=-1)


def sine_synth_last(b, M):
    """b (..., M-1) coefficients -> S_j = sum_k b_k sin(pi k j / M), j = 1..M-1."""
    z = jnp.zeros(b.shape[:-1] + (1,), b.dtype)
    o = jnp.concatenate((z, b, z, -b[..., ::-1]), -1)          # odd extension, length 2M
    return -_ext_fft(o).imag[..., 1:M] / 2.0


def cos_synth_last(g, M):
    """g (..., M-1) coefficients (k = 1..M-1) -> C_j = sum_k g_k cos(pi k j / M), j = 1..M-1."""
    z = jnp.zeros(g.shape[:-1] + (1,), g.dtype)
    e = jnp.concatenate((z, g, z, g[..., ::-1]), -1)           # even extension, length 2M
    return _ext_fft(e).real[..., 1:M] / 2.0


def along(fn, x, axis, M):
    return jnp.moveaxis(fn(jnp.moveaxis(x, axis, -1), M), -1, axis)


def analysis(f, M):
    """Trapezoid sine projection on every axis: b_k = (2/M)^d sum_j f_j prod_a sin(pi k_a j_a / M)."""
    for ax in range(f.ndim):
        f = along(sine_synth_last, f, ax, M) * (2.0 / M)
    return f


def kvec(n, d, axis):
    """k_a * pi broadcast along `axis` for coefficient arrays of side n (= M - 1 or N - 1)."""
    k = jnp.arange(1, n + 1, dtype=jnp.float64) * jnp.pi
    shp = [1] * d
    shp[axis] = n
    return k.reshape(shp)


def nonlinear(b, N, d):
    """-P[u sum_a d_a u] (coefficients, (N-1)^d) and max |u| on the padded grid."""
    M = 3 * N // 2
    pad = [(0, M - N)] * d
    bp = jnp.pad(b, pad)
    n = M - 1
    S = lambda x, ax: along(sine_synth_last, x, ax, M)
    Cc = lambda x, ax: along(cos_synth_last, x, ax, M)
    if d == 2:
        Ty = S(bp, 1)
        u = S(Ty, 0)
        g = Cc(kvec(n, 2, 0) * Ty, 0) + S(Cc(kvec(n, 2, 1) * bp, 1), 0)
    else:
        Tz = S(bp, 2)
        Tyz = S(Tz, 1)
        u = S(Tyz, 0)
        g = (Cc(kvec(n, 3, 0) * Tyz, 0) + S(Cc(kvec(n, 3, 1) * Tz, 1), 0)
             + S(S(Cc(kvec(n, 3, 2) * bp, 2), 1), 0))
    f = analysis(u * g, M)
    sl = tuple(slice(0, N - 1) for _ in range(d))
    return -f[sl], jnp.max(jnp.abs(u))


def lin_eig(N, d, nu):
    k = np.arange(1, N, dtype=np.float64)
    ks = [k.reshape([-1 if a == b else 1 for b in range(d)]) for a in range(d)]
    return -nu * np.pi ** 2 * sum(x ** 2 for x in ks)


# ------------------------------------------------------------------------------------------------ ETDRK4 -------

def etdrk4_coefs(Lk, h, Mc=32):
    """Kassam-Trefethen contour means (accumulated point by point; no (n, Mc) array)."""
    Lk = jnp.asarray(Lk)
    acc = [jnp.zeros_like(Lk) for _ in range(4)]
    for j in range(1, Mc + 1):
        r = np.exp(1j * np.pi * (j - 0.5) / Mc)
        LR = h * Lk + r
        eLR = jnp.exp(LR)
        acc[0] = acc[0] + ((jnp.exp(LR / 2) - 1) / LR).real
        acc[1] = acc[1] + ((-4 - LR + eLR * (4 - 3 * LR + LR ** 2)) / LR ** 3).real
        acc[2] = acc[2] + ((2 + LR + eLR * (-2 + LR)) / LR ** 3).real
        acc[3] = acc[3] + ((-4 - 3 * LR - LR ** 2 + eLR * (4 - LR)) / LR ** 3).real
    Q, f1, f2, f3 = (h * a / Mc for a in acc)
    return dict(E=jnp.exp(h * Lk), E2=jnp.exp(h * Lk / 2), Q=Q, f1=f1, f2=f2, f3=f3)


def make_ps(N, d, nonlin_on=True, forcing=None):
    """Return run(b0, nu, dt, lat_S, Mc=32) -> dict(lat (6, ...lattice...), umax, finite).
    forcing: None or fn(t) -> coefficient array (N-1)^d (G4)."""

    def Nfun(b, t):
        if nonlin_on:
            nb, um = nonlinear(b, N, d)
        else:
            nb, um = jnp.zeros_like(b), jnp.max(jnp.abs(b)) * 0.0
        if forcing is not None:
            nb = nb + forcing(t)
        return nb, um

    def step(carry, co, dt):
        v, t, umax = carry
        Nv, m1 = Nfun(v, t)
        a = co['E2'] * v + co['Q'] * Nv
        Na, m2 = Nfun(a, t + dt / 2)
        b = co['E2'] * v + co['Q'] * Na
        Nb, m3 = Nfun(b, t + dt / 2)
        c = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
        Nc, m4 = Nfun(c, t + dt)
        v = co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']
        return v, t + dt, jnp.maximum(umax, jnp.maximum(jnp.maximum(m1, m2), jnp.maximum(m3, m4)))

    def block(v, t, umax, co, dt, nsteps):
        def body(i, c):
            return step(c, co, dt)
        return jax.lax.fori_loop(0, nsteps, body, (v, t, umax))
    jblock = jax.jit(block, static_argnums=(5,))

    def run(b0, nu, dt, lat_S, Mc=32):
        keep = int(round(0.05 / dt))
        assert abs(keep * dt - 0.05) < 1e-12, dt
        co = etdrk4_coefs(jnp.asarray(lin_eig(N, d, nu)), dt, Mc)
        v = jnp.asarray(b0)
        t, umax = jnp.float64(0.0), jnp.float64(0.0)
        lats = [synth_lattice(v, lat_S, d)]
        for _ in range(5):
            v, t, umax = jblock(v, t, umax, co, jnp.float64(dt), keep)
            lats.append(synth_lattice(v, lat_S, d))
        lat = np.stack([np.asarray(x) for x in lats])
        fin = bool(np.isfinite(lat).all() and np.isfinite(float(umax)) and bool(jnp.all(jnp.isfinite(v))))
        return dict(lat=lat, umax=float(umax), finite=fin, coef_final=v)
    return run


# ------------------------------------------------------------------------------------------------ lattice -----

def lattice_S(N, m):
    """(m-1, N-1) synthesis matrix sin(pi k x_i), x_i = i/m interior lattice coordinates."""
    x = np.arange(1, m) / m
    k = np.arange(1, N)
    return jnp.asarray(np.sin(np.pi * np.outer(x, k)))


def synth_lattice(b, S, d):
    if d == 2:
        v = S @ b @ S.T
        return jnp.pad(v, 1)                                  # (257, 257) with zero walls (2D convention)
    v = jnp.einsum('ia,jb,kc,abc->ijk', S, S, S, b)
    return v.reshape(-1)                                      # (63^3,) flat, x slowest (3D convention)


# --------------------------------------------------------------------------------------------- quadrature -----

def composite_gl(panels, pts=32):
    """Composite Gauss-Legendre on [0, 1]: `panels` equal panels x `pts` points."""
    x, w = np.polynomial.legendre.leggauss(pts)
    edges = np.arange(panels) / panels
    X = (edges[:, None] + (x[None, :] + 1) / (2 * panels)).ravel()
    W = np.tile(w / (2 * panels), panels)
    return X, W


def sine_integrals(fvals, X, W, N):
    """I_k = 2 int_0^1 f(x) sin(k pi x) dx, k = 1..N-1, for fvals = f(X) (..., Q)."""
    k = np.arange(1, N)
    S = np.sin(np.pi * np.outer(X, k))                         # (Q, N-1)
    return 2.0 * (np.asarray(fvals) * W) @ S


def ic_coefs_2d(desc, N, panels=256):
    cx, cy, w, a, _ = [float(v) for v in desc]
    X, W = composite_gl(panels)
    gx = sine_integrals(np.exp(-(X - cx) ** 2 / (2 * w * w)), X, W, N)
    gy = sine_integrals(np.exp(-(X - cy) ** 2 / (2 * w * w)), X, W, N)
    return a * np.outer(gx, gy)


def ic_coefs_3d(tab, j, N, panels=256):
    X, W = composite_gl(panels)
    B = int(tab['B'][j])
    out = np.zeros((N - 1,) * 3)
    for b in range(B):
        c, w, rho = tab['c'][j][b], float(tab['w'][j][b]), float(tab['rho'][j][b])
        I = [sine_integrals(4 * X * (1 - X) * np.exp(-(X - c[a]) ** 2 / (2 * w * w)), X, W, N) for a in range(3)]
        out += rho * np.einsum('i,j,k->ijk', *I)
    return float(tab['A'][j]) / float(tab['s_star'][j]) * out


# ----------------------------------------------------------------------------------------------- MMS (G4) -----

MMS_NU = 0.05


def phi(x):
    return jnp.sin(jnp.pi * x) * jnp.exp(0.5 * jnp.sin(jnp.pi * x) ** 2)


dphi = jax.vmap(jax.grad(phi))
ddphi = jax.vmap(jax.grad(jax.grad(phi)))


def amp(t):
    return 1.0 + 0.5 * jnp.sin(4 * jnp.pi * t)


damp = jax.grad(amp)


def mms_exact(X, t):
    """u*(X, t) on coordinate arrays X (list of d arrays)."""
    out = amp(t)
    for x in X:
        out = out * phi(x)
    return out


def mms_forcing_grid(t, X, nu=MMS_NU):
    """f = u*_t + u* sum_a d_a u* - nu Lap u* at coordinate arrays X (FD gate G2)."""
    d = len(X)
    P = [phi(x) for x in X]
    dP = [dphi(x.ravel()).reshape(x.shape) for x in X]
    ddP = [ddphi(x.ravel()).reshape(x.shape) for x in X]
    A = amp(t)
    prod = 1.0
    for p in P:
        prod = prod * p
    ut = damp(t) * prod
    adv = 0.0
    lp = 0.0
    for a in range(d):
        term_d, term_dd = dP[a], ddP[a]
        for b in range(d):
            if b != a:
                term_d = term_d * P[b]
                term_dd = term_dd * P[b]
        adv = adv + term_d
        lp = lp + term_dd
    return ut + A * prod * A * adv - nu * A * lp


def mms_forcing_coefs(N, d, nu=MMS_NU, panels=256):
    """Returns fn(t) -> forcing sine coefficients (N-1)^d, from separable 1D integrals (independent of the DST)."""
    X, W = composite_gl(panels)
    Xj = jnp.asarray(X)
    p = np.asarray(phi(Xj))
    dp = np.asarray(dphi(Xj))
    ddp = np.asarray(ddphi(Xj))
    I_p = jnp.asarray(sine_integrals(p, X, W, N))
    I_pp = jnp.asarray(sine_integrals(p * p, X, W, N))
    I_pdp = jnp.asarray(sine_integrals(p * dp, X, W, N))
    I_ddp = jnp.asarray(sine_integrals(ddp, X, W, N))

    def outer(vs):
        if d == 2:
            return vs[0][:, None] * vs[1][None, :]
        return vs[0][:, None, None] * vs[1][None, :, None] * vs[2][None, None, :]

    base = outer([I_p] * d)
    adv = sum(outer([I_pdp if b == a else I_pp for b in range(d)]) for a in range(d))
    lp = sum(outer([I_ddp if b == a else I_p for b in range(d)]) for a in range(d))

    def fn(t):
        A = amp(t)
        return damp(t) * base + A * A * adv - nu * A * lp
    return fn


def mms_ic_coefs(N, d, panels=256):
    X, W = composite_gl(panels)
    I = jnp.asarray(sine_integrals(np.asarray(phi(jnp.asarray(X))), X, W, N))
    if d == 2:
        return amp(0.0) * I[:, None] * I[None, :]
    return amp(0.0) * I[:, None, None] * I[None, :, None] * I[None, None, :]
