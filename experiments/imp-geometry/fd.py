"""imp-geometry: second-order finite differences on L / H / B (DESIGN 3, 5.1 C-FDMG and C-RK4).

Grid: per-unit spacing h = 1/n (n a multiple of 64), nodes x_i = i h, i = 1 .. SIDE n - 1, state = (SIDE n - 1)^2 array that is zero
outside the domain mask. Laplacian and central first derivative with Shortley-Weller coefficients where a neighbour lies
inside the hole (fraction theta of h to the circle); on L and B every outside neighbour lies on a grid line (theta = 1).
`stair=True` drops the Shortley-Weller fractions (theta = 1 everywhere): the MF-FD control, first order on H.

Time integration: SBDF2 (first step IMEX Euler), Helmholtz (alpha I - nu Lap) solved by geometric multigrid V(2,2) cycles
(damped Jacobi, full weighting, bilinear prolongation, rediscretised coarse levels) to a relative residual tolerance;
or classical explicit RK4.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import geom as G


# ------------------------------------------------------------------------------------------------ stencils -----

def _theta_hole(px, py, dx, dy, h):
    """Fraction theta in (0, 1] of the step h from (px, py) along (dx, dy) to the circle (first crossing)."""
    ox, oy = px - G.HC[0], py - G.HC[1]
    # |o + t d|^2 = R^2, t = theta h
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - G.HR * G.HR
    disc = np.maximum(b * b - 4 * c, 0.0)
    t = (-b - np.sqrt(disc)) / 2
    return np.clip(t / h, 1e-12, 1.0)


class Grid:
    def __init__(self, dom, n, stair=False):
        self.dom, self.n, self.stair = dom, int(n), stair
        assert self.n % 8 == 0
        N1 = int(round(G.SIDE * self.n)) - 1
        self.N1 = N1
        h = 1.0 / self.n
        self.h = h
        x = np.arange(1, N1 + 1) * h
        X, Y = np.meshgrid(x, x, indexing='ij')
        self.X, self.Y = X, Y
        m = G.inside(dom, X, Y)
        self.mask_np = m
        th = {}
        for key, (ddx, ddy) in {'E': (1, 0), 'W': (-1, 0), 'N': (0, 1), 'S': (0, -1)}.items():
            t = np.ones_like(X)
            if dom == 'H' and not stair:
                nx, ny = X + ddx * h, Y + ddy * h
                out = ((nx - G.HC[0]) ** 2 + (ny - G.HC[1]) ** 2 <= G.HR ** 2) & m
                t[out] = _theta_hole(X[out], Y[out], ddx, ddy, h)
            th[key] = t
        tE, tW, tN, tS = th['E'], th['W'], th['N'], th['S']
        self.theta_min = float(min(t[m].min() for t in th.values()))
        mf = m.astype(float)
        c = {}
        c['lE'] = mf * 2 / (h * h * tE * (tE + tW))
        c['lW'] = mf * 2 / (h * h * tW * (tE + tW))
        c['lN'] = mf * 2 / (h * h * tN * (tN + tS))
        c['lS'] = mf * 2 / (h * h * tS * (tN + tS))
        c['lP'] = mf * (-2 / (h * h * tE * tW) - 2 / (h * h * tN * tS))
        c['dE'] = mf * tW / (h * tE * (tE + tW))
        c['dW'] = mf * (-tE / (h * tW * (tE + tW)))
        c['dN'] = mf * tS / (h * tN * (tN + tS))
        c['dS'] = mf * (-tN / (h * tS * (tN + tS)))
        c['dP'] = mf * ((tE - tW) / (h * tE * tW) + (tN - tS) / (h * tN * tS))
        self.c = {k: jnp.asarray(v) for k, v in c.items()}
        self.mask = jnp.asarray(mf)

    def lat_idx(self, lat=G.LAT):
        return G.lat_from_grid(self.dom, self.n, lat)

    def ic(self, d):
        """jnp: nodal initial condition (geom.ic), zero outside (d traced)."""
        return self.mask * G.ic(self.dom, d, jnp.asarray(self.X), jnp.asarray(self.Y), jnp)


def _sh(u):
    p = jnp.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)])
    return p[..., 2:, 1:-1], p[..., :-2, 1:-1], p[..., 1:-1, 2:], p[..., 1:-1, :-2]   # E, W, N, S


def lap(c, u):
    E, W, N, S = _sh(u)
    return c['lE'] * E + c['lW'] * W + c['lN'] * N + c['lS'] * S + c['lP'] * u


def dsum(c, u):
    """(D_x + D_y) u."""
    E, W, N, S = _sh(u)
    return c['dE'] * E + c['dW'] * W + c['dN'] * N + c['dS'] * S + c['dP'] * u


def adv(c, u):
    return u * dsum(c, u)


# ------------------------------------------------------------------------------------------------ multigrid ----

def _conv121(u):
    p = jnp.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)])
    a = p[..., :-2, :] + 2 * p[..., 1:-1, :] + p[..., 2:, :]
    return a[..., :, :-2] + 2 * a[..., :, 1:-1] + a[..., :, 2:]


def restrict(u):
    return (_conv121(u) / 16.0)[..., 1::2, 1::2]


def prolong(e, N1f):
    z = jnp.zeros(e.shape[:-2] + (N1f, N1f), e.dtype)
    z = z.at[..., 1::2, 1::2].set(e)
    return _conv121(z) / 4.0


class MG:
    """Geometric multigrid hierarchy for H u = alpha u - beta Lap u on Grid(dom, n)."""

    def __init__(self, dom, n, stair=False, nmin=8, smoother=True, nu1=2, nu2=2, omega=0.8, coarse_sweeps=40):
        self.levels = []
        m = n
        while True:
            self.levels.append(Grid(dom, m, stair))
            if m % 16 or m // 2 < nmin:
                break
            m //= 2
        self.smoother, self.nu1, self.nu2, self.omega, self.coarse_sweeps = smoother, nu1, nu2, omega, coarse_sweeps

    def H(self, l, alpha, beta, u):
        g = self.levels[l]
        return g.mask * (alpha * u - beta * lap(g.c, u))

    def diag(self, l, alpha, beta):
        g = self.levels[l]
        return alpha - beta * g.c['lP']

    def smooth(self, l, alpha, beta, u, f, k):
        if not self.smoother:                    # MF-MG control: identity "smoother"
            return u
        g = self.levels[l]
        D = self.diag(l, alpha, beta)
        for _ in range(k):
            u = u + self.omega * g.mask * (f - self.H(l, alpha, beta, u)) / D
        return u

    def vcycle(self, l, alpha, beta, u, f):
        if l == len(self.levels) - 1:
            return self.smooth(l, alpha, beta, u, f, self.coarse_sweeps) if self.smoother else u
        u = self.smooth(l, alpha, beta, u, f, self.nu1)
        r = f - self.H(l, alpha, beta, u)
        rc = self.levels[l + 1].mask * restrict(r)
        ec = self.vcycle(l + 1, alpha, beta, jnp.zeros_like(rc), rc)
        u = u + self.levels[l].mask * prolong(ec, self.levels[l].N1)
        return self.smooth(l, alpha, beta, u, f, self.nu2)

    def solve(self, alpha, beta, f, u0, tol, maxit=40):
        """V-cycles until ||f - H u|| <= tol ||f|| (Euclidean). Returns (u, cycles, converged, final ratio)."""
        fn = jnp.sqrt(jnp.sum(f * f)) + 1e-300

        def cond(s):
            u, it, rr = s
            return (rr > tol) & (it < maxit)

        def body(s):
            u, it, _ = s
            u = self.vcycle(0, alpha, beta, u, f)
            r = f - self.H(0, alpha, beta, u)
            return u, it + 1, jnp.sqrt(jnp.sum(r * r)) / fn
        r0 = f - self.H(0, alpha, beta, u0)
        u, it, rr = jax.lax.while_loop(cond, body, (u0, jnp.int32(0), jnp.sqrt(jnp.sum(r0 * r0)) / fn))
        return u, it, rr <= tol, rr


# ------------------------------------------------------------------------------------------------ solvers ------

def make_sbdf2(dom, n, nsteps, tol, out_idx=None, n_out=G.N_OUT, stair=False, maxit=40, smoother=True, ic_dom=None):
    """C-FDMG query: d -> (outputs (n_out+1, P) on out_idx (default scoring lattice), stats).
    stats = (total V-cycles, max cycles in one solve, number of non-converged solves)."""
    mg = MG(dom, n, stair, smoother=smoother)
    g = mg.levels[0]
    c = g.c
    I, J = g.lat_idx() if out_idx is None else out_idx
    I, J = jnp.asarray(I), jnp.asarray(J)
    dt = G.T_END / nsteps
    keep = nsteps // n_out
    assert keep * n_out == nsteps

    def q(d):
        nu = d[4]
        u0 = g.ic(d) if ic_dom is None else g.mask * G.ic(ic_dom, d, jnp.asarray(g.X), jnp.asarray(g.Y), jnp)
        N0 = adv(c, u0)
        f1 = g.mask * (u0 / dt - N0)
        u1, it1, ok1, _ = mg.solve(1.0 / dt, nu, f1, u0, tol, maxit)

        def step(carry, _):
            u, up, Np, tot, mx, bad = carry
            Nn = adv(c, u)
            f = g.mask * ((2 * u - 0.5 * up) / dt - (2 * Nn - Np))
            un, it, ok, _ = mg.solve(1.5 / dt, nu, f, 2 * u - up, tol, maxit)
            return (un, u, Nn, tot + it, jnp.maximum(mx, it), bad + (~ok).astype(jnp.int32)), None

        def block(carry, k):
            nst = jnp.where(k == 0, keep - 1, keep)
            carry = jax.lax.fori_loop(0, nst, lambda i, cc: step(cc, None)[0], carry)
            return carry, carry[0][..., I, J]
        carry0 = (u1, u0, N0, it1, it1, (~ok1).astype(jnp.int32))
        carry, outs = jax.lax.scan(block, carry0, jnp.arange(n_out))
        O = jnp.concatenate([u0[None, I, J], outs])
        return O, jnp.stack([carry[3], carry[4], carry[5]])
    return jax.jit(q)


def make_rk4(dom, n, nsteps, out_idx=None, n_out=G.N_OUT, stair=False, ic_dom=None):
    g = Grid(dom, n, stair)
    c = g.c
    I, J = g.lat_idx() if out_idx is None else out_idx
    I, J = jnp.asarray(I), jnp.asarray(J)
    dt = G.T_END / nsteps
    keep = nsteps // n_out
    assert keep * n_out == nsteps

    def q(d):
        nu = d[4]

        def F(u):
            return g.mask * (nu * lap(c, u) - adv(c, u))

        def st(i, u):
            k1 = F(u)
            k2 = F(u + 0.5 * dt * k1)
            k3 = F(u + 0.5 * dt * k2)
            k4 = F(u + dt * k3)
            return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u0 = g.ic(d)

        def block(u, _):
            u = jax.lax.fori_loop(0, keep, st, u)
            return u, u[I, J]
        _, outs = jax.lax.scan(block, u0, None, length=n_out)
        return jnp.concatenate([u0[None, I, J], outs]), jnp.zeros(3, jnp.int32)
    return jax.jit(q)


def rk4_stable_n(dom, n, nu_max, umax=2.0, safety=0.9, theta_min=1.0):
    """Smallest step count of the form 5*2^j with RK4 diffusion + advection stability (rough bound, used only to start
    the ladder; instability is detected anyway). RK4 real-axis bound 2.785, imaginary 2.828."""
    h = 1.0 / n
    lam_d = 8 * nu_max / (h * h * theta_min)
    lam_a = 2 * umax / (h * theta_min)
    dt = safety / (lam_d / 2.785 + lam_a / 2.828)
    nmin = G.T_END / dt
    j = max(0, math.ceil(math.log2(max(nmin / 5, 1))))
    return 5 * 2 ** j


# --------------------------------------------------------------------------------------- manufactured test -----

def mms_helmholtz(dom, n, stair=False, alpha=10.0, beta=1.0, tol=1e-12):
    """MF-FD / MF-MG: solve alpha u - beta Lap u = f for u* = envelope-like smooth function vanishing on the boundary;
    returns the max-norm error on the lattice and the MG cycles used."""
    mg = MG(dom, n, stair)
    g = mg.levels[0]
    X, Y = g.X, g.Y
    us, f = mms_fields(dom, X, Y, alpha, beta)
    u, it, ok, rr = jax.jit(lambda f_: mg.solve(alpha, beta, f_, jnp.zeros_like(f_), tol, 200))(jnp.asarray(f * g.mask_np))
    I, J = g.lat_idx()
    e = float(np.max(np.abs(np.asarray(u)[I, J] - us[I, J])))
    return e, int(it), bool(ok)


def mms_fields(dom, X, Y, alpha, beta):
    """u* vanishing on the boundary of each domain, and f = alpha u* - beta Lap u* (closed form by sympy-free
    finite composition: u* = b(x,y) * s(x,y) with b = x(2-x)y(2-y), s = 1 (B), (1-x)... see below)."""
    import sympy as sp
    x, y = sp.symbols('x y')
    b = x * (G.SIDE - x) * y * (G.SIDE - y) * sp.sin(x + 0.5 * y)
    if dom == 'H':
        b = b * ((x - G.HC[0]) ** 2 + (y - G.HC[1]) ** 2 - G.HR ** 2)
    if dom == 'L':
        b = b * (G.LC - x) * (G.LC - y)  # vanishes on x = LC and y = LC (smooth; exact on the re-entrant edges)
    fs = alpha * b - beta * (sp.diff(b, x, 2) + sp.diff(b, y, 2))
    fu, ff = sp.lambdify((x, y), b, 'numpy'), sp.lambdify((x, y), fs, 'numpy')
    return fu(X, Y), ff(X, Y)
