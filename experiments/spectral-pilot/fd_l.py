"""spectral-pilot Part B: viscous Burgers on the L-shaped domain Omega_L = (0,1)^2 minus [1/2,1]x[1/2,1], zero Dirichlet
data on all six edges (DESIGN 1, 4.3, A1-A4). PILOT: small, not a result.

Grid h = 1/L (L even, so the re-entrant edges are grid lines); state = the (L-1)^2 array of interior box nodes, zero
outside Omega_L (mask). Discretisation (the E0 box arithmetic): 5-point Laplacian, central advection
u * ((u_{i+1,j} - u_{i-1,j}) + (u_{i,j+1} - u_{i,j-1})) * L / 2.
Families:
  F-RK4   classical RK4 (JAX, compiled loop);
  F-IMEX  SBDF2 (implicit diffusion, extrapolated advection), first step IMEX Euler; both matrices
          (I - dt nu A) and (3/2 I - dt nu A) factorised by SuperLU per query (both charged).
Cohort (A1.2): a ~ U[1, 2], c ~ U[0.22, 0.30]^2, w ~ U[0.03, 0.045], log nu ~ U[log .02, log .05].
Scoring lattice x_i = i / 64 restricted to Omega_L.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import common as C

T_END = 0.25
LAT_B = 64
FAM_B = dict(a=(1.0, 2.0), c=(0.22, 0.30), w=(0.03, 0.045))
NU_B = (0.02, 0.05)
TRAIN_SEED_B, TEST_SEED_B = 3_000_001, 4_000_001
N_TRAIN_B, N_TEST_B = 96, 12


def cohorts_b():
    def draw(seed, n):
        rows = []
        for j in range(n):
            r = np.random.default_rng([int(seed), int(j)])
            cx, cy = r.uniform(*FAM_B['c']), r.uniform(*FAM_B['c'])
            w = r.uniform(*FAM_B['w'])
            a = r.uniform(*FAM_B['a'])
            nu = math.exp(r.uniform(math.log(NU_B[0]), math.log(NU_B[1])))
            rows.append((cx, cy, w, a, nu))
        return np.asarray(rows)
    assert TRAIN_SEED_B not in C.CAMPAIGN_SEEDS and TEST_SEED_B not in C.CAMPAIGN_SEEDS
    tr, te = draw(TRAIN_SEED_B, N_TRAIN_B), draw(TEST_SEED_B, N_TEST_B)
    for t in te:
        assert not np.any(np.all(np.isclose(tr, t), axis=1))
    return tr, te


def mask(L):
    assert L % 2 == 0
    x = np.arange(1, L) / L
    X, Y = np.meshgrid(x, x, indexing='ij')
    return ~((X >= 0.5) & (Y >= 0.5)), X, Y


def ic(L, d):
    m, X, Y = mask(L)
    cx, cy, w, a = d[:4]
    return np.where(m, a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w * w)), 0.0)


def lattice_idx(L, lat=LAT_B):
    """Indices (into the (L-1)^2 interior array) of the lattice nodes i/lat inside Omega_L, row-major; and the mask."""
    assert L % lat == 0
    s = L // lat
    m, _, _ = mask(L)
    ii = np.arange(1, lat) * s - 1
    sub = m[np.ix_(ii, ii)]
    I, J = np.meshgrid(ii, ii, indexing='ij')
    return I[sub], J[sub]


def lat_vals(U, L, lat=LAT_B):
    I, J = lattice_idx(L, lat)
    return U[..., I, J]


# --------------------------------------------------------------------------------------------- operators -------

def adv_np(u, L):
    p = np.pad(u, 1)
    return u * ((p[2:, 1:-1] - p[:-2, 1:-1]) + (p[1:-1, 2:] - p[1:-1, :-2])) * (L / 2.0)


def lap_matrix(L):
    """5-point Laplacian on the interior nodes of Omega_L (CSC), symmetric negative definite; and the mask."""
    m, _, _ = mask(L)
    n = int(m.sum())
    idx = -np.ones(m.shape, int)
    idx[m] = np.arange(n)
    I, J = np.nonzero(m)
    rows, cols = [], []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = I + di, J + dj
        ok = (ii >= 0) & (ii < L - 1) & (jj >= 0) & (jj < L - 1)
        ok[ok] = m[ii[ok], jj[ok]]
        rows.append(idx[I[ok], J[ok]])
        cols.append(idx[ii[ok], jj[ok]])
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    A = (A - 4 * sp.eye(n)) * (L * L)
    return A.tocsc(), m


def imex_query(d, L, n, A, m, out_lattice=True, n_out=5, flat=False):
    """F-IMEX query: IC, both factorisations, n steps, outputs at n_out+1 equispaced times (lattice values, full
    arrays, or interior vectors if flat)."""
    nu = d[4]
    dt = T_END / n
    keep = n // n_out
    assert keep * n_out == n
    I_ = sp.identity(A.shape[0], format='csc')
    lu1 = spl.splu((I_ - dt * nu * A).tocsc(), permc_spec='MMD_AT_PLUS_A')
    lu2 = spl.splu((1.5 * I_ - dt * nu * A).tocsc(), permc_spec='MMD_AT_PLUS_A')
    U0 = ic(L, d)
    u = U0[m]
    Ug = np.zeros_like(U0)

    def N(v):
        Ug[m] = v
        return adv_np(Ug, L)[m]
    outs = [U0]
    Nn = N(u)
    u_prev, N_prev = u, Nn
    u = lu1.solve(u - dt * Nn)
    for k in range(1, n):
        if k % keep == 0:
            G = np.zeros_like(U0)
            G[m] = u
            outs.append(G)
        Nn = N(u)
        u_new = lu2.solve(2 * u - 0.5 * u_prev - dt * (2 * Nn - N_prev))
        u_prev, N_prev, u = u, Nn, u_new
    G = np.zeros_like(U0)
    G[m] = u
    outs.append(G)
    O = np.stack(outs)
    if flat:
        return O[:, m]
    return lat_vals(O, L) if out_lattice else O


def make_rk4(L, n, out_lattice=True, lat=LAT_B):
    """F-RK4 query (JAX): d -> outputs at the 6 times (lattice i/lat values, or full arrays)."""
    m, X, Y = mask(L)
    mj = jnp.asarray(m, jnp.float64)
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    I, J = lattice_idx(L, lat) if out_lattice else (None, None)
    dt = T_END / n
    keep = n // 5
    assert keep * 5 == n

    def F(u, nu):
        p = jnp.pad(u, 1)
        lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * u) * (L * L)
        adv = u * ((p[2:, 1:-1] - p[:-2, 1:-1]) + (p[1:-1, 2:] - p[1:-1, :-2])) * (L / 2.0)
        return mj * (nu * lap - adv)

    def step(u, nu):
        k1 = F(u, nu)
        k2 = F(u + 0.5 * dt * k1, nu)
        k3 = F(u + 0.5 * dt * k2, nu)
        k4 = F(u + dt * k3, nu)
        return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def q(d):
        cx, cy, w, a, nu = d[0], d[1], d[2], d[3], d[4]
        u0 = mj * a * jnp.exp(-((Xj - cx) ** 2 + (Yj - cy) ** 2) / (2 * w * w))

        def blk(u, _):
            u = jax.lax.fori_loop(0, keep, lambda i, v: step(v, nu), u)
            return u, u
        _, outs = jax.lax.scan(blk, u0, None, length=5)
        O = jnp.concatenate([u0[None], outs])
        return O[:, I, J] if out_lattice else O
    return jax.jit(q)


def err_lat(F, R):
    """F, R (..., 6, P) lattice values: max_{k>=1} ||F_k - R_k|| / ||R_0||; non-finite -> inf."""
    num = np.sqrt(np.sum((F - R) ** 2, axis=-1))
    den = np.sqrt(np.sum(R[..., 0, :] ** 2, axis=-1))
    e = np.max(num[..., 1:], axis=-1) / den
    return np.where(np.all(np.isfinite(F), axis=(-1, -2)), e, np.inf)


def ref_to_lattice(Ofull, L, lat=LAT_B):
    return lat_vals(Ofull, L, lat)
