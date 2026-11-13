"""Sparse matrices for the solve-only benchmark (float64 CSR, int32 indices).

Uniform grid of n interior points per axis on (0,1)^d, h = 1/(n+1), homogeneous Dirichlet
(boundary unknowns eliminated), standard second-order finite differences, lexicographic order.

  poisson : K = -Laplacian (5-point / 7-point)
  heat    : I + (dt*kappa/2) K          Crank-Nicolson left-hand side, dt = 0.025, kappa = 0.02
  burgers : I + dt (diag(g) + diag(u) D_upw(u) + nu K)
            Newton Jacobian of the backward-Euler residual of u_t + u (sum_i du/dx_i) = nu Lap u with the
            sign-upwind stencil; g = sum_i (upwind d u/dx_i); dt = 0.005, nu = 0.01;
            linearisation state u = exp(-|x-c|^2 / (2 w^2)) + 0.5 prod_i sin(2 pi x_i), c = (0.4, 0.6[, 0.5]), w = 0.1
"""
import numpy as np
import scipy.sparse as sp

HEAT_DT, HEAT_KAPPA = 0.025, 0.02
BURGERS_DT, BURGERS_NU = 0.005, 0.01
SPD = {'poisson': True, 'heat': True, 'burgers': False}


def _assemble(n, dim, diag, lo, up):
    """diag, lo[a], up[a]: arrays of shape (n,)*dim (or scalars); lo/up couple a point to its -1/+1 neighbour on axis a."""
    shape = (n,) * dim
    N = n ** dim
    idx = np.arange(N, dtype=np.int32).reshape(shape)
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [np.broadcast_to(diag, shape).ravel().astype(np.float64)]
    for a in range(dim):
        s_hi = [slice(None)] * dim; s_hi[a] = slice(1, None)
        s_lo = [slice(None)] * dim; s_lo[a] = slice(None, -1)
        s_hi, s_lo = tuple(s_hi), tuple(s_lo)
        rows.append(idx[s_hi].ravel()); cols.append(idx[s_lo].ravel())
        vals.append(np.broadcast_to(lo[a], shape)[s_hi].ravel().astype(np.float64))
        rows.append(idx[s_lo].ravel()); cols.append(idx[s_hi].ravel())
        vals.append(np.broadcast_to(up[a], shape)[s_lo].ravel().astype(np.float64))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    A.sort_indices()
    A.indices = A.indices.astype(np.int32, copy=False)
    A.indptr = A.indptr.astype(np.int32, copy=False)
    return A


def burgers_state(n, dim):
    x = np.arange(1, n + 1) / (n + 1)
    X = np.meshgrid(*([x] * dim), indexing='ij', sparse=True)
    c = (0.4, 0.6, 0.5)[:dim]
    r2 = sum((X[i] - c[i]) ** 2 for i in range(dim))
    s = 1.0
    for i in range(dim):
        s = s * np.sin(2 * np.pi * X[i])
    return np.exp(-r2 / (2 * 0.1 ** 2)) + 0.5 * s


def build(problem, n, dim):
    h = 1.0 / (n + 1)
    ih2 = 1.0 / h ** 2
    if problem == 'poisson':
        return _assemble(n, dim, 2 * dim * ih2, [-ih2] * dim, [-ih2] * dim)
    if problem == 'heat':
        a = 0.5 * HEAT_DT * HEAT_KAPPA
        return _assemble(n, dim, 1 + a * 2 * dim * ih2, [-a * ih2] * dim, [-a * ih2] * dim)
    if problem == 'burgers':
        dt, nu = BURGERS_DT, BURGERS_NU
        u = burgers_state(n, dim)
        pos = u > 0
        diag = 1 + dt * nu * 2 * dim * ih2 + dim * dt * np.abs(u) / h
        g = np.zeros_like(u)
        lo, up = [], []
        for a in range(dim):
            pad = [(0, 0)] * dim; pad[a] = (1, 1)
            up_ = np.pad(u, pad)
            sl_m = [slice(None)] * dim; sl_m[a] = slice(0, n)
            sl_p = [slice(None)] * dim; sl_p[a] = slice(2, n + 2)
            g += np.where(pos, u - up_[tuple(sl_m)], up_[tuple(sl_p)] - u) / h
            lo.append(-dt * nu * ih2 - dt * np.where(pos, u, 0.0) / h)
            up.append(-dt * nu * ih2 + dt * np.where(pos, 0.0, u) / h)
        return _assemble(n, dim, diag + dt * g, lo, up)
    raise ValueError(problem)


if __name__ == '__main__':
    for p in SPD:
        for d, n in ((2, 16), (3, 8)):
            A = build(p, n, d)
            asym = abs(A - A.T).max()
            ev = np.linalg.eigvals(A.toarray())
            print(p, d, A.shape, A.nnz, 'asym', asym, 'min Re eig', ev.real.min(), 'nnz/row', A.nnz / A.shape[0])
