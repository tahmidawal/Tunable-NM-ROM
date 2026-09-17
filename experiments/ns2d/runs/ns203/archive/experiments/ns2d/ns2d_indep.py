"""ns2d_indep.py -- the INDEPENDENT NumPy/SciPy implementation for gate F-INDEP.

Written without importing ns2d_fom: its own stencils (index arithmetic on a padded
periodic array rather than np.roll), its own Poisson solve (real FFT with the
eigenvalues written from the 1-D second-difference symbol), its own Arakawa form
(the term-by-term Arakawa 1966 expansion, eq. 46), and an EXACT Newton with a DENSE
Jacobian assembled by applying the linearised operator to the identity, solved by
LAPACK.  It shares only the mathematical specification with the JAX FOM.

Intended for N = 64 (4096 unknowns, dense 4096 x 4096 Jacobian) over a short trajectory.
"""
from __future__ import annotations

import numpy as np

PI = np.pi


def _pad(a):
    """Periodic pad by one cell on each side: a[i-1..N] -> P[i, j] with P[1:-1,1:-1] = a."""
    return np.pad(a, 1, mode='wrap')


def laplacian_np(w, N):
    P = _pad(w)
    return (P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] - 4.0 * P[1:-1, 1:-1]) * (N * N)


def poisson_np(w, N):
    """psi = -Lap_h^{-1} w, mean zero.  1-D symbol of the second difference:
    -(2 - 2 cos(2 pi k / N)) N^2, assembled per axis with rfft2 (the JAX side uses fft2)."""
    k = np.arange(N)
    s1 = (2.0 - 2.0 * np.cos(2.0 * PI * k / N)) * N * N
    kr = np.arange(N // 2 + 1)
    s1r = (2.0 - 2.0 * np.cos(2.0 * PI * kr / N)) * N * N
    lam = s1[:, None] + s1r[None, :]
    lam[0, 0] = 1.0
    W = np.fft.rfft2(w) / lam
    W[0, 0] = 0.0
    return np.fft.irfft2(W, s=(N, N))


def arakawa_np(psi, z, N):
    """Arakawa (1966) eq. 46, written term by term on padded arrays.

    Arakawa writes eq. 46 as J(zeta, psi) with zeta the advected field; this module's
    convention is J(psi, z) = psi_x z_y - psi_y z_x = -J_Arakawa(z, psi), hence the leading
    minus sign (caught by the analytic-J check: the unsigned transcription converged to
    exactly -J, relative error 2.0)."""
    P, Z = _pad(psi), _pad(z)
    c = slice(1, -1)
    p, m = slice(2, None), slice(None, -2)
    # psi neighbours
    P_ip, P_im = P[p, c], P[m, c]            # i+1, i-1
    P_jp, P_jm = P[c, p], P[c, m]            # j+1, j-1
    P_pp, P_pm = P[p, p], P[p, m]            # i+1,j+1 ; i+1,j-1
    P_mp, P_mm = P[m, p], P[m, m]            # i-1,j+1 ; i-1,j-1
    Z0 = Z[c, c]
    Z_ip, Z_im, Z_jp, Z_jm = Z[p, c], Z[m, c], Z[c, p], Z[c, m]
    Z_pp, Z_pm, Z_mp, Z_mm = Z[p, p], Z[p, m], Z[m, p], Z[m, m]
    J = (-(P_jm + P_pm - P_jp - P_pp) * (Z_ip + Z0)
         + (P_mm + P_jm - P_mp - P_jp) * (Z0 + Z_im)
         - (P_ip + P_pp - P_im - P_mp) * (Z_jp + Z0)
         + (P_pm + P_ip - P_mm - P_im) * (Z0 + Z_jm)
         - (P_ip - P_jp) * (Z_pp + Z0)
         + (P_jm - P_im) * (Z0 + Z_mm)
         - (P_jp - P_im) * (Z_mp + Z0)
         + (P_ip - P_jm) * (Z0 + Z_pm))
    return -J * (N * N / 12.0)


def residual_np(w1, w0, nu, dt, N, f_mid=None):
    wm = 0.5 * (w1 + w0)
    psi = poisson_np(wm, N)
    r = wm * 0.0 + arakawa_np(psi, wm, N) + nu * laplacian_np(wm, N)
    if f_mid is not None:
        r = r + f_mid
    return w1 - w0 - dt * r


def lin_op_np(v, wm, psim, nu, dt, N):
    """d residual / d w1 applied to v (midpoint: half the derivative of the rhs)."""
    vm = 0.5 * v
    dpsi = poisson_np(vm, N)
    return v - dt * (arakawa_np(dpsi, wm, N) + arakawa_np(psim, vm, N) + nu * laplacian_np(vm, N))


def jacobian_dense(w1, w0, nu, dt, N):
    """Assemble the dense Newton Jacobian by applying lin_op to every unit vector.
    Vectorised: the operator is linear, so apply it to a batch (N*N, N, N)."""
    n = N * N
    wm = 0.5 * (w1 + w0)
    psim = poisson_np(wm, N)
    J = np.empty((n, n))
    E = np.eye(n).reshape(n, N, N)
    for s in range(0, n, 256):
        blk = E[s:s + 256]
        cols = np.stack([lin_op_np(b, wm, psim, nu, dt, N).ravel() for b in blk], axis=1)
        J[:, s:s + 256] = cols
    return J


def step_np(w0, nu, dt, N, ntol=1e-13, max_newton=30, f_mid=None):
    w = w0.copy()
    scale = max(np.linalg.norm(w0), 1e-300)
    its = 0
    r = residual_np(w, w0, nu, dt, N, f_mid)
    while np.linalg.norm(r) > ntol * scale and its < max_newton:
        J = jacobian_dense(w, w0, nu, dt, N)
        w = w + np.linalg.solve(J, -r.ravel()).reshape(N, N)
        r = residual_np(w, w0, nu, dt, N, f_mid)
        its += 1
    return w, its, float(np.linalg.norm(r) / scale)


def run_np(w0, nu, dt, N, nsteps, ntol=1e-13):
    states, its, rns = [w0.copy()], [], []
    w = w0.copy()
    for _ in range(nsteps):
        w, it, rn = step_np(w, nu, dt, N, ntol)
        states.append(w.copy())
        its.append(it)
        rns.append(rn)
    return np.stack(states), np.asarray(its), np.asarray(rns)
