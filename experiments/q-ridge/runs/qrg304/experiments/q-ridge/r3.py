"""R3: the exactly-integrated weak residual on held-out test modes. Pure NumPy.

This module imports NO JAX. It re-implements, in NumPy, exactly three things the job did
on the GPU:

  `modes`    `engines.modes` — the sine test modes ordered by ascending discrete Laplacian
             eigenvalue with a stable sort, and their eigenvalues;
  `spatial`  `engines.spatial` — the five-point upwind advection on the interior grid;
  `weak`     `arms.weak_dense` restricted to a supplied mode block, i.e. the row-scaled
             implicit-Euler weak residual with the advection integrated EXACTLY.

`smoke_ridge.py` gates these three against the JAX originals bitwise / to 1e-12, so the
audit can use them without a GPU.

The quantity R3 reports, for a solved trajectory u_0 .. u_S reconstructed as u_n = G c_n
from the retained per-step bank coefficients:

    r_n(Psi) = ( Psi^T u_n - Psi^T u_{n-1}
                 + dt ( Psi^T n(u_n) + nu Lambda_Psi Psi^T u_n ) ) / (1 + dt nu Lambda_Psi)

for Psi the arm's OWN M test modes (`in-space`) and for a held-out block of modes the
solve never saw (`held-out`). If the extra unknowns were buying a lower projected residual
by moving along directions the M tests barely observe, the held-out residual must be higher
at q = 16 than at q = 0 while the in-space residual is not.
"""
from __future__ import annotations

import numpy as np


def modes(L, count, skip=0):
    """`engines.modes`, NumPy, with an offset so a held-out block can be taken.

    With skip = 0 this is `engines.modes(L, count)` exactly. With skip = M it returns the
    modes ranked M+1 .. M+count in the same ordering — the held-out block.
    """
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[skip:skip + count]
    kx, ky = kx.ravel()[ind], ky.ravel()[ind]
    x = np.arange(1, L) / L
    c = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
    phi = (2. / L) * np.sin(np.pi * c[:, 0, None] * kx) * np.sin(np.pi * c[:, 1, None] * ky)
    return phi, lam.ravel()[ind], np.stack((kx, ky), axis=1)


def spatial(u, L):
    """`engines.spatial`, NumPy. Returns (advection, laplacian) on the interior."""
    p = np.pad(u.reshape(L - 1, L - 1), 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    adv = c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))
    lap = L ** 2 * (xm + xp + ym + yp - 4 * c)
    return adv.reshape(-1), lap.reshape(-1)


def weak(Phi, lam, u_now, u_prev, nu, dt, L):
    """`arms.weak_dense` on the supplied mode block, advection integrated exactly."""
    adv, _ = spatial(u_now, L)
    ah = Phi.T @ u_now
    return (ah - Phi.T @ u_prev + dt * (Phi.T @ adv + nu * lam * ah)) / (1 + dt * nu * lam)


def trajectory_residuals(G, coefficients, blocks, nu, dt, L):
    """Per-step weak residual norms for each named mode block of one trajectory.

    `G` is the frozen bank on the query grid, `coefficients` the retained (S+1, R) per-step
    bank coefficients, `blocks` a mapping name -> (Phi, lam). Returns, per block, the raw
    per-step norms, the per-mode RMS normalised by the per-node RMS of the previous state,
    and the state norms the normalisation used.
    """
    U = coefficients @ G.T                                   # (S+1, (L-1)^2)
    ng = U.shape[1]
    out = {}
    for name, (Phi, lam) in blocks.items():
        raw, rel = [], []
        for n in range(1, len(U)):
            r = weak(Phi, lam, U[n], U[n - 1], nu, dt, L)
            scale = np.linalg.norm(U[n - 1]) / np.sqrt(ng) + 1e-300
            raw.append(float(np.linalg.norm(r)))
            rel.append(float(np.linalg.norm(r) / np.sqrt(Phi.shape[1]) / scale))
        out[name] = dict(modes=int(Phi.shape[1]), raw_per_step=raw,
                         normalised_per_mode_rms_per_step=rel,
                         raw_max=float(np.max(raw)), normalised_max=float(np.max(rel)),
                         normalised_mean=float(np.mean(rel)))
    out['state_norms'] = [float(np.linalg.norm(u)) for u in U]
    return out
