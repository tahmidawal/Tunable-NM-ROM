"""How a smooth variable diffusion coefficient degrades the spectral FOM's implicit step (CPU).

The sine pseudo-spectral FOM integrates nu * Laplace(u) exactly (ETD, diagonal in the DST
basis). With a variable coefficient kappa(x), the diffusion operator div(kappa grad u) is no
longer diagonal. The standard fast-transform treatment is an implicit (Crank-Nicolson) stage
solved by conjugate gradients preconditioned with the constant-coefficient DST inverse
(I - dt/2 * kbar * Laplace)^{-1}; each iteration costs two DST pairs plus pointwise work.

This script counts those PCG iterations on the unit box (second-order FD operator on
N-1 interior nodes per axis, homogeneous Dirichlet) for kappa = nu * exp(s * g(x,y)),
g a fixed smooth field scaled to [-1/2, 1/2], so that contrast = kappa_max / kappa_min = e^s.
Iterations are recorded at relative residual 1e-10 and 1e-4 (a loose tolerance that a FOM
matched to percent-level ROM accuracy could use). Also records the explicit-diffusion step limit dt_max ~ h^2 / (4 kappa_max) for comparison.

Three right-hand-side cases: 'noisy' (smooth field plus white noise, zero start: pessimistic),
'smooth' (zero start) and 'warm' (start from a nearby smooth state, a proxy for the previous
time step). This is the FD operator with an FD-eigenvalue DST preconditioner, not the spectral
operator, and not a Burgers trajectory: the counts are indicative only.

Writes varcoef_pcg.json next to this file.
"""
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

NU = 0.01  # order of the Burgers campaign's viscosities (assumption for illustration)


def run(N, contrast, dt, tol=1e-10, case='noisy'):
    h = 1.0 / N
    xs = h * np.arange(1, N)
    xf = h * (np.arange(N) + 0.5)  # face coordinates
    g = lambda x, y: np.sin(2 * np.pi * x) * np.cos(3 * np.pi * y) + 0.5 * np.cos(5 * np.pi * x * y)
    # normalise g to [-1/2, 1/2] on a fine sample
    Xs, Ys = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401), indexing='ij')
    gs = g(Xs, Ys); lo, hi = gs.min(), gs.max()
    gn = lambda x, y: (g(x, y) - lo) / (hi - lo) - 0.5
    s = np.log(contrast)
    kap = lambda x, y: NU * np.exp(s * gn(x, y))
    # face coefficients: x-faces at (xf_i, xs_j) for i = 0..N-1, y-faces at (xs_i, xf_j)
    KX = kap(*np.meshgrid(xf, xs, indexing='ij'))  # (N, N-1)
    KY = kap(*np.meshgrid(xs, xf, indexing='ij'))  # (N-1, N)

    def Lap_k(u):  # div(kappa grad u), Dirichlet zero
        up = np.pad(u, 1)
        fx = KX * (up[1:, 1:-1] - up[:-1, 1:-1]) / h
        fy = KY * (up[1:-1, 1:] - up[1:-1, :-1]) / h
        return (fx[1:, :] - fx[:-1, :]) / h + (fy[:, 1:] - fy[:, :-1]) / h

    A = lambda u: u - 0.5 * dt * Lap_k(u)
    k = np.arange(1, N)
    lam = (4 / h ** 2) * np.sin(k * np.pi * h / 2) ** 2
    LAM = lam[:, None] + lam[None, :]
    kbar = np.sqrt(KX.min() * KX.max())  # geometric-mean scaling, a standard choice
    P = lambda r: idstn(dstn(r, type=1) / (1 + 0.5 * dt * kbar * LAM), type=1)
    rng = np.random.default_rng(0)
    sm = np.sin(np.pi * xs)[:, None] * np.sin(2 * np.pi * xs)[None, :]
    sm2 = np.sin(2 * np.pi * xs)[:, None] * np.sin(np.pi * xs)[None, :]
    if case == 'noisy':      # smooth field plus white noise, zero initial guess (pessimistic)
        u_t = sm + 0.1 * rng.standard_normal((N - 1, N - 1)); x0 = np.zeros_like(sm)
    elif case == 'smooth':   # smooth field, zero initial guess
        u_t = sm; x0 = np.zeros_like(sm)
    else:                    # 'warm': smooth field, initial guess = a nearby smooth state (proxy for the previous step)
        u_t = sm + 0.05 * sm2; x0 = sm
    b = A(u_t)
    x = x0.copy(); r = b - A(x); z = P(r); p = z.copy(); rz = np.vdot(r, z); nb = np.linalg.norm(b)
    it = 0
    it_loose = None
    for it in range(1, 5001):
        Ap = A(p); a = rz / np.vdot(p, Ap); x += a * p; r -= a * Ap
        if it_loose is None and np.linalg.norm(r) < 1e-4 * nb:
            it_loose = it
        if np.linalg.norm(r) < tol * nb:
            break
        z = P(r); rz2 = np.vdot(r, z); p = z + rz2 / rz * p; rz = rz2
    res = float(np.linalg.norm(b - A(x)) / nb)
    return dict(N=N, contrast=contrast, dt=dt, case=case, converged=bool(res < 10 * tol), final_rel_residual=res, pcg_iters=it, pcg_iters_1e4=it_loose, kappa_max=float(KX.max()),
                dt_explicit_max=float(h ** 2 / (4 * KX.max())))


def main():
    rows = [run(N, c, dt, case=k) for k in ('noisy', 'smooth', 'warm') for N in (48, 128)
            for c in (1.0001, 10, 100, 1000) for dt in (0.005, 0.0125)]
    Path(__file__).with_suffix('.json').write_text(json.dumps(dict(rows=rows, nu=NU,
        note='PCG to 1e-10 (pcg_iters) and 1e-4 (pcg_iters_1e4) relative residual, DST-I preconditioner with geometric-mean kbar'), indent=1))
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
