"""Fourier pseudo-spectral solver with Brinkman volume penalisation on a perforated box (CPU).

This is the "keep the FFT, embed the geometry" competitor (immersed boundary / volume
penalisation / Fourier embedding). Problem: -Laplace(u) = f in the fluid region of a box with
one circular hole, u = 0 on the hole and outer boundary. Exact solution is manufactured so
that it vanishes on both boundaries:
    u = sin(pi x) sin(pi y) * (1 - (R/rho)^2)  on the fluid,  rho = |x - c|,
defined on the box (0,1)^2 with hole radius R centred at c = (0.5, 0.5). u is zero on the hole
boundary and on the box walls.

The penalised problem is posed on the periodic cell [-0.5, 1.5)^2, where the solid is everything
outside the fluid (the hole and the margin around the unit box):
    -Laplace(u) + chi/eta * u = f * (1 - chi)
solved by preconditioned CG with the Fourier inverse of (-Laplace + 1/eta_bar). eta = (h/pi)^2
scaling is used (error then O(h) in the fluid). Records relative L2 error in the fluid and the
fitted order, and extrapolates the resolution needed for 1 % and 0.2 % (extrapolations, not
measurements). This is one binary-mask penalty scaling; higher-order active penalty, corrected
immersed-interface and Fourier-continuation methods are not represented.

Writes penalized_spectral.json next to this file.
"""
import json
import time
from pathlib import Path

import numpy as np

R = 0.2
C = (0.5, 0.5)


def fields(N):
    # periodic cell [-0.5, 1.5)^2 so that the physical box (0,1)^2 sits inside with solid margins
    L = 2.0
    h = L / N
    xs = -0.5 + h * np.arange(N)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    rho = np.hypot(X - C[0], Y - C[1])
    fluid = (X > 0) & (X < 1) & (Y > 0) & (Y < 1) & (rho > R)
    s = np.sin(np.pi * X) * np.sin(np.pi * Y)
    g = 1 - (R / np.maximum(rho, 1e-12)) ** 2
    u = np.where(fluid, s * g, 0.0)
    # f = -Lap(s g) = -(g Lap s + 2 grad s . grad g + s Lap g); Lap g = -4 R^2/rho^4 ... compute analytically
    lap_s = -2 * np.pi ** 2 * s
    sx = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    sy = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
    rr = np.maximum(rho, 1e-12)
    gx = 2 * R ** 2 * (X - C[0]) / rr ** 4
    gy = 2 * R ** 2 * (Y - C[1]) / rr ** 4
    lap_g = -4 * R ** 2 / rr ** 4  # Laplacian of -R^2 rho^-2 in 2D is -4R^2 rho^-4
    f = np.where(fluid, -(g * lap_s + 2 * (sx * gx + sy * gy) + s * lap_g), 0.0)
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    return X, Y, fluid, u, f, KX ** 2 + KY ** 2, h


def solve(N):
    X, Y, fluid, u_ex, f, K2, h = fields(N)
    chi = (~fluid).astype(float)
    eta = (h / np.pi) ** 2
    pen = chi / eta
    rhs = f
    shift = 1.0 / eta * chi.mean()
    Pinv = lambda r: np.real(np.fft.ifft2(np.fft.fft2(r) / (K2 + shift)))
    A = lambda v: np.real(np.fft.ifft2(K2 * np.fft.fft2(v))) + pen * v
    t0 = time.perf_counter()
    x = np.zeros_like(rhs); r = rhs - A(x); z = Pinv(r); p = z.copy(); rz = np.vdot(r, z)
    nb = np.linalg.norm(rhs); it = 0
    for it in range(1, 20001):
        Ap = A(p); a = rz / np.vdot(p, Ap); x += a * p; r -= a * Ap
        if np.linalg.norm(r) < 1e-10 * nb:
            break
        z = Pinv(r); rz2 = np.vdot(r, z); p = z + (rz2 / rz) * p; rz = rz2
    t = time.perf_counter() - t0
    e = (x - u_ex)[fluid]
    res = float(np.linalg.norm(rhs - A(x)) / nb)
    return dict(N=N, h=h, cg_iters=it, converged=bool(res < 1e-9), final_rel_residual=res, rel_l2=float(np.linalg.norm(e) / np.linalg.norm(u_ex[fluid])),
                max_err=float(np.abs(e).max() / np.abs(u_ex[fluid]).max()), solve_s=t)


def main():
    rows = [solve(N) for N in (64, 128, 256, 512, 1024)]
    Ns = np.array([r['N'] for r in rows], float)
    e = np.array([r['rel_l2'] for r in rows])
    p = float(-np.polyfit(np.log(Ns[-3:]), np.log(e[-3:]), 1)[0])
    r1, r2 = rows[-2], rows[-1]
    pl = np.log(r1['rel_l2'] / r2['rel_l2']) / np.log(2)
    need = {str(t): float(r2['N'] * (r2['rel_l2'] / t) ** (1 / pl)) for t in (1e-2, 2e-3, 1e-3, 1e-4)}
    out = dict(penalized=rows, order=p, N_for_rel_l2_periodic_cell=need,
               note='periodic cell side 2 holds the unit box: physical points per unit length = N/2; '
                    'N_for_rel_l2 values are EXTRAPOLATED from the last two N with the local order')
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))
    for r in rows:
        print(r)
    print('order', p, 'N needed (periodic cell)', need)


if __name__ == '__main__':
    main()
