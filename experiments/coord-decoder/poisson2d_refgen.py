"""Generate the N=512 reference validation solutions for the FiLM sweep.

Must use the SAME parameter draw as poisson2d_film.py: seed 0,
m = N_TRAIN + N_VAL = 2048 + 256; the val set is the trailing 256.
Saves float64 solutions to ref512_val.npz.
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

N_REF = 512
N_TRAIN, N_VAL = 2048, 256
SEED = 0
CG_TOL = 1e-11
CG_MAXITER = 60_000


def neg_lap_interior(u_int, n):
    dx = 1.0 / (n - 1)
    u = jnp.pad(u_int, 1)
    lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
           - 4.0 * u[1:-1, 1:-1]) / dx**2
    return -lap


def main():
    print(f"jax_backend={jax.default_backend()}", flush=True)
    rng = np.random.default_rng(SEED)
    m = N_TRAIN + N_VAL
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), m))
    a = rng.uniform(0.5, 2.0, m)
    cx, cy, w, a = cx[N_TRAIN:], cy[N_TRAIN:], w[N_TRAIN:], a[N_TRAIN:]

    n = N_REF
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]

    op = lambda v: neg_lap_interior(v, n)

    @jax.jit
    def solve_one(F_int):
        u, _ = jax.scipy.sparse.linalg.cg(op, F_int, tol=CG_TOL, maxiter=CG_MAXITER)
        return u

    U = np.zeros((N_VAL, n, n))
    t0 = time.time()
    res_max = 0.0
    for i in range(N_VAL):
        F = a[i] * np.exp(-((Xi - cx[i]) ** 2 + (Yi - cy[i]) ** 2) / (2 * w[i] ** 2))
        u_int = np.asarray(solve_one(jnp.asarray(F)))
        U[i, 1:-1, 1:-1] = u_int
        if i % 64 == 0:
            r = np.asarray(neg_lap_interior(jnp.asarray(u_int), n)) - F
            res_max = max(res_max, np.linalg.norm(r) / np.linalg.norm(F))
            print(f"  {i}/{N_VAL} [{time.time()-t0:.0f}s] res {res_max:.2e}", flush=True)

    np.savez_compressed("ref512_val.npz", U=U.reshape(N_VAL, n * n),
                        cx=cx, cy=cy, w=w, a=a)
    print(f"DONE {N_VAL} solves in {time.time()-t0:.0f}s, "
          f"max spot residual {res_max:.2e}", flush=True)


if __name__ == "__main__":
    main()
