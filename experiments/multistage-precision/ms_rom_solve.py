"""Experiment C: Gauss-Newton ROM solve on the multi-stage parametric decoder.

The user's question: does multi-stage decoder precision SURVIVE the reduced-
order solve, or does something else (GN convergence, collocation sampling,
conditioning) become the floor first?

Setup (Poisson 2D, matching Experiment B): for each test parameter, the online
solver knows ONLY the source f(x; cx,cy,w,a) and the trained decoder. It
minimizes the discrete FD residual of the combined decoder over the 4-dim
conditioning latent z with damped Gauss-Newton (the coord-net GN/EQ skeleton
from exp/2026-08-13-cost-scaling-coordnet, upgraded from cost-timing to a real
solve): residual rows = 5-point stencil rows at m interior nodes + boundary
rows weighted 1/dx^2 (matching stencil-row magnitude). Because the training
fields are exact discrete FD solutions, a perfect decoder at the true z zeros
this residual exactly — so ROM-solve error is bounded below by decoder
representation error, and the gap above that floor is solver-induced.

Arms: stages used in {1, 2, ..., K} x collocation {full interior, m=512 EQ-
style random}. Metrics per test sample: field rel-L2 vs FD truth at the
GN-solved z, rel-L2 at the TRUE z (representation floor), latent error.

Usage: [N_TEST=16] [GN_ITERS=40] [M_EQ=512] python ms_rom_solve.py [outdir]
(run AFTER ms_parametric.py with the same env; reads its pkl/json from outdir)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import ms_parametric as mp

N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "40"))
M_EQ = int(os.environ.get("M_EQ", "512"))
DAMPING = float(os.environ.get("DAMPING", "1e-10"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))

F64 = jnp.float64


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}",
          flush=True)
    with open(os.path.join(OUTDIR, "ms_parametric_report.json")) as f:
        brep = json.load(f)
    N = brep["N"]
    assert mp.HIDDEN == brep["hidden"] and mp.N_LAYERS == brep["n_layers"], (
        "run with the same HIDDEN env as ms_parametric.py")
    with open(os.path.join(OUTDIR, "ms_parametric_stages.pkl"), "rb") as f:
        raw = pickle.load(f)
    stages = [{"params": jax.tree_util.tree_map(jnp.asarray, s["params"]),
               "n_freq": s["n_freq"], "eps": s["eps"]} for s in raw]
    K = len(stages)
    print(f"loaded {K} stages, N={N}", flush=True)

    # test parameters = first N_TEST val samples of the B family (exact split)
    cx, cy, w, a, z_all = mp.sample_params()
    n_tr = brep["n_train"]
    sl = slice(n_tr, n_tr + N_TEST)
    cx, cy, w, a, z_true = cx[sl], cy[sl], w[sl], a[sl], z_all[sl]

    # FD truths (f64 CG, same solver as B)
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    op = lambda v: mp.neg_lap_interior(v, N)
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        op, F, tol=1e-13, maxiter=100_000)[0])
    F_all = np.stack([a[i] * np.exp(-((Xi - cx[i]) ** 2 + (Yi - cy[i]) ** 2)
                                    / (2 * w[i] ** 2)) for i in range(N_TEST)])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(F_all)))
    U_true = np.zeros((N_TEST, N, N))
    U_true[:, 1:-1, 1:-1] = U_int
    U_true = U_true.reshape(N_TEST, N * N)
    coords = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1))

    dx = 1.0 / (N - 1)
    rng = np.random.default_rng(SEED)

    def stencil_pts(ix, iy):
        px, py = ix * dx, iy * dx
        return np.stack([
            np.stack([px, py], axis=1),
            np.stack([px + dx, py], axis=1),
            np.stack([px - dx, py], axis=1),
            np.stack([px, py + dx], axis=1),
            np.stack([px, py - dx], axis=1),
        ])                                              # (5, m, 2)

    # full-interior and EQ-subset collocation
    ii, jj = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1),
                         indexing="ij")
    ix_full, iy_full = ii.reshape(-1), jj.reshape(-1)
    sub = rng.choice(len(ix_full), size=min(M_EQ, len(ix_full)), replace=False)
    colls = {"full": (ix_full, iy_full),
             f"m{M_EQ}": (ix_full[sub], iy_full[sub])}

    # boundary points, weighted like stencil rows
    bmask = np.zeros((N, N), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bxy = np.stack([X[bmask], Y[bmask]], axis=1)
    bw = 1.0 / dx**2

    def make_solver(n_stages, pts, f_vals, bpts):
        st = stages[:n_stages]

        def dec(z, xy):
            return mp.combined_apply(st, z, xy)

        def residual(z):
            u = dec(z, pts.reshape(-1, 2)).reshape(5, -1)
            lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (dx * dx)
            r_int = -lap - f_vals
            r_bnd = bw * dec(z, bpts)
            return jnp.concatenate([r_int, r_bnd])

        @jax.jit
        def gn_step(z):
            r = residual(z)
            J = jax.jacfwd(residual)(z)
            H = J.T @ J + DAMPING * jnp.eye(J.shape[1], dtype=F64)
            g = J.T @ r
            return z - jnp.linalg.solve(H, g), jnp.linalg.norm(r)

        return gn_step, dec

    results = []
    for name, (ix, iy) in colls.items():
        pts = jnp.asarray(stencil_pts(ix, iy))
        bpts = jnp.asarray(bxy)
        for n_stages in range(1, K + 1):
            errs, errs_true_z, z_errs, rnorms = [], [], [], []
            t0 = time.time()
            for i in range(N_TEST):
                f_vals = jnp.asarray(
                    a[i] * np.exp(-((ix * dx - cx[i]) ** 2
                                    + (iy * dx - cy[i]) ** 2)
                                  / (2 * w[i] ** 2)))
                gn_step, dec = make_solver(n_stages, pts, f_vals, bpts)
                z = jnp.zeros((4,), dtype=F64)
                for _ in range(GN_ITERS):
                    z, rn = gn_step(z)
                pred = np.asarray(dec(z, coords))
                pred_tz = np.asarray(dec(jnp.asarray(z_true[i]), coords))
                nrm = np.linalg.norm(U_true[i])
                errs.append(np.linalg.norm(pred - U_true[i]) / nrm)
                errs_true_z.append(np.linalg.norm(pred_tz - U_true[i]) / nrm)
                z_errs.append(float(np.linalg.norm(np.asarray(z) - z_true[i])))
                rnorms.append(float(rn))
            row = {"colloc": name, "n_stages": n_stages,
                   "rom_rel_l2_mean": float(np.mean(errs)),
                   "rom_rel_l2_med": float(np.median(errs)),
                   "truez_rel_l2_mean": float(np.mean(errs_true_z)),
                   "z_err_med": float(np.median(z_errs)),
                   "gn_resid_med": float(np.median(rnorms)),
                   "secs": time.time() - t0}
            results.append(row)
            print(f"RESULT colloc={name:6s} stages={n_stages}  "
                  f"ROM rel-L2 mean {row['rom_rel_l2_mean']:.3e} "
                  f"(med {row['rom_rel_l2_med']:.3e})  "
                  f"true-z floor {row['truez_rel_l2_mean']:.3e}  "
                  f"z-err med {row['z_err_med']:.2e}", flush=True)

    with open(os.path.join(OUTDIR, "ms_rom_solve_report.json"), "w") as f:
        json.dump({"N": N, "n_test": N_TEST, "gn_iters": GN_ITERS,
                   "m_eq": M_EQ, "damping": DAMPING, "results": results},
                  f, indent=2)
    print("wrote ms_rom_solve_report.json", flush=True)


if __name__ == "__main__":
    main()
