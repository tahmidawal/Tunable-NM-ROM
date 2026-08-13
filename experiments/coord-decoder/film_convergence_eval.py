"""Evaluate FiLM nets (trained at N=16..256) against the N=512 reference.

Also computes the per-N data-floor: the coarse FD val solutions themselves,
bilinearly interpolated to the 512 grid, vs the reference — the bound set
by the training data's discretization error.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
from scipy.ndimage import map_coordinates
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("N", "16")
spec = importlib.util.spec_from_file_location(
    "film", os.path.join(HERE, "poisson2d_film.py"))
film = importlib.util.module_from_spec(spec)
sys.modules["film"] = film
spec.loader.exec_module(film)

NS = [16, 32, 64, 128, 256]
N_REF = 512
N_TRAIN, N_VAL = 2048, 256

ref = np.load(os.path.join(HERE, "film", "ref512_val.npz"))
U_ref = ref["U"]  # (256, 512*512) float64

# sanity: parameter draw must match
cx, cy, w, a, z = film.sample_params(seed=0, m=N_TRAIN + N_VAL)
assert np.allclose(cx[N_TRAIN:], ref["cx"]), "param mismatch vs refgen"
z_va = z[N_TRAIN:]

x_ref = np.linspace(0.0, 1.0, N_REF)
Xr, Yr = np.meshgrid(x_ref, x_ref, indexing="ij")
coords_ref = jnp.asarray(np.stack([Xr.reshape(-1), Yr.reshape(-1)], axis=1),
                         dtype=jnp.float32)

ref_norm = np.linalg.norm(U_ref, axis=1)

results = {}
for N in NS:
    t0 = time.time()
    # --- net eval natively on the 512 grid (chunked over points) ---
    with open(os.path.join(HERE, "film", f"film_params_N{N}.pkl"), "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params)

    @jax.jit
    def net_at(z_i, pts):
        return film.film_apply(params, z_i, pts)

    errs = np.empty(N_VAL)
    CH = 65536
    for i in range(N_VAL):
        z_i = jnp.asarray(z_va[i])
        pred = np.concatenate([
            np.asarray(net_at(z_i, coords_ref[s:s + CH]), dtype=np.float64)
            for s in range(0, N_REF * N_REF, CH)])
        errs[i] = np.linalg.norm(pred - U_ref[i]) / ref_norm[i]
    e_net = float(errs.mean())

    # --- data floor: coarse FD val solutions interpolated to 512 ---
    os.environ["N"] = str(N)
    U_c, _, _, _, _ = None, None, None, None, None
    # regenerate only the val slice at resolution N
    x_c = np.linspace(0.0, 1.0, N)
    Xc, Yc = np.meshgrid(x_c, x_c, indexing="ij")
    Xi, Yi = Xc[1:-1, 1:-1], Yc[1:-1, 1:-1]
    cxv, cyv, wv, av = cx[N_TRAIN:], cy[N_TRAIN:], w[N_TRAIN:], a[N_TRAIN:]
    F = np.stack([av[i] * np.exp(-((Xi - cxv[i]) ** 2 + (Yi - cyv[i]) ** 2)
                                 / (2 * wv[i] ** 2)) for i in range(N_VAL)])
    U_int = np.asarray(film.fd_solve_batch(jnp.asarray(F), N))
    U_c = np.zeros((N_VAL, N, N))
    U_c[:, 1:-1, 1:-1] = U_int

    # bilinear interp each coarse field onto the 512 grid
    scale = (N - 1) / (N_REF - 1)
    ci = np.stack([Xr * (N - 1), Yr * (N - 1)])  # index coords in coarse grid
    floors = np.empty(N_VAL)
    for i in range(N_VAL):
        interp = map_coordinates(U_c[i], ci.reshape(2, -1), order=1)
        floors[i] = np.linalg.norm(interp - U_ref[i]) / ref_norm[i]
    e_floor = float(floors.mean())

    results[N] = {"net_vs_ref512": e_net, "data_floor": e_floor}
    print(f"N={N:4d}  data-floor={e_floor:.3e}  film-net-vs-ref512={e_net:.3e}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

with open(os.path.join(HERE, "film", "film_convergence.json"), "w") as f:
    json.dump(results, f, indent=2)
print("wrote film/film_convergence.json", flush=True)
