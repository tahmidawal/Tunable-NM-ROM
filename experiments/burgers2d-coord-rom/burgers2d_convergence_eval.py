"""Evaluate Burgers-2D FiLM nets (trained at N=16..256) vs the N=512 reference.

Per training resolution N:
  - net_vs_ref512: the FiLM net evaluated NATIVELY on the 512 grid at the
    EVAL_TIMES slices, vs the f64 reference trajectories (mesh transfer +
    convergence in one number).
  - data_floor: the coarse-grid FD trajectories themselves, bilinearly
    interpolated to 512, vs the reference -- the spatial discretization bound
    on anything trained at that resolution.

burgers2d_film.py computes its Nyquist-capped N_FREQ from the N env var at
import time, so the module is re-imported per checkpoint (checkpoints and
features must agree). N_TRAIN/N_VAL must match the training runs (ambient env).

Usage:  python burgers2d_convergence_eval.py <ckpt-dir> [ref-npz] [out-json]
        ckpt-dir contains burgers2d_film_N{16,32,64,128,256}.pkl
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else HERE
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "ref512_val.npz")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    CKPT_DIR, "burgers2d_convergence.json")
NS = [16, 32, 64, 128, 256]
N_REF = 512

print(f"jax_backend={jax.default_backend()}", flush=True)
ref = np.load(REF)
U_ref = ref["U"]                       # (n_val, n_times, 512^2) f64
eval_times = ref["eval_times"].tolist()
n_val, n_times = U_ref.shape[:2]
ref_norm = np.linalg.norm(U_ref, axis=2)          # (n_val, n_times)

x_ref = np.linspace(0.0, 1.0, N_REF)
Xr, Yr = np.meshgrid(x_ref, x_ref, indexing="ij")
coords_ref = jnp.asarray(np.stack([Xr.reshape(-1), Yr.reshape(-1)], axis=1),
                         dtype=jnp.float32)

results = {}
for N in NS:
    ck = os.path.join(CKPT_DIR, f"burgers2d_film_N{N}.pkl")
    if not os.path.exists(ck):
        print(f"N={N}: no checkpoint, skipping", flush=True)
        continue
    t0 = time.time()
    os.environ["N"] = str(N)
    spec = importlib.util.spec_from_file_location(
        f"bf{N}", os.path.join(HERE, "burgers2d_film.py"))
    bf = importlib.util.module_from_spec(spec)
    sys.modules[f"bf{N}"] = bf
    spec.loader.exec_module(bf)
    assert bf.EVAL_TIMES == eval_times, "eval-time mismatch vs refgen"

    cx, cy, w, a, nu, z = bf.sample_params(seed=bf.SEED,
                                           m=bf.N_TRAIN + bf.N_VAL)
    sl = slice(bf.N_TRAIN, bf.N_TRAIN + bf.N_VAL)
    assert np.allclose(cx[sl], ref["cx"]), "param mismatch vs refgen"
    assert np.allclose(nu[sl], ref["nu"]), "param mismatch vs refgen"
    z_va = z[sl]
    taus = np.asarray(eval_times, dtype=np.float64) / bf.NUM_STEPS

    with open(ck, "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    net = jax.jit(lambda z_i, tau_i, pts: bf.film_apply(params, z_i, tau_i, pts))

    # --- net natively on the 512 grid ---
    errs = np.empty((n_val, n_times))
    CH = 65536
    for i in range(n_val):
        z_i = jnp.asarray(z_va[i])
        for kt, tau in enumerate(taus):
            tau_i = jnp.asarray(tau, dtype=jnp.float32)
            pred = np.concatenate([
                np.asarray(net(z_i, tau_i, coords_ref[s:s + CH]),
                           dtype=np.float64)
                for s in range(0, N_REF * N_REF, CH)])
            errs[i, kt] = (np.linalg.norm(pred - U_ref[i, kt])
                           / ref_norm[i, kt])
    e_net = float(errs.mean())

    # --- data floor: coarse FD trajectories interpolated to 512 ---
    rollout, _ = bf.make_rollout(N)
    ci = np.stack([Xr * (N - 1), Yr * (N - 1)]).reshape(2, -1)
    floors = np.empty((n_val, n_times))
    CHUNK = max(1, 4096 // N)
    for s in range(0, n_val, CHUNK):
        e = min(s + CHUNK, n_val)
        U0 = np.stack([bf.blob_ic(N, ref["cx"][i], ref["cy"][i],
                                  ref["w"][i], ref["a"][i])
                       for i in range(s, e)])
        snaps, _ = rollout(jnp.asarray(U0), jnp.asarray(ref["nu"][s:e]))
        snaps = np.asarray(snaps)
        for bi, i in enumerate(range(s, e)):
            for kt, k in enumerate(eval_times):
                U_c = snaps[k, bi].reshape(N, N)
                interp = map_coordinates(U_c, ci, order=1)
                floors[i, kt] = (np.linalg.norm(interp - U_ref[i, kt])
                                 / ref_norm[i, kt])
    e_floor = float(floors.mean())

    results[N] = {
        "net_vs_ref512": e_net,
        "data_floor": e_floor,
        "net_per_time": {str(k): float(errs[:, kt].mean())
                         for kt, k in enumerate(eval_times)},
        "n_freq": bf.N_FREQ,
    }
    print(f"N={N:4d}  data-floor={e_floor:.3e}  net-vs-ref512={e_net:.3e}  "
          f"(n_freq={bf.N_FREQ})  [{time.time()-t0:.0f}s]", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {OUT}", flush=True)
