"""Evaluate wave-2D FiLM nets (trained at N=16..256) against the N=512 reference.

Per training resolution N:
  - net_vs_ref512: the FiLM net evaluated NATIVELY on the 512 grid at the
    EVAL_TIMES slices, vs the f64 reference trajectories (mesh transfer +
    convergence in one number).
  - data_floor: the coarse-grid FD trajectories themselves, bilinearly
    interpolated to 512, vs the reference — the spatial discretization bound
    on anything trained at that resolution.

Both metrics are reported: traj (normalized by the trajectory-RMS norm over
the EVAL_TIMES slices — primary for wave, see wave2d_film.py) and snap
(heat-style per-snapshot relative L2).

wave2d_film.py computes its Nyquist-capped N_FREQ from the N env var at import
time, so the module is re-imported per checkpoint (the aliasing landmine from
the Poisson round; checkpoints and features must agree).

Usage:  python wave2d_convergence_eval.py <checkpoint-dir> [ref-npz] [out-json]
        checkpoint-dir contains wave2d_film_N{16,32,64,128,256}.pkl
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
    CKPT_DIR, "wave2d_convergence.json")
NS = [16, 32, 64, 128, 256]
N_REF = 512

print(f"jax_backend={jax.default_backend()}", flush=True)
ref = np.load(REF)
U_ref = ref["U"]                       # (n_val, n_times, 512^2) f64
eval_times = ref["eval_times"].tolist()
n_val, n_times = U_ref.shape[:2]
ref_norm = np.linalg.norm(U_ref, axis=2)               # (n_val, n_times)
ref_rms = np.sqrt(np.mean(ref_norm**2, axis=1))        # (n_val,)

x_ref = np.linspace(0.0, 1.0, N_REF)
Xr, Yr = np.meshgrid(x_ref, x_ref, indexing="ij")
coords_ref = jnp.asarray(np.stack([Xr.reshape(-1), Yr.reshape(-1)], axis=1),
                         dtype=jnp.float32)

results = {}
for N in NS:
    ck = os.path.join(CKPT_DIR, f"wave2d_film_N{N}.pkl")
    if not os.path.exists(ck):
        print(f"N={N}: no checkpoint, skipping", flush=True)
        continue
    t0 = time.time()
    os.environ["N"] = str(N)
    spec = importlib.util.spec_from_file_location(
        f"wf{N}", os.path.join(HERE, "wave2d_film.py"))
    wf = importlib.util.module_from_spec(spec)
    sys.modules[f"wf{N}"] = wf
    spec.loader.exec_module(wf)
    assert wf.EVAL_TIMES == eval_times, "eval-time mismatch vs refgen"

    cx, cy, w, a, c, z = wf.sample_params(seed=wf.SEED,
                                          m=wf.N_TRAIN + wf.N_VAL)
    sl = slice(wf.N_TRAIN, wf.N_TRAIN + wf.N_VAL)
    assert np.allclose(cx[sl], ref["cx"]), "param mismatch vs refgen"
    assert np.allclose(c[sl], ref["c"]), "param mismatch vs refgen (c)"
    z_va = z[sl]
    taus = np.asarray(eval_times, dtype=np.float64) / wf.NUM_STEPS

    with open(ck, "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    net = jax.jit(lambda z_i, tau_i, pts: wf.film_apply(params, z_i, tau_i, pts))

    # --- net natively on the 512 grid ---
    errs = np.empty((n_val, n_times))                  # per-snap relative
    errs_d = np.empty((n_val, n_times))                # raw diff norms
    CH = 65536
    for i in range(n_val):
        z_i = jnp.asarray(z_va[i])
        for kt, tau in enumerate(taus):
            tau_i = jnp.asarray(tau, dtype=jnp.float32)
            pred = np.concatenate([
                np.asarray(net(z_i, tau_i, coords_ref[s:s + CH]),
                           dtype=np.float64)
                for s in range(0, N_REF * N_REF, CH)])
            d = np.linalg.norm(pred - U_ref[i, kt])
            errs_d[i, kt] = d
            errs[i, kt] = d / max(ref_norm[i, kt], 1e-300)
    e_net_snap = float(errs.mean())
    e_net_traj = float((errs_d / ref_rms[:, None]).mean())

    # --- data floor: coarse FD trajectories interpolated to 512 ---
    rollout, _ = wf.make_rollout(N)
    x_c = np.linspace(0.0, 1.0, N)
    Xc, Yc = np.meshgrid(x_c, x_c, indexing="ij")
    mask_c = np.asarray(wf.boundary_mask(N))
    ci = np.stack([Xr * (N - 1), Yr * (N - 1)]).reshape(2, -1)
    floors = np.empty((n_val, n_times))
    floors_d = np.empty((n_val, n_times))
    CHUNK = max(1, 4096 // N)
    for s in range(0, n_val, CHUNK):
        e = min(s + CHUNK, n_val)
        U0 = np.stack([
            (ref["a"][i] * np.exp(-((Xc - ref["cx"][i]) ** 2
                                    + (Yc - ref["cy"][i]) ** 2)
                                  / (2 * ref["w"][i] ** 2)) * mask_c).reshape(-1)
            for i in range(s, e)])
        snaps, _ens = rollout(jnp.asarray(U0), jnp.asarray(ref["c"][s:e]))
        snaps = np.asarray(snaps)
        for bi, i in enumerate(range(s, e)):
            for kt, k in enumerate(eval_times):
                U_c = snaps[k, bi].reshape(N, N)
                interp = map_coordinates(U_c, ci, order=1)
                d = np.linalg.norm(interp - U_ref[i, kt])
                floors_d[i, kt] = d
                floors[i, kt] = d / max(ref_norm[i, kt], 1e-300)
    e_floor_snap = float(floors.mean())
    e_floor_traj = float((floors_d / ref_rms[:, None]).mean())

    results[N] = {
        "net_vs_ref512": e_net_traj,
        "net_vs_ref512_snap": e_net_snap,
        "data_floor": e_floor_traj,
        "data_floor_snap": e_floor_snap,
        "net_per_time": {str(k): float((errs_d[:, kt] / ref_rms).mean())
                         for kt, k in enumerate(eval_times)},
        "net_per_time_snap": {str(k): float(errs[:, kt].mean())
                              for kt, k in enumerate(eval_times)},
        "n_freq": wf.N_FREQ,
    }
    print(f"N={N:4d}  data-floor={e_floor_traj:.3e}  "
          f"net-vs-ref512={e_net_traj:.3e}  (snap: {e_floor_snap:.3e} / "
          f"{e_net_snap:.3e}, n_freq={wf.N_FREQ})  [{time.time()-t0:.0f}s]",
          flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {OUT}", flush=True)
