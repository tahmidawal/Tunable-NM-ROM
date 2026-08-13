"""Evaluate the Nyquist-fixed FiLM checkpoints (N=16, 32) against the N=512 reference.

The original film/film_convergence.json holds the pre-fix N=16/32 entries, whose
checkpoints were trained with N_FREQ=32 Fourier features — above the Nyquist limit
of those coarse training grids — so they exploited aliased features that look
correct on-grid and blow up when evaluated on the fine reference grid
(net_vs_ref512 of 3.7e4 and 1.8e3).

The retrained checkpoints in film-fix/ cap the bandwidth at the training grid's
Nyquist limit (N=16 -> n_freq=8, N=32 -> n_freq=16). This script evaluates them on
the 512 reference and writes film/film_convergence_fixed.json: the corrected
N=16/32 entries merged with the (already valid) N>=64 entries from the original
eval. This reproduces the final convergence table of the 2026-08-12 session.

Run locally (eval only, a few minutes):
    source /etc/profile.d/jax-mem.sh
    JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun \
        /home/tahmid/Dev/.venv/bin/python film_fix_eval.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
N_REF = 512
N_TRAIN, N_VAL = 2048, 256

ref = np.load(os.path.join(HERE, "film", "ref512_val.npz"))
U_ref = ref["U"]
ref_norm = np.linalg.norm(U_ref, axis=1)

x_ref = np.linspace(0.0, 1.0, N_REF)
Xr, Yr = np.meshgrid(x_ref, x_ref, indexing="ij")
coords_ref = jnp.asarray(np.stack([Xr.reshape(-1), Yr.reshape(-1)], axis=1),
                         dtype=jnp.float32)

with open(os.path.join(HERE, "film", "film_convergence.json")) as f:
    merged = json.load(f)

# poisson2d_film.py reads N and N_FREQ from the environment at import time, so
# the module is re-imported per checkpoint with the bandwidth it was trained at.
for N, NF in [(16, 8), (32, 16)]:
    os.environ["N"] = str(N)
    os.environ["N_FREQ"] = str(NF)
    spec = importlib.util.spec_from_file_location(
        f"film{N}", os.path.join(HERE, "poisson2d_film.py"))
    film = importlib.util.module_from_spec(spec)
    sys.modules[f"film{N}"] = film
    spec.loader.exec_module(film)

    cx, _, _, _, z = film.sample_params(seed=0, m=N_TRAIN + N_VAL)
    assert np.allclose(cx[N_TRAIN:], ref["cx"]), "param mismatch vs refgen"
    z_va = z[N_TRAIN:]

    with open(os.path.join(HERE, "film-fix", f"film_params_N{N}.pkl"), "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    net = jax.jit(lambda z_i, pts: film.film_apply(params, z_i, pts))

    errs = np.empty(N_VAL)
    CH = 65536
    for i in range(N_VAL):
        z_i = jnp.asarray(z_va[i])
        pred = np.concatenate([
            np.asarray(net(z_i, coords_ref[s:s + CH]), dtype=np.float64)
            for s in range(0, N_REF * N_REF, CH)])
        errs[i] = np.linalg.norm(pred - U_ref[i]) / ref_norm[i]

    merged[str(N)]["net_vs_ref512"] = float(errs.mean())
    merged[str(N)]["n_freq"] = NF
    merged[str(N)]["checkpoint"] = f"film-fix/film_params_N{N}.pkl"
    print(f"N={N} (n_freq={NF})  film-net-vs-ref512={errs.mean():.3e}", flush=True)

out = os.path.join(HERE, "film", "film_convergence_fixed.json")
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"wrote {out}", flush=True)
