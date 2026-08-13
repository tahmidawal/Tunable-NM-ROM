"""Mesh-transfer eval: 2D coord nets trained at coarse N, evaluated at N=256.

Loads round2 checkpoints (trained with N_FREQ=16 - the round-2 config), regenerates the N=256
validation set from seed 0, and evaluates each coarse-trained net natively
on the fine grid. Eval-only — no training.
"""
import os
import pickle

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

os.environ["N_FREQ"] = os.environ.get("CKPT_N_FREQ", "16")  # must match checkpoint
import importlib.util, sys
spec = importlib.util.spec_from_file_location("d2", os.path.join(
    os.path.dirname(__file__), "poisson2d_diag_nf.py"))
d2 = importlib.util.module_from_spec(spec)
sys.modules["d2"] = d2
# poisson2d_diag_nf reads env N; set before exec so constants are consistent
os.environ.setdefault("N", "256")
spec.loader.exec_module(d2)

import jax.numpy as jnp

N_EVAL = 256

U, z = d2.build_snapshots(N_EVAL)
U_va, z_va = U[d2.N_TRAIN:], z[d2.N_TRAIN:]

x = np.linspace(0.0, 1.0, N_EVAL)
X, Y = np.meshgrid(x, x, indexing="ij")
coords32 = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1),
                       dtype=jnp.float32)
apply_fine = d2.make_coord_apply(coords32)

import json
results = {"eval_grid": N_EVAL, "n_freq": int(os.environ["N_FREQ"]),
           "checkpoints": "round2", "seed": 0, "cells": {}}
for n_src in [32, 64, 128, 256]:
    with open(f"round2/coord_params_2d_N{n_src}.pkl", "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    err = d2.eval_full(apply_fine, params, U_va, z_va, N_EVAL * N_EVAL)
    results["cells"][str(n_src)] = float(err)
    print(f"coord net trained N={n_src:3d} -> evaluated on N=256 grid: "
          f"rel_l2={err:.3e}", flush=True)
out = os.path.join(os.path.dirname(__file__), "round2", "transfer2d_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {out}", flush=True)
