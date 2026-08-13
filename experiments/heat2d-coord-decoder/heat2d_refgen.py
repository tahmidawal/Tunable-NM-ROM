"""Generate the N=512 heat-2D reference trajectories for the val parameters.

Same seed-0 parameter draw as heat2d_film.py (val = last N_VAL entries), same
backward-Euler dt/steps — so the reference isolates SPATIAL discretization
error; time-stepping error is common to every resolution and cancels in the
comparison. Stores only the EVAL_TIMES snapshot slices in f64.

Output: ref512_val.npz with U (n_val, len(EVAL_TIMES), 512*512), eval_times,
and the val params for sanity checks.

Usage:  python heat2d_refgen.py [outdir]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heat2d_film as hf

N_REF = 512
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
CHUNK = 8   # trajectories per rollout batch at 512^2

backend = jax.default_backend()
print(f"jax_backend={backend}", flush=True)

cx, cy, w, a, kappa, z = hf.sample_params(seed=hf.SEED,
                                          m=hf.N_TRAIN + hf.N_VAL)
sl = slice(hf.N_TRAIN, hf.N_TRAIN + hf.N_VAL)
cx, cy, w, a, kappa = cx[sl], cy[sl], w[sl], a[sl], kappa[sl]

rollout, _ = hf.make_rollout(N_REF)
x = np.linspace(0.0, 1.0, N_REF)
X, Y = np.meshgrid(x, x, indexing="ij")
mask = np.asarray(hf.boundary_mask(N_REF))

U_ref = np.zeros((hf.N_VAL, len(hf.EVAL_TIMES), N_REF * N_REF))
t0 = time.time()
for s in range(0, hf.N_VAL, CHUNK):
    e = min(s + CHUNK, hf.N_VAL)
    U0 = np.stack([
        (a[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2)
                       / (2 * w[i] ** 2)) * mask).reshape(-1)
        for i in range(s, e)])
    snaps = np.asarray(rollout(jnp.asarray(U0), jnp.asarray(kappa[s:e])))
    U_ref[s:e] = snaps[hf.EVAL_TIMES].transpose(1, 0, 2)
    print(f"  {e}/{hf.N_VAL} trajectories [{time.time()-t0:.0f}s]", flush=True)

out = os.path.join(OUTDIR, "ref512_val.npz")
np.savez(out, U=U_ref, eval_times=np.asarray(hf.EVAL_TIMES),
         cx=cx, cy=cy, w=w, a=a, kappa=kappa)
print(f"wrote {out} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)
