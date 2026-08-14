"""Generate the N=512 wave-2D reference trajectories for the val parameters.

Same seed-0 parameter draw as wave2d_film.py (val = last N_VAL entries), same
Crank-Nicolson dt/substeps — so the reference isolates SPATIAL discretization
error; time-stepping error is common to every resolution and cancels in the
comparison (and is independently bounded by wave2d_selfconv.py's dt-halving
check).  Stores only the EVAL_TIMES snapshot slices in f64.

Output: ref512_val.npz with U (n_val, len(EVAL_TIMES), 512*512), eval_times,
the val params, and the per-trajectory energy drift of the reference solve.

Usage:  python wave2d_refgen.py [outdir]
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
import wave2d_film as wf

N_REF = 512
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
CHUNK = 8   # trajectories per rollout batch at 512^2

backend = jax.default_backend()
print(f"jax_backend={backend}", flush=True)

cx, cy, w, a, c, z = wf.sample_params(seed=wf.SEED, m=wf.N_TRAIN + wf.N_VAL)
sl = slice(wf.N_TRAIN, wf.N_TRAIN + wf.N_VAL)
cx, cy, w, a, c = cx[sl], cy[sl], w[sl], a[sl], c[sl]

rollout, _ = wf.make_rollout(N_REF)
x = np.linspace(0.0, 1.0, N_REF)
X, Y = np.meshgrid(x, x, indexing="ij")
mask = np.asarray(wf.boundary_mask(N_REF))

U_ref = np.zeros((wf.N_VAL, len(wf.EVAL_TIMES), N_REF * N_REF))
drifts = np.zeros(wf.N_VAL)
t0 = time.time()
for s in range(0, wf.N_VAL, CHUNK):
    e = min(s + CHUNK, wf.N_VAL)
    U0 = np.stack([
        (a[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2)
                       / (2 * w[i] ** 2)) * mask).reshape(-1)
        for i in range(s, e)])
    snaps, ens = rollout(jnp.asarray(U0), jnp.asarray(c[s:e]))
    U_ref[s:e] = np.asarray(snaps)[wf.EVAL_TIMES].transpose(1, 0, 2)
    ens = np.asarray(ens)                              # (T+1, B)
    drifts[s:e] = np.max(np.abs(ens - ens[0])
                         / np.maximum(ens[0], 1e-300), axis=0)
    print(f"  {e}/{wf.N_VAL} trajectories, max drift so far "
          f"{drifts[:e].max():.2e} [{time.time()-t0:.0f}s]", flush=True)

out = os.path.join(OUTDIR, "ref512_val.npz")
np.savez(out, U=U_ref, eval_times=np.asarray(wf.EVAL_TIMES),
         cx=cx, cy=cy, w=w, a=a, c=c, energy_drift=drifts,
         substeps=wf.SUBSTEPS, t_final=wf.T_FINAL)
print(f"wrote {out} ({os.path.getsize(out)/1e9:.2f} GB), "
      f"max energy drift {drifts.max():.2e}", flush=True)
