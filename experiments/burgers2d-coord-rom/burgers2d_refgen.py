"""Generate the N=512 Burgers-2D reference trajectories for the val parameters.

Same seed-0 parameter draw as burgers2d_film.py (val = last N_VAL entries),
same backward-Euler dt/steps and Newton/BiCGStab solver -- so the reference
isolates SPATIAL discretization error; time-stepping error is common to every
resolution and cancels in the comparison. Stores only the EVAL_TIMES snapshot
slices in f64, plus the max Newton relative residual as a solver audit.

Output: ref512_val.npz with U (n_val, len(EVAL_TIMES), 512*512), eval_times,
the val params, and newton_res_max.

Usage:  python burgers2d_refgen.py [outdir]
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
import burgers2d_film as bf

N_REF = 512
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
CHUNK = 8   # trajectories per rollout batch at 512^2

backend = jax.default_backend()
print(f"jax_backend={backend}", flush=True)

cx, cy, w, a, nu, z = bf.sample_params(seed=bf.SEED,
                                       m=bf.N_TRAIN + bf.N_VAL)
sl = slice(bf.N_TRAIN, bf.N_TRAIN + bf.N_VAL)
cx, cy, w, a, nu = cx[sl], cy[sl], w[sl], a[sl], nu[sl]

rollout, _ = bf.make_rollout(N_REF)

U_ref = np.zeros((bf.N_VAL, len(bf.EVAL_TIMES), N_REF * N_REF))
res_max = 0.0
t0 = time.time()
for s in range(0, bf.N_VAL, CHUNK):
    e = min(s + CHUNK, bf.N_VAL)
    U0 = np.stack([bf.blob_ic(N_REF, cx[i], cy[i], w[i], a[i])
                   for i in range(s, e)])
    snaps, res = rollout(jnp.asarray(U0), jnp.asarray(nu[s:e]))
    U_ref[s:e] = np.asarray(snaps)[bf.EVAL_TIMES].transpose(1, 0, 2)
    res_max = max(res_max, float(jnp.max(res)))
    print(f"  {e}/{bf.N_VAL} trajectories [{time.time()-t0:.0f}s] "
          f"(max Newton rel res so far {res_max:.2e})", flush=True)

if not np.isfinite(res_max) or res_max > 1e-8:
    print(f"WARNING: Newton residual {res_max:.2e} above 1e-8", flush=True)

out = os.path.join(OUTDIR, "ref512_val.npz")
np.savez(out, U=U_ref, eval_times=np.asarray(bf.EVAL_TIMES),
         cx=cx, cy=cy, w=w, a=a, nu=nu, newton_res_max=res_max)
print(f"wrote {out} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)
