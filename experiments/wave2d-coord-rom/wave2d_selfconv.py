"""Self-convergence validation of the wave-2D Crank-Nicolson FOM.

Two fixed parameter points — a representative one (median width, c=1) and the
resolution stress case (sharpest blob w=0.05 at the fastest speed c=2,
dissipation-free propagation) — each rolled out at N in {64,128,256} with the
SAME dt and compared against the N=512 run (cubic interpolation to the 512
grid, so interpolation error O(h^4) does not contaminate the O(h^2) scheme
order).  Reports the observed spatial order log2(e_N / e_2N).

Also the explicit CN-dispersion check: N=512 rerun with dt halved
(2*BASE_SUBSTEPS); the difference to the BASE_SUBSTEPS run is the temporal
error at the reference resolution and must sit well below the N=256 -> 512
spatial difference.

Errors are per-snapshot relative L2 vs the reference, averaged over EVAL_TIMES
slices EXCLUDING t=0 (t=0 only measures IC sampling/interp error).

Usage:  python wave2d_selfconv.py [out-json]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "selfconv.json")
N_REF = 512
NS = [64, 128, 256]
BASE_SUBSTEPS = 80          # must match the wave2d_film.py default


CASES = {
    "representative": dict(cx=0.42, cy=0.57, w=0.125, a=5.0, c=1.0),
    "stress":         dict(cx=0.42, cy=0.57, w=0.05,  a=5.0, c=2.0),
}


def load_wf(n, substeps):
    os.environ["N"] = str(n)
    os.environ["SUBSTEPS"] = str(substeps)
    spec = importlib.util.spec_from_file_location(
        f"wf{n}_{substeps}", os.path.join(HERE, "wave2d_film.py"))
    wf = importlib.util.module_from_spec(spec)
    sys.modules[f"wf{n}_{substeps}"] = wf
    spec.loader.exec_module(wf)
    return wf


def run(n, substeps, p):
    wf = load_wf(n, substeps)
    rollout, _ = wf.make_rollout(n)
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.asarray(wf.boundary_mask(n))
    U0 = (p["a"] * np.exp(-((X - p["cx"]) ** 2 + (Y - p["cy"]) ** 2)
                          / (2 * p["w"] ** 2)) * mask).reshape(1, -1)
    snaps, ens = rollout(jnp.asarray(U0), jnp.asarray([p["c"]]))
    snaps = np.asarray(snaps)[:, 0]                    # (T+1, n^2)
    ens = np.asarray(ens)[:, 0]
    drift = float(np.max(np.abs(ens - ens[0]) / max(ens[0], 1e-300)))
    return snaps[wf.EVAL_TIMES], drift, wf.EVAL_TIMES


def to_ref(U_slices, n):
    x_ref = np.linspace(0.0, 1.0, N_REF)
    Xr, Yr = np.meshgrid(x_ref, x_ref, indexing="ij")
    ci = np.stack([Xr * (n - 1), Yr * (n - 1)]).reshape(2, -1)
    return np.stack([map_coordinates(u.reshape(n, n), ci, order=3)
                     for u in U_slices])


print(f"jax_backend={jax.default_backend()}", flush=True)
results = {}
for name, p in CASES.items():
    t0 = time.time()
    ref, drift_ref, eval_times = run(N_REF, BASE_SUBSTEPS, p)
    ref_norm = np.linalg.norm(ref, axis=1)
    print(f"[{name}] N=512 ref done, energy drift {drift_ref:.2e} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    errs = {}
    drifts = {N_REF: drift_ref}
    for n in NS:
        U_c, drift, _ = run(n, BASE_SUBSTEPS, p)
        interp = to_ref(U_c, n)
        rel = np.linalg.norm(interp - ref, axis=1) / ref_norm
        errs[n] = float(rel[1:].mean())               # exclude t=0
        drifts[n] = drift
        print(f"[{name}] N={n:4d}  err={errs[n]:.3e}  drift={drift:.2e}",
              flush=True)
    orders = {f"{a}->{b}": float(np.log2(errs[a] / errs[b]))
              for a, b in zip(NS, NS[1:])}

    # dt check at the reference resolution
    ref_dt2, drift_dt2, _ = run(N_REF, 2 * BASE_SUBSTEPS, p)
    dt_err = float((np.linalg.norm(ref_dt2 - ref, axis=1)
                    / ref_norm)[1:].mean())
    spatial_256 = errs[NS[-1]]
    print(f"[{name}] orders {orders} | CN dt-halving diff at N=512: "
          f"{dt_err:.3e} vs N=256 spatial err {spatial_256:.3e} "
          f"(ratio {dt_err / spatial_256:.3f})", flush=True)
    results[name] = {"params": p, "errs_vs_ref512": errs, "orders": orders,
                     "energy_drift": drifts, "cn_dt_halving_diff": dt_err,
                     "dt_over_spatial256_ratio": dt_err / spatial_256,
                     "eval_times": list(eval_times)}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {OUT}", flush=True)
