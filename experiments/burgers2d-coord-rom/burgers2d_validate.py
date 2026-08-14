"""Pre-training FOM validation for the Burgers-2D testbed. Two checks:

1. SELF-CONVERGENCE: solve representative parameters at N=64/128/256/512 with
   the FIXED dt, bilinearly interpolate each coarse solution to the 512 grid,
   and report the error vs the N=512 solution at the EVAL_TIMES slices plus
   the observed order log2(e_N / e_2N). With first-order upwind advection the
   expected asymptotic order is ~1 in advection-dominated regions (up to ~2
   where diffusion dominates); the ratios should sit clearly between.

2. NEWTON CONVERGENCE at parameter-box corners (highest a, lowest nu,
   near-wall center, width extremes): per-step Newton residual histories at
   N=256 and N=512; reports iterations needed to drive the relative residual
   below 1e-10 (max/median over the 50 steps).

Runs locally on the GB10 in a few minutes (single trajectories only).

Usage:  python burgers2d_validate.py [out-json]
"""
from __future__ import annotations

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
sys.path.insert(0, HERE)
import burgers2d_film as bf

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "burgers2d_validation.json")
NS = [64, 128, 256, 512]
N_REF = 512

print(f"jax_backend={jax.default_backend()}", flush=True)
results = {"dt": bf.DT, "num_steps": bf.NUM_STEPS,
           "newton_iters": bf.NEWTON_ITERS, "lin_tol": bf.LIN_TOL}

# ---------------- 1. self-convergence ----------------
# (cx, cy, w, a, nu): a mid-box case and the sharp advective corner.
CONV_PARAMS = {
    "mid (a=1.2, nu=0.03)": (0.50, 0.50, 0.120, 1.2, 0.030),
    "sharp corner (a=2, nu=0.01)": (0.40, 0.40, 0.080, 2.0, 0.010),
}

x_ref = np.linspace(0.0, 1.0, N_REF)
Xr, Yr = np.meshgrid(x_ref, x_ref, indexing="ij")

results["self_convergence"] = {}
for label, (pcx, pcy, pw, pa, pnu) in CONV_PARAMS.items():
    sols = {}
    for n in NS:
        rollout, _ = bf.make_rollout(n)
        U0 = bf.blob_ic(n, pcx, pcy, pw, pa)[None]
        t0 = time.time()
        snaps, res = rollout(jnp.asarray(U0), jnp.asarray([pnu]))
        snaps = np.asarray(snaps)[bf.EVAL_TIMES, 0]      # (n_times, n*n)
        rmax = float(jnp.max(res))
        sols[n] = snaps
        print(f"  [{label}] N={n:4d} solved in {time.time()-t0:.0f}s "
              f"(max Newton rel res {rmax:.2e})", flush=True)
        assert np.isfinite(rmax) and rmax < 1e-8, "Newton failed to converge"
    ref = sols[N_REF]
    ref_norm = np.linalg.norm(ref, axis=1)
    errs = {}
    for n in NS[:-1]:
        ci = np.stack([Xr * (n - 1), Yr * (n - 1)]).reshape(2, -1)
        e_t = []
        for kt in range(len(bf.EVAL_TIMES)):
            interp = map_coordinates(sols[n][kt].reshape(n, n), ci, order=1)
            e_t.append(np.linalg.norm(interp - ref[kt]) / ref_norm[kt])
        errs[n] = e_t
    # aggregate over t>0 slices (t=0 is pure IC interpolation error)
    agg = {n: float(np.mean(e[1:])) for n, e in errs.items()}
    orders = {f"{n}->{2*n}": float(np.log2(agg[n] / agg[2 * n]))
              for n in NS[:-2]}
    results["self_convergence"][label] = {
        "err_vs_512_per_time": {str(n): [float(v) for v in e]
                                for n, e in errs.items()},
        "err_vs_512_aggregate_tpos": agg,
        "observed_order": orders,
    }
    print(f"  [{label}] aggregate err vs 512: "
          + "  ".join(f"N{n}={e:.3e}" for n, e in agg.items())
          + "   orders: " + "  ".join(f"{k}:{v:.2f}"
                                      for k, v in orders.items()), flush=True)

# ---------------- 2. Newton behavior at the corners ----------------
CORNER_PARAMS = {
    "near-wall sharp (0.85,0.85,w=0.05,a=2,nu=0.01)": (0.85, 0.85, 0.05, 2.0, 0.01),
    "wide sharp (0.5,0.5,w=0.20,a=2,nu=0.01)": (0.50, 0.50, 0.20, 2.0, 0.01),
    "low-corner sharp (0.15,0.15,w=0.05,a=2,nu=0.01)": (0.15, 0.15, 0.05, 2.0, 0.01),
    "mild (0.5,0.5,w=0.125,a=1,nu=0.0316)": (0.50, 0.50, 0.125, 1.0, 0.0316),
}
NEWTON_PROBE_ITERS = 12
TOL = 1e-10


def newton_history(n, u0_flat, nu):
    """Per-step residual-norm history: (NUM_STEPS, NEWTON_PROBE_ITERS)."""
    _, residual = bf.make_rollout(n)

    def newton_step(u_prev):
        def body(u, _):
            r = residual(u, u_prev, nu)
            Jv = lambda v: jax.jvp(
                lambda uu: residual(uu, u_prev, nu), (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=bf.LIN_TOL, maxiter=bf.LIN_MAXITER)
            u2 = u + du
            rel = (jnp.linalg.norm(residual(u2, u_prev, nu))
                   / jnp.maximum(jnp.linalg.norm(u_prev), 1e-300))
            return u2, rel
        return jax.lax.scan(body, u_prev, None, length=NEWTON_PROBE_ITERS)

    @jax.jit
    def run(u0):
        def body(u, _):
            u2, hist = newton_step(u)
            return u2, hist
        _, H = jax.lax.scan(body, u0, None, length=bf.NUM_STEPS)
        return H

    return np.asarray(run(jnp.asarray(u0_flat)))


results["newton_corners"] = {}
for label, (pcx, pcy, pw, pa, pnu) in CORNER_PARAMS.items():
    for n in [256, 512]:
        u0 = bf.blob_ic(n, pcx, pcy, pw, pa)
        t0 = time.time()
        H = newton_history(n, u0, pnu)                   # (steps, iters)
        conv = H < TOL
        iters = np.where(conv.any(axis=1), conv.argmax(axis=1) + 1,
                         NEWTON_PROBE_ITERS + 1)
        entry = {
            "iters_to_1e-10_max": int(iters.max()),
            "iters_to_1e-10_median": float(np.median(iters)),
            "unconverged_steps": int((iters > NEWTON_PROBE_ITERS).sum()),
            "final_res_max": float(H[:, -1].max()),
        }
        results["newton_corners"][f"{label} @N={n}"] = entry
        print(f"  [{label}] N={n}: iters to {TOL:.0e} max={entry['iters_to_1e-10_max']} "
              f"median={entry['iters_to_1e-10_median']:.1f} "
              f"unconverged={entry['unconverged_steps']} "
              f"[{time.time()-t0:.0f}s]", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {OUT}", flush=True)
