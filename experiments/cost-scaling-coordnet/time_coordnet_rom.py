"""Online-cost timing skeleton for the coordinate-decoder ROM (Poisson 2D).

Measures the work of the future coord-net ROM's ONLINE solve — Gauss-Newton
over the latent z in R^k with an EQ-sampled FD residual — using random weights
(cost depends on tensor shapes, not values; no training involved).

The claim under test: mesh size N appears in NO tensor shape inside the timed
loop. N enters only as the VALUE dx = 1/(N-1) (stencil spacing) and through
which interior nodes the fixed m_eq EQ points land on (offline setup). The
script proves shape-level N-independence by compiling the GN step at the
smallest and largest N of the sweep (same k) and asserting the jaxpr shape
multisets and XLA FLOP counts are identical.

Residual at each of m EQ interior nodes: 5-point FD stencil of decoder point
evaluations (5m evals total), minus an analytic Gaussian source at the node.
GN step: J = jacfwd(residual)(z) (m x k), Levenberg-damped normal equations
(k x k), update z.

Decoder: FiLM coord-net from the Poisson testbed
(exp/2026-08-12-coord-decoder/experiments/coord-decoder/poisson2d_film.py),
z-dimension generalized to k; hidden 256 x 5 trunk, n_freq=32 FIXED across all
cells so feature width never varies with N.

Usage:
  [NS=32,64,128,256,512] [KS=2,4,8,16,32] [REPEATS=30] [GN_ITERS=10]
  [M_EQ=100] python time_coordnet_rom.py [outdir]
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

NS = [int(s) for s in os.environ.get("NS", "32,64,128,256,512").split(",")]
KS = [int(s) for s in os.environ.get("KS", "2,4,8,16,32").split(",")]
REPEATS = int(os.environ.get("REPEATS", "30"))
WARMUP = int(os.environ.get("WARMUP", "3"))
GN_ITERS = int(os.environ.get("GN_ITERS", "10"))
M_EQ = int(os.environ.get("M_EQ", "100"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))

N_FREQ = 32          # FIXED across cells — feature width must not vary with N
HIDDEN = 256
N_LAYERS = 5
DAMPING = 1e-6
F32 = jnp.float32

# Fixed Gaussian source (values only; play no role in cost)
SRC_CX, SRC_CY, SRC_W, SRC_A = 0.4, 0.6, 0.05, 1.5

COORD_FEATS = 2 * (2 * N_FREQ + 1)


# ------------------- FiLM coord net (from poisson2d_film.py) -------------------

def init_dense(key, d_in, d_out):
    W = jax.random.normal(key, (d_in, d_out), dtype=F32) * np.sqrt(1.0 / d_in)
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F32)}


def init_film_net(key, k):
    keys = jax.random.split(key, N_LAYERS + 4)
    trunk = [init_dense(keys[0], COORD_FEATS, HIDDEN)]
    for i in range(1, N_LAYERS):
        trunk.append(init_dense(keys[i], HIDDEN, HIDDEN))
    out = init_dense(keys[N_LAYERS], HIDDEN, 1)
    z_embed = init_dense(keys[N_LAYERS + 1], k, 64)
    film = init_dense(keys[N_LAYERS + 2], 64, N_LAYERS * 2 * HIDDEN)
    film["W"] = film["W"] * 0.01
    return {"trunk": trunk, "out": out, "z_embed": z_embed, "film": film}


def coord_features(xy):
    j = jnp.arange(1, N_FREQ + 1, dtype=F32)

    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)

    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1])], axis=1)


def film_apply(params, z, xy):
    g = jax.nn.swish(z @ params["z_embed"]["W"] + params["z_embed"]["b"])
    film = (g @ params["film"]["W"] + params["film"]["b"]).reshape(
        N_LAYERS, 2, HIDDEN)
    h = coord_features(xy)
    for i, lyr in enumerate(params["trunk"]):
        h = h @ lyr["W"] + lyr["b"]
        h = h * (1.0 + film[i, 0]) + film[i, 1]
        h = jax.nn.swish(h)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]


# ------------------------------ GN/EQ online solve ------------------------------

def make_gn_step():
    """One damped Gauss-Newton step. All shapes depend on (m_eq, k) only."""

    def residual(z, params, pts, dx, f_vals):
        # pts: (5, m, 2) — center, +x, -x, +y, -y stencil coordinates
        u = film_apply(params, z, pts.reshape(-1, 2)).reshape(5, -1)
        lap = (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / (dx * dx)
        return -lap - f_vals

    def gn_step(z, params, pts, dx, f_vals):
        r = residual(z, params, pts, dx, f_vals)
        J = jax.jacfwd(residual)(z, params, pts, dx, f_vals)   # (m, k)
        H = J.T @ J + DAMPING * jnp.eye(J.shape[1], dtype=F32)
        g = J.T @ r
        dz = jnp.linalg.solve(H, g)
        return z - dz

    return jax.jit(gn_step)


def build_cell_inputs(N, rng):
    """Offline setup: EQ node choice + stencil coords + source values."""
    dx = 1.0 / (N - 1)
    ix = rng.integers(1, N - 1, M_EQ)
    iy = rng.integers(1, N - 1, M_EQ)
    x, y = ix * dx, iy * dx
    pts = np.stack([
        np.stack([x, y], axis=1),
        np.stack([x + dx, y], axis=1),
        np.stack([x - dx, y], axis=1),
        np.stack([x, y + dx], axis=1),
        np.stack([x, y - dx], axis=1),
    ])                                                          # (5, m, 2)
    f_vals = SRC_A * np.exp(-((x - SRC_CX) ** 2 + (y - SRC_CY) ** 2)
                            / (2 * SRC_W ** 2))
    return (jnp.asarray(pts, dtype=F32), jnp.asarray(dx, dtype=F32),
            jnp.asarray(f_vals, dtype=F32))


def jaxpr_shape_multiset(fn, *args):
    """Multiset of all intermediate/const/invar shapes in the closed jaxpr."""
    closed = jax.make_jaxpr(fn)(*args)
    shapes = []

    def walk(jx):
        for v in list(jx.invars) + list(jx.constvars) + list(jx.outvars):
            if hasattr(v, "aval") and hasattr(v.aval, "shape"):
                shapes.append(tuple(v.aval.shape))
        for eqn in jx.eqns:
            for v in list(eqn.invars) + list(eqn.outvars):
                if hasattr(v, "aval") and hasattr(v.aval, "shape"):
                    shapes.append(tuple(v.aval.shape))
            for sub in eqn.params.values():
                if hasattr(sub, "jaxpr"):
                    walk(sub.jaxpr)

    walk(closed.jaxpr)
    return sorted(shapes)


def flops_of(fn, *args):
    try:
        ca = fn.lower(*args).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return float(ca.get("flops")) if ca and "flops" in ca else None
    except Exception as e:  # noqa: BLE001
        print(f"  cost_analysis unavailable: {e}", flush=True)
        return None


def main():
    backend = jax.default_backend()
    dev = jax.devices()[0]
    print(f"jax_backend={backend}", flush=True)
    print(f"device={dev.device_kind}  m_eq={M_EQ}  gn_iters={GN_ITERS}  "
          f"repeats={REPEATS}  n_freq={N_FREQ} (fixed)", flush=True)

    rng = np.random.default_rng(SEED)

    # ---- shape-level N-independence proof (per k, smallest vs largest N) ----
    gn_step = make_gn_step()
    n_lo, n_hi = min(NS), max(NS)
    for k in KS:
        params = init_film_net(jax.random.PRNGKey(SEED), k)
        z = jnp.zeros((k,), dtype=F32)
        args_lo = (z, params) + build_cell_inputs(n_lo, np.random.default_rng(1))
        args_hi = (z, params) + build_cell_inputs(n_hi, np.random.default_rng(1))
        sh_lo = jaxpr_shape_multiset(gn_step, *args_lo)
        sh_hi = jaxpr_shape_multiset(gn_step, *args_hi)
        assert sh_lo == sh_hi, (
            f"N LEAK at k={k}: jaxpr shapes differ between N={n_lo} and N={n_hi}")
    print(f"SHAPE CHECK PASSED: GN-step jaxpr shape multisets identical for "
          f"N={n_lo} vs N={n_hi} at every k in {KS} — N appears in no tensor "
          f"shape in the timed loop", flush=True)

    cells = []
    for k in KS:
        params = init_film_net(jax.random.PRNGKey(SEED), k)
        z0s = jax.random.normal(jax.random.PRNGKey(SEED + 7),
                                (WARMUP + REPEATS, k), dtype=F32)
        for N in NS:
            pts, dx, f_vals = build_cell_inputs(N, rng)
            flops = flops_of(gn_step, z0s[0], params, pts, dx, f_vals)

            def solve(z0):
                zz = z0
                for _ in range(GN_ITERS):
                    zz = gn_step(zz, params, pts, dx, f_vals)
                return zz

            for i in range(WARMUP):                     # excluded from timing
                jax.block_until_ready(solve(z0s[i]))
            times = np.empty(REPEATS)
            for i in range(REPEATS):
                t0 = time.perf_counter()
                jax.block_until_ready(solve(z0s[WARMUP + i]))
                times[i] = (time.perf_counter() - t0) / GN_ITERS
            med = float(np.median(times))
            iqr = [float(np.percentile(times, 25)),
                   float(np.percentile(times, 75))]
            zf = solve(z0s[WARMUP])
            assert bool(jnp.all(jnp.isfinite(zf))), f"non-finite z at N={N},k={k}"
            cells.append({"N": N, "k": k, "median_s_per_gn_iter": med,
                          "iqr_s": iqr, "flops_per_gn_iter": flops,
                          "repeats": REPEATS})
            print(f"RESULT N={N:4d} k={k:3d}  median={med*1e3:.4f} ms/GN-iter  "
                  f"iqr=[{iqr[0]*1e3:.4f},{iqr[1]*1e3:.4f}]  "
                  f"flops={flops if flops is not None else 'n/a'}", flush=True)

    # FLOP invariance across N at fixed k (the designed outcome of this arm)
    for k in KS:
        fl = {c["N"]: c["flops_per_gn_iter"] for c in cells if c["k"] == k}
        vals = [v for v in fl.values() if v is not None]
        if vals and len(set(vals)) > 1:
            raise AssertionError(f"FLOP LEAK at k={k}: {fl}")
    if any(c["flops_per_gn_iter"] is not None for c in cells):
        print("FLOP CHECK PASSED: identical FLOPs/GN-iter across all N at "
              "every fixed k", flush=True)

    out = {"arm": "coordnet", "gpu": dev.device_kind, "backend": backend,
           "hostname": socket.gethostname(), "gn_iters": GN_ITERS,
           "m_eq": M_EQ, "n_freq": N_FREQ, "hidden": HIDDEN,
           "n_layers": N_LAYERS, "seed": SEED, "cells": cells}
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"coordnet_timing_{socket.gethostname()}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
