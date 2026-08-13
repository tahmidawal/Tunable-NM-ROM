"""Online-cost scaling of the REAL ViT-CP NM-ROM solve (poisson package).

Measures, per (N, k) cell, the wall-clock and XLA-FLOP cost of one GN-LM
iteration of `NMROMSolver.solve` — the actual online solve path imported from
poisson/src — plus two expected-O(n) contrast series: the final full-field
decode and the FOM CG solve.

Protocol (see AUDIT.md): random decoder weights of the true architecture
shapes; m EQ nodes fixed across all cells; fixed GN iteration count via
gn_rel_tol=0; jit + >=3 warm-up solves excluded; R timed repeats with
block_until_ready; medians + IQR reported. Compiler FLOPs via
jit(...).lower(...).compile().cost_analysis().

Usage:
  NS=32,64,128,256,512 KS=2,4,8,16,32 python time_cp_rom.py [outdir]
  (defaults: full grid; REPEATS=30, GN_ITERS=10, M_EQ=640)
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "poisson", "src"))

from tunable_rom_poisson.solver.nm_rom import NMROMSolver          # noqa: E402
from tunable_rom_poisson.eq.nnls import build_v_eq                 # noqa: E402
from tunable_rom_poisson.fom.poisson import PoissonFOM, source_field  # noqa: E402
from tunable_rom_poisson.models.decoder import LinearCPDecoder     # noqa: E402

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
NS = [int(s) for s in os.environ.get("NS", "32,64,128,256,512").split(",")]
KS = [int(s) for s in os.environ.get("KS", "2,4,8,16,32").split(",")]
REPEATS = int(os.environ.get("REPEATS", "30"))
FOM_REPEATS = int(os.environ.get("FOM_REPEATS", "10"))
GN_ITERS = int(os.environ.get("GN_ITERS", "10"))
M_EQ = int(os.environ.get("M_EQ", "640"))
RANK = int(os.environ.get("RANK", "128"))
HIDDEN = int(os.environ.get("HIDDEN", "256"))
SEED = int(os.environ.get("SEED", "0"))

F32 = np.float32


def random_decoder_params(rng, N, k):
    """Random weights with the exact LinearCPDecoder parameter shapes."""
    def dense(d_in, d_out, bias=True):
        p = {"kernel": rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_out)).astype(F32)}
        if bias:
            p["bias"] = np.zeros(d_out, dtype=F32)
        return p

    return {
        "W1": dense(k, HIDDEN),
        "W2": dense(HIDDEN, HIDDEN),
        "W_rank": dense(HIDDEN, RANK),
        "W_direct": {"kernel": rng.normal(0, 1 / np.sqrt(k), (k, RANK)).astype(F32)},
        "W_x": rng.normal(0, 0.01, (RANK, N)).astype(F32),
        "W_y": rng.normal(0, 0.01, (RANK, N)).astype(F32),
        "bias": np.zeros((), dtype=F32),
    }


def pick_eq_nodes(rng, N, m):
    """m distinct strictly-interior flat indices (stencil stays in-grid)."""
    rr = np.arange(1, N - 1)
    coords = np.stack(np.meshgrid(rr, rr, indexing="ij"), axis=-1).reshape(-1, 2)
    sel = rng.choice(coords.shape[0], size=min(m, coords.shape[0]), replace=False)
    ij = coords[sel]
    return (ij[:, 0] * N + ij[:, 1]).astype(np.int64)


def flops_of(compiled):
    """Best-effort FLOP count from XLA cost analysis (None if unavailable)."""
    try:
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return float(ca["flops"])
    except Exception as e:  # pragma: no cover - jax-version dependent
        print(f"  cost_analysis unavailable: {e}", flush=True)
        return None


def time_median(fn, out_block, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_block(fn())
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return float(np.median(ts)), [float(np.percentile(ts, 25)),
                                  float(np.percentile(ts, 75))]


def main():
    backend = jax.default_backend()
    print(f"jax_backend={backend}", flush=True)
    gpu = jax.devices()[0].device_kind
    print(f"device={gpu}  NS={NS}  KS={KS}  m_eq={M_EQ}  gn_iters={GN_ITERS}  "
          f"rank={RANK} hidden={HIDDEN}  repeats={REPEATS}", flush=True)

    rng = np.random.default_rng(SEED)
    cells = []
    fom_done = {}

    for N in NS:
        fom = PoissonFOM(N=N, spatial_dim=2)
        eq_flat = pick_eq_nodes(rng, N, M_EQ)
        eq_w = np.full(eq_flat.shape[0], 1.0 / eq_flat.shape[0], dtype=F32)
        F_full = source_field(fom, [2.0, 2.0])
        F_eq = jnp.asarray(np.asarray(F_full)[eq_flat])

        for k in KS:
            dec = random_decoder_params(rng, N, k)
            v_eq_st, stencil_idx = build_v_eq(dec, eq_flat, N, 2)   # real repo fn
            solver = NMROMSolver(
                autoencoder=None, params={"decoder": dec}, N=N, spatial_dim=2,
                dx=fom.dx, eq_flat_indices=eq_flat, eq_weights=eq_w,
                v_eq_stencil=v_eq_st, stencil_indices=stencil_idx,
                gn_max_iters=GN_ITERS, gn_rel_tol=0.0,   # fixed iteration count
            )

            jsolve = jax.jit(solver.solve)
            lowered = jsolve.lower(F_eq)
            flops = flops_of(lowered.compile())
            # warm-up (compile + >=3 executions)
            for _ in range(3):
                z, gn, iters = jsolve(F_eq)
                z.block_until_ready()
            assert int(iters) == GN_ITERS, f"iters={int(iters)} != {GN_ITERS}"

            med, iqr = time_median(lambda: jsolve(F_eq),
                                   lambda out: out[0].block_until_ready(), REPEATS)

            # full-field decode (expected O(n)) via the real flax decoder
            dec_mod = LinearCPDecoder(N=N, spatial_dim=2, latent_dim=k,
                                      rank=RANK, hidden_dim=HIDDEN)
            zf = jnp.asarray(np.asarray(z))
            jdec = jax.jit(lambda zz: dec_mod.apply({"params": dec}, zz))
            jdec(zf).block_until_ready()
            dmed, diqr = time_median(lambda: jdec(zf),
                                     lambda out: out.block_until_ready(), REPEATS)

            # FOM CG (once per N)
            if N not in fom_done:
                jcg = jax.jit(fom.cg_solve)
                jcg(F_full).block_until_ready()
                fmed, fiqr = time_median(lambda: jcg(F_full),
                                         lambda out: out.block_until_ready(),
                                         FOM_REPEATS)
                fom_done[N] = fmed
            else:
                fmed = None

            cell = {
                "N": N, "k": k, "m_eq": int(eq_flat.shape[0]),
                "median_s_per_gn_iter": med / GN_ITERS,
                "iqr_s": [iqr[0] / GN_ITERS, iqr[1] / GN_ITERS],
                "solve_median_s": med,
                "flops_per_gn_iter": (flops / GN_ITERS) if flops else None,
                "decode_median_s": dmed,
                "repeats": REPEATS,
                "fom_median_s": fmed,
            }
            cells.append(cell)
            fl = f"{cell['flops_per_gn_iter']:.3e}" if flops else "n/a"
            print(f"RESULT N={N:4d} k={k:3d}  gn_iter={cell['median_s_per_gn_iter']:.3e}s"
                  f"  flops/iter={fl}  decode={dmed:.3e}s"
                  f"  fom={fmed if fmed else '-'}", flush=True)

    out = {
        "arm": "cp", "gpu": gpu, "backend": backend, "hostname": socket.gethostname(),
        "gn_iters": GN_ITERS, "m_eq": M_EQ, "rank": RANK, "hidden": HIDDEN,
        "seed": SEED, "jax": jax.__version__, "cells": cells,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"cp_timing_{socket.gethostname()}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
