#!/usr/bin/env python
"""Build the EQ quadrature and benchmark the NM-ROM rollout.

Usage:
    python -m scripts.run_rom --config configs/heat2d_n64.yaml \
        --data data/heat2d_n64.npz --ckpt checkpoints/heat2d_n64.pkl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tunable_rom_heat.models.autoencoder import ViTCPAutoencoder
from tunable_rom_heat.fom.heat import HeatFOM, NUM_STEPS
from tunable_rom_heat.eq.nnls import compute_eq_weights, build_v_eq
from tunable_rom_heat.solver.nm_rom import NMROMSolver
from tunable_rom_heat.utils.config import load_config
from tunable_rom_heat.utils.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--num-trajectories", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train, U_val = data["U_train"], data["U_val"]
    val_kappa = data["val_kappa"]

    ckpt = load_checkpoint(args.ckpt)
    params = ckpt["params"]

    model = ViTCPAutoencoder(
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        rank=cfg.rank,
        hidden_dim=cfg.hidden_dim,
    )

    # 1. Build EQ quadrature from training snapshots.
    print("Computing EQ weights via NNLS")
    eq_flat, eq_w = compute_eq_weights(
        model=model,
        params=params,
        snapshots=U_train,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        n_eq_samples=cfg.n_eq_samples,
        min_eq_points=cfg.min_eq_points,
    )
    v_eq_st, stencil_idx = build_v_eq(params["decoder"], eq_flat, cfg.N, cfg.spatial_dim)
    print(f"  EQ nodes selected: {eq_flat.shape[0]}")

    # 2. Build solver.
    fom = HeatFOM(N=cfg.N, spatial_dim=cfg.spatial_dim)
    solver = NMROMSolver(
        autoencoder=model,
        params=params,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        dx=fom.dx,
        eq_flat_indices=eq_flat,
        eq_weights=eq_w,
        v_eq_stencil=v_eq_st,
        stencil_indices=stencil_idx,
        gn_max_iters=cfg.gn_max_iters,
        gn_rel_tol=cfg.gn_rel_tol,
    )
    rom_rollout = jax.jit(
        lambda u0, k: solver.rollout(u0, k, num_steps=cfg.num_rom_steps),
        static_argnames=(),
    )

    # 3. Benchmark.
    rng = np.random.default_rng(0)
    val_starts = np.arange(0, U_val.shape[0], NUM_STEPS + 1)[: args.num_trajectories]
    rel_l2s = []
    speedups = []
    for i, s in enumerate(val_starts):
        u0 = jnp.asarray(U_val[s])
        kappa = jnp.float32(val_kappa[i])
        u_T_true = jnp.asarray(U_val[s + NUM_STEPS])
        # Warm up.
        u_rom, _ = rom_rollout(u0, kappa)
        u_rom.block_until_ready()
        t0 = time.perf_counter()
        u_rom, _ = rom_rollout(u0, kappa)
        u_rom.block_until_ready()
        t_rom = time.perf_counter() - t0
        # FOM baseline.
        fom_rollout = jax.jit(lambda u, k: fom.rollout(u, k, NUM_STEPS))
        _ = fom_rollout(u0, kappa).block_until_ready()
        t0 = time.perf_counter()
        u_fom = fom_rollout(u0, kappa)
        u_fom.block_until_ready()
        t_fom = time.perf_counter() - t0
        rel = float(jnp.linalg.norm(u_rom - u_T_true) / jnp.linalg.norm(u_T_true))
        rel_l2s.append(rel)
        speedups.append(t_fom / t_rom)
        print(f"  traj {i}: kappa={float(kappa):.4f}  rel_l2={rel:.4e}  speedup={t_fom/t_rom:.2f}x")

    print(f"\nMean rel_l2: {np.mean(rel_l2s):.4e}")
    print(f"Median speedup: {np.median(speedups):.2f}x")


if __name__ == "__main__":
    main()
