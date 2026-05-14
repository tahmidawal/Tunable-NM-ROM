#!/usr/bin/env python
"""Build EQ and benchmark NM-ROM vs FOM on held-out parameters."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tunable_rom_vit_poisson.models.autoencoder import ViTViTAutoencoder
from tunable_rom_vit_poisson.fom.poisson import PoissonFOM, source_field
from tunable_rom_vit_poisson.eq.nnls import compute_eq_weights, build_v_eq
from tunable_rom_vit_poisson.solver.nm_rom import NMROMSolver
from tunable_rom_vit_poisson.utils.config import load_config
from tunable_rom_vit_poisson.utils.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = np.load(args.data)
    U_train, U_val = data["U_train"], data["U_val"]
    freqs_val = data["freqs_val"]

    ckpt = load_checkpoint(args.ckpt)
    params = ckpt["params"]

    model = ViTViTAutoencoder(
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        dec_patch_size=cfg.dec_patch_size,
        dec_embed_dim=cfg.dec_embed_dim,
        dec_num_heads=cfg.dec_num_heads,
        dec_num_layers=cfg.dec_num_layers,
    )

    # Build EQ via NNLS on |K u|.
    fom = PoissonFOM(N=cfg.N, spatial_dim=cfg.spatial_dim)
    K_op_jit = jax.jit(lambda u: fom.K_op(u))
    def K_op_numpy(u):
        return np.asarray(K_op_jit(jnp.asarray(u)))

    print("Computing EQ weights via NNLS")
    eq_flat, eq_w = compute_eq_weights(
        snapshots=U_train,
        K_op_numpy=K_op_numpy,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        n_eq_samples=cfg.n_eq_samples,
        min_eq_points=cfg.min_eq_points,
    )
    v_eq_st, stencil_idx = build_v_eq(params["decoder"], eq_flat, cfg.N, cfg.spatial_dim)
    print(f"  EQ nodes selected: {eq_flat.shape[0]}")

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

    # Build F at the EQ stencil centres for each test param.
    rel_l2s = []
    speedups = []
    for i in range(min(args.num_samples, U_val.shape[0])):
        freqs = list(freqs_val[i])
        F_full = source_field(fom, freqs)
        F_eq = F_full[eq_flat]
        u_true = jnp.asarray(U_val[i])

        # Warm up.
        z, _, _ = solver.solve(F_eq)
        u_rom = solver.decode(z)
        u_rom.block_until_ready()

        t0 = time.perf_counter()
        z, _, _ = solver.solve(F_eq)
        u_rom = solver.decode(z)
        u_rom.block_until_ready()
        t_rom = time.perf_counter() - t0

        # FOM.
        cg_jit = jax.jit(lambda F: fom.cg_solve(F))
        _ = cg_jit(F_full).block_until_ready()
        t0 = time.perf_counter()
        u_fom = cg_jit(F_full)
        u_fom.block_until_ready()
        t_fom = time.perf_counter() - t0

        rel = float(jnp.linalg.norm(u_rom - u_true) / jnp.linalg.norm(u_true))
        sp = t_fom / t_rom
        rel_l2s.append(rel)
        speedups.append(sp)
        print(f"  sample {i}: freqs={freqs}  rel_l2={rel:.4e}  speedup={sp:.2f}x")

    print(f"\nMean rel_l2: {np.mean(rel_l2s):.4e}")
    print(f"Median speedup: {np.median(speedups):.2f}x")


if __name__ == "__main__":
    main()
