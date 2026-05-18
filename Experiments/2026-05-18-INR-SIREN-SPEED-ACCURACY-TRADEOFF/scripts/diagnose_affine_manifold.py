#!/usr/bin/env python
"""For an affine_z checkpoint, measure the gap between:
  - oracle rel-L2: encode test sample u_i, decode(encode(u_i)) — best the manifold can do
  - ROM rel-L2:    cold-start NM-ROM solve from z=0

This is the manifold-capacity vs solver-pathology diagnostic.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.fom.poisson_cg import PoissonFOM
from tunable_rom_speed.solver.nm_rom_affine import AffineNMROMSolver


def _build_model(cfg):
    return INRAutoencoder(
        decoder_kind=cfg["decoder_kind"],
        N=cfg["N"], spatial_dim=cfg["spatial_dim"],
        patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_enc_layers=cfg["num_enc_layers"],
        latent_dim=cfg["latent_dim"], coord_dim=cfg["spatial_dim"],
        hidden_dim=cfg["siren_hidden_dim"],
        siren_num_layers=cfg["siren_num_layers"],
        omega_0=cfg["omega_0"], omega=cfg["omega"],
        modulator_hidden=cfg["modulator_hidden"],
        d_attn=cfg.get("d_attn", 64),
        num_fourier=cfg.get("num_fourier", 16),
        xattn_num_layers=cfg.get("xattn_num_layers", 3),
        fourier_scale=cfg.get("fourier_scale", 4.0),
        affine_bias_hidden=cfg.get("affine_bias_hidden", 128),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--eq", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--n-samples", type=int, default=16)
    args = p.parse_args()

    ck = pickle.load(open(args.ckpt, "rb"))
    cfg = ck["config"]
    model = _build_model(cfg)
    params = ck["params"]
    fom = PoissonFOM(N=cfg["N"], spatial_dim=cfg["spatial_dim"])
    data = np.load(args.data)
    F_test = data["F_test"][: args.n_samples]
    U_test = data["U_test"][: args.n_samples]
    mask_flat = np.asarray(fom.mask).reshape(-1)
    print(f"[diag] ckpt {args.ckpt.name} | latent_dim {cfg['latent_dim']}", flush=True)

    eq = np.load(args.eq)
    solver = AffineNMROMSolver(
        autoencoder=model, params=params,
        N=cfg["N"], spatial_dim=cfg["spatial_dim"], dx=float(fom.dx),
        eq_flat_indices=eq["eq_flat_indices"], eq_weights=eq["eq_weights"],
        stencil_indices=eq["stencil_indices"],
        tokens_ref=None, gn_max_iters=30,
        latent_dim_override=cfg["latent_dim"],
    )
    solve = solver.make_solve()

    @jax.jit
    def encode_then_decode_grid(u_flat):
        z, _T = model.apply({"params": params}, u_flat, method=model.encode)
        d = cfg["spatial_dim"]; N = cfg["N"]
        rng = jnp.linspace(0.0, 1.0, N)
        xx, yy = jnp.meshgrid(rng, rng, indexing="ij")
        x_q = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)
        u = model.apply({"params": params}, z, None, x_q, method=model.decode_points)
        return z, u

    print(f"{'i':>3} {'||z_enc||':>10} {'||z_rom||':>10} "
          f"{'oracle':>10} {'rom':>10} {'oracle p':>10} ", flush=True)
    print("="*80)
    for i in range(args.n_samples):
        u_true = U_test[i]
        z_enc, u_dec = encode_then_decode_grid(jnp.asarray(u_true))
        z_enc.block_until_ready()
        rl_oracle = float(np.linalg.norm(np.asarray(u_dec) * mask_flat - u_true) /
                          (np.linalg.norm(u_true) + 1e-12))
        Fe = jnp.asarray(data["F_test"][i, eq["eq_flat_indices"]])
        z_rom, _, _ = solve(Fe); z_rom.block_until_ready()
        u_rom = np.asarray(solver.decode_full_grid(z_rom)) * mask_flat
        rl_rom = float(np.linalg.norm(u_rom - u_true) / (np.linalg.norm(u_true) + 1e-12))
        # Plug z_enc into decode and check what it gives us (verify pipeline).
        u_via_enc = np.asarray(solver.decode_full_grid(z_enc)) * mask_flat
        rl_via_enc = float(np.linalg.norm(u_via_enc - u_true) /
                           (np.linalg.norm(u_true) + 1e-12))
        print(f"{i:3d} {float(np.linalg.norm(z_enc)):10.4e} "
              f"{float(np.linalg.norm(z_rom)):10.4e} "
              f"{rl_oracle:10.3e} {rl_rom:10.3e} {rl_via_enc:10.3e}")


if __name__ == "__main__":
    main()
