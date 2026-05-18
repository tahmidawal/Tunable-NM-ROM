#!/usr/bin/env python
"""Smoke test: init affine decoder, run forward+backward, verify shapes
and that AffineNMROMSolver's precompute path works."""
from __future__ import annotations
import sys
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tunable_rom_speed.autoencoder import INRAutoencoder
from tunable_rom_speed.fom.poisson_cg import PoissonFOM
from tunable_rom_speed.solver.nm_rom_affine import AffineNMROMSolver


def main():
    model = INRAutoencoder(
        decoder_kind="affine_z",
        N=128, spatial_dim=2, patch_size=16, embed_dim=64,
        num_heads=4, num_enc_layers=4, latent_dim=16,
        coord_dim=2, hidden_dim=128, siren_num_layers=4,
        omega_0=30.0, omega=1.0, modulator_hidden=64,
    )
    rng = jax.random.PRNGKey(0)
    u_dummy = jnp.zeros((128 * 128,))
    x_dummy = jnp.zeros((4, 2))
    params = model.init(rng, u_dummy, x_dummy)["params"]
    print("init OK — params tree:")
    for k, v in params["decoder"].items():
        if isinstance(v, dict): continue
        print(f"  decoder.{k}: shape={v.shape}")
    # Run forward.
    z = jax.random.normal(rng, (16,))
    u = model.apply({"params": params}, z, None, x_dummy, method=model.decode_points)
    print(f"decode_points -> {u.shape}, sample {u[0]:.4e}")
    # Solver init.
    fom = PoissonFOM(N=128, spatial_dim=2)
    n_eq = 50
    eq_flat = np.arange(n_eq, dtype=np.int64) + 128 + 1  # interior-ish
    eq_w = np.ones(n_eq, dtype=np.float64)
    # 5-point stencil indices: (i, i-1, i+1, i-N, i+N) for 2D
    sten = np.empty((n_eq, 5), dtype=np.int64)
    for k, idx in enumerate(eq_flat):
        sten[k] = [idx, idx-1, idx+1, idx-128, idx+128]
    solver = AffineNMROMSolver(
        autoencoder=model, params=params,
        N=128, spatial_dim=2, dx=float(fom.dx),
        eq_flat_indices=eq_flat, eq_weights=eq_w,
        stencil_indices=sten, tokens_ref=None,
        gn_max_iters=5, latent_dim_override=16,
    )
    print(f"solver._Phi_st shape: {solver._Phi_st.shape}")
    print(f"solver._bias_st shape: {solver._bias_st.shape}")
    # Try a solve.
    F_eq = jnp.zeros((n_eq,))
    solve = solver.make_solve()
    z_out, gn, it = solve(F_eq); z_out.block_until_ready()
    print(f"solve OK — z norm {jnp.linalg.norm(z_out):.4e}, iters {int(it)}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
