"""End-to-end smoke test on a tiny N=8 2D problem.

Verifies that the entire pipeline (FOM, AE init, decode, EQ NNLS,
NM-ROM solve) runs without crashing. Does NOT validate accuracy —
training is too short to converge.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from tunable_rom_heat.fom.heat import (
    HeatFOM,
    NUM_STEPS,
    sample_parameters,
    generate_trajectory,
)
from tunable_rom_heat.models.autoencoder import ViTCPAutoencoder
from tunable_rom_heat.eq.nnls import compute_eq_weights, build_v_eq
from tunable_rom_heat.solver.nm_rom import NMROMSolver


N = 8
D = 2


def _gen_trajectories(fom, n, rng):
    snapshots = []
    kappas = []
    for _ in range(n):
        p = sample_parameters(rng, D)
        traj = np.asarray(generate_trajectory(fom, p, num_steps=NUM_STEPS))
        snapshots.append(traj)
        kappas.append(p["kappa"])
    return np.concatenate(snapshots, axis=0).astype(np.float32), kappas


def test_fom_runs():
    fom = HeatFOM(N=N, spatial_dim=D)
    rng = np.random.default_rng(0)
    p = sample_parameters(rng, D)
    traj = generate_trajectory(fom, p, num_steps=5)
    assert traj.shape == (6, N**D)
    assert np.all(np.isfinite(np.asarray(traj)))


def test_autoencoder_init():
    model = ViTCPAutoencoder(
        N=N, spatial_dim=D, patch_size=4, embed_dim=16,
        num_heads=2, num_enc_layers=2, latent_dim=8, rank=16, hidden_dim=32,
    )
    rng = jax.random.PRNGKey(0)
    u = jnp.ones((N**D,))
    params = model.init(rng, u)["params"]
    u_hat = model.apply({"params": params}, u)
    assert u_hat.shape == (N**D,)


def test_end_to_end_pipeline():
    fom = HeatFOM(N=N, spatial_dim=D)
    rng = np.random.default_rng(0)
    U_train, kappas = _gen_trajectories(fom, 4, rng)

    model = ViTCPAutoencoder(
        N=N, spatial_dim=D, patch_size=4, embed_dim=16,
        num_heads=2, num_enc_layers=2, latent_dim=8, rank=16, hidden_dim=32,
    )
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(U_train[0]))["params"]

    eq_flat, eq_w = compute_eq_weights(
        model=model,
        params=params,
        snapshots=U_train,
        N=N,
        spatial_dim=D,
        n_eq_samples=2,
        min_eq_points=4,
    )
    v_eq_st, stencil_idx = build_v_eq(params["decoder"], eq_flat, N, D)
    solver = NMROMSolver(
        autoencoder=model,
        params=params,
        N=N,
        spatial_dim=D,
        dx=fom.dx,
        eq_flat_indices=eq_flat,
        eq_weights=eq_w,
        v_eq_stencil=v_eq_st,
        stencil_indices=stencil_idx,
        gn_max_iters=2,
    )
    u0 = jnp.asarray(U_train[0])
    u_T, iters = solver.rollout(u0, jnp.float32(kappas[0]), num_steps=3)
    assert u_T.shape == (N**D,)
    assert iters.shape == (3,)
