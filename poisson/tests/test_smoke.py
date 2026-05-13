"""End-to-end smoke test on a tiny N=8 2D problem."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tunable_rom_poisson.fom.poisson import PoissonFOM, source_field
from tunable_rom_poisson.fom.data import generate_cg
from tunable_rom_poisson.models.autoencoder import ViTLinearCPAutoencoder
from tunable_rom_poisson.eq.nnls import compute_eq_weights, build_v_eq
from tunable_rom_poisson.solver.nm_rom import NMROMSolver


N = 8
D = 2


def test_fom_cg_runs():
    fom = PoissonFOM(N=N, spatial_dim=D)
    F = source_field(fom, [2.0, 2.0])
    u = fom.cg_solve(F)
    assert u.shape == (N**D,)
    assert np.all(np.isfinite(np.asarray(u)))


def test_data_generators():
    U_cg, freqs_cg = generate_cg(N=N, spatial_dim=D, n_samples=4, seed=0)
    assert U_cg.shape == (4, N**D)
    assert freqs_cg.shape == (4, D)


def test_autoencoder_init_and_decode():
    model = ViTLinearCPAutoencoder(
        N=N, spatial_dim=D, patch_size=4, embed_dim=16,
        num_heads=2, num_enc_layers=2, latent_dim=8, rank=16, hidden_dim=32,
    )
    rng = jax.random.PRNGKey(0)
    u = jnp.ones((N**D,))
    params = model.init(rng, u)["params"]
    u_hat = model.apply({"params": params}, u)
    assert u_hat.shape == (N**D,)
    # Linear skip means decode(z=0) is bias only, not necessarily zero, but finite.
    z0 = jnp.zeros((8,))
    u0 = model.apply({"params": params}, z0, method=model.decode)
    assert jnp.all(jnp.isfinite(u0))


def test_end_to_end_pipeline():
    fom = PoissonFOM(N=N, spatial_dim=D)
    U_train, freqs_train = generate_cg(N=N, spatial_dim=D, n_samples=4, seed=0)
    model = ViTLinearCPAutoencoder(
        N=N, spatial_dim=D, patch_size=4, embed_dim=16,
        num_heads=2, num_enc_layers=2, latent_dim=8, rank=16, hidden_dim=32,
    )
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(U_train[0]))["params"]

    K_op_numpy = lambda u: np.asarray(fom.K_op(jnp.asarray(u)))
    eq_flat, eq_w = compute_eq_weights(
        snapshots=U_train,
        K_op_numpy=K_op_numpy,
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
    F_full = source_field(fom, list(freqs_train[0]))
    F_eq = F_full[eq_flat]
    z, _, iters = solver.solve(F_eq)
    u_rom = solver.decode(z)
    assert u_rom.shape == (N**D,)
    assert iters.shape == ()
