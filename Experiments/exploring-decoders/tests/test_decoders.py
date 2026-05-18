"""Smoke tests — verify shapes, dtype, gradient flow for both decoders."""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make sure src/ is importable when pytest is invoked from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from tunable_rom_decoders.autoencoder import INRAutoencoder
from tunable_rom_decoders.fom.poisson_analytical import PoissonAnalytical, generate_dataset


@pytest.fixture(scope="module")
def tiny_dataset():
    N = 64
    U, freqs = generate_dataset(N=N, spatial_dim=2, n_samples=4, seed=0)
    return N, U, freqs


def _build(kind, N):
    return INRAutoencoder(
        decoder_kind=kind,
        N=N,
        spatial_dim=2,
        patch_size=8,
        embed_dim=32,
        num_heads=2,
        num_enc_layers=2,
        latent_dim=8,
        coord_dim=2,
        hidden_dim=64,
        siren_num_layers=3,
        omega_0=30.0,
        omega=1.0,
        modulator_hidden=32,
        d_attn=32,
        num_fourier=8,
        xattn_num_layers=2,
        fourier_scale=4.0,
    )


@pytest.mark.parametrize("kind", ["siren", "xattn"])
def test_forward_shapes(tiny_dataset, kind):
    N, U, _ = tiny_dataset
    model = _build(kind, N)
    u_in = jnp.asarray(U[0])
    x_q = jnp.asarray(np.random.uniform(0, 1, (32, 2)).astype(np.float32))
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, u_in, x_q)["params"]
    out = model.apply({"params": params}, u_in, x_q)
    assert out.shape == (32,)
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize("kind", ["siren", "xattn"])
def test_grad_flow(tiny_dataset, kind):
    N, U, freqs = tiny_dataset
    model = _build(kind, N)
    u_in = jnp.asarray(U[0])
    x_q = jnp.asarray(np.random.uniform(0, 1, (32, 2)).astype(np.float32))
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, u_in, x_q)["params"]

    fom = PoissonAnalytical(N=N, spatial_dim=2)
    u_target = jnp.asarray(fom.u_at_points(freqs[0], np.asarray(x_q)))

    def loss_fn(p):
        u_pred = model.apply({"params": p}, u_in, x_q)
        return jnp.mean((u_pred - u_target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert jnp.isfinite(loss)
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    grad_norm = sum(jnp.sum(g ** 2) for g in flat_grads) ** 0.5
    assert jnp.isfinite(grad_norm)
    assert float(grad_norm) > 0.0
