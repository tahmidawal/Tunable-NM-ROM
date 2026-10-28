"""Frozen-POD-lift + residual-MLP decoder for the wave NM-ROM.

    x_norm_hat(z) = mean + V z + W_out * MLP(z)        (MLP optional)

with V a FROZEN stacked-state POD basis of the channel-balanced
normalised snapshots x_norm = (u/s, v/(s*V_SCALE)), mean their mean,
and a tiny-init 2-layer swish residual MLP so training starts AT the
POD solution. The encoder is the exact projection z = V^T (x_norm -
mean) — nothing to train there. See PLAN-frozen-pod-lift.md.

Implemented as pure functions over a plain params dict (no flax): the
only trainable pieces are the MLP weights and W_out; V and mean stay
frozen numpy arrays inside the dict.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .autoencoder import V_SCALE


def init_params(V: np.ndarray, mean: np.ndarray, mlp_width: int, seed: int = 0,
                use_mlp: bool = True) -> dict:
    """Build the params dict. V: (2*N^2, k), mean: (2*N^2,)."""
    k = V.shape[1]
    params = {
        "V": jnp.asarray(V, dtype=jnp.float32),
        "mean": jnp.asarray(mean, dtype=jnp.float32),
        "use_mlp": use_mlp,
    }
    if use_mlp:
        rng = np.random.default_rng(seed)
        W = mlp_width
        params.update(
            W1=jnp.asarray(rng.normal(0, np.sqrt(2.0 / k), (k, W)).astype(np.float32)),
            b1=jnp.zeros((W,), jnp.float32),
            W2=jnp.asarray(rng.normal(0, np.sqrt(2.0 / W), (W, W)).astype(np.float32)),
            b2=jnp.zeros((W,), jnp.float32),
            # Tiny-init output so decode(z) == POD lift at step 0.
            W_out=jnp.asarray(rng.normal(0, 1e-4, (V.shape[0], W)).astype(np.float32)),
        )
    return params


def mlp_h(params: dict, z: jnp.ndarray) -> jnp.ndarray:
    """Hidden feature h(z) of the residual MLP, shape (W,)."""
    h = jax.nn.swish(z @ params["W1"] + params["b1"])
    h = jax.nn.swish(h @ params["W2"] + params["b2"])
    return h


def decode_norm(params: dict, z: jnp.ndarray) -> jnp.ndarray:
    """Normalised stacked reconstruction mean + V z (+ W_out h)."""
    x = params["mean"] + params["V"] @ z
    if params["use_mlp"]:
        x = x + params["W_out"] @ mlp_h(params, z)
    return x


def encode(params: dict, x_flat: jnp.ndarray, nn: int):
    """(z, scale) from a raw stacked (2*N^2,) pair; exact POD projection."""
    u, v = x_flat[:nn], x_flat[nn:]
    scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(u)), jnp.max(jnp.abs(v)) / V_SCALE), 1e-6)
    x_norm = jnp.concatenate([u / scale, v / (scale * V_SCALE)])
    z = params["V"].T @ (x_norm - params["mean"])
    return z, scale


def decode(params: dict, z: jnp.ndarray, scale: jnp.ndarray, nn: int) -> jnp.ndarray:
    """Raw stacked (u, v) from (z, scale)."""
    x_norm = decode_norm(params, z)
    u = x_norm[:nn] * scale
    v = x_norm[nn:] * scale * V_SCALE
    return jnp.concatenate([u, v])


def reconstruct(params: dict, x_flat: jnp.ndarray, nn: int) -> jnp.ndarray:
    z, s = encode(params, x_flat, nn)
    return decode(params, z, s, nn)
