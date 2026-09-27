"""Small separable transported residual decoder for Poisson warm starts.

The model represents only the tail left after an exact q-by-q sine correction.  It is a
nonlinear manifold: the latent moves and dilates one-dimensional stems before their rank-R
outer products are assembled.  There is no fixed ambient output basis.  Full-grid evaluation
cost is O(R N^2 + width R N), while arbitrary EQ points remain O(width R m).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

F64 = jnp.float64


def config(rank=8, width=24, depth=2, k_lat=6):
    return dict(name="separable_transport_tail", rank=int(rank), width=int(width),
                depth=int(depth), k_lat=int(k_lat), hard_bc=True)


def _dense(key, din, dout, scale=1.0):
    return dict(
        W=jax.random.normal(key, (din, dout), dtype=F64)
        * (scale / np.sqrt(float(din))),
        b=jnp.zeros((dout,), dtype=F64),
    )


def _init_mlp(keys, din, width, depth, dout, out_scale=0.1):
    layers = [_dense(next(keys), din, width)]
    layers.extend(_dense(next(keys), width, width) for _ in range(depth - 1))
    layers.append(_dense(next(keys), width, dout, scale=out_scale))
    return layers


def init(key, cfg):
    """Initialize two width-by-depth 1D stems and one latent coefficient net."""
    rank, width, depth, k_lat = (int(cfg[k]) for k in
                                  ("rank", "width", "depth", "k_lat"))
    # Each stem gets depth hidden layers + output; the coefficient net gets one hidden + output.
    keys = iter(jax.random.split(key, 2 * (depth + 1) + 2))
    params = dict(
        x=_init_mlp(keys, 5, width, depth, rank, out_scale=1.0),
        y=_init_mlp(keys, 5, width, depth, rank, out_scale=1.0),
        coef=_init_mlp(keys, k_lat, width, 1, rank, out_scale=0.5),
    )
    # An untrained learned tail must be exactly the q-base, not high-frequency noise.
    # The zero coefficient head still has nonzero gradients through its random input stem.
    params["coef"][-1]["W"] = jnp.zeros_like(params["coef"][-1]["W"])
    params["coef"][-1]["b"] = jnp.zeros_like(params["coef"][-1]["b"])
    return params


def _mlp(layers, x):
    for layer in layers[:-1]:
        x = jax.nn.silu(x @ layer["W"] + layer["b"])
    return x @ layers[-1]["W"] + layers[-1]["b"]


def _axis_features(x, center, width):
    t = (x - center) / jnp.maximum(width, 0.008)
    g = jnp.exp(-0.5 * jnp.square(t))
    return jnp.stack([x, t, g, jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1)


def physical_transform(z):
    """Turn standardized source-like latents into a bounded transported chart."""
    cx = 0.5 + 0.35 * z[0] + 0.04 * jnp.tanh(z[4])
    cy = 0.5 + 0.35 * z[1] + 0.04 * jnp.tanh(z[5])
    width = jnp.exp(jnp.log(0.045) + 0.8 * z[2])
    return cx, cy, width


def _stems(params, z, x, y):
    cx, cy, width = physical_transform(z)
    sx = _mlp(params["x"], _axis_features(x, cx, width))
    sy = _mlp(params["y"], _axis_features(y, cy, width))
    coef = _mlp(params["coef"], z) / jnp.sqrt(jnp.asarray(sx.shape[-1], F64))
    return sx, sy, coef


def apply_grid(params, z, x, eps):
    """Decode one endpoint-including square grid from a cached 1D coordinate vector."""
    sx, sy, coef = _stems(params, z, x, x)
    raw = (sx * coef[None, :]) @ sy.T
    b = 4.0 * x * (1.0 - x)
    return eps * b[:, None] * raw * b[None, :]


def apply_points(params, z, xy, eps):
    """Decode paired arbitrary points for the empirical-quadrature weak objective."""
    sx, sy, coef = _stems(params, z, xy[:, 0], xy[:, 1])
    raw = jnp.sum(sx * sy * coef[None, :], axis=1)
    b = 16.0 * xy[:, 0] * (1.0 - xy[:, 0]) * xy[:, 1] * (1.0 - xy[:, 1])
    return eps * b * raw


def parameter_count(params):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
