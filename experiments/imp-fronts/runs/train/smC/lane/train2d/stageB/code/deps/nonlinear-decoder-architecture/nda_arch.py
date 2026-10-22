"""Pure nonlinear coordinate decoders for the 2026-08-19 architecture study.

The control decoder remains ``ms_parametric.film_apply``.  This module owns the
new residual/grouped-FiLM arm so Poisson and Burgers use byte-identical model
code.  All configuration is explicit and is stored in every checkpoint.

The residual decoder has a coordinate-only nonlinear stem followed by
latent-modulated residual blocks.  FiLM parameters are shared by fixed-size
channel groups.  Since nonlinear activations follow the joint (x, z)
modulation, its image is a nonlinear manifold; there is no fixed spatial
basis or POD component.
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

F64 = jnp.float64


def coord_features(xy, n_freq):
    """Match the frozen control's sin/cos(pi*j*x) coordinate convention."""
    j = jnp.arange(1, n_freq + 1, dtype=F64)

    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)

    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1])], axis=1)


def z_features(z, z_ff):
    if z_ff <= 0:
        return z
    j = jnp.arange(1, z_ff + 1, dtype=F64)
    return jnp.concatenate(
        [z, jnp.sin(jnp.pi * j[:, None] * z[None, :]).reshape(-1),
         jnp.cos(jnp.pi * j[:, None] * z[None, :]).reshape(-1)])


def _dense(key, d_in, d_out, scale=1.0):
    W = (jax.random.normal(key, (d_in, d_out), dtype=F64)
         * (scale * np.sqrt(1.0 / d_in)))
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F64)}


def validate_config(config):
    required = ("name", "hidden", "n_layers", "group_size", "film_start",
                "z_width", "residual_scale", "warp_max_shift",
                "warp_max_log_scale")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"decoder config missing {missing}")
    if config["name"] not in ("resfilm", "groupfilm", "warp_resfilm"):
        raise ValueError(f"unknown nonlinear decoder {config['name']}")
    h = int(config["hidden"])
    group = int(config["group_size"])
    layers = int(config["n_layers"])
    start = int(config["film_start"])
    if h <= 0 or group <= 0 or h % group:
        raise ValueError(f"hidden={h} must be divisible by group_size={group}")
    if layers < 2:
        raise ValueError("n_layers counts the stem and residual transforms; need >=2")
    if not 0 <= start < layers:
        raise ValueError(f"film_start must be in [0,{layers - 1}], got {start}")
    return config


def config(name, hidden, n_layers, *, group_size=8, film_start=1,
           z_width=64, residual_scale=None, warp_max_shift=0.15,
           warp_max_log_scale=0.25):
    """Create a JSON-safe architecture manifest."""
    n_blocks = int(n_layers) - 1
    if residual_scale is None:
        residual_scale = 1.0 / math.sqrt(n_blocks)
    return validate_config(dict(
        name=str(name), hidden=int(hidden), n_layers=int(n_layers),
        group_size=int(group_size), film_start=int(film_start),
        z_width=int(z_width), residual_scale=float(residual_scale),
        warp_max_shift=float(warp_max_shift),
        warp_max_log_scale=float(warp_max_log_scale)))


def init(key, n_freq, k_lat, config_, z_ff=0):
    """Initialize a residual/grouped-FiLM coordinate decoder."""
    cfg = validate_config(dict(config_))
    hidden = int(cfg["hidden"])
    n_layers = int(cfg["n_layers"])
    n_blocks = n_layers - 1
    n_mod = n_layers - int(cfg["film_start"])
    n_groups = hidden // int(cfg["group_size"])
    d_in = 2 * (2 * int(n_freq) + 1)
    d_z = int(k_lat) * (1 + 2 * int(z_ff))
    n_keys = 1 + n_blocks + 1 + 1 + 1 + (1 if cfg["name"] == "warp_resfilm" else 0)
    keys = iter(jax.random.split(key, n_keys))
    stem = _dense(next(keys), d_in, hidden)
    blocks = [_dense(next(keys), hidden, hidden) for _ in range(n_blocks)]
    out = _dense(next(keys), hidden, 1)
    z_embed = _dense(next(keys), d_z, int(cfg["z_width"]))
    film = _dense(next(keys), int(cfg["z_width"]), n_mod * 2 * n_groups,
                  scale=0.01)
    params = {"stem": stem, "blocks": blocks, "out": out,
              "z_embed": z_embed, "film": film}
    if cfg["name"] == "warp_resfilm":
        # Zero initialization makes the first forward pass exactly the unwarped
        # residual decoder while retaining nonzero gradients through tanh.
        params["warp"] = _dense(next(keys), int(cfg["z_width"]), 4, scale=0.0)
    return params


def parameter_count(params):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def _latent_embedding(params, z, z_ff):
    return jax.nn.swish(z_features(z, z_ff) @ params["z_embed"]["W"]
                        + params["z_embed"]["b"])


def prepare_coords(params, xy, n_freq, config_):
    """Cache the coordinate-only stem at a fixed point set.

    This is valid only for the unwarped residual decoder.  It changes no
    arithmetic downstream of the stem and imposes no output basis.
    """
    cfg = validate_config(config_)
    if cfg["name"] not in ("resfilm", "groupfilm"):
        raise ValueError("coordinate caching is unavailable when coordinates depend on z")
    # Cache the affine map, not its activation: film_start=0 still modulates the
    # stem before the first nonlinearity and remains exactly cacheable.
    return coord_features(xy, n_freq) @ params["stem"]["W"] + params["stem"]["b"]


def apply_prepared(params, z, h, config_, z_ff=0):
    """Evaluate from a coordinate stem produced by :func:`prepare_coords`."""
    cfg = validate_config(config_)
    hidden = int(cfg["hidden"])
    group_size = int(cfg["group_size"])
    film_start = int(cfg["film_start"])
    n_layers = int(cfg["n_layers"])
    n_groups = hidden // group_size
    n_mod = n_layers - film_start

    ze = _latent_embedding(params, z, z_ff)
    film = (ze @ params["film"]["W"] + params["film"]["b"]).reshape(
        n_mod, 2, n_groups)
    if film_start == 0:
        gamma = jnp.repeat(film[0, 0], group_size)
        beta = jnp.repeat(film[0, 1], group_size)
        h = h * (1.0 + gamma) + beta
    h = jax.nn.swish(h)
    scale = float(cfg["residual_scale"])
    for block_index, lyr in enumerate(params["blocks"], start=1):
        r = h @ lyr["W"] + lyr["b"]
        if block_index >= film_start:
            mod_index = block_index - film_start
            gamma = jnp.repeat(film[mod_index, 0], group_size)
            beta = jnp.repeat(film[mod_index, 1], group_size)
            r = r * (1.0 + gamma) + beta
        if cfg["name"] == "groupfilm":
            h = jax.nn.swish(r)
        else:
            h = h + scale * jax.nn.swish(r)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]


def apply(params, z, xy, n_freq, config_, z_ff=0):
    """Evaluate the nonlinear decoder at arbitrary coordinates ``xy``."""
    cfg = validate_config(config_)
    xy_net = xy
    if cfg["name"] == "warp_resfilm":
        ze = _latent_embedding(params, z, z_ff)
        raw = ze @ params["warp"]["W"] + params["warp"]["b"]
        shift = float(cfg["warp_max_shift"]) * jnp.tanh(raw[:2])
        log_scale = float(cfg["warp_max_log_scale"]) * jnp.tanh(raw[2:])
        xy_net = 0.5 + (xy - 0.5) * jnp.exp(log_scale)[None, :] + shift[None, :]

    h = coord_features(xy_net, n_freq) @ params["stem"]["W"] + params["stem"]["b"]
    # Keep the warp's z-dependent stem in this function; the unwarped arm is
    # bitwise-equivalent to prepare_coords + apply_prepared.
    if cfg["name"] in ("resfilm", "groupfilm"):
        return apply_prepared(params, z, h, cfg, z_ff)

    hidden = int(cfg["hidden"])
    group_size = int(cfg["group_size"])
    film_start = int(cfg["film_start"])
    n_layers = int(cfg["n_layers"])
    n_groups = hidden // group_size
    n_mod = n_layers - film_start
    film = (ze @ params["film"]["W"] + params["film"]["b"]).reshape(
        n_mod, 2, n_groups)
    if film_start == 0:
        gamma = jnp.repeat(film[0, 0], group_size)
        beta = jnp.repeat(film[0, 1], group_size)
        h = h * (1.0 + gamma) + beta
    h = jax.nn.swish(h)
    scale = float(cfg["residual_scale"])
    for block_index, lyr in enumerate(params["blocks"], start=1):
        r = h @ lyr["W"] + lyr["b"]
        if block_index >= film_start:
            mod_index = block_index - film_start
            gamma = jnp.repeat(film[mod_index, 0], group_size)
            beta = jnp.repeat(film[mod_index, 1], group_size)
            r = r * (1.0 + gamma) + beta
        h = h + scale * jax.nn.swish(r)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]
