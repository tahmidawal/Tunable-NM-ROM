"""One-shot trajectory-surrogate correction around live cubic FOM history.

The frozen coordinate decoder was trained on complete Burgers trajectories as
``u(x,y,t;z)``.  Online, ``z`` is recovered once from the available Gaussian
initial condition and the known viscosity.  A single batched decode on a fixed
coarse grid produces all 51 states.  Temporal differences of those states are
then used only as corrections to the FOM's live cubic predictor.  Consequently
smooth decoder bias cancels instead of being injected as an absolute state.

This module is a supervised surrogate control, not by itself a genuine weak
NM-ROM.  It is the cheap gate for whether its temporal-difference representation
is worth wrapping in a one-time weak latent solve.  All online arithmetic is
f64.  The historical checkpoint contains f32 weights;
they are cast once on load and every accepted run records its content hash.
"""
from __future__ import annotations

import hashlib
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc

F64 = jnp.float64
N_LAYERS = 5
N_FREQ = 31
T_FREQ = 8


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    params = jax.tree_util.tree_map(lambda value: jnp.asarray(value, F64), raw)
    leaves = jax.tree_util.tree_leaves(params)
    if len(params.get("trunk", ())) != N_LAYERS:
        raise ValueError("trajectory checkpoint has the wrong trunk depth")
    if any(leaf.dtype != F64 for leaf in leaves):
        raise ValueError("trajectory checkpoint was not promoted to f64")
    return params


def _dense(layer, values):
    return values @ layer["W"] + layer["b"]


def _coord_features(coords, tau):
    spatial_frequencies = jnp.arange(1, N_FREQ + 1, dtype=F64)

    def spatial_features(values):
        phase = jnp.pi * values[:, None] * spatial_frequencies[None]
        return jnp.concatenate(
            (values[:, None], jnp.sin(phase), jnp.cos(phase)), axis=1
        )

    temporal_frequencies = jnp.arange(1, T_FREQ + 1, dtype=F64)
    temporal_phase = jnp.pi * tau * temporal_frequencies
    temporal = jnp.concatenate(
        (
            jnp.asarray([tau], F64),
            jnp.sin(temporal_phase),
            jnp.cos(temporal_phase),
        )
    )
    temporal = jnp.broadcast_to(temporal[None], (coords.shape[0], temporal.size))
    return jnp.concatenate(
        (spatial_features(coords[:, 0]), spatial_features(coords[:, 1]), temporal),
        axis=1,
    )


def _decode_one(params, latent, tau, coords):
    conditioning = jnp.concatenate((latent, jnp.asarray([2.0 * tau - 1.0], F64)))
    embedded = jax.nn.swish(_dense(params["z_embed"], conditioning))
    film = _dense(params["film"], embedded).reshape(N_LAYERS, 2, -1)
    hidden = _coord_features(coords, tau)
    for index, layer in enumerate(params["trunk"]):
        hidden = _dense(layer, hidden)
        hidden = hidden * (1.0 + film[index, 0]) + film[index, 1]
        hidden = jax.nn.swish(hidden)
    return _dense(params["out"], hidden)[:, 0]


def _sample_indices(target_n, coarse_n):
    locations = np.rint(np.linspace(0, target_n - 1, coarse_n)).astype(np.int32)
    return (locations[:, None] * target_n + locations[None, :]).reshape(-1)


def _gaussian_fit_matrix(coarse_n):
    axis = np.linspace(0.0, 1.0, coarse_n)
    x, y = np.meshgrid(axis[1:-1], axis[1:-1], indexing="ij")
    design = np.stack(
        (
            np.ones(x.size),
            x.reshape(-1),
            y.reshape(-1),
            (x * x + y * y).reshape(-1),
        ),
        axis=1,
    )
    return np.linalg.pinv(design).astype(np.float64)


def _infer_latent(sampled_u0, nu, fit_matrix, coarse_n):
    interior = sampled_u0.reshape(coarse_n, coarse_n)[1:-1, 1:-1].reshape(-1)
    coefficients = fit_matrix @ jnp.log(jnp.maximum(interior, 1e-300))
    inverse_width = jnp.minimum(coefficients[3], -1e-12)
    width2 = -0.5 / inverse_width
    width = jnp.sqrt(width2)
    cx = coefficients[1] * width2
    cy = coefficients[2] * width2
    log_amplitude = coefficients[0] + (cx * cx + cy * cy) / (2.0 * width2)
    amplitude = jnp.exp(log_amplitude)
    return jnp.stack(
        (
            (cx - 0.5) / 0.35,
            (cy - 0.5) / 0.35,
            (width - 0.125) / 0.075,
            (amplitude - 1.25) / 0.75,
            (jnp.log(nu) - jnp.log(jnp.sqrt(0.001))) / (0.5 * jnp.log(10.0)),
        )
    )


def _cubic_corrections(states):
    """Corrections relative to the same startup/cubic rule as ``make_chain``."""
    corrections = []
    for step in range(bc.T):
        if step == 0:
            base = states[0]
        elif step == 1:
            base = 2.0 * states[1] - states[0]
        elif step == 2:
            base = 3.0 * states[2] - 3.0 * states[1] + states[0]
        else:
            base = (
                4.0 * states[step]
                - 6.0 * states[step - 1]
                + 4.0 * states[step - 2]
                - states[step - 3]
            )
        corrections.append(states[step + 1] - base)
    return jnp.stack(corrections)


def _resize_fields(fields, target_n):
    source_n = fields.shape[1]
    if source_n == target_n:
        return fields
    positions = jnp.linspace(0.0, source_n - 1.0, target_n)
    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, source_n - 1)
    fraction = positions - lower
    rows = (
        fields[:, lower, :] * (1.0 - fraction)[None, :, None]
        + fields[:, upper, :] * fraction[None, :, None]
    )
    resized = (
        rows[:, :, lower] * (1.0 - fraction)[None, None, :]
        + rows[:, :, upper] * fraction[None, None, :]
    )
    return resized


def make_constructor(target_n, coarse_n, checkpoint_path):
    if coarse_n < 8 or coarse_n > target_n:
        raise ValueError("coarse decode size must lie in [8,target_n]")
    params = load_checkpoint(checkpoint_path)
    indices = jnp.asarray(_sample_indices(target_n, coarse_n))
    fit_matrix = jnp.asarray(_gaussian_fit_matrix(coarse_n), F64)
    axis = jnp.linspace(0.0, 1.0, coarse_n, dtype=F64)
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    coords = jnp.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    taus = jnp.linspace(0.0, 1.0, bc.T + 1, dtype=F64)
    boundary = jnp.ones((coarse_n, coarse_n), F64)
    boundary = boundary.at[0, :].set(0.0).at[-1, :].set(0.0)
    boundary = boundary.at[:, 0].set(0.0).at[:, -1].set(0.0)

    def constructor(u0, nu):
        sampled = u0[indices]
        latent = _infer_latent(sampled, nu, fit_matrix, coarse_n)
        states = jax.vmap(lambda tau: _decode_one(params, latent, tau, coords))(taus)
        states = states.reshape(bc.T + 1, coarse_n, coarse_n) * boundary[None]
        corrections = _cubic_corrections(states)
        corrections = _resize_fields(corrections, target_n)
        corrections = corrections.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
        corrections = corrections.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        return corrections.reshape(bc.T, target_n * target_n), latent

    return jax.jit(constructor)


def default_checkpoint():
    candidates = (
        os.environ.get("TRAJECTORY_CHECKPOINT", ""),
        os.path.join(
            os.path.dirname(__file__),
            "deps",
            "burgers2d-coord-rom",
            "burgers2d_film_N64.pkl",
        ),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "2026-08-14-burgers2d-coord-rom",
            "experiments",
            "burgers2d-coord-rom",
            "sweep",
            "burgers2d_film_N64.pkl",
        ),
    )
    for candidate in candidates:
        path = os.path.abspath(candidate) if candidate else ""
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("cannot locate N=64 Burgers trajectory checkpoint")
