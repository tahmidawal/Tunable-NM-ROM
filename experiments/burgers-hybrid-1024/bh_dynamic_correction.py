"""Fixed-coarse preconditioned residual correction around live cubic history.

This is a classical reduced correction control, not a learned NM-ROM.  It uses
the exact coarse-grid upwind Burgers residual and one exact coarse Dirichlet
Helmholtz inverse.  Its purpose is to test whether a dynamic correction can
remove FOM Newton work cheaply enough to license a learned/weak reduced chart.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc

F64 = jnp.float64


def _sample_indices(target_n, coarse_n):
    points = np.rint(np.linspace(0, target_n - 1, coarse_n)).astype(np.int32)
    return (points[:, None] * target_n + points[None, :]).reshape(-1)


def _resize(field, target_n):
    source_n = field.shape[0]
    if source_n == target_n:
        return field
    positions = jnp.linspace(0.0, source_n - 1.0, target_n)
    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, source_n - 1)
    fraction = positions - lower
    rows = (
        field[lower, :] * (1.0 - fraction)[:, None]
        + field[upper, :] * fraction[:, None]
    )
    return (
        rows[:, lower] * (1.0 - fraction)[None, :]
        + rows[:, upper] * fraction[None, :]
    )


def make_predictor(target_n, coarse_n=32, relaxation=1.0):
    if coarse_n < 8 or coarse_n > target_n:
        raise ValueError("coarse_n must lie in [8,target_n]")
    relaxation = float(relaxation)
    indices = jnp.asarray(_sample_indices(target_n, coarse_n))
    _, residual = bc.bf.make_rollout(coarse_n)
    dx = 1.0 / (coarse_n - 1)
    interior_n = coarse_n - 2
    frequencies = jnp.arange(1, interior_n + 1, dtype=F64)
    eigenvalues = (4.0 / dx**2) * (
        jnp.sin(jnp.pi * frequencies[:, None] / (2.0 * (coarse_n - 1))) ** 2
        + jnp.sin(jnp.pi * frequencies[None, :] / (2.0 * (coarse_n - 1))) ** 2
    )

    def dst_axis(values, axis):
        zeros_shape = list(values.shape)
        zeros_shape[axis] = 1
        zeros = jnp.zeros(zeros_shape, values.dtype)
        extension = jnp.concatenate(
            (zeros, values, zeros, -jnp.flip(values, axis=axis)), axis=axis
        )
        transformed = -jnp.fft.fft(extension, axis=axis).imag
        index = [slice(None)] * values.ndim
        index[axis] = slice(1, values.shape[axis] + 1)
        return transformed[tuple(index)]

    def solve_helmholtz(vector, nu):
        field = vector.reshape(coarse_n, coarse_n)
        coefficients = dst_axis(dst_axis(field[1:-1, 1:-1], 0), 1)
        interior = dst_axis(
            dst_axis(coefficients / (1.0 + bc.bf.DT * nu * eigenvalues), 0), 1
        ) / (4.0 * (interior_n + 1) ** 2)
        return jnp.zeros_like(field).at[1:-1, 1:-1].set(interior).reshape(-1)

    def cubic(u, u2, u3, u4, step_index):
        return jax.lax.cond(
            step_index < 2,
            lambda: jax.lax.cond(step_index == 0, lambda: u, lambda: 2.0 * u - u2),
            lambda: jax.lax.cond(
                step_index == 2,
                lambda: 3.0 * u - 3.0 * u2 + u3,
                lambda: 4.0 * u - 6.0 * u2 + 4.0 * u3 - u4,
            ),
        )

    def predictor(u_prev, u_prev2, u_prev3, u_prev4, nu, step_index):
        base = cubic(u_prev, u_prev2, u_prev3, u_prev4, step_index)
        coarse_base = base[indices]
        coarse_previous = u_prev[indices]
        coarse_residual = residual(coarse_base, coarse_previous, nu)
        coarse_correction = -solve_helmholtz(coarse_residual, nu)
        correction = _resize(
            coarse_correction.reshape(coarse_n, coarse_n), target_n
        )
        correction = correction.at[0, :].set(0.0).at[-1, :].set(0.0)
        correction = correction.at[:, 0].set(0.0).at[:, -1].set(0.0)
        return base + relaxation * correction.reshape(-1)

    return predictor
