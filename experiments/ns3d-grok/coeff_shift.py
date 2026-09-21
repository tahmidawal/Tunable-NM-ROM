"""Coefficient-space form of the rank-64 center tracker.

The startup step matches diag04. The nonlinearity is the same full-grid
dealiased map, stored as a quadratic tensor. The energy centroid is the same
discrete circular mean, stored as six quadratic forms. Removing that centroid
is a Fourier shift of the POD reconstruction followed by reprojection. A
Fourier basis uses a 2 by 2 phase rotation instead, because a shift stays in
that span.
"""
from __future__ import annotations

import numpy as np

import diag_floor as D
import jax
import jax.numpy as jnp
import ns3d_fom as F


def assemble_advection_tensor(basis, n, geom, chunk=32):
    """T[m, j, k] so that T contracted with c, c equals the grid advection."""
    basis_j = jax.device_put(jnp.asarray(basis))
    rank = int(basis.shape[1])

    @jax.jit
    def apply_batch(coeffs):
        def one(coeff):
            field = (basis_j @ coeff).reshape(3, n, n, n)
            return basis_j.T @ F.ifft(F.nonlinear(F.fft(field), geom)).ravel()
        return jax.vmap(one)(coeffs)

    diagonal = np.asarray(apply_batch(jnp.eye(rank, dtype=jnp.float64)))
    tensor = np.zeros((rank, rank, rank), dtype=np.float64)
    for index in range(rank):
        tensor[:, index, index] = diagonal[index]
    left, right = np.triu_indices(rank, k=1)
    for start in range(0, len(left), chunk):
        rows = left[start:start + chunk]
        cols = right[start:start + chunk]
        probes = np.zeros((len(rows), rank), dtype=np.float64)
        probes[np.arange(len(rows)), rows] = 1.0
        probes[np.arange(len(rows)), cols] = 1.0
        values = np.asarray(apply_batch(jnp.asarray(probes)))
        for value, row, col in zip(values, rows, cols):
            cross = (value - diagonal[row] - diagonal[col]) / 2.0
            tensor[:, row, col] = cross
            tensor[:, col, row] = cross
    sample = np.random.default_rng(1).normal(size=(4, rank))
    got = np.einsum("mjk,bj,bk->bm", tensor, sample, sample)
    expected = np.asarray(apply_batch(jnp.asarray(sample)))
    gap = float(np.max(np.linalg.norm(got - expected, axis=1)
                       / np.maximum(np.linalg.norm(expected, axis=1), 1e-300)))
    if gap > 1e-8:
        raise RuntimeError(f"advection tensor misses the grid map: {gap}")
    return tensor, gap


def centroid_matrices(basis, n):
    """Quadratic forms of the discrete circular mean of |G c|^2."""
    rank = int(basis.shape[1])
    modes = np.asarray(basis, dtype=np.float64).T.reshape(rank, 3, n, n, n)
    angle = 2 * np.pi * np.arange(n, dtype=np.float64) / n
    cos_m = np.empty((3, rank, rank), dtype=np.float64)
    sin_m = np.empty((3, rank, rank), dtype=np.float64)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = n
        cos_m[axis] = np.einsum("acxyz,bcxyz,xyz->ab", modes, modes,
                                np.cos(angle).reshape(shape) * np.ones((n, n, n)))
        sin_m[axis] = np.einsum("acxyz,bcxyz,xyz->ab", modes, modes,
                                np.sin(angle).reshape(shape) * np.ones((n, n, n)))
    coeff = np.random.default_rng(2).normal(size=rank)
    field = (np.asarray(basis) @ coeff).reshape(3, n, n, n)
    reference = D.energy_centroid(field)
    got = centroid_from_coeff_np(coeff, sin_m, cos_m)
    gap = float(np.max(D.torus_delta(got, reference)))
    if gap > 1e-8:
        raise RuntimeError(f"coefficient centroid misses the grid centroid: {gap}")
    return sin_m, cos_m, gap


def centroid_from_coeff_np(coeff, sin_m, cos_m):
    sine = np.einsum("aij,i,j->a", sin_m, coeff, coeff)
    cosine = np.einsum("aij,i,j->a", cos_m, coeff, coeff)
    return np.arctan2(sine, cosine) / (2 * np.pi) % 1.0


def fourier_pack(basis, n):
    """Unnormalized FFTs of the basis, packed as (frequency, component, mode)."""
    rank = int(basis.shape[1])
    modes = np.asarray(basis, dtype=np.float64).T.reshape(rank, 3, n, n, n)
    hat = np.fft.fftn(modes, axes=(-3, -2, -1))
    packed = np.moveaxis(np.moveaxis(hat, 0, -1), 0, 3).reshape(n ** 3, 3, rank)
    freq = np.fft.fftfreq(n) * n
    grid_x, grid_y, grid_z = np.meshgrid(freq, freq, freq, indexing="ij")
    return np.ascontiguousarray(packed), grid_x.ravel(), grid_y.ravel(), grid_z.ravel()


def truncate_pack(packed, kx, ky, kz, tail):
    power = np.sum(np.abs(packed) ** 2, axis=(1, 2))
    order = np.argsort(power)[::-1]
    cumulative = np.cumsum(power[order])
    total = float(cumulative[-1])
    count = int(np.searchsorted(cumulative, (1.0 - tail) * total) + 1)
    count = min(max(count, 1), packed.shape[0])
    chosen = order[:count]
    return (np.ascontiguousarray(packed[chosen]), np.ascontiguousarray(kx[chosen]),
            np.ascontiguousarray(ky[chosen]), np.ascontiguousarray(kz[chosen]),
            int(count), float(power[chosen].sum() / total))


def _shift_field(field, offset_samples):
    spec = jnp.fft.fftn(field, axes=(-3, -2, -1))
    n = field.shape[-1]
    for axis in range(3):
        modes = jnp.fft.fftfreq(n) * n
        phase = jnp.exp(-2j * jnp.pi * modes * (offset_samples[axis] / n))
        shape = [1, 1, 1]
        shape[axis] = n
        spec = spec * phase.reshape((1, *shape))
    return jnp.fft.ifftn(spec, axes=(-3, -2, -1)).real


def _grid_centroid(field):
    weight = jnp.sum(field * field, axis=0)
    n = field.shape[-1]
    pi = jnp.asarray(jnp.pi, dtype=field.dtype)
    angle = 2 * pi * jnp.arange(n, dtype=field.dtype) / n
    one = jnp.asarray(1, dtype=field.dtype)
    centers = []
    for axis in range(3):
        marginal = weight.sum(axis=tuple(i for i in range(3) if i != axis))
        centers.append(jnp.arctan2(jnp.sum(marginal * jnp.sin(angle)),
                                   jnp.sum(marginal * jnp.cos(angle))) / (2 * pi) % one)
    return jnp.stack(centers)


def make_coeff_run(dt, nsteps, out_every, n, transport):
    """Startup-step tracker. `transport` is 'pod' or 'fourier'."""
    assert nsteps % out_every == 0
    assert transport in ("pod", "fourier")

    @jax.jit
    def run(u0, nu, basis, linear, tensor, packed, kx, ky, kz, sin_m, cos_m, pair_wave):
        if linear.dtype != jnp.float64:
            linear = 0.5 * (linear + jnp.swapaxes(linear, 0, 1))
        u0 = jnp.asarray(u0, dtype=linear.dtype)
        nu = jnp.asarray(nu, dtype=linear.dtype)
        dt_j = jnp.asarray(dt, dtype=linear.dtype)
        half_j = jnp.asarray(0.5, dtype=linear.dtype)
        identity = jnp.eye(linear.shape[0], dtype=linear.dtype)
        half = jnp.linalg.cholesky(identity - half_j * dt_j * nu * linear)
        full = jnp.linalg.cholesky(identity - dt_j * nu * linear)

        def solve(factor, rhs):
            return jax.scipy.linalg.cho_solve((factor, True), rhs)

        def advect(coeff):
            return jnp.einsum("mjk,j,k->m", tensor, coeff, coeff)

        def centroid(coeff):
            sine = jnp.einsum("aij,i,j->a", sin_m, coeff, coeff)
            cosine = jnp.einsum("aij,i,j->a", cos_m, coeff, coeff)
            pi = jnp.asarray(jnp.pi, dtype=coeff.dtype)
            one = jnp.asarray(1, dtype=coeff.dtype)
            return jnp.arctan2(sine, cosine) / (2 * pi) % one

        def remove_centroid(coeff, delta):
            if transport == "pod":
                mixed = jnp.einsum("fck,k->fc", packed, coeff)
                phase_dtype = jnp.complex64 if delta.dtype == jnp.float32 else jnp.complex128
                phase = jnp.exp(jnp.asarray(-2j * np.pi, dtype=phase_dtype) * (
                    kx * (-delta[0]) + ky * (-delta[1]) + kz * (-delta[2])))
                mixed = mixed * phase[:, None]
                scale = jnp.asarray(n ** 3, dtype=coeff.dtype)
                return jnp.real(jnp.einsum("fck,fc->k", jnp.conj(packed), mixed)) / scale
            theta = -2 * jnp.asarray(jnp.pi, dtype=coeff.dtype) * (pair_wave @ delta)
            cosine = jnp.cos(theta)
            sine = jnp.sin(theta)
            first = coeff[0::2]
            second = coeff[1::2]
            return jnp.stack((first * cosine - second * sine,
                              first * sine + second * cosine), axis=1).ravel()

        def step(carry, _index):
            coeff, center = carry
            current = advect(coeff)
            predicted = solve(full, coeff + dt_j * current)
            updated = solve(half, coeff + half_j * dt_j * (
                nu * (linear @ coeff) + current + advect(predicted)))
            delta = centroid(updated)
            lab = _shift_field((basis @ updated).reshape(3, n, n, n), center * n)
            one = jnp.asarray(1, dtype=center.dtype)
            new_coeff = remove_centroid(updated, delta).astype(coeff.dtype)
            new_center = ((center + delta) % one).astype(center.dtype)
            return (new_coeff, new_center), lab

        def block(carry, _index):
            new, frames = jax.lax.scan(step, carry, jnp.arange(out_every))
            return new, frames[-1]

        center0 = _grid_centroid(u0)
        shifted0 = _shift_field(u0, -center0 * n)
        coeff0 = basis.T @ shifted0.ravel()
        frame0 = _shift_field((basis @ coeff0).reshape(3, n, n, n), center0 * n)
        _, frames = jax.lax.scan(block, (coeff0, center0), jnp.arange(nsteps // out_every))
        return jnp.concatenate((frame0[None], frames))

    return run


def check_shift_identity(basis, packed, kx, ky, kz, n):
    """Zero shift reproduces coefficients. A random shift matches the grid."""
    rank = int(basis.shape[1])
    coeff = np.random.default_rng(3).normal(size=rank)
    packed_j = jnp.asarray(packed)
    axes = (jnp.asarray(kx), jnp.asarray(ky), jnp.asarray(kz))
    basis_j = jnp.asarray(basis)

    def reproject(vector, delta):
        mixed = jnp.einsum("fck,k->fc", packed_j, vector)
        phase = jnp.exp(-2j * jnp.pi * (
            axes[0] * (-delta[0]) + axes[1] * (-delta[1]) + axes[2] * (-delta[2])))
        mixed = mixed * phase[:, None]
        return jnp.real(jnp.einsum("fck,fc->k", jnp.conj(packed_j), mixed)) / n ** 3

    identity = np.asarray(reproject(jnp.asarray(coeff), jnp.zeros(3)))
    identity_gap = float(np.linalg.norm(identity - coeff) / np.linalg.norm(coeff))
    delta = np.array([0.17, -0.08, 0.23])
    got = np.asarray(reproject(jnp.asarray(coeff), jnp.asarray(delta)))
    field = (basis @ coeff).reshape(3, n, n, n)
    expected = basis.T @ D.fourier_shift(field, -delta * n).ravel()
    shift_gap = float(np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-300))
    if identity_gap > 1e-8 or shift_gap > 1e-8:
        raise RuntimeError(f"shift reprojection failed: identity {identity_gap}, shift {shift_gap}")
    del basis_j
    return dict(identity_gap=identity_gap, shift_gap=shift_gap)


def fourier_pair_waves(ids):
    if len(ids) % 2:
        raise RuntimeError("Fourier modes are not cos/sin pairs")
    waves = []
    for index in range(0, len(ids), 2):
        if ids[index]["kind"] != "cos" or ids[index + 1]["kind"] != "sin":
            raise RuntimeError("expected cos then sin")
        if ids[index]["wave"] != ids[index + 1]["wave"]:
            raise RuntimeError("cos/sin pair does not share a wave")
        waves.append(ids[index]["wave"])
    return np.asarray(waves, dtype=np.float64)


def check_phase_rotation(basis, ids, n):
    rank = int(basis.shape[1])
    if rank % 2:
        raise RuntimeError("phase rotation needs an even rank")
    pair_wave = jnp.asarray(fourier_pair_waves(ids))
    coeff = np.random.default_rng(4).normal(size=rank)
    delta = np.array([0.11, -0.07, 0.19])
    theta = -2 * np.pi * (np.asarray(pair_wave) @ delta)
    first = coeff[0::2]
    second = coeff[1::2]
    rotated = np.stack((first * np.cos(theta) - second * np.sin(theta),
                        first * np.sin(theta) + second * np.cos(theta)), axis=1).ravel()
    field = (basis @ coeff).reshape(3, n, n, n)
    expected = basis.T @ D.fourier_shift(field, -delta * n).ravel()
    gap = float(np.linalg.norm(rotated - expected) / max(np.linalg.norm(expected), 1e-300))
    if gap > 1e-8:
        raise RuntimeError(f"phase rotation misses the grid shift: {gap}")
    return gap
