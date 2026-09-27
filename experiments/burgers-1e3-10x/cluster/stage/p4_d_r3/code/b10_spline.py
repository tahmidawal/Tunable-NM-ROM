"""Locked transported cubic B-spline hyperdecoder machinery for Phase 2."""
from __future__ import annotations

import math
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import b10_common as c

F64 = jnp.float64
DEGREE = 3
DOMAIN_HALF_WIDTH = 4.5
DOMAIN_LENGTH = 9.0
ORACLE_RIDGE = 1e-12
ORACLE_NORMAL_TOL = 1e-8
ORACLE_LSMR_TOL = 1e-12
ORACLE_LSMR_CONLIM = 1e12
ORACLE_LSMR_MAXITER = 500
HYPER_WIDTH = 32

CANDIDATES = (
    {"arm": "A", "R": 24, "k": 12, "M": 64, "m": 256},
    {"arm": "B", "R": 32, "k": 16, "M": 64, "m": 256},
    {"arm": "C", "R": 48, "k": 24, "M": 96, "m": 384},
)


def knots_np(r):
    """Open-uniform clamped cubic knots on the locked aligned domain."""
    r = int(r)
    if r < DEGREE + 1:
        raise ValueError("cubic spline needs at least four controls")
    spacing = DOMAIN_LENGTH / (r - DEGREE)
    interior = -DOMAIN_HALF_WIDTH + spacing * np.arange(1, r - DEGREE)
    return np.concatenate((
        np.full(DEGREE + 1, -DOMAIN_HALF_WIDTH, np.float64),
        interior.astype(np.float64),
        np.full(DEGREE + 1, DOMAIN_HALF_WIDTH, np.float64),
    ))


def knots_jax(r):
    return jnp.asarray(knots_np(r), F64)


def analytic_screen(r):
    spacing = DOMAIN_LENGTH / (int(r) - DEGREE)
    tail = math.sqrt(1.0 - math.erf(DOMAIN_HALF_WIDTH) ** 2)
    delta = (5.0 / 384.0) * spacing**4 * math.sqrt(105.0 / 16.0)
    return {
        "R": int(r),
        "spacing": float(spacing),
        "relative_l2_tail_bound": float(tail),
        "one_axis_cubic_bound": float(delta),
        "two_dimensional_tail_plus_cubic_bound": float(tail + 2.0 * delta + delta**2),
    }


def hyperdecoder_parameter_count(r, k):
    q = int(k) - 5
    return (q * HYPER_WIDTH + HYPER_WIDTH
            + HYPER_WIDTH * HYPER_WIDTH + HYPER_WIDTH
            + HYPER_WIDTH * int(r) ** 2 + int(r) ** 2)


def predictor_parameter_count(k):
    return (7 * HYPER_WIDTH + HYPER_WIDTH
            + HYPER_WIDTH * HYPER_WIDTH + HYPER_WIDTH
            + HYPER_WIDTH * int(k) + int(k))


def _basis_1d_jax(value, r):
    """Return the four active control indices/weights for one scalar."""
    knots = knots_jax(r)
    valid = (value >= -DOMAIN_HALF_WIDTH) & (value <= DOMAIN_HALF_WIDTH)
    safe = jnp.clip(value, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH)
    spacing = DOMAIN_LENGTH / (int(r) - DEGREE)
    scaled = (safe + DOMAIN_HALF_WIDTH) / spacing
    cell = jnp.floor(scaled).astype(jnp.int32)
    span = jnp.clip(cell + DEGREE, DEGREE, int(r) - 1)
    # Division can collapse a nextafter neighbor onto an integer.  Two O(1) knot
    # gathers correct the tentative uniform span without a binary search.
    span = span - (safe < knots[span]).astype(jnp.int32)
    span = span + (safe >= knots[span + 1]).astype(jnp.int32)
    span = jnp.where(safe == DOMAIN_HALF_WIDTH, int(r) - 1, span)
    span = jnp.clip(span, DEGREE, int(r) - 1)

    values = jnp.zeros((DEGREE + 1,), F64).at[0].set(1.0)
    left = jnp.zeros((DEGREE + 1,), F64)
    right = jnp.zeros((DEGREE + 1,), F64)
    for order in range(1, DEGREE + 1):
        left = left.at[order].set(safe - knots[span + 1 - order])
        right = right.at[order].set(knots[span + order] - safe)
        saved = jnp.asarray(0.0, F64)
        for index in range(order):
            denominator = right[index + 1] + left[order - index]
            temporary = jnp.where(denominator != 0.0, values[index] / denominator, 0.0)
            updated = saved + right[index + 1] * temporary
            saved = left[order - index] * temporary
            values = values.at[index].set(updated)
        values = values.at[order].set(saved)
    indices = span - DEGREE + jnp.arange(DEGREE + 1, dtype=jnp.int32)
    return indices, values * valid.astype(F64)


def local_rows_jax(aligned_coords, r):
    """Exact 16-entry sparse tensor rows for aligned coordinates."""
    ix, wx = jax.vmap(lambda value: _basis_1d_jax(value, r))(aligned_coords[:, 0])
    iy, wy = jax.vmap(lambda value: _basis_1d_jax(value, r))(aligned_coords[:, 1])
    indices = (ix[:, :, None] * int(r) + iy[:, None, :]).reshape(-1, 16)
    weights = (wx[:, :, None] * wy[:, None, :]).reshape(-1, 16)
    return indices, weights


def _basis_1d_np(values, r):
    """Vectorized NumPy equivalent of the locked Cox-de Boor convention."""
    values = np.asarray(values, np.float64)
    knots = knots_np(r)
    valid = (values >= -DOMAIN_HALF_WIDTH) & (values <= DOMAIN_HALF_WIDTH)
    safe = np.clip(values, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH)
    span = np.searchsorted(knots, safe, side="right") - 1
    span = np.where(safe == DOMAIN_HALF_WIDTH, int(r) - 1, span)
    span = np.clip(span, DEGREE, int(r) - 1)
    basis = np.zeros((safe.size, DEGREE + 1), np.float64)
    basis[:, 0] = 1.0
    left = np.zeros_like(basis)
    right = np.zeros_like(basis)
    rows = np.arange(safe.size)
    for order in range(1, DEGREE + 1):
        left[:, order] = safe - knots[span + 1 - order]
        right[:, order] = knots[span + order] - safe
        saved = np.zeros(safe.size, np.float64)
        for index in range(order):
            denominator = right[:, index + 1] + left[:, order - index]
            temporary = np.divide(
                basis[:, index], denominator,
                out=np.zeros_like(denominator), where=denominator != 0.0,
            )
            basis[:, index] = saved + right[:, index + 1] * temporary
            saved = left[:, order - index] * temporary
        basis[:, order] = saved
    indices = span[:, None] - DEGREE + np.arange(DEGREE + 1)[None]
    return indices.astype(np.int64), basis * valid[:, None], rows


def local_rows_np(aligned_coords, r):
    aligned_coords = np.asarray(aligned_coords, np.float64)
    ix, wx, _ = _basis_1d_np(aligned_coords[:, 0], r)
    iy, wy, _ = _basis_1d_np(aligned_coords[:, 1], r)
    indices = (ix[:, :, None] * int(r) + iy[:, None, :]).reshape(-1, 16)
    weights = (wx[:, :, None] * wy[:, None, :]).reshape(-1, 16)
    return indices, weights


def affine_state_from_field(field, coords):
    """Locked positive-field L2 full-covariance moment alignment."""
    field = np.asarray(field, np.float64).reshape(-1)
    coords = np.asarray(coords, np.float64)
    weights = np.square(np.maximum(field, 0.0))
    mass = float(np.sum(weights))
    if not np.isfinite(mass) or mass <= 1e-300:
        raise ValueError("moment transport has nonfinite/zero positive L2 mass")
    center = np.sum(weights[:, None] * coords, axis=0) / mass
    delta = coords - center[None]
    covariance = 2.0 * np.einsum("n,ni,nj->ij", weights, delta, delta) / mass
    covariance += 1e-8 * np.eye(2, dtype=np.float64)
    cholesky = np.linalg.cholesky(covariance)
    state = np.asarray((
        center[0], center[1], np.log(cholesky[0, 0]),
        cholesky[1, 0], np.log(cholesky[1, 1]),
    ), np.float64)
    if not np.all(np.isfinite(state)):
        raise ValueError("nonfinite moment transport")
    return state


def aligned_coords_np(coords, affine_state):
    coords = np.asarray(coords, np.float64)
    state = np.asarray(affine_state, np.float64)
    first = (coords[:, 0] - state[0]) / np.exp(state[2])
    second = (coords[:, 1] - state[1] - state[3] * first) / np.exp(state[4])
    return np.column_stack((first, second))


def aligned_coords_jax(coords, affine_state):
    first = (coords[:, 0] - affine_state[0]) / jnp.exp(affine_state[2])
    second = (
        coords[:, 1] - affine_state[1] - affine_state[3] * first
    ) / jnp.exp(affine_state[4])
    return jnp.stack((first, second), axis=1)


def normalized_state_from_affine(affine_state):
    """Map physical moment transport into the locked dimensionless state."""
    affine = np.asarray(affine_state, np.float64)
    l11 = np.exp(affine[2])
    l22 = np.exp(affine[4])
    state = np.asarray((
        (affine[0] - 0.5) / 0.75,
        (affine[1] - 0.5) / 0.75,
        2.0 * (np.log(l11) - np.log(0.015)) / np.log(1.0 / 0.015) - 1.0,
        affine[3] / (2.0 * np.sqrt(l11 * l22)),
        2.0 * (np.log(l22) - np.log(0.015)) / np.log(1.0 / 0.015) - 1.0,
    ), np.float64)
    if not np.all(np.isfinite(state)) or np.any(np.abs(state) > 1.0):
        raise ValueError(f"moment transport outside locked normalized range: {state}")
    return state


def affine_from_normalized_jax(state):
    """Decode the first five normalized variables to physical affine state."""
    state = jnp.asarray(state, F64)
    mu = 0.5 + 0.75 * state[:2]
    log_ratio = math.log(1.0 / 0.015)
    ell11 = math.log(0.015) + 0.5 * (state[2] + 1.0) * log_ratio
    ell22 = math.log(0.015) + 0.5 * (state[4] + 1.0) * log_ratio
    offdiag = 2.0 * state[3] * jnp.sqrt(jnp.exp(ell11 + ell22))
    return jnp.asarray((mu[0], mu[1], ell11, offdiag, ell22), F64)


def sparse_design(field, coords, boundary_mask, r):
    affine = affine_state_from_field(field, coords)
    indices, weights = local_rows_np(aligned_coords_np(coords, affine), r)
    weights = weights * np.asarray(boundary_mask, np.float64)[:, None]
    n_points = weights.shape[0]
    rows = np.repeat(np.arange(n_points, dtype=np.int64), 16)
    design = sp.coo_matrix(
        (weights.reshape(-1), (rows, indices.reshape(-1))),
        shape=(n_points, int(r) ** 2), dtype=np.float64,
    ).tocsr()
    design.sum_duplicates()
    return affine, design, indices, weights


def fit_free_oracle(field, coords, boundary_mask, r, maxiter=ORACLE_LSMR_MAXITER):
    """Fit one deterministic sparse free-spline spatial projection oracle."""
    started = time.perf_counter()
    field = np.asarray(field, np.float64).reshape(-1)
    affine, design, indices, weights = sparse_design(field, coords, boundary_mask, r)
    diagonal = np.asarray(design.power(2).sum(axis=0)).reshape(-1)
    ridge = ORACLE_RIDGE * max(float(np.mean(diagonal)), 1e-300)
    result = spla.lsmr(
        design, field, damp=math.sqrt(ridge),
        atol=ORACLE_LSMR_TOL, btol=ORACLE_LSMR_TOL,
        conlim=ORACLE_LSMR_CONLIM, maxiter=int(maxiter), show=False,
    )
    coefficients = np.asarray(result[0], np.float64)
    prediction = np.asarray(design @ coefficients, np.float64)
    normal = np.asarray(design.T @ (prediction - field), np.float64) + ridge * coefficients
    rhs = np.asarray(design.T @ field, np.float64)
    relative_normal = float(np.linalg.norm(normal) / max(np.linalg.norm(rhs), 1e-300))
    boundary_values = prediction[np.asarray(boundary_mask) == 0.0]
    healthy = bool(
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(prediction))
        and np.isfinite(relative_normal)
        and relative_normal <= ORACLE_NORMAL_TOL
        and np.all(boundary_values == 0.0)
    )
    info = {
        "healthy": healthy,
        "lsmr_istop": int(result[1]),
        "lsmr_iterations": int(result[2]),
        "lsmr_normr": float(result[3]),
        "lsmr_normar": float(result[4]),
        "lsmr_norma": float(result[5]),
        "lsmr_conda": float(result[6]),
        "lsmr_normx": float(result[7]),
        "ridge": float(ridge),
        "relative_normal_residual": relative_normal,
        "exact_boundary": bool(np.all(boundary_values == 0.0)),
        "local_entries_per_point": 16,
        "max_partition_sum_error_in_domain": float(np.max(np.abs(
            np.sum(weights, axis=1)[
                np.all(np.abs(aligned_coords_np(coords, affine)) <= DOMAIN_HALF_WIDTH, axis=1)
                & (np.asarray(boundary_mask) > 0.0)
            ] - 1.0
        ), initial=0.0)),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return affine, coefficients, prediction, info


def decode_one_jax(state, coefficients, coords, boundary_mask, r):
    affine_state = affine_from_normalized_jax(state[:5])
    indices, weights = local_rows_jax(aligned_coords_jax(coords, affine_state), r)
    values = jnp.sum(coefficients[indices] * weights, axis=1)
    return values * boundary_mask


def init_mlp(key, dimensions):
    """Deterministic Xavier-normal MLP pytree, all f64."""
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = []
    for one, n_in, n_out in zip(keys, dimensions[:-1], dimensions[1:]):
        scale = math.sqrt(2.0 / (n_in + n_out))
        layers.append({
            "W": scale * jax.random.normal(one, (n_in, n_out), dtype=F64),
            "b": jnp.zeros((n_out,), F64),
        })
    return tuple(layers)


def apply_mlp(parameters, inputs):
    hidden = inputs
    for layer in parameters[:-1]:
        hidden = jax.nn.swish(hidden @ layer["W"] + layer["b"])
    return hidden @ parameters[-1]["W"] + parameters[-1]["b"]


def init_cost_oracle_parameters(r, k, seed):
    one, two = jax.random.split(jax.random.PRNGKey(int(seed)))
    predictor = init_mlp(one, (7, HYPER_WIDTH, HYPER_WIDTH, int(k)))
    hyper = init_mlp(two, (int(k) - 5, HYPER_WIDTH, HYPER_WIDTH, int(r) ** 2))
    return predictor, hyper


def predictor_raw_to_state(raw):
    """Stable fixed map used only by the untrained S0 structural cost oracle."""
    return jnp.tanh(raw)


def charged_construction(features, coords, boundary_mask, predictor, hyper, r):
    """Predict, generate all coefficient grids, and decode every requested slice."""
    states = predictor_raw_to_state(apply_mlp(predictor, features))
    coefficients = apply_mlp(hyper, states[:, 5:])
    # Sequential over time: each spatial slice remains a fused 16-support query,
    # avoiding a 51 x N^2 x 16 materialized index/weight tensor at N=1024.
    fields = jax.lax.map(
        lambda pair: decode_one_jax(
            pair[0], pair[1], coords, boundary_mask, r
        ),
        (states, coefficients),
    )
    return fields, states, coefficients


def assert_locked_math():
    """Cheap implementation invariants used by both smoke and scientific S0."""
    for candidate in CANDIDATES:
        r, k = candidate["R"], candidate["k"]
        knots = knots_np(r)
        assert knots.size == r + DEGREE + 1
        assert np.all(knots[:4] == -DOMAIN_HALF_WIDTH)
        assert np.all(knots[-4:] == DOMAIN_HALF_WIDTH)
        assert hyperdecoder_parameter_count(r, k) < 100000
        assert candidate["M"] >= max(64, 4 * k)
        assert candidate["m"] == 4 * candidate["M"]
    probe = np.asarray((
        (-4.5, -4.5), (-4.0, 0.0), (0.0, 0.0),
        (4.5, 4.5), (-4.5001, 0.0), (0.0, 4.5001),
    ), np.float64)
    for candidate in CANDIDATES:
        _, weights = local_rows_np(probe, candidate["R"])
        if not np.allclose(np.sum(weights[:4], axis=1), 1.0, atol=2e-15, rtol=0.0):
            raise AssertionError("in-domain B-spline rows do not partition unity")
        if not np.all(weights[4:] == 0.0):
            raise AssertionError("outside-domain B-spline rows are nonzero")
        # The uniform-span arithmetic must reproduce the searchsorted NumPy
        # reference at every knot and its immediate f64 neighbors.
        knots = np.unique(knots_np(candidate["R"]))
        knot_probe = np.unique(np.concatenate((
            knots,
            np.nextafter(knots, -np.inf),
            np.nextafter(knots, np.inf),
        )))
        reference_indices, reference_weights, _ = _basis_1d_np(
            knot_probe, candidate["R"]
        )
        got_indices, got_weights = jax.vmap(
            lambda value: _basis_1d_jax(value, candidate["R"])
        )(jnp.asarray(knot_probe, F64))
        got_indices, got_weights = np.asarray(got_indices), np.asarray(got_weights)
        if not np.array_equal(got_indices, reference_indices):
            raise AssertionError("uniform span indices disagree with knot reference")
        tolerance = 8.0 * np.finfo(np.float64).eps
        if not np.allclose(got_weights, reference_weights, atol=tolerance, rtol=tolerance):
            raise AssertionError("uniform span weights disagree beyond 8 f64 eps")
