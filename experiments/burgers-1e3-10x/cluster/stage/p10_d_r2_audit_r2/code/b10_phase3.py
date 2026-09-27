"""Prospective exact solver/kernel candidates for bounded Burgers Phase 3.

This module preserves the Phase-2 R=48 transported spline space.  It only
changes how its already-locked cubic basis is evaluated and how the identical
ridge projection is solved.
"""
from __future__ import annotations

import math
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import b10_spline as s

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton
except ImportError:  # pragma: no cover - cluster/local preflight records this
    pl = None
    pltriton = None


F64 = jnp.float64
R = 48
K = 24
M = 96
Q = 384
TIME_CHUNK = 3
PALLAS_BLOCK = 128
PALLAS_STATE_BLOCK = 32
SOLVER_MAXITER = 4 * R * R


def _multiply_linear(polynomial, constant, slope):
    """Multiply an ascending-power polynomial by constant+slope*t."""
    result = np.zeros_like(polynomial)
    result += constant * polynomial
    result[1:] += slope * polynomial[:-1]
    return result


def span_polynomial_table_np(r=R):
    """Return exact per-span cubic coefficients derived from the locked knots.

    ``table[span, local_basis, power]`` represents the same Cox--de Boor basis
    on that nonempty knot span in the local coordinate t in [0,1].  Coefficients
    are constructed by the defining recurrence, not by fitting samples.
    """
    r = int(r)
    knots = s.knots_np(r)
    table = np.zeros((r, s.DEGREE + 1, s.DEGREE + 1), np.float64)
    for span in range(s.DEGREE, r):
        left_edge = knots[span]
        width = knots[span + 1] - left_edge
        if not width > 0.0:
            raise AssertionError("locked active knot span is not positive")
        previous = np.zeros((knots.size - 1, s.DEGREE + 1), np.float64)
        previous[span, 0] = 1.0
        for degree in range(1, s.DEGREE + 1):
            current = np.zeros(
                (knots.size - degree - 1, s.DEGREE + 1), np.float64
            )
            for index in range(current.shape[0]):
                left_denominator = knots[index + degree] - knots[index]
                if left_denominator != 0.0:
                    current[index] += _multiply_linear(
                        previous[index],
                        (left_edge - knots[index]) / left_denominator,
                        width / left_denominator,
                    )
                right_denominator = (
                    knots[index + degree + 1] - knots[index + 1]
                )
                if right_denominator != 0.0:
                    current[index] += _multiply_linear(
                        previous[index + 1],
                        (knots[index + degree + 1] - left_edge)
                        / right_denominator,
                        -width / right_denominator,
                    )
            previous = current
        table[span] = previous[span - s.DEGREE:span + 1]
    return table


def _span_safe_valid_jax(value, r):
    knots = s.knots_jax(r)
    valid = (
        (value >= -s.DOMAIN_HALF_WIDTH)
        & (value <= s.DOMAIN_HALF_WIDTH)
    )
    safe = jnp.clip(value, -s.DOMAIN_HALF_WIDTH, s.DOMAIN_HALF_WIDTH)
    spacing = s.DOMAIN_LENGTH / (int(r) - s.DEGREE)
    scaled = (safe + s.DOMAIN_HALF_WIDTH) / spacing
    cell = jnp.floor(scaled).astype(jnp.int32)
    span = jnp.clip(cell + s.DEGREE, s.DEGREE, int(r) - 1)
    span = span - (safe < knots[span]).astype(jnp.int32)
    span = span + (safe >= knots[span + 1]).astype(jnp.int32)
    span = jnp.where(safe == s.DOMAIN_HALF_WIDTH, int(r) - 1, span)
    span = jnp.clip(span, s.DEGREE, int(r) - 1)
    return span, safe, valid


def _basis_1d_polynomial_jax(value, table, r=R):
    """Evaluate four active weights by a precomputed per-span cubic Horner map."""
    span, safe, valid = _span_safe_valid_jax(value, r)
    knots = s.knots_jax(r)
    local = (safe - knots[span]) / (knots[span + 1] - knots[span])
    coefficients = table[span]
    weights = (
        (coefficients[:, 3] * local + coefficients[:, 2]) * local
        + coefficients[:, 1]
    ) * local + coefficients[:, 0]
    indices = span - s.DEGREE + jnp.arange(s.DEGREE + 1, dtype=jnp.int32)
    return indices, weights * valid.astype(F64)


def local_rows_polynomial_jax(aligned_coords, table, r=R):
    ix, wx = jax.vmap(
        lambda value: _basis_1d_polynomial_jax(value, table, r)
    )(aligned_coords[:, 0])
    iy, wy = jax.vmap(
        lambda value: _basis_1d_polynomial_jax(value, table, r)
    )(aligned_coords[:, 1])
    indices = (ix[:, :, None] * int(r) + iy[:, None, :]).reshape(-1, 16)
    weights = (wx[:, :, None] * wy[:, None, :]).reshape(-1, 16)
    return indices, weights


def decode_one_polynomial_jax(
    state, coefficients, coords, boundary_mask, table, r=R
):
    affine_state = s.affine_from_normalized_jax(state[:5])
    aligned = s.aligned_coords_jax(coords, affine_state)
    indices, weights = local_rows_polynomial_jax(aligned, table, r)
    values = jnp.sum(coefficients[indices] * weights, axis=1)
    return values * boundary_mask


def decode_states_polynomial_sequential(
    states, coefficients, coords, boundary_mask, table, r=R
):
    """Shape-faithful sequential-time replacement for the Phase-2 decoder."""
    return jax.lax.map(
        lambda pair: decode_one_polynomial_jax(
            pair[0], pair[1], coords, boundary_mask, table, r
        ),
        (states, coefficients),
    )


def decode_states_polynomial_chunk3(
    states, coefficients, coords, boundary_mask, table, r=R
):
    """Fuse exactly three time slices while keeping bounded live work arrays."""
    if states.shape[0] % TIME_CHUNK:
        raise ValueError("locked chunk-3 route requires time count divisible by three")
    state_chunks = states.reshape(-1, TIME_CHUNK, states.shape[1])
    coefficient_chunks = coefficients.reshape(
        -1, TIME_CHUNK, coefficients.shape[1]
    )

    def one_chunk(pair):
        return jax.vmap(
            lambda state, coefficient: decode_one_polynomial_jax(
                state, coefficient, coords, boundary_mask, table, r
            )
        )(pair[0], pair[1])

    return jax.lax.map(one_chunk, (state_chunks, coefficient_chunks)).reshape(
        states.shape[0], coords.shape[0]
    )


def pallas_is_available():
    return pl is not None and pltriton is not None and jax.default_backend() == "gpu"


def make_pallas_polynomial_decoder(time_count, point_count, r=R):
    """One Triton grid over time and spatial blocks for the identical decoder."""
    if not pallas_is_available():
        raise RuntimeError("Pallas/Triton GPU backend is unavailable")
    time_count, point_count, r = int(time_count), int(point_count), int(r)
    if point_count % PALLAS_BLOCK:
        raise ValueError("locked Pallas route requires point count divisible by block")
    blocks = (point_count + PALLAS_BLOCK - 1) // PALLAS_BLOCK

    def kernel(states_ref, coefficients_ref, coords_ref, mask_ref, table_ref, out_ref):
        offsets = jnp.arange(PALLAS_BLOCK, dtype=jnp.int32)
        x = coords_ref[offsets, 0]
        y = coords_ref[offsets, 1]
        boundary = mask_ref[offsets]

        log_ratio = math.log(1.0 / 0.015)
        center_x = 0.5 + 0.75 * states_ref[0, 0]
        center_y = 0.5 + 0.75 * states_ref[0, 1]
        ell11 = math.log(0.015) + 0.5 * (states_ref[0, 2] + 1.0) * log_ratio
        ell22 = math.log(0.015) + 0.5 * (states_ref[0, 4] + 1.0) * log_ratio
        l11 = jnp.exp(ell11)
        l22 = jnp.exp(ell22)
        offdiag = 2.0 * states_ref[0, 3] * jnp.sqrt(l11 * l22)
        aligned_x = (x - center_x) / l11
        aligned_y = (y - center_y - offdiag * aligned_x) / l22
        spacing = s.DOMAIN_LENGTH / (r - s.DEGREE)

        def span_and_weights(value):
            in_domain = (
                (value >= -s.DOMAIN_HALF_WIDTH)
                & (value <= s.DOMAIN_HALF_WIDTH)
            )
            safe = jnp.clip(value, -s.DOMAIN_HALF_WIDTH, s.DOMAIN_HALF_WIDTH)
            cell = jnp.floor(
                (safe + s.DOMAIN_HALF_WIDTH) / spacing
            ).astype(jnp.int32)
            span = jnp.clip(cell + s.DEGREE, s.DEGREE, r - 1)
            # The table interval endpoints are uniform and exactly the knots
            # used to generate the polynomial coefficients.
            left = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE) * spacing
            span = jnp.where(safe < left, span - 1, span)
            right = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE + 1) * spacing
            span = jnp.where(safe >= right, span + 1, span)
            span = jnp.where(safe == s.DOMAIN_HALF_WIDTH, r - 1, span)
            span = jnp.clip(span, s.DEGREE, r - 1)
            left = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE) * spacing
            local = (safe - left) / spacing
            basis = []
            for local_basis in range(4):
                polynomial = [
                    table_ref[span, local_basis, power]
                    for power in range(4)
                ]
                weight = (
                    (polynomial[3] * local + polynomial[2]) * local
                    + polynomial[1]
                ) * local + polynomial[0]
                basis.append(weight * in_domain.astype(F64))
            return span, tuple(basis)

        span_x, weight_x = span_and_weights(aligned_x)
        span_y, weight_y = span_and_weights(aligned_y)
        value = jnp.zeros((PALLAS_BLOCK,), F64)
        for one in range(4):
            for two in range(4):
                control_index = (
                    (span_x - s.DEGREE + one) * r
                    + span_y - s.DEGREE + two
                )
                control = coefficients_ref[0, control_index]
                value += control * weight_x[one] * weight_y[two]
        out_ref[0, offsets] = value * boundary

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((time_count, point_count), F64),
        grid=(time_count, blocks),
        in_specs=(
            pl.BlockSpec(
                (1, PALLAS_STATE_BLOCK),
                lambda time_index, point_block: (time_index, 0),
            ),
            pl.BlockSpec(
                (1, r * r), lambda time_index, point_block: (time_index, 0)
            ),
            pl.BlockSpec(
                (PALLAS_BLOCK, 2),
                lambda time_index, point_block: (point_block, 0),
            ),
            pl.BlockSpec(
                (PALLAS_BLOCK,),
                lambda time_index, point_block: (point_block,),
            ),
            pl.BlockSpec(
                (r, 4, 4), lambda time_index, point_block: (0, 0, 0)
            ),
        ),
        out_specs=pl.BlockSpec(
            (1, PALLAS_BLOCK),
            lambda time_index, point_block: (time_index, point_block),
        ),
        compiler_params=pltriton.CompilerParams(num_warps=4),
        name="b10_phase3_cubic_decode",
    )

    def padded_call(states, coefficients, coords, boundary_mask, table):
        padded_states = jnp.pad(
            states, ((0, 0), (0, PALLAS_STATE_BLOCK - K))
        )
        return call(padded_states, coefficients, coords, boundary_mask, table)

    return padded_call


def make_pallas_basis_probe(point_count, r=R):
    """Return K3's actual one-dimensional support/weight calculation."""
    if not pallas_is_available():
        raise RuntimeError("Pallas/Triton GPU backend is unavailable")
    point_count, r = int(point_count), int(r)
    if point_count % PALLAS_BLOCK:
        raise ValueError("Pallas basis probe count must be block divisible")
    blocks = point_count // PALLAS_BLOCK

    def kernel(values_ref, table_ref, indices_ref, weights_ref):
        offsets = jnp.arange(PALLAS_BLOCK, dtype=jnp.int32)
        value = values_ref[offsets]
        in_domain = (
            (value >= -s.DOMAIN_HALF_WIDTH)
            & (value <= s.DOMAIN_HALF_WIDTH)
        )
        safe = jnp.clip(value, -s.DOMAIN_HALF_WIDTH, s.DOMAIN_HALF_WIDTH)
        spacing = s.DOMAIN_LENGTH / (r - s.DEGREE)
        cell = jnp.floor(
            (safe + s.DOMAIN_HALF_WIDTH) / spacing
        ).astype(jnp.int32)
        span = jnp.clip(cell + s.DEGREE, s.DEGREE, r - 1)
        left = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE) * spacing
        span = jnp.where(safe < left, span - 1, span)
        right = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE + 1) * spacing
        span = jnp.where(safe >= right, span + 1, span)
        span = jnp.where(safe == s.DOMAIN_HALF_WIDTH, r - 1, span)
        span = jnp.clip(span, s.DEGREE, r - 1)
        left = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE) * spacing
        local = (safe - left) / spacing
        for local_basis in range(4):
            polynomial = [
                table_ref[span, local_basis, power] for power in range(4)
            ]
            weight = (
                (polynomial[3] * local + polynomial[2]) * local
                + polynomial[1]
            ) * local + polynomial[0]
            indices_ref[offsets, local_basis] = span - s.DEGREE + local_basis
            weights_ref[offsets, local_basis] = weight * in_domain.astype(F64)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((point_count, 4), jnp.int32),
            jax.ShapeDtypeStruct((point_count, 4), F64),
        ),
        grid=(blocks,),
        in_specs=(
            pl.BlockSpec((PALLAS_BLOCK,), lambda point_block: (point_block,)),
            pl.BlockSpec((r, 4, 4), lambda point_block: (0, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec(
                (PALLAS_BLOCK, 4), lambda point_block: (point_block, 0)
            ),
            pl.BlockSpec(
                (PALLAS_BLOCK, 4), lambda point_block: (point_block, 0)
            ),
        ),
        compiler_params=pltriton.CompilerParams(num_warps=4),
        name="b10_phase3_cubic_basis_probe",
    )


def _ridge_components(field, coords, boundary_mask, r=R):
    affine, design, _, _ = s.sparse_design(field, coords, boundary_mask, r)
    diagonal = np.asarray(design.power(2).sum(axis=0)).reshape(-1)
    ridge = s.ORACLE_RIDGE * max(float(np.mean(diagonal)), 1e-300)
    rhs = np.asarray(design.T @ np.asarray(field, np.float64).reshape(-1), np.float64)
    return affine, design, ridge, rhs


def _grade_ridge_solution(field, design, ridge, rhs, coefficients, started, solver):
    field = np.asarray(field, np.float64).reshape(-1)
    coefficients = np.asarray(coefficients, np.float64)
    prediction = np.asarray(design @ coefficients, np.float64)
    normal = np.asarray(design.T @ (prediction - field), np.float64) + ridge * coefficients
    relative_normal = float(np.linalg.norm(normal) / max(np.linalg.norm(rhs), 1e-300))
    return coefficients, prediction, {
        "solver": solver,
        "ridge": float(ridge),
        "relative_normal_residual": relative_normal,
        "healthy": bool(
            np.all(np.isfinite(coefficients))
            and np.all(np.isfinite(prediction))
            and np.isfinite(relative_normal)
            and relative_normal <= s.ORACLE_NORMAL_TOL
        ),
        "elapsed_s": float(time.perf_counter() - started),
    }


def fit_augmented_column_lsmr(field, coords, boundary_mask, r=R):
    """Solve the identical ridge objective after deterministic column scaling."""
    started = time.perf_counter()
    affine, design, ridge, rhs = _ridge_components(
        field, coords, boundary_mask, r
    )
    column_norm = np.sqrt(
        np.asarray(design.power(2).sum(axis=0)).reshape(-1) + ridge
    )
    inverse = 1.0 / np.maximum(column_norm, np.finfo(np.float64).tiny)
    scaled = design @ sp.diags(inverse, format="csr")
    augmented = sp.vstack((
        scaled,
        math.sqrt(ridge) * sp.diags(inverse, format="csr"),
    ), format="csr")
    augmented_rhs = np.concatenate((
        np.asarray(field, np.float64).reshape(-1),
        np.zeros(design.shape[1], np.float64),
    ))
    result = spla.lsmr(
        augmented, augmented_rhs, atol=s.ORACLE_LSMR_TOL,
        btol=s.ORACLE_LSMR_TOL, conlim=s.ORACLE_LSMR_CONLIM,
        maxiter=SOLVER_MAXITER, show=False,
    )
    coefficients = inverse * np.asarray(result[0], np.float64)
    coefficients, prediction, info = _grade_ridge_solution(
        field, design, ridge, rhs, coefficients, started,
        "augmented_column_preconditioned_lsmr",
    )
    info.update({"iterations": int(result[2]), "istop": int(result[1])})
    return affine, coefficients, prediction, info


def fit_sparse_normal_lu(field, coords, boundary_mask, r=R):
    """Deterministic sparse LU of the identical ridge normal equations."""
    started = time.perf_counter()
    affine, design, ridge, rhs = _ridge_components(
        field, coords, boundary_mask, r
    )
    normal = (
        design.T @ design
        + ridge * sp.eye(design.shape[1], dtype=np.float64, format="csc")
    ).tocsc()
    factor = spla.splu(
        normal, permc_spec="COLAMD", diag_pivot_thresh=1.0,
        options={"Equil": False},
    )
    coefficients = factor.solve(rhs)
    coefficients, prediction, info = _grade_ridge_solution(
        field, design, ridge, rhs, coefficients, started,
        "sparse_normal_equation_superlu_colamd",
    )
    info.update({"nnz_normal": int(normal.nnz), "nnz_factors": int(factor.nnz)})
    return affine, coefficients, prediction, info
