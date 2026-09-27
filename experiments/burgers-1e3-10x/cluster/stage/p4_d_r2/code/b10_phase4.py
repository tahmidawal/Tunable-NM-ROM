"""Exact hierarchical local-support spline machinery for Burgers Phase 4."""
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
import b10_phase3 as p3
import b10_s0_spline as base
import b10_spline as s


F64 = jnp.float64
GLOBAL_R = 48
FINE_HALF_WIDTH = 2.0
FINE_CORE_HALF_WIDTH = 1.5
FINE_COORDINATE_SCALE = s.DOMAIN_HALF_WIDTH / FINE_HALF_WIDTH
PALLAS_BLOCK = 128
PALLAS_STATE_BLOCK = 32
IDENTITY_TOL = 2e-14

CANDIDATES = (
    {
        "arm": "H1", "R": GLOBAL_R, "P": 32, "k": 24,
        "M": 96, "m": 384, "q": 19,
        "hyperdecoder_parameters": 111_520,
        "predictor_parameters": 2_104,
    },
    {
        "arm": "H2", "R": GLOBAL_R, "P": 48, "k": 24,
        "M": 96, "m": 384, "q": 19,
        "hyperdecoder_parameters": 153_760,
        "predictor_parameters": 2_104,
    },
)


def coefficient_count(candidate):
    return int(candidate["R"]) ** 2 + int(candidate["P"]) ** 2


def taper_1d_np(value):
    absolute = np.abs(np.asarray(value, np.float64))
    ratio = np.clip(
        (FINE_HALF_WIDTH - absolute)
        / (FINE_HALF_WIDTH - FINE_CORE_HALF_WIDTH),
        0.0, 1.0,
    )
    transition = 6.0 * ratio**5 - 15.0 * ratio**4 + 10.0 * ratio**3
    return np.where(
        absolute <= FINE_CORE_HALF_WIDTH, 1.0,
        np.where(absolute >= FINE_HALF_WIDTH, 0.0, transition),
    )


def taper_1d_jax(value):
    absolute = jnp.abs(value)
    ratio = jnp.clip(
        (FINE_HALF_WIDTH - absolute)
        / (FINE_HALF_WIDTH - FINE_CORE_HALF_WIDTH),
        0.0, 1.0,
    )
    transition = 6.0 * ratio**5 - 15.0 * ratio**4 + 10.0 * ratio**3
    return jnp.where(
        absolute <= FINE_CORE_HALF_WIDTH, 1.0,
        jnp.where(absolute >= FINE_HALF_WIDTH, 0.0, transition),
    )


def tensor_window_np(aligned):
    aligned = np.asarray(aligned, np.float64)
    return taper_1d_np(aligned[:, 0]) * taper_1d_np(aligned[:, 1])


def tensor_window_jax(aligned):
    return taper_1d_jax(aligned[:, 0]) * taper_1d_jax(aligned[:, 1])


def hierarchical_sparse_design(field, coords, boundary_mask, candidate):
    """Return the locked joint sparse design with the binary mask applied last."""
    field = np.asarray(field, np.float64).reshape(-1)
    coords = np.asarray(coords, np.float64)
    mask = np.asarray(boundary_mask, np.float64).reshape(-1)
    r, fine = int(candidate["R"]), int(candidate["P"])
    affine = s.affine_state_from_field(field, coords)
    aligned = s.aligned_coords_np(coords, affine)
    coarse_indices, coarse_weights = s.local_rows_np(aligned, r)
    fine_indices, fine_weights = s.local_rows_np(
        aligned * FINE_COORDINATE_SCALE, fine
    )
    window = tensor_window_np(aligned)
    coarse_weights = coarse_weights * mask[:, None]
    fine_weights = fine_weights * window[:, None] * mask[:, None]
    rows = np.repeat(np.arange(field.size, dtype=np.int64), 16)
    coarse_design = sp.coo_matrix(
        (coarse_weights.reshape(-1), (rows, coarse_indices.reshape(-1))),
        shape=(field.size, r * r), dtype=np.float64,
    ).tocsr()
    fine_design = sp.coo_matrix(
        (fine_weights.reshape(-1), (rows, fine_indices.reshape(-1))),
        shape=(field.size, fine * fine), dtype=np.float64,
    ).tocsr()
    design = sp.hstack((coarse_design, fine_design), format="csr")
    design.sum_duplicates()
    design.eliminate_zeros()
    local_support = np.sum(coarse_weights != 0.0, axis=1)
    local_support += np.sum(fine_weights != 0.0, axis=1)
    return affine, design, {
        "coarse_indices": coarse_indices,
        "coarse_weights": coarse_weights,
        "fine_indices": fine_indices,
        "fine_weights": fine_weights,
        "window": window,
        "pou_max_abs": float(np.max(np.abs((1.0 - window) + window - 1.0))),
        "max_local_support": int(np.max(local_support)),
        "local_support_all": local_support,
    }


def fit_hierarchical_oracle(field, coords, boundary_mask, candidate):
    """Solve the exact joint ridge projection with locked sparse SuperLU."""
    started = time.perf_counter()
    field = np.asarray(field, np.float64).reshape(-1)
    mask = np.asarray(boundary_mask, np.float64).reshape(-1)
    affine, design, basis = hierarchical_sparse_design(
        field, coords, mask, candidate
    )
    diagonal = np.asarray(design.power(2).sum(axis=0)).reshape(-1)
    ridge = s.ORACLE_RIDGE * max(float(np.mean(diagonal)), 1e-300)
    rhs = np.asarray(design.T @ field, np.float64)
    normal = (
        design.T @ design
        + ridge * sp.eye(design.shape[1], dtype=np.float64, format="csc")
    ).tocsc()
    factor = spla.splu(
        normal, permc_spec="COLAMD", diag_pivot_thresh=1.0,
        options={"Equil": False},
    )
    coefficients = np.asarray(factor.solve(rhs), np.float64)
    prediction = np.asarray(design @ coefficients, np.float64)
    residual = np.asarray(design.T @ (prediction - field), np.float64)
    residual += ridge * coefficients
    relative_normal = float(
        np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)
    )
    boundary_exact = bool(np.all(prediction[mask == 0.0] == 0.0))
    finite = bool(
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(prediction))
        and np.isfinite(relative_normal)
    )
    info = {
        "solver": "joint_sparse_normal_equation_superlu_colamd",
        "ridge": float(ridge),
        "relative_normal_residual": relative_normal,
        "finite": finite,
        "boundary_exact": boundary_exact,
        "pou_max_abs": basis["pou_max_abs"],
        "max_local_support": basis["max_local_support"],
        "nnz_design": int(design.nnz),
        "nnz_normal": int(normal.nnz),
        "nnz_factors": int(factor.nnz),
        "healthy": bool(
            finite and boundary_exact
            and basis["pou_max_abs"] <= 1e-15
            and basis["max_local_support"] == 32
            and relative_normal <= s.ORACLE_NORMAL_TOL
        ),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return affine, coefficients, prediction, info


def decode_one_polynomial_jax(
    state, coefficients, coords, boundary_mask, candidate,
    coarse_table, fine_table,
):
    affine = s.affine_from_normalized_jax(state[:5])
    aligned = s.aligned_coords_jax(coords, affine)
    coarse_indices, coarse_weights = p3.local_rows_polynomial_jax(
        aligned, coarse_table, int(candidate["R"])
    )
    fine_indices, fine_weights = p3.local_rows_polynomial_jax(
        aligned * FINE_COORDINATE_SCALE, fine_table, int(candidate["P"])
    )
    r2 = int(candidate["R"]) ** 2
    coarse = jnp.sum(coefficients[coarse_indices] * coarse_weights, axis=1)
    fine = jnp.sum(
        coefficients[r2 + fine_indices] * fine_weights, axis=1
    )
    return (coarse + tensor_window_jax(aligned) * fine) * boundary_mask


def decode_one_cox_jax(state, coefficients, coords, boundary_mask, candidate):
    """Independent clamped Cox--de Boor control for the polynomial/K3 route."""
    affine = s.affine_from_normalized_jax(state[:5])
    aligned = s.aligned_coords_jax(coords, affine)
    coarse_indices, coarse_weights = s.local_rows_jax(
        aligned, int(candidate["R"])
    )
    fine_indices, fine_weights = s.local_rows_jax(
        aligned * FINE_COORDINATE_SCALE, int(candidate["P"])
    )
    r2 = int(candidate["R"]) ** 2
    coarse = jnp.sum(coefficients[coarse_indices] * coarse_weights, axis=1)
    fine = jnp.sum(
        coefficients[r2 + fine_indices] * fine_weights, axis=1
    )
    return (coarse + tensor_window_jax(aligned) * fine) * boundary_mask


def decode_states_polynomial_sequential(
    states, coefficients, coords, boundary_mask, candidate,
    coarse_table, fine_table,
):
    return jax.lax.map(
        lambda pair: decode_one_polynomial_jax(
            pair[0], pair[1], coords, boundary_mask, candidate,
            coarse_table, fine_table,
        ),
        (states, coefficients),
    )


def decode_states_cox_sequential(
    states, coefficients, coords, boundary_mask, candidate,
):
    return jax.lax.map(
        lambda pair: decode_one_cox_jax(
            pair[0], pair[1], coords, boundary_mask, candidate,
        ),
        (states, coefficients),
    )


def init_cost_parameters(candidate, seed):
    first, second = jax.random.split(jax.random.PRNGKey(int(seed)))
    predictor = s.init_mlp(first, (7, s.HYPER_WIDTH, s.HYPER_WIDTH, 24))
    hyper = s.init_mlp(
        second,
        (19, s.HYPER_WIDTH, s.HYPER_WIDTH, coefficient_count(candidate)),
    )
    return predictor, hyper


def _pallas_span_weights(value, count, table_ref):
    """Shared exact K3 span/weight implementation for decoder and probe."""
    in_domain = (
        (value >= -s.DOMAIN_HALF_WIDTH)
        & (value <= s.DOMAIN_HALF_WIDTH)
    )
    safe = jnp.clip(value, -s.DOMAIN_HALF_WIDTH, s.DOMAIN_HALF_WIDTH)
    spacing = s.DOMAIN_LENGTH / (count - s.DEGREE)
    cell = jnp.floor(
        (safe + s.DOMAIN_HALF_WIDTH) / spacing
    ).astype(jnp.int32)
    span = jnp.clip(cell + s.DEGREE, s.DEGREE, count - 1)
    left = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE) * spacing
    span = jnp.where(safe < left, span - 1, span)
    right = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE + 1) * spacing
    span = jnp.where(safe >= right, span + 1, span)
    span = jnp.where(safe == s.DOMAIN_HALF_WIDTH, count - 1, span)
    span = jnp.clip(span, s.DEGREE, count - 1)
    left = -s.DOMAIN_HALF_WIDTH + (span - s.DEGREE) * spacing
    local = (safe - left) / spacing
    basis = []
    for local_basis in range(4):
        polynomial = [
            table_ref[span, local_basis, power] for power in range(4)
        ]
        weight = (
            (polynomial[3] * local + polynomial[2]) * local
            + polynomial[1]
        ) * local + polynomial[0]
        basis.append(weight * in_domain.astype(F64))
    return span, tuple(basis)


def make_pallas_hierarchical_basis_probe(point_count, count):
    """Probe the exact helper invoked by the hierarchical Pallas decoder."""
    if not p3.pallas_is_available():
        raise RuntimeError("Pallas/Triton GPU backend is unavailable")
    pl, pltriton = p3.pl, p3.pltriton
    point_count, count = int(point_count), int(count)
    if point_count % PALLAS_BLOCK:
        raise ValueError("Pallas basis probe count must be block divisible")

    def kernel(values_ref, table_ref, indices_ref, weights_ref):
        offsets = jnp.arange(PALLAS_BLOCK, dtype=jnp.int32)
        span, weights = _pallas_span_weights(
            values_ref[offsets], count, table_ref
        )
        for local_basis in range(4):
            indices_ref[offsets, local_basis] = (
                span - s.DEGREE + local_basis
            )
            weights_ref[offsets, local_basis] = weights[local_basis]

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((point_count, 4), jnp.int32),
            jax.ShapeDtypeStruct((point_count, 4), F64),
        ),
        grid=(point_count // PALLAS_BLOCK,),
        in_specs=(
            pl.BlockSpec((PALLAS_BLOCK,), lambda block: (block,)),
            pl.BlockSpec((count, 4, 4), lambda block: (0, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec((PALLAS_BLOCK, 4), lambda block: (block, 0)),
            pl.BlockSpec((PALLAS_BLOCK, 4), lambda block: (block, 0)),
        ),
        compiler_params=pltriton.CompilerParams(num_warps=4),
        name=f"b10_phase4_hierarchical_basis_p{count}",
    )


def make_pallas_hierarchical_decoder(time_count, point_count, candidate):
    """Fused f64 global-plus-windowed-fine decoder with bounded support."""
    if not p3.pallas_is_available():
        raise RuntimeError("Pallas/Triton GPU backend is unavailable")
    pl, pltriton = p3.pl, p3.pltriton
    time_count, point_count = int(time_count), int(point_count)
    r, fine = int(candidate["R"]), int(candidate["P"])
    total = coefficient_count(candidate)
    if point_count % PALLAS_BLOCK:
        raise ValueError("hierarchical Pallas point count must be block divisible")
    blocks = point_count // PALLAS_BLOCK

    def kernel(
        states_ref, coefficients_ref, coords_ref, mask_ref,
        coarse_table_ref, fine_table_ref, out_ref,
    ):
        offsets = jnp.arange(PALLAS_BLOCK, dtype=jnp.int32)
        x, y = coords_ref[offsets, 0], coords_ref[offsets, 1]
        boundary = mask_ref[offsets]
        log_ratio = math.log(1.0 / 0.015)
        center_x = 0.5 + 0.75 * states_ref[0, 0]
        center_y = 0.5 + 0.75 * states_ref[0, 1]
        ell11 = math.log(0.015) + 0.5 * (states_ref[0, 2] + 1.0) * log_ratio
        ell22 = math.log(0.015) + 0.5 * (states_ref[0, 4] + 1.0) * log_ratio
        l11, l22 = jnp.exp(ell11), jnp.exp(ell22)
        offdiag = 2.0 * states_ref[0, 3] * jnp.sqrt(l11 * l22)
        aligned_x = (x - center_x) / l11
        aligned_y = (y - center_y - offdiag * aligned_x) / l22

        coarse_x, coarse_wx = _pallas_span_weights(
            aligned_x, r, coarse_table_ref
        )
        coarse_y, coarse_wy = _pallas_span_weights(
            aligned_y, r, coarse_table_ref
        )
        scaled_x = aligned_x * FINE_COORDINATE_SCALE
        scaled_y = aligned_y * FINE_COORDINATE_SCALE
        fine_x, fine_wx = _pallas_span_weights(
            scaled_x, fine, fine_table_ref
        )
        fine_y, fine_wy = _pallas_span_weights(
            scaled_y, fine, fine_table_ref
        )

        value = jnp.zeros((PALLAS_BLOCK,), F64)
        fine_value = jnp.zeros((PALLAS_BLOCK,), F64)
        for one in range(4):
            for two in range(4):
                coarse_index = (
                    (coarse_x - s.DEGREE + one) * r
                    + coarse_y - s.DEGREE + two
                )
                value += (
                    coefficients_ref[0, coarse_index]
                    * coarse_wx[one] * coarse_wy[two]
                )
                fine_index = (
                    (fine_x - s.DEGREE + one) * fine
                    + fine_y - s.DEGREE + two
                )
                fine_value += (
                    coefficients_ref[0, r * r + fine_index]
                    * fine_wx[one] * fine_wy[two]
                )
        window = taper_1d_jax(aligned_x) * taper_1d_jax(aligned_y)
        out_ref[0, offsets] = (value + window * fine_value) * boundary

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
                (1, total), lambda time_index, point_block: (time_index, 0)
            ),
            pl.BlockSpec(
                (PALLAS_BLOCK, 2),
                lambda time_index, point_block: (point_block, 0),
            ),
            pl.BlockSpec(
                (PALLAS_BLOCK,),
                lambda time_index, point_block: (point_block,),
            ),
            pl.BlockSpec((r, 4, 4), lambda time_index, point_block: (0, 0, 0)),
            pl.BlockSpec(
                (fine, 4, 4), lambda time_index, point_block: (0, 0, 0)
            ),
        ),
        out_specs=pl.BlockSpec(
            (1, PALLAS_BLOCK),
            lambda time_index, point_block: (time_index, point_block),
        ),
        compiler_params=pltriton.CompilerParams(num_warps=4),
        name=f"b10_phase4_hierarchical_p{fine}",
    )

    def padded_call(
        states, coefficients, coords, boundary_mask, coarse_table, fine_table
    ):
        padded_states = jnp.pad(
            states, ((0, 0), (0, PALLAS_STATE_BLOCK - 24))
        )
        return call(
            padded_states, coefficients, coords, boundary_mask,
            coarse_table, fine_table,
        )

    return padded_call


def make_online_kernels(n, candidate, predictor, hyper, num_steps):
    """Build locked mandatory/max-one K3 routes plus an independent Cox probe."""
    geometry = base.weak_geometry(n, candidate)
    stencil_coords = jnp.asarray(geometry["stencil_coords"], F64)
    stencil_mask = jnp.asarray(geometry["stencil_mask"], F64)
    phi_weighted = jnp.asarray(geometry["phi_weighted"], F64)
    eigenvalues = jnp.asarray(geometry["eigenvalues"], F64)
    coarse_table = jnp.asarray(p3.span_polynomial_table_np(GLOBAL_R), F64)
    fine_table = jnp.asarray(
        p3.span_polynomial_table_np(int(candidate["P"])), F64
    )
    m = int(candidate["m"])
    full_pallas = make_pallas_hierarchical_decoder(
        num_steps + 1, n * n, candidate
    )

    def coefficients(states):
        return jax.vmap(lambda state: s.apply_mlp(hyper, state[5:]))(states)

    def decode_one(state, coefficient, query_coords, query_mask, cox=False):
        if cox:
            return decode_one_cox_jax(
                state, coefficient, query_coords, query_mask, candidate
            )
        return decode_one_polynomial_jax(
            state, coefficient, query_coords, query_mask, candidate,
            coarse_table, fine_table,
        )

    def weak_residual(
        state, state_coefficients, previous_state, previous_coefficients,
        viscosity, cox=False,
    ):
        current = decode_one(
            state, state_coefficients,
            stencil_coords, stencil_mask, cox,
        ).reshape(m, 5)
        previous = decode_one(
            previous_state, previous_coefficients,
            stencil_coords[::5], stencil_mask[::5], cox,
        )
        center, xp, xm, yp, ym = [current[:, index] for index in range(5)]
        dx = 1.0 / (n - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected_u = phi_weighted.T @ center
        mode_preconditioner = (1.0 + c.DT * viscosity * eigenvalues) ** -1.0
        residual = mode_preconditioner * (
            phi_weighted.T @ (center - previous)
            + c.DT * (
                phi_weighted.T @ advection
                + viscosity * eigenvalues * projected_u
            )
        )
        denominator = jnp.maximum(
            jnp.linalg.norm(phi_weighted.T @ previous), 1e-12
        )
        return residual, jnp.linalg.norm(residual) / denominator

    def predict_states(features):
        return jnp.tanh(s.apply_mlp(predictor, features[:num_steps + 1]))

    def full_decode(states, coefficient_values, coords, mask, cox=False):
        if cox:
            return decode_states_cox_sequential(
                states, coefficient_values, coords, mask, candidate
            )
        return full_pallas(
            states, coefficient_values, coords, mask, coarse_table, fine_table
        )

    def mandatory(features, coords, mask, viscosity):
        states = predict_states(features)
        coefficient_values = coefficients(states)
        residual, rho = jax.vmap(
            lambda state, state_coeff, previous, previous_coeff: weak_residual(
                state, state_coeff, previous, previous_coeff, viscosity, False
            )
        )(
            states[1:], coefficient_values[1:],
            states[:-1], coefficient_values[:-1],
        )
        fields = full_decode(states, coefficient_values, coords, mask, False)
        return fields, states, coefficient_values, residual, rho

    def maximum_one(features, coords, mask, viscosity):
        predicted = predict_states(features)

        def step(previous, target):
            previous_coefficients = s.apply_mlp(hyper, previous[5:])
            target_coefficients = s.apply_mlp(hyper, target[5:])
            residual, rho_before = weak_residual(
                target, target_coefficients, previous,
                previous_coefficients, viscosity, False
            )
            jacobian = jax.jacfwd(
                lambda state: weak_residual(
                    state, s.apply_mlp(hyper, state[5:]), previous,
                    previous_coefficients, viscosity, False
                )[0]
            )(target)
            hessian = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            diagonal = jnp.diag(jnp.diag(hessian)) + 1e-12 * jnp.eye(
                target.size, dtype=F64
            )
            raw_step = jnp.linalg.solve(
                hessian + 1e-6 * diagonal, -gradient
            )
            raw_step = jnp.where(
                jnp.all(jnp.isfinite(raw_step)), raw_step,
                jnp.zeros_like(raw_step),
            )
            scale = jnp.minimum(
                1.0, 0.25 / jnp.maximum(jnp.linalg.norm(raw_step), 1e-300)
            )
            bounded = raw_step * scale
            factors = jnp.asarray((1.0, 0.5, 0.25, 0.0), F64)
            trials = jnp.clip(
                target[None] + factors[:, None] * bounded[None],
                -1.0 + 1e-8, 1.0 - 1e-8,
            )
            trial_rho = jax.vmap(
                lambda state: weak_residual(
                    state, s.apply_mlp(hyper, state[5:]), previous,
                    previous_coefficients, viscosity, False
                )[1]
            )(trials)
            trial_rho = jnp.where(jnp.isfinite(trial_rho), trial_rho, jnp.inf)
            best = jnp.argmin(trial_rho)
            accepted = trial_rho[best] < rho_before
            corrected = jnp.where(accepted, trials[best], target)
            return corrected, (
                corrected, residual, rho_before, trial_rho[best], jacobian,
                factors[best], jnp.linalg.norm(bounded), accepted,
            )

        _, outputs = jax.lax.scan(step, predicted[0], predicted[1:])
        corrected = jnp.concatenate((predicted[:1], outputs[0]), axis=0)
        coefficient_values = coefficients(corrected)
        fields = full_decode(
            corrected, coefficient_values, coords, mask, False
        )
        return (fields, corrected, coefficient_values) + outputs[1:]

    def identity_probe(features, coords, mask, viscosity, cox):
        states = predict_states(features)
        coefficient_values = coefficients(states)
        residual, rho = jax.vmap(
            lambda state, state_coeff, previous, previous_coeff: weak_residual(
                state, state_coeff, previous, previous_coeff, viscosity, cox
            )
        )(
            states[1:], coefficient_values[1:],
            states[:-1], coefficient_values[:-1],
        )
        current = jax.lax.map(
            lambda pair: decode_one(
                pair[0], pair[1], stencil_coords, stencil_mask, cox
            ),
            (states[1:], coefficient_values[1:]),
        )
        previous = jax.lax.map(
            lambda pair: decode_one(
                pair[0], pair[1], stencil_coords[::5],
                stencil_mask[::5], cox,
            ),
            (states[:-1], coefficient_values[:-1]),
        )
        fields = full_decode(states, coefficient_values, coords, mask, cox)
        return fields, current, previous, residual, rho

    return {
        "mandatory": jax.jit(mandatory),
        "maximum_one": jax.jit(maximum_one),
        "identity_k3": jax.jit(
            lambda features, coords, mask, viscosity: identity_probe(
                features, coords, mask, viscosity, False
            )
        ),
        "identity_cox": jax.jit(
            lambda features, coords, mask, viscosity: identity_probe(
                features, coords, mask, viscosity, True
            )
        ),
        "geometry": geometry,
    }


def compile_online_kernels(n, candidate, features, viscosity, num_steps, seed):
    predictor, hyper = init_cost_parameters(candidate, seed)
    kernels = make_online_kernels(
        n, candidate, predictor, hyper, num_steps
    )
    arguments = (
        jnp.asarray(features, F64),
        jnp.asarray(c.grid_coords(n), F64),
        jnp.asarray(c.binary_boundary_mask(n), F64),
        jnp.asarray(viscosity, F64),
    )
    compiled, setup = {}, {}
    for name in ("mandatory", "maximum_one", "identity_k3", "identity_cox"):
        lower_started = time.perf_counter()
        lowered = kernels[name].lower(*arguments)
        lower_s = time.perf_counter() - lower_started
        compile_started = time.perf_counter()
        executable = lowered.compile()
        compile_s = time.perf_counter() - compile_started
        compiled[name] = executable
        setup[name] = {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "lower_plus_compile_s": float(lower_s + compile_s),
            "memory_analysis": base.memory_analysis(executable),
        }
    return compiled, setup, kernels["geometry"], arguments
