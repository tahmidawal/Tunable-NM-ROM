"""Excluded local execution/identity feasibility for bounded Phase 3."""
from __future__ import annotations

import argparse
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase3 as p3
import b10_spline as s


def relative_l2(left, right):
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(np.linalg.norm(np.asarray(right)), 1e-300)
    )


def invoke_timed(function, *arguments):
    started = time.perf_counter()
    output = function(*arguments)
    jax.block_until_ready(output)
    return output, float(time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh", type=int, default=32)
    args = parser.parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    table_np = p3.span_polynomial_table_np()
    table = jnp.asarray(table_np, jnp.float64)

    knots = np.unique(s.knots_np(p3.R))
    rng = np.random.default_rng(20260820)
    probes = np.unique(np.concatenate((
        knots,
        np.nextafter(knots, -np.inf),
        np.nextafter(knots, np.inf),
        rng.uniform(-4.6, 4.6, size=4096),
    )))
    reference_indices, reference_weights = jax.vmap(
        lambda value: s._basis_1d_jax(value, p3.R)
    )(jnp.asarray(probes, jnp.float64))
    polynomial_indices, polynomial_weights = jax.vmap(
        lambda value: p3._basis_1d_polynomial_jax(value, table)
    )(jnp.asarray(probes, jnp.float64))
    reference_indices = np.asarray(reference_indices)
    reference_weights = np.asarray(reference_weights)
    polynomial_indices = np.asarray(polynomial_indices)
    polynomial_weights = np.asarray(polynomial_weights)
    if not np.array_equal(reference_indices, polynomial_indices):
        raise SystemExit("span-polynomial support indices differ from Cox-de Boor")
    weight_abs = float(np.max(np.abs(polynomial_weights - reference_weights)))
    weight_scaled_eps = float(np.max(
        np.abs(polynomial_weights - reference_weights)
        / (np.finfo(np.float64).eps * np.maximum(
            1.0, np.maximum(np.abs(polynomial_weights), np.abs(reference_weights))
        ))
    ))
    if weight_scaled_eps > 64.0:
        raise SystemExit(
            f"span-polynomial weight mismatch: {weight_scaled_eps} scaled eps"
        )

    n = int(args.mesh)
    coords_np = c.grid_coords(n)
    mask_np = c.binary_boundary_mask(n)
    states_np = rng.uniform(-0.7, 0.7, size=(3, p3.K)).astype(np.float64)
    coefficients_np = rng.normal(size=(3, p3.R * p3.R)).astype(np.float64)
    coords = jnp.asarray(coords_np, jnp.float64)
    mask = jnp.asarray(mask_np, jnp.float64)
    states = jnp.asarray(states_np, jnp.float64)
    coefficients = jnp.asarray(coefficients_np, jnp.float64)

    current = jax.jit(lambda st, cf, xy, mk: jax.lax.map(
        lambda pair: s.decode_one_jax(pair[0], pair[1], xy, mk, p3.R),
        (st, cf),
    ))
    sequential = jax.jit(p3.decode_states_polynomial_sequential)
    chunk3 = jax.jit(p3.decode_states_polynomial_chunk3)
    current_fields, current_first = invoke_timed(
        current, states, coefficients, coords, mask
    )
    sequential_fields, sequential_first = invoke_timed(
        sequential, states, coefficients, coords, mask, table
    )
    chunk_fields, chunk_first = invoke_timed(
        chunk3, states, coefficients, coords, mask, table
    )
    current_fields = np.asarray(current_fields)
    sequential_fields = np.asarray(sequential_fields)
    chunk_fields = np.asarray(chunk_fields)
    sequential_relative = relative_l2(sequential_fields, current_fields)
    chunk_relative = relative_l2(chunk_fields, current_fields)
    field_abs = float(np.max(np.abs(sequential_fields - current_fields)))
    if sequential_relative > 2e-14 or chunk_relative > 2e-14:
        raise SystemExit("polynomial field identity gate failed")
    boundary = mask_np == 0.0
    if not (
        np.all(current_fields[:, boundary] == 0.0)
        and np.all(sequential_fields[:, boundary] == 0.0)
        and np.all(chunk_fields[:, boundary] == 0.0)
    ):
        raise SystemExit("exact binary boundary failed")

    aligned = s.aligned_coords_jax(
        coords, s.affine_from_normalized_jax(states[0, :5])
    )
    support_indices, support_weights = p3.local_rows_polynomial_jax(
        aligned, table
    )
    support_indices = np.asarray(support_indices)
    support_weights = np.asarray(support_weights)
    if support_indices.shape != (n * n, 16) or support_weights.shape != (n * n, 16):
        raise SystemExit("local support is not exactly 16 entries")

    pallas_record = {
        "available": p3.pallas_is_available(), "executed": False,
    }
    if p3.pallas_is_available():
        pallas_decoder = jax.jit(p3.make_pallas_polynomial_decoder(3, n * n))
        pallas_fields, pallas_first = invoke_timed(
            pallas_decoder, states, coefficients, coords, mask, table
        )
        pallas_fields = np.asarray(pallas_fields)
        pallas_relative = relative_l2(pallas_fields, current_fields)
        pallas_abs = float(np.max(np.abs(pallas_fields - current_fields)))
        if pallas_relative > 2e-14 or not np.all(pallas_fields[:, boundary] == 0.0):
            raise SystemExit("Pallas polynomial field identity gate failed")
        pallas_record.update({
            "executed": True, "first_lower_compile_execute_s": pallas_first,
            "relative_l2_vs_cox": pallas_relative,
            "max_abs_vs_cox": pallas_abs,
        })

    # The solver smoke is synthetic and excluded.  It checks exact objective
    # preservation and numerical health without touching any locked draw.
    field = np.asarray(c.bf.blob_ic(n, 0.42, 0.58, 0.12, 1.20), np.float64)
    solver_rows = {}
    solutions = {}
    for name, solver in (
        ("augmented_column_lsmr", p3.fit_augmented_column_lsmr),
        ("sparse_normal_lu", p3.fit_sparse_normal_lu),
    ):
        affine, one_coefficients, prediction, info = solver(
            field, coords_np, mask_np
        )
        if not info["healthy"] or not np.all(np.isfinite(affine)):
            raise SystemExit(f"{name} failed the synthetic health gate: {info}")
        solver_rows[name] = info
        solutions[name] = {
            "coefficients": one_coefficients, "prediction": prediction,
        }
    solver_prediction_difference = relative_l2(
        solutions["augmented_column_lsmr"]["prediction"],
        solutions["sparse_normal_lu"]["prediction"],
    )
    if solver_prediction_difference > 2e-8:
        raise SystemExit("solver routes do not preserve the same ridge minimizer")

    report = {
        "status": "excluded_execution_feasibility_pass",
        "scientific_result": False,
        "provenance": c.provenance(),
        "mesh": n,
        "identity": {
            "probe_count": int(probes.size),
            "includes_knots_nextafter_random": True,
            "support_indices_exact": True,
            "local_support": 16,
            "weight_max_abs": weight_abs,
            "weight_max_scaled_eps": weight_scaled_eps,
            "sequential_field_relative_l2_vs_cox": sequential_relative,
            "chunk3_field_relative_l2_vs_cox": chunk_relative,
            "sequential_field_max_abs_vs_cox": field_abs,
            "exact_binary_boundary": True,
        },
        "first_lower_compile_execute_s": {
            "cox_sequential": current_first,
            "polynomial_sequential": sequential_first,
            "polynomial_chunk3": chunk_first,
        },
        "pallas": pallas_record,
        "solver": solver_rows,
        "solver_prediction_relative_l2": solver_prediction_difference,
        "elapsed_s": float(time.perf_counter() - started),
        "model_validation_touched": False,
        "confirmation_touched": False,
    }
    c.save_json(args.output, report)
    c.log(report)


if __name__ == "__main__":
    main()
