"""Phase-3 solver/kernel diagnostic; selection data only, never model validation."""
from __future__ import annotations

import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase3 as p3
import b10_s0_spline as base
import b10_spline as s


SOLVER_SUBSETS = {
    64: (512, 533, 554, 575),
    128: (512, 523, 530, 543),
    256: (512, 517, 522, 527),
}
SOLVER_TIMES = (0, 1, 10, 20, 30, 40, 50)
TARGET_N128_INDEX = 530
SOLVERS = (
    ("S1", p3.fit_augmented_column_lsmr),
    ("S2", p3.fit_sparse_normal_lu),
)
KERNELS = ("K0", "K1", "K2", "K3")
IDENTITY_TOL = 2e-14
NO_REGRESSION_ABSOLUTE = 1e-5
FOM_CASES = 4
TIME_REPS = 20
TIME_WARM = 1
BURN_SECONDS = 3.0
ARM_C = {"arm": "C", "R": 48, "k": 24, "M": 96, "m": 384}


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def validate_s0_chain(json_path, npz_path, audit_path):
    report, audit = load_json(json_path), load_json(audit_path)
    if report.get("status") != "complete" or audit.get("status") != "pass":
        raise SystemExit("S0 scientific/audit status is not complete/pass")
    if report.get("npz", {}).get("sha256") != c.sha256(npz_path):
        raise SystemExit("S0 NPZ internal checksum mismatch")
    if audit.get("source_json_sha256") != c.sha256(json_path):
        raise SystemExit("S0 independent audit JSON binding mismatch")
    if audit.get("source_npz_sha256") != c.sha256(npz_path):
        raise SystemExit("S0 independent audit NPZ binding mismatch")
    if not audit.get("decision", {}).get("phase2_hard_stop"):
        raise SystemExit("Phase 3 requires the audited Phase-2 hard stop")
    return report, audit


def relative_field_error(prediction, truth):
    return float(
        np.linalg.norm(np.asarray(prediction) - np.asarray(truth))
        / max(np.linalg.norm(np.asarray(truth)), 1e-300)
    )


def basis_identity_record():
    table = jnp.asarray(p3.span_polynomial_table_np(), jnp.float64)
    knots = np.unique(s.knots_np(p3.R))
    rng = np.random.default_rng(20260820)
    probes = np.unique(np.concatenate((
        knots, np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf),
        rng.uniform(-4.6, 4.6, size=4096),
    )))
    reference_indices, reference_weights = jax.vmap(
        lambda value: s._basis_1d_jax(value, p3.R)
    )(jnp.asarray(probes, jnp.float64))
    polynomial_indices, polynomial_weights = jax.vmap(
        lambda value: p3._basis_1d_polynomial_jax(value, table)
    )(jnp.asarray(probes, jnp.float64))
    reference_indices, reference_weights = map(
        np.asarray, (reference_indices, reference_weights)
    )
    polynomial_indices, polynomial_weights = map(
        np.asarray, (polynomial_indices, polynomial_weights)
    )
    difference = np.abs(polynomial_weights - reference_weights)
    max_abs = float(np.max(difference))
    max_scaled_eps = float(np.max(
        difference / (np.finfo(np.float64).eps * np.maximum(
            1.0, np.maximum(np.abs(polynomial_weights), np.abs(reference_weights))
        ))
    ))
    indices_exact = bool(np.array_equal(reference_indices, polynomial_indices))
    return {
        "seed": 20260820, "probe_count": int(probes.size),
        "unique_knots_nextafter_random": True,
        "support_indices_exact": indices_exact, "local_support": 16,
        "weight_max_abs": max_abs, "weight_max_scaled_eps": max_scaled_eps,
        "pass": bool(indices_exact and max_abs <= IDENTITY_TOL),
    }


def pallas_basis_identity_record():
    if not p3.pallas_is_available():
        return {"available": False, "executed": False, "pass": False}
    knots = np.unique(s.knots_np(p3.R))
    rng = np.random.default_rng(20260820)
    probes = np.unique(np.concatenate((
        knots, np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf),
        rng.uniform(-4.6, 4.6, size=4096),
    )))
    reference_indices, reference_weights = jax.vmap(
        lambda value: s._basis_1d_jax(value, p3.R)
    )(jnp.asarray(probes, jnp.float64))
    padded_count = (
        (probes.size + p3.PALLAS_BLOCK - 1) // p3.PALLAS_BLOCK
        * p3.PALLAS_BLOCK
    )
    padded = np.pad(probes, (0, padded_count - probes.size), constant_values=0.0)
    probe = jax.jit(p3.make_pallas_basis_probe(padded_count))
    indices, weights = probe(
        jnp.asarray(padded, jnp.float64),
        jnp.asarray(p3.span_polynomial_table_np(), jnp.float64),
    )
    jax.block_until_ready((indices, weights))
    indices = np.asarray(indices)[:probes.size]
    weights = np.asarray(weights)[:probes.size]
    reference_indices = np.asarray(reference_indices)
    reference_weights = np.asarray(reference_weights)
    difference = np.abs(weights - reference_weights)
    indices_exact = bool(np.array_equal(indices, reference_indices))
    max_abs = float(np.max(difference))
    return {
        "available": True, "executed": True, "seed": 20260820,
        "probe_count": int(probes.size), "padded_probe_count": int(padded_count),
        "unique_knots_nextafter_random": True,
        "support_indices_exact": indices_exact, "local_support": 4,
        "weight_max_abs": max_abs,
        "pass": bool(indices_exact and max_abs <= IDENTITY_TOL),
    }


def run_solver_diagnostic(s0_npz_path, smoke=False):
    records = {name: [] for name, _ in SOLVERS}
    arrays = {}
    if smoke:
        n = 32
        coords = c.grid_coords(n)
        mask = c.binary_boundary_mask(n)
        field = np.asarray(c.bf.blob_ic(n, 0.42, 0.58, 0.12, 1.2), np.float64)
        tasks = [(n, 0, 0, field, np.nan)]
    else:
        tasks = []
        with np.load(s0_npz_path, allow_pickle=False) as old:
            for n, indices in SOLVER_SUBSETS.items():
                fields, _, health = c.generate_population(
                    n, 0, 704, np.asarray(indices), chunk=max(1, len(indices))
                )
                if (
                    health["reported_max_relative_residual"] > 1e-8
                    or health["independent_max_relative_residual"] > 1e-8
                ):
                    raise SystemExit(f"N={n} diagnostic truth health failed")
                old_snapshot = np.asarray(old[f"C_N{n}_snapshot_error"], np.float64)
                for local_index, draw_index in enumerate(indices):
                    times = set(SOLVER_TIMES)
                    if n == 128 and draw_index == TARGET_N128_INDEX:
                        times.update(range(c.NUM_STEPS + 1))
                    for time_index in sorted(times):
                        old_error = old_snapshot[draw_index - 512, time_index]
                        tasks.append((
                            n, draw_index, time_index,
                            fields[local_index, time_index], float(old_error),
                        ))
    for solver_name, solver in SOLVERS:
        coefficients = []
        predictions = []
        for n, draw_index, time_index, field, old_error in tasks:
            coords, mask = c.grid_coords(n), c.binary_boundary_mask(n)
            affine, one_coefficients, prediction, info = solver(
                field, coords, mask
            )
            difference = np.asarray(prediction) - np.asarray(field)
            error_numerator_sq = float(np.vdot(difference, difference).real)
            truth_norm_sq = float(np.vdot(field, field).real)
            error = float(np.sqrt(error_numerator_sq / max(truth_norm_sq, 1e-300)))
            boundary_exact = bool(np.all(prediction[mask == 0.0] == 0.0))
            no_regression = bool(
                smoke or error <= old_error + NO_REGRESSION_ABSOLUTE
            )
            row = {
                "N": int(n), "draw_index": int(draw_index),
                "time_index": int(time_index), "field_relative_l2": error,
                "error_numerator_sq": error_numerator_sq,
                "truth_norm_sq": truth_norm_sq,
                "s0_field_relative_l2": None if smoke else float(old_error),
                "no_regression": no_regression,
                "boundary_exact": boundary_exact, **info,
            }
            records[solver_name].append(row)
            coefficients.append(one_coefficients)
            predictions.append(prediction)
        arrays[f"{solver_name}_coefficients"] = np.stack(coefficients)
        # Synthetic predictions have one mesh; scientific meshes differ, so
        # persist their exact per-fit field errors/health in JSON and only the
        # fixed-size coefficient vectors in NPZ.
        if smoke:
            arrays[f"{solver_name}_predictions"] = np.stack(predictions)

    summaries = {}
    for solver_name, _ in SOLVERS:
        rows = records[solver_name]
        targeted = [
            row for row in rows
            if row["N"] == 128 and row["draw_index"] == TARGET_N128_INDEX
        ]
        if smoke:
            targeted_trajectory = 0.0
            targeted_s0 = None
        else:
            targeted = sorted(targeted, key=lambda row: row["time_index"])
            if [row["time_index"] for row in targeted] != list(range(51)):
                raise SystemExit(f"{solver_name} target trajectory is incomplete")
            targeted_trajectory = float(np.sqrt(
                sum(row["error_numerator_sq"] for row in targeted)
                / max(sum(row["truth_norm_sq"] for row in targeted), 1e-300)
            ))
            with np.load(s0_npz_path, allow_pickle=False) as old:
                targeted_s0 = float(np.asarray(
                    old["C_N128_trajectory_error"], np.float64
                )[TARGET_N128_INDEX - 512])
        all_health = bool(all(
            row["healthy"] and row["relative_normal_residual"] <= 1e-8
            and row["boundary_exact"] for row in rows
        ))
        no_regression = bool(all(row["no_regression"] for row in rows))
        target_pass = bool(
            smoke or (
                targeted_trajectory <= 7e-4
                and targeted_trajectory <= targeted_s0 + NO_REGRESSION_ABSOLUTE
            )
        )
        summaries[solver_name] = {
            "fit_count": len(rows), "all_fit_health_pass": all_health,
            "no_snapshot_regression_pass": no_regression,
            "worst_relative_normal_residual": float(max(
                row["relative_normal_residual"] for row in rows
            )),
            "target_N128_draw530_trajectory_relative_l2": targeted_trajectory,
            "target_N128_draw530_s0_trajectory_relative_l2": targeted_s0,
            "target_N128_draw530_pass": target_pass,
            "pass": bool(all_health and no_regression and target_pass),
        }
    passing = [name for name, _ in SOLVERS if summaries[name]["pass"]]
    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda name: (
                summaries[name]["worst_relative_normal_residual"],
                0 if name == "S1" else 1,
            ),
        )
    return {
        "config": {
            "subsets": {str(key): list(value) for key, value in SOLVER_SUBSETS.items()},
            "times": list(SOLVER_TIMES),
            "target_full_trajectory": {"N": 128, "draw_index": 530},
            "normal_tolerance": 1e-8,
            "no_regression_absolute": NO_REGRESSION_ABSOLUTE,
        },
        "records": records, "summaries": summaries,
        "selected_solver": selected,
    }, arrays


def make_mandatory_kernel(route, n, predictor, hyper, num_steps):
    geometry = base.weak_geometry(n, ARM_C)
    stencil_coords = jnp.asarray(geometry["stencil_coords"], jnp.float64)
    stencil_mask = jnp.asarray(geometry["stencil_mask"], jnp.float64)
    phi_weighted = jnp.asarray(geometry["phi_weighted"], jnp.float64)
    eigenvalues = jnp.asarray(geometry["eigenvalues"], jnp.float64)
    table = jnp.asarray(p3.span_polynomial_table_np(), jnp.float64)
    m = ARM_C["m"]
    pallas_full = pallas_current = pallas_previous = None
    if route == "K3":
        pallas_full = p3.make_pallas_polynomial_decoder(num_steps + 1, n * n)
        pallas_current = p3.make_pallas_polynomial_decoder(num_steps, m * 5)
        pallas_previous = p3.make_pallas_polynomial_decoder(num_steps, m)

    def coefficients(states):
        return jax.vmap(lambda state: s.apply_mlp(hyper, state[5:]))(states)

    def decode_batch(states, coefficient_values, coords, mask, purpose):
        if route == "K0":
            return jax.lax.map(
                lambda pair: s.decode_one_jax(
                    pair[0], pair[1], coords, mask, p3.R
                ),
                (states, coefficient_values),
            )
        if route == "K1" or (route == "K2" and purpose != "full"):
            return p3.decode_states_polynomial_sequential(
                states, coefficient_values, coords, mask, table
            )
        if route == "K2":
            return p3.decode_states_polynomial_chunk3(
                states, coefficient_values, coords, mask, table
            )
        decoder = {
            "full": pallas_full,
            "current": pallas_current,
            "previous": pallas_previous,
        }[purpose]
        return decoder(states, coefficient_values, coords, mask, table)

    def mandatory(features, coords, mask, viscosity):
        states = jnp.tanh(s.apply_mlp(predictor, features[:num_steps + 1]))
        coefficient_values = coefficients(states)
        current = decode_batch(
            states[1:], coefficient_values[1:], stencil_coords,
            stencil_mask, "current",
        ).reshape(num_steps, m, 5)
        previous = decode_batch(
            states[:-1], coefficient_values[:-1], stencil_coords[::5],
            stencil_mask[::5], "previous",
        )
        center, xp, xm, yp, ym = [current[:, :, index] for index in range(5)]
        dx = 1.0 / (n - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected_u = center @ phi_weighted
        mode_preconditioner = (1.0 + c.DT * viscosity * eigenvalues) ** -1.0
        residual = mode_preconditioner[None] * (
            (center - previous) @ phi_weighted
            + c.DT * (advection @ phi_weighted + viscosity * projected_u * eigenvalues)
        )
        denominator = jnp.maximum(
            jnp.linalg.norm(previous @ phi_weighted, axis=1), 1e-12
        )
        rho = jnp.linalg.norm(residual, axis=1) / denominator
        fields = decode_batch(
            states, coefficient_values, coords, mask, "full"
        )
        return fields, states, coefficient_values, residual, rho

    def identity_probe(features, coords, mask, viscosity):
        states = jnp.tanh(s.apply_mlp(predictor, features[:num_steps + 1]))
        coefficient_values = coefficients(states)
        current = decode_batch(
            states[1:], coefficient_values[1:], stencil_coords,
            stencil_mask, "current",
        )
        previous = decode_batch(
            states[:-1], coefficient_values[:-1], stencil_coords[::5],
            stencil_mask[::5], "previous",
        )
        fields, _, _, residual, rho = mandatory(
            features, coords, mask, viscosity
        )
        return fields, current, previous, residual, rho

    return jax.jit(mandatory), jax.jit(identity_probe), geometry


def compile_kernel(route, n, features, viscosity, num_steps):
    predictor, hyper = s.init_cost_oracle_parameters(
        p3.R, p3.K, base.ARM_SEEDS["C"]
    )
    kernel, identity_probe, geometry = make_mandatory_kernel(
        route, n, predictor, hyper, num_steps
    )
    arguments = (
        jnp.asarray(features, jnp.float64),
        jnp.asarray(c.grid_coords(n), jnp.float64),
        jnp.asarray(c.binary_boundary_mask(n), jnp.float64),
        jnp.asarray(viscosity, jnp.float64),
    )
    lower_started = time.perf_counter()
    lowered = kernel.lower(*arguments)
    lower_s = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - compile_started
    identity_lower_started = time.perf_counter()
    identity_lowered = identity_probe.lower(*arguments)
    identity_lower_s = time.perf_counter() - identity_lower_started
    identity_compile_started = time.perf_counter()
    identity_executable = identity_lowered.compile()
    identity_compile_s = time.perf_counter() - identity_compile_started
    return executable, identity_executable, arguments, {
        "lower_s": float(lower_s), "compile_s": float(compile_s),
        "lower_plus_compile_s": float(lower_s + compile_s),
        "identity_lower_s": float(identity_lower_s),
        "identity_compile_s": float(identity_compile_s),
        "memory_analysis": base.memory_analysis(executable),
        "weak_geometry": {
            "M": p3.M, "m": p3.Q, "rule": geometry["rule"],
        },
    }


def kernel_work_record(route, case, output):
    fields, _, _, residual, rho = [np.asarray(value) for value in output]
    return {
        "route": route, "case_index": int(case),
        "finite": bool(
            np.all(np.isfinite(fields)) and np.all(np.isfinite(residual))
            and np.all(np.isfinite(rho))
        ),
        "output_shape": list(fields.shape),
        "rho_all": rho.tolist(),
        "weak_residual_norm_all": np.linalg.norm(residual, axis=1).tolist(),
        "weak_objective_evaluations": int(rho.size),
        "weak_jacobian_evaluations": 0,
        "trial_residual_evaluations": 0,
    }


def run_cost_diagnostic(smoke=False):
    basis_identity = basis_identity_record()
    if not basis_identity["pass"]:
        raise SystemExit("span-polynomial basis identity gate failed")
    try:
        pallas_basis_identity = pallas_basis_identity_record()
    except Exception as error:
        pallas_basis_identity = {
            "available": p3.pallas_is_available(), "executed": False,
            "pass": False, "error": f"{type(error).__name__}: {error}",
        }
    if smoke:
        n, num_steps = 32, 2
        parameters = {
            "cx": np.asarray((0.42,)), "cy": np.asarray((0.58,)),
            "width": np.asarray((0.12,)), "amplitude": np.asarray((1.2,)),
            "nu": np.asarray((0.01,)),
        }
        features_by_case = [c.trajectory_features(parameters, n)[0, :3]]
        routes, case_count = KERNELS, 1
        reference = None
    else:
        n, num_steps = 1024, c.NUM_STEPS
        truth, parameters, reference_health, dummy = base.generate_live_reference(n)
        reference = {
            "truth": truth, "health": reference_health, "dummy": dummy,
        }
        features_by_case = []
        for case in range(FOM_CASES):
            recovered, sample_indices = c.recover_blob_parameters_fixed_sample(
                truth[case, 0], n
            )
            expected = np.asarray((
                parameters["cx"][case], parameters["cy"][case],
                parameters["width"][case], parameters["amplitude"][case],
            ))
            recovery_error = float(
                np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
            )
            if sample_indices.size > 4096 or recovery_error > 1e-9:
                raise SystemExit("charged fixed-sample recovery gate failed")
            one = {
                "cx": recovered[0:1], "cy": recovered[1:2],
                "width": recovered[2:3], "amplitude": recovered[3:4],
                "nu": parameters["nu"][case:case + 1],
            }
            features_by_case.append(c.trajectory_features(one, n)[0])
        routes, case_count = KERNELS, FOM_CASES

    compiled, identity_compiled, setup, compile_failures = {}, {}, {}, {}
    cached_coords_mask = None
    for route in routes:
        if route == "K3" and not pallas_basis_identity["pass"]:
            compile_failures[route] = pallas_basis_identity.get(
                "error", "K3-specific basis identity did not pass"
            )
            continue
        try:
            executable, identity_executable, arguments, one_setup = compile_kernel(
                route, n, features_by_case[0], parameters["nu"][0], num_steps
            )
        except Exception as error:
            if route != "K3":
                raise
            compile_failures[route] = f"{type(error).__name__}: {error}"
            continue
        compiled[route] = executable
        identity_compiled[route] = identity_executable
        setup[route] = one_setup
        cached_coords_mask = arguments[1:3]
    if not all(route in compiled for route in ("K0", "K1", "K2")):
        raise SystemExit("noncustom kernel bracket did not compile")

    fom = None
    if not smoke:
        fom, _ = base.bc.make_chain(
            n, base.FOM_OUTER, lin_tol=base.FOM_INNER,
            preconditioner="helmholtz",
        )
    methods = (["fom"] if not smoke else []) + list(compiled)

    def invoke(method, case):
        started = time.perf_counter()
        if method == "fom":
            output = fom(
                jnp.asarray(reference["truth"][case, 0]),
                parameters["nu"][case], reference["dummy"], jnp.int32(5),
            )
        else:
            if smoke:
                features = features_by_case[case]
            else:
                recovered, _ = c.recover_blob_parameters_fixed_sample(
                    reference["truth"][case, 0], n
                )
                one = {
                    "cx": recovered[0:1], "cy": recovered[1:2],
                    "width": recovered[2:3], "amplitude": recovered[3:4],
                    "nu": parameters["nu"][case:case + 1],
                }
                features = c.trajectory_features(one, n)[0]
            output = compiled[method](
                jnp.asarray(features, jnp.float64), cached_coords_mask[0],
                cached_coords_mask[1],
                jnp.asarray(parameters["nu"][case], jnp.float64),
            )
        jax.block_until_ready(output)
        return output, float(time.perf_counter() - started)

    first_execution, canonical = {}, {route: [] for route in compiled}
    for method in methods:
        output, elapsed = invoke(method, 0)
        first_execution[method] = elapsed
    for route in compiled:
        for case in range(case_count):
            output, _ = invoke(route, case)
            canonical[route].append(kernel_work_record(route, case, output))
    identity = {route: [] for route in compiled if route != "K0"}
    for case in range(case_count):
        features = jnp.asarray(features_by_case[case], jnp.float64)
        identity_arguments = (
            features, cached_coords_mask[0], cached_coords_mask[1],
            jnp.asarray(parameters["nu"][case], jnp.float64),
        )
        control = identity_compiled["K0"](*identity_arguments)
        jax.block_until_ready(control)
        control_values = [np.asarray(value) for value in control]
        for route in identity:
            output = identity_compiled[route](*identity_arguments)
            jax.block_until_ready(output)
            values = [np.asarray(value) for value in output]
            names = ("full_fields", "current_stencils", "previous_centers",
                     "weak_residual", "rho")
            relative = {
                name: relative_field_error(value, reference_value)
                for name, value, reference_value in zip(names, values, control_values)
            }
            max_relative = float(max(relative.values()))
            identity[route].append({
                "case_index": case, "relative_l2": relative,
                "max_relative_l2": max_relative,
                "full_field_max_abs": float(np.max(np.abs(
                    values[0] - control_values[0]
                ))),
                "exact_boundary": bool(np.all(
                    values[0][:, c.binary_boundary_mask(n) == 0.0] == 0.0
                )),
                "pass": bool(
                    max_relative <= IDENTITY_TOL
                    and np.all(values[0][:, c.binary_boundary_mask(n) == 0.0] == 0.0)
                ),
            })
    if smoke:
        return {
            "status": "excluded_execution_smoke_pass",
            "setup": setup, "compile_failures": compile_failures,
            "first_execution_s": first_execution,
            "basis_identity": basis_identity,
            "pallas_basis_identity": pallas_basis_identity,
            "identity": identity, "canonical_work": canonical,
        }

    for _ in range(TIME_WARM):
        for case in range(case_count):
            for method in methods:
                invoke(method, case)
    burn_count = c.gpu_burn(BURN_SECONDS)
    records = {method: [] for method in methods}
    timing_orders = []
    for repetition in range(TIME_REPS):
        offset = repetition % len(methods)
        order = methods[offset:] + methods[:offset]
        if (repetition // len(methods)) % 2:
            order = list(reversed(order))
        timing_orders.append(order)
        cases = list(range(case_count))
        cases = cases[repetition % case_count:] + cases[:repetition % case_count]
        for case in cases:
            for method in order:
                output, elapsed = invoke(method, case)
                row = {"case_index": case, "repetition": repetition, "elapsed_s": elapsed}
                if method == "fom":
                    row.update(base.fom_grade(output, reference["truth"][case]))
                else:
                    row["finite"] = bool(np.all(np.isfinite(np.asarray(output[0]))))
                records[method].append(row)
    position_counts = {
        method: [
            int(sum(order[position] == method for order in timing_orders))
            for position in range(len(methods))
        ] for method in methods
    }
    expected_position = TIME_REPS // len(methods)
    exact_balance = bool(
        TIME_REPS % len(methods) == 0
        and all(
            count == expected_position
            for counts in position_counts.values() for count in counts
        )
    )
    if not exact_balance:
        raise SystemExit("Phase-3 timing position balance failed")
    summaries = {
        method: base.summarize_timing(rows, case_count)
        for method, rows in records.items()
    }
    fom_accuracy_rows = [row for row in records["fom"] if row["repetition"] == 0]
    fom_mean = float(np.mean([row["trajectory_relative_l2"] for row in fom_accuracy_rows]))
    fom_worst = float(np.max([row["trajectory_relative_l2"] for row in fom_accuracy_rows]))
    fom_healthy = bool(all(
        row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
        and row["max_returned_relative_residual"] <= base.FOM_OUTER
        for row in records["fom"]
    ))
    fom_eligible = bool(fom_healthy and fom_mean <= 1e-3 and fom_worst <= 3e-3)
    gates = {}
    for index, route in enumerate(compiled):
        speedup = summaries["fom"]["median_elapsed_s"] / summaries[route]["median_elapsed_s"]
        ci = base.clustered_speedup_ci(
            summaries["fom"]["per_case_median_elapsed_s"],
            summaries[route]["per_case_median_elapsed_s"], 20263000 + index,
        )
        identity_pass = bool(
            route != "K0" and basis_identity["pass"]
            and (route != "K3" or pallas_basis_identity["pass"])
            and all(row["pass"] for row in identity[route])
        )
        work_pass = bool(len(canonical[route]) == 4 and all(
            row["finite"] and row["weak_objective_evaluations"] == 50
            and row["weak_jacobian_evaluations"] == 0
            and row["trial_residual_evaluations"] == 0
            for row in canonical[route]
        ))
        memory_ok = setup[route]["memory_analysis"]["eligibility_device_bytes"] <= 20_000_000_000
        gates[route] = {
            "paired_median_speedup": float(speedup),
            "clustered_speedup_ci": ci, "identity_pass": identity_pass,
            "memory_pass": memory_ok, "fom_eligible": fom_eligible,
            "canonical_work_pass": work_pass,
            "pass": bool(
                identity_pass and work_pass and memory_ok and fom_eligible
                and speedup >= 10.0 and ci[0] >= 8.0
            ),
        }
    selected = None
    passing = [route for route in ("K1", "K2", "K3") if gates.get(route, {}).get("pass")]
    if passing:
        selected = min(
            passing, key=lambda route: (summaries[route]["median_elapsed_s"], KERNELS.index(route))
        )
    return {
        "status": "complete", "reference_health": reference["health"],
        "setup": setup, "compile_failures": compile_failures,
        "first_execution_after_compile_s": first_execution,
        "canonical_work": canonical, "basis_identity": basis_identity,
        "pallas_basis_identity": pallas_basis_identity,
        "identity": identity,
        "burn_count": burn_count, "timing_orders": timing_orders,
        "position_counts": position_counts, "exact_position_balance": exact_balance,
        "records": records, "summaries": summaries,
        "fom_accuracy": {
            "mean": fom_mean, "worst": fom_worst,
            "healthy": fom_healthy, "eligible": fom_eligible,
        },
        "gates": gates, "selected_kernel": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--s0-json")
    parser.add_argument("--s0-npz")
    parser.add_argument("--s0-audit")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    if args.smoke:
        solver, arrays = run_solver_diagnostic("", smoke=True)
        cost = run_cost_diagnostic(smoke=True)
        status = "excluded_execution_smoke_pass"
        s0_binding = None
    else:
        if not args.s0_json or not args.s0_npz or not args.s0_audit:
            raise SystemExit("scientific Phase3-D requires the complete S0 artifact chain")
        s0, audit = validate_s0_chain(args.s0_json, args.s0_npz, args.s0_audit)
        solver, arrays = run_solver_diagnostic(args.s0_npz, smoke=False)
        cost = run_cost_diagnostic(smoke=False)
        status = "complete"
        s0_binding = {
            "json": os.path.abspath(args.s0_json), "json_sha256": c.sha256(args.s0_json),
            "npz": os.path.abspath(args.s0_npz), "npz_sha256": c.sha256(args.s0_npz),
            "audit": os.path.abspath(args.s0_audit), "audit_sha256": c.sha256(args.s0_audit),
            "commit": s0["provenance"]["commit"], "job_id": s0["provenance"]["slurm_job_id"],
            "audit_decision": audit["decision"],
        }
    decision = {
        "selected_solver": solver["selected_solver"],
        "selected_kernel": cost.get("selected_kernel"),
    }
    decision["run_P3_F"] = bool(
        status == "complete" and decision["selected_solver"]
        and decision["selected_kernel"]
    )
    decision["phase3_hard_stop"] = bool(status == "complete" and not decision["run_P3_F"])
    np.savez_compressed(args.output_npz, **arrays)
    report = {
        "status": status, "provenance": c.provenance(),
        "config": {
            "candidate": ARM_C, "smoke": args.smoke,
            "data_seed": None if args.smoke else 0,
            "draw_count": None if args.smoke else 704,
            "solver_subsets": {
                str(key): list(value) for key, value in SOLVER_SUBSETS.items()
            },
            "solver_times": list(SOLVER_TIMES),
            "target_N128_full_trajectory_index": TARGET_N128_INDEX,
            "solver_normal_tolerance": 1e-8,
            "no_regression_absolute": NO_REGRESSION_ABSOLUTE,
            "kernel_routes": list(KERNELS),
            "identity_relative_l2_tolerance": IDENTITY_TOL,
            "basis_identity_seed": 20260820,
            "cost_parameter_seed": base.ARM_SEEDS["C"],
            "fom_seed": base.FOM_SEED,
            "fom_draw_count": base.FOM_DRAW_COUNT,
            "fom_cases": FOM_CASES,
            "fom_outer": base.FOM_OUTER,
            "fom_inner": base.FOM_INNER,
            "reference_outer": base.REFERENCE_OUTER,
            "reference_inner": base.REFERENCE_INNER,
            "audit_outer": base.AUDIT_OUTER,
            "audit_inner": base.AUDIT_INNER,
            "time_repetitions": TIME_REPS,
            "time_warmups": TIME_WARM,
            "burn_seconds": BURN_SECONDS,
            "model_validation_touched": False,
            "confirmation_touched": False,
            "f64": True, "matmul_precision": "highest",
        },
        "s0_binding": s0_binding, "solver_diagnostic": solver,
        "kernel_diagnostic": cost, "decision": decision,
        "elapsed_s": float(time.perf_counter() - started),
        "npz": {
            "basename": os.path.basename(args.output_npz),
            "sha256": c.sha256(args.output_npz),
        },
    }
    c.save_json(args.output_json, report)
    c.log({"status": status, "decision": decision, "elapsed_s": report["elapsed_s"]})


if __name__ == "__main__":
    main()
