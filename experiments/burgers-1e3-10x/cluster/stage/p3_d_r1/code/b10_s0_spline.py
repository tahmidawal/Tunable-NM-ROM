"""Phase-2 S0: free-spline projection and paired charged cost falsification.

Scientific mode regenerates only the locked exposed selection and FOM-calibration
draws.  ``SMOKE=1`` is an excluded, small execution check and opens no new draw.

Usage: b10_s0_spline.py OUTPUT_JSON OUTPUT_NPZ
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_spline as s

HERE = os.path.dirname(os.path.abspath(__file__))
HYBRID = os.path.abspath(os.path.join(HERE, "..", "burgers-hybrid-1024"))
if os.path.isfile(os.path.join(HYBRID, "bh_common.py")):
    sys.path.insert(0, HYBRID)
import bh_common as bc  # noqa: E402

OUTPUT_JSON, OUTPUT_NPZ = sys.argv[1:3]
SMOKE = os.environ.get("SMOKE", "0") == "1"
TIME_REPS = int(os.environ.get("TIME_REPS", "10"))
TIME_WARM = int(os.environ.get("TIME_WARM", "1"))
BURN_SECONDS = float(os.environ.get("BURN_SECONDS", "3"))
REFERENCE_OUTER = 1e-12
REFERENCE_INNER = 1e-7
AUDIT_OUTER = 3e-13
AUDIT_INNER = 3e-8
FOM_OUTER = 3e-3
FOM_INNER = 1e-1
FOM_SEED = 20260822
FOM_DRAW_COUNT = 32
FOM_CASES = 4
ORACLE_WORKERS = int(os.environ.get("ORACLE_WORKERS", "1" if SMOKE else "8"))

SELECTION = (
    (64, 512, 64),
    (128, 512, 32),
    (256, 512, 16),
)
ARM_SEEDS = {"A": 20262012, "B": 20262016, "C": 20262024}


def interior_indices(n):
    ii, jj = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1), indexing="ij")
    return (ii * n + jj).reshape(-1)


def deterministic_weak_points(n, m):
    """Locked lexicographic rounded tensor lattice, exactly m unique centers."""
    side = int(math.ceil(math.sqrt(int(m))))
    axis = np.unique(np.rint(np.linspace(1, n - 2, side)).astype(np.int64))
    while axis.size * axis.size < int(m):
        side += 1
        axis = np.unique(np.rint(np.linspace(1, n - 2, side)).astype(np.int64))
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    flat = np.sort((ii * n + jj).reshape(-1))[:int(m)]
    if np.unique(flat).size != int(m):
        raise ValueError("weak lattice does not contain m unique centers")
    return flat


def selected_test_modes(n, count, flat_indices):
    """Lowest sine modes evaluated only at named target-grid centers."""
    dx = 1.0 / (n - 1)
    kk = np.arange(1, n - 1)
    kx_grid, ky_grid = np.meshgrid(kk, kk, indexing="ij")
    eigenvalues = (4.0 / dx**2) * (
        np.sin(np.pi * kx_grid / (2 * (n - 1))) ** 2
        + np.sin(np.pi * ky_grid / (2 * (n - 1))) ** 2
    )
    order = np.argsort(eigenvalues.reshape(-1), kind="stable")[:int(count)]
    kx = kx_grid.reshape(-1)[order]
    ky = ky_grid.reshape(-1)[order]
    ii = np.asarray(flat_indices) // n
    jj = np.asarray(flat_indices) % n
    normalization = (n - 1) / 2.0
    phi = (
        np.sin(np.pi * ii[:, None] * kx[None] / (n - 1))
        * np.sin(np.pi * jj[:, None] * ky[None] / (n - 1))
        / normalization
    )
    return phi.astype(np.float64), eigenvalues.reshape(-1)[order].astype(np.float64)


def weak_geometry(n, candidate):
    m, mode_count = int(candidate["m"]), int(candidate["M"])
    centers = deterministic_weak_points(n, m)
    stencil = np.stack(
        (centers, centers + n, centers - n, centers + 1, centers - 1), axis=1
    )
    coords = c.grid_coords(n)
    mask = c.binary_boundary_mask(n)
    phi, eigenvalues = selected_test_modes(n, mode_count, centers)
    structural_weights = np.full(m, (n - 2) ** 2 / m, np.float64)
    return {
        "center_indices": centers,
        "stencil_coords": coords[stencil.reshape(-1)],
        "stencil_mask": mask[stencil.reshape(-1)],
        "phi_weighted": phi * structural_weights[:, None],
        "eigenvalues": eigenvalues,
        "weight": structural_weights,
        "rule": (
            "lexicographically sorted rounded uniform interior tensor lattice; "
            "constant structural weights (N-2)^2/m; not fitted EQ"
        ),
    }


def make_online_kernels(n, candidate, predictor, hyper, num_steps):
    """Build direct, mandatory-zero, and maximum-one-update cost kernels."""
    r = int(candidate["R"])
    geometry = weak_geometry(n, candidate)
    stencil_coords = jnp.asarray(geometry["stencil_coords"], jnp.float64)
    stencil_mask = jnp.asarray(geometry["stencil_mask"], jnp.float64)
    phi_weighted = jnp.asarray(geometry["phi_weighted"], jnp.float64)
    eigenvalues = jnp.asarray(geometry["eigenvalues"], jnp.float64)
    m = int(candidate["m"])

    def coefficients(state):
        return s.apply_mlp(hyper, state[5:])

    def weak_residual(state, previous_state, viscosity):
        current_values = s.decode_one_jax(
            state, coefficients(state), stencil_coords, stencil_mask, r
        ).reshape(m, 5)
        previous_values = s.decode_one_jax(
            previous_state, coefficients(previous_state),
            stencil_coords[::5], stencil_mask[::5], r,
        )
        center, xp, xm, yp, ym = [current_values[:, index] for index in range(5)]
        dx = 1.0 / (n - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected_u = phi_weighted.T @ center
        mode_preconditioner = (1.0 + c.DT * viscosity * eigenvalues) ** -1.0
        residual = mode_preconditioner * (
            phi_weighted.T @ (center - previous_values)
            + c.DT * (
                phi_weighted.T @ advection
                + viscosity * eigenvalues * projected_u
            )
        )
        denominator = jnp.maximum(
            jnp.linalg.norm(phi_weighted.T @ previous_values), 1e-12
        )
        return residual, jnp.linalg.norm(residual) / denominator

    def predict_states(features):
        return jnp.tanh(s.apply_mlp(predictor, features))

    def decode_states(states, coords, mask):
        coeff = jax.vmap(coefficients)(states)
        # Keep time sequential so only one N^2 x 16 local-support work array is
        # live; all 51 decoded fields are still written and charged.
        fields = jax.lax.map(
            lambda pair: s.decode_one_jax(pair[0], pair[1], coords, mask, r),
            (states, coeff),
        )
        return fields, coeff

    def direct(features, coords, mask, viscosity):
        del viscosity
        states = predict_states(features[:num_steps + 1])
        fields, coeff = decode_states(states, coords, mask)
        return fields, states, coeff

    def mandatory(features, coords, mask, viscosity):
        states = predict_states(features[:num_steps + 1])
        residual, rho = jax.vmap(
            lambda state, previous: weak_residual(state, previous, viscosity)
        )(states[1:], states[:-1])
        fields, coeff = decode_states(states, coords, mask)
        return fields, states, coeff, residual, rho

    def maximum_one(features, coords, mask, viscosity):
        predicted = predict_states(features[:num_steps + 1])

        def step(previous, target):
            residual, rho_before = weak_residual(target, previous, viscosity)
            jacobian = jax.jacfwd(
                lambda state: weak_residual(state, previous, viscosity)[0]
            )(target)
            hessian = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            diagonal = jnp.diag(jnp.diag(hessian)) + 1e-12 * jnp.eye(
                target.size, dtype=jnp.float64
            )
            raw_step = jnp.linalg.solve(hessian + 1e-6 * diagonal, -gradient)
            raw_step = jnp.where(
                jnp.all(jnp.isfinite(raw_step)), raw_step, jnp.zeros_like(raw_step)
            )
            scale = jnp.minimum(1.0, 0.25 / jnp.maximum(jnp.linalg.norm(raw_step), 1e-300))
            bounded = raw_step * scale
            factors = jnp.asarray((1.0, 0.5, 0.25, 0.0), jnp.float64)
            trials = jnp.clip(
                target[None] + factors[:, None] * bounded[None],
                -1.0 + 1e-8, 1.0 - 1e-8,
            )
            trial_rho = jax.vmap(
                lambda state: weak_residual(state, previous, viscosity)[1]
            )(trials)
            trial_rho = jnp.where(jnp.isfinite(trial_rho), trial_rho, jnp.inf)
            best = jnp.argmin(trial_rho)
            candidate_state = trials[best]
            accepted = trial_rho[best] < rho_before
            corrected = jnp.where(accepted, candidate_state, target)
            return corrected, (
                corrected, residual, rho_before, trial_rho[best], jacobian,
                factors[best], jnp.linalg.norm(bounded), accepted,
            )

        _, outputs = jax.lax.scan(step, predicted[0], predicted[1:])
        corrected = jnp.concatenate((predicted[:1], outputs[0]), axis=0)
        fields, coeff = decode_states(corrected, coords, mask)
        return (fields, corrected, coeff) + outputs[1:]

    return {
        "direct": jax.jit(direct),
        "mandatory": jax.jit(mandatory),
        "maximum_one": jax.jit(maximum_one),
        "geometry": geometry,
    }


def memory_analysis(compiled):
    analysis = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
        "alias_size_in_bytes", "host_argument_size_in_bytes",
        "host_output_size_in_bytes", "host_temp_size_in_bytes",
    )
    values = {name: int(getattr(analysis, name, 0) or 0) for name in names}
    values["eligibility_device_bytes"] = int(
        values["argument_size_in_bytes"] + values["output_size_in_bytes"]
        + values["temp_size_in_bytes"] - values["alias_size_in_bytes"]
    )
    return values


def allocator_stats():
    values = jax.devices()[0].memory_stats() or {}
    return {
        str(key): int(value) if isinstance(value, (int, np.integer)) else str(value)
        for key, value in values.items()
    }


def compile_online_kernels(n, candidate, features, viscosity, num_steps):
    predictor, hyper = s.init_cost_oracle_parameters(
        candidate["R"], candidate["k"], ARM_SEEDS[candidate["arm"]]
    )
    kernels = make_online_kernels(n, candidate, predictor, hyper, num_steps)
    coords = jnp.asarray(c.grid_coords(n), jnp.float64)
    mask = jnp.asarray(c.binary_boundary_mask(n), jnp.float64)
    arguments = (jnp.asarray(features, jnp.float64), coords, mask,
                 jnp.asarray(viscosity, jnp.float64))
    compiled = {}
    memory = {}
    build = {}
    for name in ("direct", "mandatory", "maximum_one"):
        lower_started = time.perf_counter()
        lowered = kernels[name].lower(*arguments)
        lower_s = time.perf_counter() - lower_started
        compile_started = time.perf_counter()
        executable = lowered.compile()
        compile_s = time.perf_counter() - compile_started
        compiled[name] = executable
        memory[name] = memory_analysis(executable)
        build[name] = {
            "lower_s": float(lower_s), "compile_s": float(compile_s),
            "lower_plus_compile_s": float(lower_s + compile_s),
        }
    return compiled, memory, build, kernels["geometry"], arguments


def fom_grade(output, truth, elapsed_s=None):
    fields, newton, linear, breakdowns, flags, residuals = [
        np.asarray(value) for value in output
    ]
    record = {
        "trajectory_relative_l2": float(
            np.linalg.norm(fields - truth[1:]) / np.linalg.norm(truth[1:])
        ),
        "max_returned_relative_residual": float(np.max(residuals)),
        "newton_total": int(np.sum(newton)),
        "linear_total": int(np.sum(linear)),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "finite": bool(np.all(np.isfinite(fields)) and np.all(np.isfinite(residuals))),
    }
    if elapsed_s is not None:
        record["elapsed_s"] = float(elapsed_s)
    return record


def generate_live_reference(n):
    cx, cy, width, amplitude, nu, normalized = c.bf.sample_params(
        seed=FOM_SEED, m=FOM_DRAW_COUNT
    )
    selected = np.arange(FOM_CASES)
    u0 = np.stack([
        c.bf.blob_ic(n, cx[index], cy[index], width[index], amplitude[index])
        for index in selected
    ]).astype(np.float64)
    dummy = jnp.zeros((c.NUM_STEPS, n * n), jnp.float64)
    reference, _ = bc.make_chain(
        n, REFERENCE_OUTER, lin_tol=REFERENCE_INNER, preconditioner="helmholtz"
    )
    audit, _ = bc.make_chain(
        n, AUDIT_OUTER, lin_tol=AUDIT_INNER, preconditioner="helmholtz"
    )
    truth, reference_records, audit_records, differences = [], [], [], []
    for case in selected:
        output = reference(jnp.asarray(u0[case]), nu[case], dummy, jnp.int32(5))
        jax.block_until_ready(output)
        trajectory = np.concatenate((u0[case:case + 1], np.asarray(output[0])), axis=0)
        reference_record = fom_grade(output, trajectory)
        audit_output = audit(jnp.asarray(u0[case]), nu[case], dummy, jnp.int32(5))
        jax.block_until_ready(audit_output)
        audit_record = fom_grade(audit_output, trajectory)
        reference_records.append(reference_record)
        audit_records.append(audit_record)
        differences.append(audit_record["trajectory_relative_l2"])
        truth.append(trajectory)
    reference_healthy = all(
        item["finite"] and item["breakdowns"] == 0 and item["flags_nonzero"] == 0
        and item["max_returned_relative_residual"] <= REFERENCE_OUTER
        for item in reference_records
    )
    audit_healthy = all(
        item["finite"] and item["breakdowns"] == 0 and item["flags_nonzero"] == 0
        and item["max_returned_relative_residual"] <= AUDIT_OUTER
        for item in audit_records
    )
    if not reference_healthy or not audit_healthy or max(differences) > 1e-4:
        raise SystemExit("S0 live reference/audit health gate failed")
    parameters = {
        "cx": np.asarray(cx[selected]), "cy": np.asarray(cy[selected]),
        "width": np.asarray(width[selected]), "amplitude": np.asarray(amplitude[selected]),
        "nu": np.asarray(nu[selected]), "normalized": np.asarray(normalized[selected]),
    }
    health = {
        "reference_outer": REFERENCE_OUTER, "reference_inner": REFERENCE_INNER,
        "audit_outer": AUDIT_OUTER, "audit_inner": AUDIT_INNER,
        "reference_records": reference_records, "audit_records": audit_records,
        "cross_chain_trajectory_relative_l2_all": differences,
        "cross_chain_worst": float(max(differences)),
        "all_finite_zero_flags_breakdowns": True,
    }
    return np.stack(truth), parameters, health, dummy


def trajectory_metrics(prediction, truth):
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    trajectory = np.linalg.norm(
        (prediction - truth).reshape(truth.shape[0], -1), axis=1
    ) / np.maximum(np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1), 1e-300)
    snapshot = np.linalg.norm(prediction - truth, axis=2) / np.maximum(
        np.linalg.norm(truth, axis=2), 1e-300
    )
    return trajectory, snapshot


def run_free_oracles(report, arrays):
    if not 1 <= ORACLE_WORKERS <= 8:
        raise SystemExit("ORACLE_WORKERS must be between one and eight")
    pooled = {item["arm"]: [] for item in s.CANDIDATES}
    for n, start, count in SELECTION:
        c.log("S0 free oracle N", n, "cases", count)
        truth, _, health = c.generate_population(
            n, 0, 704, np.arange(start, start + count), chunk=max(1, min(8, count))
        )
        if (
            not np.isfinite(health["reported_max_relative_residual"])
            or not np.isfinite(health["independent_max_relative_residual"])
            or health["reported_max_relative_residual"] > 1e-8
            or health["independent_max_relative_residual"] > 1e-8
        ):
            raise SystemExit(f"N={n} exposed selection legacy reference health failed")
        report["selection_reference_health"][str(n)] = health
        coords = c.grid_coords(n)
        mask = c.binary_boundary_mask(n)
        flat_truth = truth.reshape(-1, n * n)
        for candidate in s.CANDIDATES:
            arm, r = candidate["arm"], candidate["R"]
            predicted = np.empty_like(flat_truth)
            affine = np.empty((flat_truth.shape[0], 5), np.float64)
            normalized_affine = np.empty((flat_truth.shape[0], 5), np.float64)
            coefficients = np.empty((flat_truth.shape[0], r * r), np.float64)
            fit_iterations = np.empty(flat_truth.shape[0], np.int64)
            fit_normal = np.empty(flat_truth.shape[0], np.float64)
            fit_elapsed = np.empty(flat_truth.shape[0], np.float64)
            fit_healthy = np.empty(flat_truth.shape[0], bool)
            exact_boundary = np.empty(flat_truth.shape[0], bool)
            partition_error = np.empty(flat_truth.shape[0], np.float64)
            started = time.perf_counter()
            def fit_one(field):
                return s.fit_free_oracle(field, coords, mask, r)

            if ORACLE_WORKERS == 1:
                fitted = map(fit_one, flat_truth)
                executor = None
            else:
                executor = ThreadPoolExecutor(max_workers=ORACLE_WORKERS)
                fitted = executor.map(fit_one, flat_truth, chunksize=1)
            for index, (one_affine, one_coeff, one_prediction, info) in enumerate(fitted):
                affine[index] = one_affine
                normalized_affine[index] = s.normalized_state_from_affine(one_affine)
                coefficients[index] = one_coeff
                predicted[index] = one_prediction
                fit_iterations[index] = info["lsmr_iterations"]
                fit_normal[index] = info["relative_normal_residual"]
                fit_elapsed[index] = info["elapsed_s"]
                fit_healthy[index] = info["healthy"]
                exact_boundary[index] = info["exact_boundary"]
                partition_error[index] = info["max_partition_sum_error_in_domain"]
            if executor is not None:
                executor.shutdown(wait=True)
            prediction = predicted.reshape(truth.shape)
            trajectory, snapshot = trajectory_metrics(prediction, truth)
            pooled[arm].extend(trajectory.tolist())
            key = f"{arm}_N{n}"
            arrays[f"{key}_affine"] = affine.reshape(count, c.NUM_STEPS + 1, 5)
            arrays[f"{key}_normalized_affine"] = normalized_affine.reshape(
                count, c.NUM_STEPS + 1, 5
            )
            arrays[f"{key}_coefficients"] = coefficients.reshape(
                count, c.NUM_STEPS + 1, r * r
            )
            arrays[f"{key}_trajectory_error"] = trajectory
            arrays[f"{key}_snapshot_error"] = snapshot
            arrays[f"{key}_fit_iterations"] = fit_iterations.reshape(count, -1)
            arrays[f"{key}_relative_normal"] = fit_normal.reshape(count, -1)
            arrays[f"{key}_fit_elapsed_s"] = fit_elapsed.reshape(count, -1)
            arrays[f"{key}_fit_healthy"] = fit_healthy.reshape(count, -1)
            report["free_oracle"][arm]["meshes"][str(n)] = {
                "N": n, "indices": list(range(start, start + count)),
                "trajectory_error_mean": float(np.mean(trajectory)),
                "trajectory_error_worst": float(np.max(trajectory)),
                "trajectory_error_all": trajectory.tolist(),
                "snapshot_error_mean": float(np.mean(snapshot)),
                "snapshot_error_worst": float(np.max(snapshot)),
                "healthy_fits": int(np.sum(fit_healthy)),
                "fit_count": int(fit_healthy.size),
                "zero_unhealthy_fits": bool(np.all(fit_healthy)),
                "exact_binary_boundary": bool(np.all(exact_boundary)),
                "max_partition_sum_error": float(np.max(partition_error)),
                "relative_normal_worst": float(np.max(fit_normal)),
                "lsmr_iterations_median": float(np.median(fit_iterations)),
                "lsmr_iterations_max": int(np.max(fit_iterations)),
                "per_fit_elapsed_median_s": float(np.median(fit_elapsed)),
                "per_fit_elapsed_worst_s": float(np.max(fit_elapsed)),
                "oracle_workers": ORACLE_WORKERS,
                "elapsed_s": float(time.perf_counter() - started),
            }
            c.log(key, report["free_oracle"][arm]["meshes"][str(n)])
    for candidate in s.CANDIDATES:
        arm = candidate["arm"]
        values = np.asarray(pooled[arm], np.float64)
        meshes = report["free_oracle"][arm]["meshes"].values()
        summary = {
            "trajectory_error_mean": float(np.mean(values)),
            "trajectory_error_worst": float(np.max(values)),
            "trajectory_error_all": values.tolist(),
            "every_mesh_and_pooled_accuracy_pass": bool(
                np.mean(values) <= 2e-4 and np.max(values) <= 7e-4
                and all(item["trajectory_error_mean"] <= 2e-4
                        and item["trajectory_error_worst"] <= 7e-4 for item in meshes)
            ),
            "zero_unhealthy_fits": bool(all(
                item["zero_unhealthy_fits"] for item in meshes
            )),
            "exact_binary_boundary": bool(all(
                item["exact_binary_boundary"] for item in meshes
            )),
        }
        report["free_oracle"][arm]["summary"] = summary


def clustered_speedup_ci(fom_case_medians, method_case_medians, seed):
    fom = np.asarray(fom_case_medians, np.float64)
    method = np.asarray(method_case_medians, np.float64)
    ratios = fom / method
    rng = np.random.default_rng(int(seed))
    sample = ratios[rng.integers(0, ratios.size, size=(10000, ratios.size))]
    return [float(value) for value in np.quantile(np.median(sample, axis=1), (0.025, 0.975))]


def summarize_timing(records, n_cases):
    elapsed = np.asarray([item["elapsed_s"] for item in records], np.float64)
    case_medians = [float(np.median([
        item["elapsed_s"] for item in records if item["case_index"] == case
    ])) for case in range(n_cases)]
    median = float(np.median(elapsed))
    outliers_by_case = []
    for case, case_median in enumerate(case_medians):
        values = np.asarray([
            item["elapsed_s"] for item in records if item["case_index"] == case
        ], np.float64)
        outliers_by_case.append(int(np.sum(values > 1.5 * case_median)))
    return {
        "median_elapsed_s": median,
        "elapsed_all_s": elapsed.tolist(),
        "per_case_median_elapsed_s": case_medians,
        "outliers_gt_1p5_within_trajectory_all": outliers_by_case,
        "outliers_gt_1p5_within_trajectory_total": int(sum(outliers_by_case)),
    }


def online_work_record(method, case, output):
    fields = np.asarray(output[0])
    item = {
        "method": method, "case_index": int(case),
        "finite": bool(np.all(np.isfinite(fields))),
        "output_shape": list(fields.shape),
    }
    if method.endswith("_mandatory"):
        residual = np.asarray(output[3])
        rho = np.asarray(output[4])
        item.update({
            "rho_all": rho.tolist(),
            "weak_residual_norm_all": np.linalg.norm(residual, axis=1).tolist(),
            "weak_objective_evaluations": int(rho.size),
            "weak_jacobian_evaluations": 0,
            "trial_residual_evaluations": 0,
        })
    elif method.endswith("_maximum_one"):
        residual = np.asarray(output[3])
        rho_before = np.asarray(output[4])
        rho_after = np.asarray(output[5])
        jacobian = np.asarray(output[6])
        factor = np.asarray(output[7])
        step_norm = np.asarray(output[8])
        accepted = np.asarray(output[9])
        item.update({
            "rho_before_all": rho_before.tolist(),
            "rho_after_best_trial_all": rho_after.tolist(),
            "weak_residual_norm_all": np.linalg.norm(residual, axis=1).tolist(),
            "jacobian_frobenius_all": np.linalg.norm(
                jacobian.reshape(jacobian.shape[0], -1), axis=1
            ).tolist(),
            "trial_factor_all": factor.tolist(),
            "bounded_step_norm_all": step_norm.tolist(),
            "accepted_all": accepted.tolist(),
            "weak_objective_evaluations": int(rho_before.size),
            "weak_jacobian_evaluations": int(rho_before.size),
            "trial_residual_evaluations": int(4 * rho_before.size),
        })
    return item


def run_cost_panel(report):
    n = 1024
    truth, parameters, reference_health, dummy = generate_live_reference(n)
    report["cost_panel"]["reference_health"] = reference_health
    fom, _ = bc.make_chain(
        n, FOM_OUTER, lin_tol=FOM_INNER, preconditioner="helmholtz"
    )
    architecture = {}
    setup = {}
    for candidate in s.CANDIDATES:
        arm = candidate["arm"]
        architecture[arm] = []
        setup[arm] = {
            "memory_analysis": {}, "compile": {}, "weak_geometry": None,
            "representative_inputs": [],
        }
        arm_compiled = None
        cached_coords_mask = None
        for case in range(FOM_CASES):
            recovered, sample_indices = c.recover_blob_parameters_fixed_sample(
                truth[case, 0], n
            )
            expected = np.asarray((
                parameters["cx"][case], parameters["cy"][case],
                parameters["width"][case], parameters["amplitude"][case],
            ), np.float64)
            recovery_relative_error = float(
                np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
            )
            if not np.all(np.isfinite(recovered)) or recovery_relative_error > 1e-9:
                raise SystemExit(
                    f"case {case} fixed-sample recovery gate failed: "
                    f"{recovery_relative_error:.3e}"
                )
            one = {
                "cx": recovered[0:1], "cy": recovered[1:2],
                "width": recovered[2:3], "amplitude": recovered[3:4],
                "nu": parameters["nu"][case:case + 1],
            }
            features = c.trajectory_features(one, n)[0]
            if arm_compiled is None:
                compiled, memory, build, geometry, arguments = compile_online_kernels(
                    n, candidate, features, parameters["nu"][case], c.NUM_STEPS
                )
                arm_compiled = compiled
                cached_coords_mask = arguments[1:3]
                setup[arm]["memory_analysis"] = memory
                setup[arm]["compile"] = build
                setup[arm]["weak_geometry"] = {
                    "rule": geometry["rule"], "M": candidate["M"], "m": candidate["m"],
                    "center_indices": geometry["center_indices"].tolist(),
                    "structural_weight": float(geometry["weight"][0]),
                }
            else:
                arguments = (
                    jnp.asarray(features, jnp.float64), cached_coords_mask[0],
                    cached_coords_mask[1], jnp.asarray(parameters["nu"][case], jnp.float64),
                )
            architecture[arm].append({
                "compiled": arm_compiled, "arguments": arguments,
                "cold_sample_count": int(sample_indices.size),
                "recovered": recovered.tolist(),
            })
            setup[arm]["representative_inputs"].append({
                "case_index": int(case), "source_draw_index": int(case),
                "cold_sample_count": int(sample_indices.size),
                "recovered_parameters": recovered.tolist(),
                "expected_parameters": expected.tolist(),
                "recovery_relative_error": recovery_relative_error,
                "viscosity": float(parameters["nu"][case]),
                "features": features.tolist(),
            })

    methods = ["fom"] + [
        f"{candidate['arm']}_{suffix}" for candidate in s.CANDIDATES
        for suffix in ("direct", "mandatory", "maximum_one")
    ]

    def invoke(method, case):
        started = time.perf_counter()
        if method == "fom":
            output = fom(
                jnp.asarray(truth[case, 0]), parameters["nu"][case], dummy, jnp.int32(5)
            )
        else:
            arm, suffix = method.split("_", 1)
            item = architecture[arm][case]
            # Feature recovery and all host feature construction are inside the
            # charged interval.  Only target-grid coordinates/masks and compiled
            # executables are cached setup.
            recovered, _ = c.recover_blob_parameters_fixed_sample(truth[case, 0], n)
            one = {
                "cx": recovered[0:1], "cy": recovered[1:2],
                "width": recovered[2:3], "amplitude": recovered[3:4],
                "nu": parameters["nu"][case:case + 1],
            }
            features = c.trajectory_features(one, n)[0]
            arguments = (
                jnp.asarray(features, jnp.float64), item["arguments"][1],
                item["arguments"][2], jnp.asarray(parameters["nu"][case], jnp.float64),
            )
            output = item["compiled"][suffix](*arguments)
        jax.block_until_ready(output)
        return output, float(time.perf_counter() - started)

    first_use = {}
    allocator_first_use = {}
    for method in methods:
        before = allocator_stats()
        output, elapsed = invoke(method, 0)
        after = allocator_stats()
        first_use[method] = elapsed
        allocator_first_use[method] = {"before": before, "after": after}
    canonical_work = {}
    for method in methods:
        if not (method.endswith("_mandatory") or method.endswith("_maximum_one")):
            continue
        canonical_work[method] = []
        for case in range(FOM_CASES):
            output, _ = invoke(method, case)
            canonical_work[method].append(online_work_record(method, case, output))
    for _ in range(TIME_WARM):
        for case in range(FOM_CASES):
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
        cases = list(range(FOM_CASES))
        cases = cases[repetition % FOM_CASES:] + cases[:repetition % FOM_CASES]
        for case in cases:
            for method in order:
                output, elapsed = invoke(method, case)
                item = {
                    "case_index": case, "repetition": repetition,
                    "elapsed_s": elapsed,
                }
                if method == "fom":
                    item.update(fom_grade(output, truth[case]))
                else:
                    fields = np.asarray(output[0])
                    item.update({
                        "finite": bool(np.all(np.isfinite(fields))),
                        "output_shape": list(fields.shape),
                    })
                records[method].append(item)
    summaries = {
        method: summarize_timing(items, FOM_CASES) for method, items in records.items()
    }
    position_counts = {
        method: [int(sum(order[position] == method for order in timing_orders))
                 for position in range(len(methods))]
        for method in methods
    }
    exact_position_balance = bool(all(
        counts == [1] * len(methods) for counts in position_counts.values()
    ))
    if not exact_position_balance:
        raise SystemExit("ten-method cyclic timing position balance failed")
    fom_summary = summaries["fom"]
    fom_accuracy = [item for item in records["fom"] if item["repetition"] == 0]
    fom_healthy = all(
        item["finite"] and item["breakdowns"] == 0 and item["flags_nonzero"] == 0
        and item["max_returned_relative_residual"] <= FOM_OUTER
        for item in records["fom"]
    )
    fom_mean_error = float(np.mean([
        item["trajectory_relative_l2"] for item in fom_accuracy
    ]))
    fom_worst_error = float(np.max([
        item["trajectory_relative_l2"] for item in fom_accuracy
    ]))
    fom_eligible = bool(
        fom_healthy and fom_mean_error <= 1e-3 and fom_worst_error <= 3e-3
    )
    report["cost_panel"].update({
        "setup": setup,
        "first_execution_after_explicit_compile_s": first_use,
        "runtime_allocator_first_execution_context": allocator_first_use,
        "burn_count": burn_count,
        "timing_orders": timing_orders, "timing_position_counts": position_counts,
        "exact_ten_method_position_balance": exact_position_balance,
        "records": records, "canonical_untimed_work": canonical_work,
        "summaries": summaries,
        "fom_same_invocation_accuracy": {
            "trajectory_error_mean": fom_mean_error,
            "trajectory_error_worst": fom_worst_error,
            "healthy": fom_healthy, "accuracy_eligible": fom_eligible,
        },
    })
    for candidate in s.CANDIDATES:
        arm = candidate["arm"]
        method = f"{arm}_mandatory"
        point_speedup = (
            fom_summary["median_elapsed_s"] / summaries[method]["median_elapsed_s"]
        )
        ci = clustered_speedup_ci(
            fom_summary["per_case_median_elapsed_s"],
            summaries[method]["per_case_median_elapsed_s"], 20260822 + candidate["R"],
        )
        memory_ok = all(
            item["eligibility_device_bytes"] <= 20_000_000_000
            for item in setup[arm]["memory_analysis"].values()
        )
        report["cost_panel"].setdefault("gates", {})[arm] = {
            "paired_median_speedup_mandatory_lower_bound": float(point_speedup),
            "clustered_speedup_ci": ci,
            "point_ge_10": bool(point_speedup >= 10.0),
            "clustered_lower_ge_8": bool(ci[0] >= 8.0),
            "compiled_peak_memory_le_20GB": bool(memory_ok),
            "paired_fom_accuracy_eligible": fom_eligible,
            "pass": bool(
                point_speedup >= 10.0 and ci[0] >= 8.0 and memory_ok and fom_eligible
            ),
        }


def smoke(report, arrays):
    n = 24
    coords = c.grid_coords(n)
    mask = c.binary_boundary_mask(n)
    fields = np.stack((
        c.bf.blob_ic(n, 0.42, 0.57, 0.11, 1.2),
        c.bf.blob_ic(n, 0.46, 0.54, 0.13, 1.0),
    ))
    for candidate in s.CANDIDATES:
        affine, coeff, prediction, info = s.fit_free_oracle(
            fields[0], coords, mask, candidate["R"]
        )
        normalized_affine = s.normalized_state_from_affine(affine)
        arrays[f"smoke_{candidate['arm']}_coeff"] = coeff
        report["free_oracle"][candidate["arm"]]["smoke"] = {
            "finite": bool(np.all(np.isfinite(prediction))),
            "fit_info": info, "affine": affine.tolist(),
            "normalized_affine": normalized_affine.tolist(),
        }
    candidate = s.CANDIDATES[0]
    recovered, sample_indices = c.recover_blob_parameters_fixed_sample(fields[0], n)
    expected = np.asarray((0.42, 0.57, 0.11, 1.2), np.float64)
    recovery_relative_error = float(
        np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
    )
    if recovery_relative_error > 1e-9:
        raise AssertionError("smoke fixed-sample 3x3 recovery failed")
    one = {
        "cx": recovered[0:1], "cy": recovered[1:2],
        "width": recovered[2:3], "amplitude": recovered[3:4],
        "nu": np.asarray((0.03,), np.float64),
    }
    features = c.trajectory_features(one, n)[0, :3]
    compiled, memory, build, geometry, arguments = compile_online_kernels(
        n, candidate, features, 0.03, 2
    )
    outputs = {}
    for name, executable in compiled.items():
        started = time.perf_counter()
        output = executable(*arguments)
        jax.block_until_ready(output)
        outputs[name] = {
            "elapsed_s": float(time.perf_counter() - started),
            "finite": bool(np.all(np.isfinite(np.asarray(output[0])))),
            "shape": list(np.asarray(output[0]).shape),
        }
    chain, _ = bc.make_chain(
        n, FOM_OUTER, lin_tol=FOM_INNER, preconditioner="helmholtz"
    )
    dummy = jnp.zeros((c.NUM_STEPS, n * n), jnp.float64)
    fom_output = chain(jnp.asarray(fields[0]), 0.03, dummy, jnp.int32(5))
    jax.block_until_ready(fom_output)
    report["smoke"] = {
        "N": n, "num_weak_steps": 2, "arm": candidate["arm"],
        "cold_sample_count": int(sample_indices.size), "online": outputs,
        "cold_recovery_relative_error": recovery_relative_error,
        "memory_analysis": memory, "weak_geometry": {
            "M": candidate["M"], "m": candidate["m"],
            "center_count": int(geometry["center_indices"].size),
            "rule": geometry["rule"],
        },
        "compile": build,
        "fom_finite": bool(all(np.all(np.isfinite(np.asarray(value))) for value in fom_output)),
    }


def main():
    c.require_gpu_highest()
    s.assert_locked_math()
    _, sample_indices_1024, sample_coords_1024 = c.fixed_sample_geometry(1024)
    if sample_indices_1024.size > 4096 or sample_coords_1024.shape[0] > 4096:
        raise AssertionError("N=1024 cold-start geometry is not hyper-reduced")
    if not SMOKE and TIME_REPS != 10:
        raise SystemExit("scientific S0 requires ten cyclic timing repetitions")
    report = {
        "stage": "adaptive Phase-2 S0 free-spline and charged weak/FOM cost oracle",
        "status": "excluded_execution_smoke" if SMOKE else "running",
        "config": {
            "smoke": SMOKE, "f64": True, "dt": c.DT, "num_steps": c.NUM_STEPS,
            "domain": [-s.DOMAIN_HALF_WIDTH, s.DOMAIN_HALF_WIDTH], "degree": s.DEGREE,
            "selection": [
                {"N": n, "seed": 0, "draw_count": 704,
                 "indices": list(range(start, start + count))}
                for n, start, count in SELECTION
            ],
            "fom_panel": {
                "N": 1024, "seed": FOM_SEED, "draw_count": FOM_DRAW_COUNT,
                "indices": list(range(FOM_CASES)), "outer": FOM_OUTER,
                "inner": FOM_INNER, "history": "cubic",
                "preconditioner": "exact Helmholtz",
            },
            "candidate_bracket": [
                {
                    **candidate,
                    "q": candidate["k"] - 5,
                    "hyperdecoder_parameters": s.hyperdecoder_parameter_count(
                        candidate["R"], candidate["k"]
                    ),
                    "predictor_parameters": s.predictor_parameter_count(candidate["k"]),
                    "analytic_screen": s.analytic_screen(candidate["R"]),
                    "initialization_seed": ARM_SEEDS[candidate["arm"]],
                } for candidate in s.CANDIDATES
            ],
            "model_validation_touched": False, "confirmation_touched": False,
            "cold_start_geometry_assertion": {
                "N": 1024, "sample_count": int(sample_indices_1024.size),
                "coordinate_count": int(sample_coords_1024.shape[0]),
                "full_grid_coordinates_allocated": False,
                "recovery": "interior maximum then log-quadratic 3x3 sampled patch",
                "recovery_relative_error_gate": 1e-9,
            },
        },
        "provenance": c.provenance(),
        "selection_reference_health": {},
        "free_oracle": {
            candidate["arm"]: {"candidate": candidate, "meshes": {}}
            for candidate in s.CANDIDATES
        },
        "cost_panel": {},
    }
    arrays = {}
    c.save_json(OUTPUT_JSON, report)
    if SMOKE:
        smoke(report, arrays)
    else:
        run_free_oracles(report, arrays)
        c.save_json(OUTPUT_JSON, report)
        run_cost_panel(report)
        for candidate in s.CANDIDATES:
            arm = candidate["arm"]
            free = report["free_oracle"][arm]["summary"]
            cost = report["cost_panel"]["gates"][arm]
            report["free_oracle"][arm]["promote"] = bool(
                free["every_mesh_and_pooled_accuracy_pass"]
                and free["zero_unhealthy_fits"] and free["exact_binary_boundary"]
                and cost["pass"]
            )
        promoted = [
            candidate["arm"] for candidate in s.CANDIDATES
            if report["free_oracle"][candidate["arm"]]["promote"]
        ]
        report["decision"] = {
            "promoted_arms": promoted,
            "selected_smallest_promoted": promoted[0] if promoted else None,
            "phase2_hard_stop": not bool(promoted),
        }
        report["status"] = "complete"
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **arrays)
    report["npz"] = {
        "path": os.path.basename(OUTPUT_NPZ), "sha256": c.sha256(OUTPUT_NPZ)
    }
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps({
        "status": report["status"], "decision": report.get("decision"),
        "smoke": report.get("smoke"),
    }, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
