#!/usr/bin/env python
"""Phase-4 diagnostic: hierarchical spline oracle and paired K3 cost."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase3 as p3
import b10_phase4 as p4
import b10_s0_spline as base
import b10_spline as s


SELECTION = {64: (512, 64), 128: (512, 32), 256: (512, 16)}
FOM_CASES = 4
TIME_REPS = 20
TIME_WARM = 1
BURN_SECONDS = 3.0
NO_REGRESSION = 1e-5
IDENTITY_TOL = 2e-14
ORACLE_WORKERS = int(os.environ.get("B10_ORACLE_WORKERS", "8"))
ARM_SEEDS = {"H1": 20264032, "H2": 20264048}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            rows[relative.removeprefix("./")] = digest
    return rows


def validate_bound_artifact(json_path, npz_path, audit_path, phase):
    report, audit = load_json(json_path), load_json(audit_path)
    if report.get("status") != "complete" or audit.get("status") != "pass":
        raise SystemExit(f"{phase} artifact chain is not independently complete")
    if report["npz"]["sha256"] != c.sha256(npz_path):
        raise SystemExit(f"{phase} NPZ checksum mismatch")
    if audit.get("source_json_sha256") != c.sha256(json_path):
        raise SystemExit(f"{phase} audit JSON binding mismatch")
    if audit.get("source_npz_sha256") != c.sha256(npz_path):
        raise SystemExit(f"{phase} audit NPZ binding mismatch")
    if audit.get("decision") != report.get("decision"):
        raise SystemExit(f"{phase} audit decision mismatch")
    expected = (
        phase == "S0" and report["decision"].get("phase2_hard_stop") is True
    ) or (
        phase == "P3" and report["decision"] == {
            "selected_solver": None, "selected_kernel": "K3",
            "run_P3_F": False, "phase3_hard_stop": True,
        }
    )
    if not expected:
        raise SystemExit(f"{phase} immutable decision is unexpected")
    return report, audit


def binding(
    json_path, npz_path, audit_path, report, audit, manifest_path=None,
):
    value = {
        "json": os.path.abspath(json_path),
        "json_sha256": c.sha256(json_path),
        "npz": os.path.abspath(npz_path),
        "npz_sha256": c.sha256(npz_path),
        "audit": os.path.abspath(audit_path),
        "audit_sha256": c.sha256(audit_path),
        "commit": report["provenance"]["commit"],
        "job_id": report["provenance"]["slurm_job_id"],
        "audit_decision": audit["decision"],
    }
    if manifest_path is not None:
        value.update({
            "staged_manifest": os.path.abspath(manifest_path),
            "staged_manifest_sha256": c.sha256(manifest_path),
        })
    return value


def trajectory_metrics(prediction, truth):
    differences = np.asarray(prediction) - np.asarray(truth)
    numerator = np.sum(differences * differences, axis=2)
    denominator = np.sum(np.asarray(truth) ** 2, axis=2)
    snapshot = np.sqrt(numerator / np.maximum(denominator, 1e-300))
    trajectory = np.sqrt(
        np.sum(numerator, axis=1) / np.maximum(np.sum(denominator, axis=1), 1e-300)
    )
    return trajectory, snapshot, numerator, denominator


def basis_identity(candidate, smoke=False):
    rng = np.random.default_rng(20260820 + int(candidate["P"]))
    probes = np.concatenate((
        np.asarray((-4.5, 4.5, -2.0, -1.5, 0.0, 1.5, 2.0)),
        np.nextafter(np.asarray((-4.5, -2.0, -1.5, 1.5, 2.0, 4.5)), -np.inf),
        np.nextafter(np.asarray((-4.5, -2.0, -1.5, 1.5, 2.0, 4.5)), np.inf),
        rng.uniform(-4.6, 4.6, 16 if smoke else 128),
    )).astype(np.float64)
    window_np = p4.taper_1d_np(probes)
    window_jax = np.asarray(jax.jit(p4.taper_1d_jax)(jnp.asarray(probes)))
    window_error = float(np.max(np.abs(window_np - window_jax)))
    rows = {}
    for name, count, values in (
        ("coarse", p4.GLOBAL_R, probes),
        ("fine", int(candidate["P"]), probes * p4.FINE_COORDINATE_SCALE),
    ):
        table = jnp.asarray(p3.span_polynomial_table_np(count), jnp.float64)
        control_index, control_weight = jax.vmap(
            lambda value: s._basis_1d_jax(value, count)
        )(jnp.asarray(values))
        poly_index, poly_weight = jax.vmap(
            lambda value: p3._basis_1d_polynomial_jax(value, table, count)
        )(jnp.asarray(values))
        control_index, control_weight = map(np.asarray, (control_index, control_weight))
        poly_index, poly_weight = map(np.asarray, (poly_index, poly_weight))
        rows[name] = {
            "probe_count": int(values.size),
            "support_indices_exact": bool(np.array_equal(control_index, poly_index)),
            "weight_max_abs": float(np.max(np.abs(control_weight - poly_weight))),
            "max_nonzero_support": int(np.max(np.sum(control_weight != 0.0, axis=1))),
        }
    passed = bool(
        window_error <= IDENTITY_TOL
        and p4.taper_1d_np(np.asarray((-2.0, 2.0))).tolist() == [0.0, 0.0]
        and p4.taper_1d_np(np.asarray((-1.5, 1.5))).tolist() == [1.0, 1.0]
        and all(
            row["support_indices_exact"]
            and row["weight_max_abs"] <= IDENTITY_TOL
            and row["max_nonzero_support"] <= 4
            for row in rows.values()
        )
    )
    return {
        "seed": 20260820 + int(candidate["P"]), "routes": rows,
        "window_max_abs": window_error, "max_tensor_support": 32,
        "exact_taper_endpoints": True, "pass": passed,
    }


def pallas_basis_identity(candidate, smoke=False):
    """Exercise the exact support/weight arithmetic used by hierarchical K3."""
    if not p3.pallas_is_available():
        return {"available": False, "executed": False, "pass": False}
    result = {"available": True, "executed": True, "routes": {}}
    for name, count in (("coarse", p4.GLOBAL_R), ("fine", int(candidate["P"]))):
        knots = np.unique(s.knots_np(count))
        rng = np.random.default_rng(20264150 + count)
        probes = np.unique(np.concatenate((
            knots, np.nextafter(knots, -np.inf),
            np.nextafter(knots, np.inf),
            rng.uniform(-4.6, 4.6, 32 if smoke else 4096),
        )))
        reference_indices, reference_weights = jax.vmap(
            lambda value: s._basis_1d_jax(value, count)
        )(jnp.asarray(probes))
        padded_count = (
            (probes.size + p3.PALLAS_BLOCK - 1) // p3.PALLAS_BLOCK
            * p3.PALLAS_BLOCK
        )
        padded = np.pad(probes, (0, padded_count - probes.size))
        probe = jax.jit(
            p4.make_pallas_hierarchical_basis_probe(padded_count, count)
        )
        indices, weights = probe(
            jnp.asarray(padded),
            jnp.asarray(p3.span_polynomial_table_np(count)),
        )
        jax.block_until_ready((indices, weights))
        indices, weights = np.asarray(indices)[:probes.size], np.asarray(weights)[:probes.size]
        reference_indices, reference_weights = np.asarray(reference_indices), np.asarray(reference_weights)
        row = {
            "probe_count": int(probes.size),
            "support_indices_exact": bool(np.array_equal(indices, reference_indices)),
            "weight_max_abs": float(np.max(np.abs(weights - reference_weights))),
            "local_support": 4,
        }
        row["pass"] = bool(row["support_indices_exact"] and row["weight_max_abs"] <= IDENTITY_TOL)
        result["routes"][name] = row
    result["pass"] = bool(all(row["pass"] for row in result["routes"].values()))
    return result


def fit_fields(candidate, fields, coords, mask):
    shape = fields.shape
    flat = fields.reshape(-1, fields.shape[-1])
    coefficient = np.empty((flat.shape[0], p4.coefficient_count(candidate)))
    affine = np.empty((flat.shape[0], 5))
    predicted = np.empty_like(flat)
    normal = np.empty(flat.shape[0])
    elapsed = np.empty(flat.shape[0])
    healthy = np.empty(flat.shape[0], bool)
    boundary = np.empty(flat.shape[0], bool)
    pou = np.empty(flat.shape[0])
    support = np.empty(flat.shape[0], np.int64)

    def fit_one(field):
        return p4.fit_hierarchical_oracle(field, coords, mask, candidate)

    if ORACLE_WORKERS == 1:
        fitted, executor = map(fit_one, flat), None
    else:
        executor = ThreadPoolExecutor(max_workers=ORACLE_WORKERS)
        fitted = executor.map(fit_one, flat, chunksize=1)
    for index, (one_affine, one_coefficient, prediction, info) in enumerate(fitted):
        affine[index], coefficient[index], predicted[index] = (
            one_affine, one_coefficient, prediction
        )
        normal[index], elapsed[index] = (
            info["relative_normal_residual"], info["elapsed_s"]
        )
        healthy[index], boundary[index] = info["healthy"], info["boundary_exact"]
        pou[index], support[index] = info["pou_max_abs"], info["max_local_support"]
    if executor is not None:
        executor.shutdown(wait=True)
    return {
        "affine": affine.reshape(shape[:-1] + (5,)),
        "coefficients": coefficient.reshape(shape[:-1] + (coefficient.shape[-1],)),
        "prediction": predicted.reshape(shape),
        "normal": normal.reshape(shape[:-1]),
        "elapsed": elapsed.reshape(shape[:-1]),
        "healthy": healthy.reshape(shape[:-1]),
        "boundary": boundary.reshape(shape[:-1]),
        "pou": pou.reshape(shape[:-1]),
        "support": support.reshape(shape[:-1]),
    }


def run_free_oracle(s0_npz_path=None, p3_report=None, smoke=False):
    arrays, report = {}, {}
    candidates = p4.CANDIDATES[:1] if smoke else p4.CANDIDATES
    pooled = {item["arm"]: [] for item in candidates}
    old = None if smoke else np.load(s0_npz_path)
    p3_s2 = {} if smoke else {
        (int(row["N"]), int(row["draw_index"]), int(row["time_index"])):
            float(row["field_relative_l2"])
        for row in p3_report["solver_diagnostic"]["records"]["S2"]
    }
    if smoke:
        meshes = {24: (0, 1)}
    else:
        meshes = SELECTION
    for candidate in candidates:
        arm = candidate["arm"]
        report[arm] = {"candidate": candidate, "basis_identity": basis_identity(candidate, smoke), "meshes": {}}
    for n, (start, count) in meshes.items():
        if smoke:
            coords, mask = c.grid_coords(n), c.binary_boundary_mask(n)
            parameters = {
                "cx": np.asarray((0.42,)), "cy": np.asarray((0.58,)),
                "width": np.asarray((0.12,)), "amplitude": np.asarray((1.2,)),
                "nu": np.asarray((0.01,)),
            }
            truth = np.asarray(c.bf.blob_ic(
                n, parameters["cx"][0], parameters["cy"][0],
                parameters["width"][0], parameters["amplitude"][0],
            ))[None, None, :]
            health = {"smoke": True}
        else:
            truth, _, health = c.generate_population(
                n, 0, 704, np.arange(start, start + count), chunk=4
            )
            coords, mask = c.grid_coords(n), c.binary_boundary_mask(n)
            if not (
                np.isfinite(health["reported_max_relative_residual"])
                and np.isfinite(health["independent_max_relative_residual"])
                and health["reported_max_relative_residual"] <= 1e-8
                and health["independent_max_relative_residual"] <= 1e-8
            ):
                raise SystemExit(f"N{n} legacy selection truth health gate failed")
        flat_truth = truth.reshape(truth.shape[0], truth.shape[1], n * n)
        for candidate in candidates:
            arm = candidate["arm"]
            started = time.perf_counter()
            fitted = fit_fields(candidate, flat_truth, coords, mask)
            trajectory, snapshot, numerator, denominator = trajectory_metrics(
                fitted["prediction"], flat_truth
            )
            pooled[arm].extend(trajectory.tolist())
            key = f"{arm}_N{n}"
            for name in ("affine", "coefficients", "normal", "elapsed", "healthy", "boundary", "pou", "support"):
                arrays[f"{key}_{name}"] = fitted[name]
            arrays[f"{key}_trajectory_error"] = trajectory
            arrays[f"{key}_snapshot_error"] = snapshot
            arrays[f"{key}_error_numerator_sq"] = numerator
            arrays[f"{key}_truth_norm_sq"] = denominator
            s0_pass = True
            p3_pass = True
            p3_comparisons = []
            if not smoke:
                old_error = np.asarray(old[f"C_N{n}_snapshot_error"])
                arrays[f"{key}_s0_snapshot_error"] = old_error
                arrays[f"{key}_s0_delta"] = snapshot - old_error
                s0_pass = bool(np.all(snapshot <= old_error + NO_REGRESSION))
                for (mesh, draw, time_index), value in p3_s2.items():
                    if mesh != n:
                        continue
                    row = draw - start
                    observed = float(snapshot[row, time_index])
                    p3_comparisons.append({
                        "draw_index": draw, "time_index": time_index,
                        "p4_error": observed, "p3_s2_error": value,
                        "no_regression": bool(observed <= value + NO_REGRESSION),
                    })
                p3_pass = bool(all(row["no_regression"] for row in p3_comparisons))
            report[arm]["meshes"][str(n)] = {
                "indices": list(range(start, start + count)),
                "trajectory_mean": float(np.mean(trajectory)),
                "trajectory_worst": float(np.max(trajectory)),
                "snapshot_mean": float(np.mean(snapshot)),
                "snapshot_worst": float(np.max(snapshot)),
                "fit_count": int(fitted["healthy"].size),
                "healthy_count": int(np.sum(fitted["healthy"])),
                "zero_unhealthy": bool(np.all(fitted["healthy"])),
                "normal_worst": float(np.max(fitted["normal"])),
                "exact_boundary": bool(np.all(fitted["boundary"])),
                "pou_worst": float(np.max(fitted["pou"])),
                "max_support": int(np.max(fitted["support"])),
                "s0_no_regression": s0_pass,
                "p3_fixed_subset_no_regression": p3_pass,
                "p3_fixed_subset": p3_comparisons,
                "truth_health": health,
                "elapsed_s": float(time.perf_counter() - started),
            }
    for candidate in candidates:
        arm = candidate["arm"]
        values = np.asarray(pooled[arm])
        meshes = report[arm]["meshes"].values()
        target = None
        if not smoke:
            target = report[arm]["meshes"]["128"]["indices"].index(530)
            target = float(arrays[f"{arm}_N128_trajectory_error"][target])
        summary = {
            "trajectory_mean": float(np.mean(values)),
            "trajectory_worst": float(np.max(values)),
            "all_mesh_and_pooled_accuracy_pass": bool(
                smoke or (
                    np.mean(values) <= 2e-4 and np.max(values) <= 7e-4
                    and all(row["trajectory_mean"] <= 2e-4 and row["trajectory_worst"] <= 7e-4 for row in meshes)
                )
            ),
            "zero_unhealthy": bool(all(row["zero_unhealthy"] for row in meshes)),
            "exact_boundary_pou_support": bool(
                report[arm]["basis_identity"]["pass"]
                and all(row["exact_boundary"] and row["pou_worst"] <= 1e-15 and row["max_support"] == 32 for row in meshes)
            ),
            "s0_no_regression": bool(all(row["s0_no_regression"] for row in meshes)),
            "p3_fixed_subset_no_regression": bool(all(row["p3_fixed_subset_no_regression"] for row in meshes)),
            "N128_draw530_trajectory": target,
            "N128_draw530_pass": bool(smoke or (
                target <= 7e-4
                and target <= p3_report["solver_diagnostic"]["summaries"]["S2"]["target_N128_draw530_trajectory_relative_l2"] + NO_REGRESSION
            )),
        }
        summary["pass"] = bool(
            summary["all_mesh_and_pooled_accuracy_pass"]
            and summary["zero_unhealthy"]
            and summary["exact_boundary_pou_support"]
            and summary["s0_no_regression"]
            and summary["p3_fixed_subset_no_regression"]
            and summary["N128_draw530_pass"]
        )
        report[arm]["summary"] = summary
    return report, arrays


def relative_l2(value, control):
    return float(
        np.linalg.norm(np.asarray(value) - np.asarray(control))
        / max(np.linalg.norm(np.asarray(control)), 1e-300)
    )


def work_record(method, case, output):
    fields = np.asarray(output[0])
    item = {"method": method, "case_index": case, "finite": bool(np.all(np.isfinite(fields))), "output_shape": list(fields.shape)}
    residual = np.asarray(output[3])
    if method.endswith("mandatory"):
        rho = np.asarray(output[4])
        item.update({
            "rho_all": rho.tolist(), "weak_residual_norm_all": np.linalg.norm(residual, axis=1).tolist(),
            "weak_objective_evaluations": int(rho.size), "weak_jacobian_evaluations": 0,
            "trial_residual_evaluations": 0, "zero_failures": bool(np.all(np.isfinite(rho))),
            "coefficient_grid_evaluations": int(rho.size + 1),
        })
    else:
        rho_before, rho_after, jacobian = map(np.asarray, output[4:7])
        factors, step_norm, accepted = map(np.asarray, output[7:10])
        item.update({
            "rho_before_all": rho_before.tolist(), "rho_after_all": rho_after.tolist(),
            "weak_residual_norm_all": np.linalg.norm(residual, axis=1).tolist(),
            "jacobian_frobenius_all": np.linalg.norm(jacobian.reshape(jacobian.shape[0], -1), axis=1).tolist(),
            "trial_factor_all": factors.tolist(), "bounded_step_norm_all": step_norm.tolist(),
            "accepted_all": accepted.tolist(), "weak_objective_evaluations": int(rho_before.size),
            "weak_jacobian_evaluations": int(rho_before.size),
            "trial_residual_evaluations": int(4 * rho_before.size),
            # Per step: previous+target, one differentiated target, four
            # trials = seven; final corrected full decode adds (steps+1).
            "coefficient_grid_evaluations": int(8 * rho_before.size + 1),
            "zero_failures": bool(np.all(np.isfinite(rho_before)) and np.all(np.isfinite(rho_after)) and np.all(np.isfinite(jacobian))),
        })
    item["finite"] = bool(item["finite"] and np.all(np.isfinite(residual)))
    return item


def run_cost(smoke=False):
    candidates = p4.CANDIDATES[:1] if smoke else p4.CANDIDATES
    n, num_steps = (32, 1) if smoke else (1024, c.NUM_STEPS)
    if smoke:
        parameters = {"cx": np.asarray((0.42,)), "cy": np.asarray((0.58,)), "width": np.asarray((0.12,)), "amplitude": np.asarray((1.2,)), "nu": np.asarray((0.01,))}
        truth = np.asarray(c.bf.blob_ic(
            n, parameters["cx"][0], parameters["cy"][0],
            parameters["width"][0], parameters["amplitude"][0],
        ))[None, None, :]
        features_by_case = [c.trajectory_features(parameters, n)[0, :2]]
        case_count, reference, dummy = 1, None, None
    else:
        truth, parameters, health, dummy = base.generate_live_reference(n)
        reference = health
        features_by_case = []
        representative_inputs = []
        for case in range(FOM_CASES):
            recovered, sample_indices = c.recover_blob_parameters_fixed_sample(truth[case, 0], n)
            expected = np.asarray((parameters["cx"][case], parameters["cy"][case], parameters["width"][case], parameters["amplitude"][case]))
            error = float(np.linalg.norm(recovered - expected) / np.linalg.norm(expected))
            if sample_indices.size > 4096 or error > 1e-9:
                raise SystemExit("cold recovery gate failed")
            one = {"cx": recovered[0:1], "cy": recovered[1:2], "width": recovered[2:3], "amplitude": recovered[3:4], "nu": parameters["nu"][case:case + 1]}
            features_by_case.append(c.trajectory_features(one, n)[0])
            representative_inputs.append({
                "case_index": case, "source_draw_index": case,
                "cold_sample_count": int(sample_indices.size),
                "recovered_parameters": recovered.tolist(),
                "expected_parameters": expected.tolist(),
                "recovery_relative_error": error,
                "viscosity": float(parameters["nu"][case]),
                "features": features_by_case[-1].tolist(),
            })
        case_count = FOM_CASES
    if smoke:
        representative_inputs = [{
            "case_index": 0, "synthetic": True,
            "features": features_by_case[0].tolist(),
        }]
    architecture, setup, identities = {}, {}, {}
    for candidate in candidates:
        arm = candidate["arm"]
        try:
            pallas_probe = pallas_basis_identity(candidate, smoke)
        except Exception as error:
            pallas_probe = {"available": p3.pallas_is_available(), "executed": False, "pass": False, "error": f"{type(error).__name__}: {error}"}
        if not pallas_probe["pass"]:
            raise SystemExit(f"{arm} hierarchical K3 support/weight probe failed")
        compiled, one_setup, geometry, arguments = p4.compile_online_kernels(
            n, candidate, features_by_case[0], parameters["nu"][0], num_steps, ARM_SEEDS[arm]
        )
        architecture[arm] = {"compiled": compiled, "coords": arguments[1], "mask": arguments[2]}
        setup[arm] = {"kernels": one_setup, "pallas_basis_identity": pallas_probe, "representative_inputs": representative_inputs, "weak_geometry": {"M": candidate["M"], "m": candidate["m"], "rule": geometry["rule"]}}
        identities[arm] = []
        for case in range(case_count):
            args = (jnp.asarray(features_by_case[case]), arguments[1], arguments[2], jnp.asarray(parameters["nu"][case]))
            control = compiled["identity_cox"](*args)
            observed = compiled["identity_k3"](*args)
            jax.block_until_ready((control, observed))
            names = ("full_fields", "current_stencils", "previous_centers", "weak_residual", "rho")
            relative = {name: relative_l2(value, baseline) for name, value, baseline in zip(names, observed, control)}
            exact_boundary = bool(np.all(np.asarray(observed[0])[:, c.binary_boundary_mask(n) == 0] == 0.0))
            identities[arm].append({"case_index": case, "relative_l2": relative, "max_relative_l2": max(relative.values()), "exact_boundary": exact_boundary, "pass": bool(max(relative.values()) <= IDENTITY_TOL and exact_boundary)})
    methods = [] if smoke else ["fom"]
    methods += [f"{candidate['arm']}_{suffix}" for candidate in candidates for suffix in ("mandatory", "maximum_one")]
    fom = None if smoke else base.bc.make_chain(n, base.FOM_OUTER, lin_tol=base.FOM_INNER, preconditioner="helmholtz")[0]

    def invoke(method, case):
        started = time.perf_counter()
        if method == "fom":
            output = fom(jnp.asarray(truth[case, 0]), parameters["nu"][case], dummy, jnp.int32(5))
        else:
            arm, suffix = method.split("_", 1)
            if smoke:
                features = features_by_case[case]
            else:
                recovered, sample_indices = c.recover_blob_parameters_fixed_sample(
                    truth[case, 0], n
                )
                if sample_indices.size > 4096 or not np.all(np.isfinite(recovered)):
                    raise SystemExit("charged cold recovery failed")
                one = {
                    "cx": recovered[0:1], "cy": recovered[1:2],
                    "width": recovered[2:3], "amplitude": recovered[3:4],
                    "nu": parameters["nu"][case:case + 1],
                }
                features = c.trajectory_features(one, n)[0]
            output = architecture[arm]["compiled"][suffix](jnp.asarray(features), architecture[arm]["coords"], architecture[arm]["mask"], jnp.asarray(parameters["nu"][case]))
        jax.block_until_ready(output)
        return output, float(time.perf_counter() - started)

    first_execution, canonical = {}, {}
    for method in methods:
        _, first_execution[method] = invoke(method, 0)
    for method in methods:
        if method == "fom":
            continue
        canonical[method] = [work_record(method, case, invoke(method, case)[0]) for case in range(case_count)]
    if smoke:
        gates = {}
        for candidate in candidates:
            arm = candidate["arm"]
            gates[arm] = {"scientific_promotion_allowed": False, "identity_pass": all(row["pass"] for row in identities[arm]), "canonical_work_pass": all(row["finite"] and row["zero_failures"] and row["weak_objective_evaluations"] == num_steps for suffix in ("mandatory", "maximum_one") for row in canonical[f"{arm}_{suffix}"])}
        return {"status": "excluded_execution_smoke_pass", "setup": setup, "identity": identities, "first_execution_after_compile_s": first_execution, "canonical_work": canonical, "gates": gates, "selected_cost_arm": None}
    for _ in range(TIME_WARM):
        for case in range(case_count):
            for method in methods:
                invoke(method, case)
    burn_count = c.gpu_burn(BURN_SECONDS)
    records = {method: [] for method in methods}
    orders = []
    for repetition in range(TIME_REPS):
        offset = repetition % len(methods)
        order = methods[offset:] + methods[:offset]
        if (repetition // len(methods)) % 2:
            order = list(reversed(order))
        orders.append(order)
        cases = list(range(case_count))
        cases = cases[repetition % case_count:] + cases[:repetition % case_count]
        for case in cases:
            for method in order:
                output, elapsed = invoke(method, case)
                row = {"case_index": case, "repetition": repetition, "elapsed_s": elapsed}
                if method == "fom":
                    row.update(base.fom_grade(output, truth[case]))
                else:
                    row["finite"] = bool(np.all(np.isfinite(np.asarray(output[0]))))
                records[method].append(row)
    position_counts = {method: [sum(order[position] == method for order in orders) for position in range(len(methods))] for method in methods}
    exact_balance = bool(all(value == 4 for counts in position_counts.values() for value in counts))
    if not exact_balance:
        raise SystemExit("P4-D exact timing balance failed")
    summaries = {method: base.summarize_timing(rows, case_count) for method, rows in records.items()}
    fom_rows = [row for row in records["fom"] if row["repetition"] == 0]
    fom_mean = float(np.mean([row["trajectory_relative_l2"] for row in fom_rows]))
    fom_worst = float(np.max([row["trajectory_relative_l2"] for row in fom_rows]))
    fom_healthy = bool(all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0 and row["max_returned_relative_residual"] <= base.FOM_OUTER for row in records["fom"]))
    fom_eligible = bool(fom_healthy and fom_mean <= 1e-3 and fom_worst <= 3e-3)
    gates = {}
    for candidate in candidates:
        arm = candidate["arm"]
        one = {"fom_eligible": fom_eligible, "identity_pass": all(row["pass"] for row in identities[arm])}
        for suffix in ("mandatory", "maximum_one"):
            method = f"{arm}_{suffix}"
            speed = summaries["fom"]["median_elapsed_s"] / summaries[method]["median_elapsed_s"]
            ci = base.clustered_speedup_ci(summaries["fom"]["per_case_median_elapsed_s"], summaries[method]["per_case_median_elapsed_s"], 20264100 + (0 if arm == "H1" else 20) + (0 if suffix == "mandatory" else 1))
            work_pass = bool(len(canonical[method]) == 4 and all(row["finite"] and row["zero_failures"] and row["weak_objective_evaluations"] == 50 and row["weak_jacobian_evaluations"] == (0 if suffix == "mandatory" else 50) and row["trial_residual_evaluations"] == (0 if suffix == "mandatory" else 200) for row in canonical[method]))
            memory_pass = setup[arm]["kernels"][suffix]["memory_analysis"]["eligibility_device_bytes"] <= 20_000_000_000
            one[suffix] = {"paired_median_speedup": float(speed), "clustered_speedup_ci": ci, "canonical_work_pass": work_pass, "memory_pass": memory_pass, "pass": bool(fom_eligible and one["identity_pass"] and work_pass and memory_pass and speed >= 10 and ci[0] >= 8)}
        one["training_cost_license"] = one["mandatory"]["pass"]
        one["correction_classification"] = "correction-capable" if one["maximum_one"]["pass"] else ("conditional-zero-or-occasional-attempt" if one["mandatory"]["pass"] else "cost-fail")
        gates[arm] = one
    return {"status": "complete", "reference_health": reference, "setup": setup, "identity": identities, "first_execution_after_compile_s": first_execution, "canonical_work": canonical, "burn_count": burn_count, "timing_orders": orders, "position_counts": position_counts, "exact_position_balance": exact_balance, "records": records, "summaries": summaries, "fom_accuracy": {"mean": fom_mean, "worst": fom_worst, "healthy": fom_healthy, "eligible": fom_eligible}, "gates": gates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--s0-json")
    parser.add_argument("--s0-npz")
    parser.add_argument("--s0-audit")
    parser.add_argument("--p3-json")
    parser.add_argument("--p3-npz")
    parser.add_argument("--p3-audit")
    parser.add_argument("--p3-manifest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    if args.smoke:
        free, arrays = run_free_oracle(smoke=True)
        cost = run_cost(smoke=True)
        bindings = None
        status = "excluded_execution_smoke_pass"
    else:
        required = (args.s0_json, args.s0_npz, args.s0_audit, args.p3_json, args.p3_npz, args.p3_audit, args.p3_manifest)
        if any(value is None for value in required):
            raise SystemExit("scientific P4-D requires complete S0 and P3 chains")
        s0_report, s0_audit = validate_bound_artifact(args.s0_json, args.s0_npz, args.s0_audit, "S0")
        p3_report, p3_audit = validate_bound_artifact(args.p3_json, args.p3_npz, args.p3_audit, "P3")
        prior_manifest = manifest_rows(args.p3_manifest)
        film_key = "code/deps/burgers2d-coord-rom/burgers2d_film.py"
        bh_key = "code/bh_common.py"
        film_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "deps", "burgers2d-coord-rom", "burgers2d_film.py",
        )
        bh_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "bh_common.py"
        )
        if prior_manifest.get(film_key) != c.sha256(film_path):
            raise SystemExit("external Burgers FOM source differs from immutable P3 manifest")
        if prior_manifest.get(bh_key) != c.sha256(bh_path):
            raise SystemExit("bh_common differs from immutable P3 manifest")
        free, arrays = run_free_oracle(args.s0_npz, p3_report, smoke=False)
        cost = run_cost(smoke=False)
        bindings = {
            "S0": binding(
                args.s0_json, args.s0_npz, args.s0_audit,
                s0_report, s0_audit,
            ),
            "P3": binding(
                args.p3_json, args.p3_npz, args.p3_audit,
                p3_report, p3_audit, args.p3_manifest,
            ),
            "runtime_dependencies": {
                "burgers2d_film_sha256": c.sha256(film_path),
                "bh_common_sha256": c.sha256(bh_path),
                "source_manifest": "P3",
            },
        }
        status = "complete"
    selected = None
    if not args.smoke:
        passing = [candidate["arm"] for candidate in p4.CANDIDATES if free[candidate["arm"]]["summary"]["pass"] and cost["gates"][candidate["arm"]]["training_cost_license"]]
        selected = passing[0] if passing else None
    decision = {"selected_spatial_arm": selected, "training_seed11_k24_licensed": selected is not None, "correction_classification": None if selected is None else cost["gates"][selected]["correction_classification"], "phase4_hard_stop": bool(not args.smoke and selected is None), "scientific_promotion_allowed": not args.smoke}
    np.savez_compressed(args.output_npz, **arrays)
    report = {
        "status": status, "provenance": c.provenance(),
        "config": {"candidates": list(p4.CANDIDATES), "selection": {str(n): {"start": start, "count": count, "times": 51} for n, (start, count) in SELECTION.items()}, "oracle_workers": ORACLE_WORKERS, "normal_tolerance": s.ORACLE_NORMAL_TOL, "no_regression_absolute": NO_REGRESSION, "identity_tolerance": IDENTITY_TOL, "fom_cases": FOM_CASES, "fom_seed": base.FOM_SEED, "time_repetitions": TIME_REPS, "time_warmups": TIME_WARM, "burn_seconds": BURN_SECONDS, "correction_radius": 0.25, "lm": 1e-6, "trial_factors": [1.0, 0.5, 0.25, 0.0], "model_validation_touched": False, "confirmation_touched": False, "smoke": args.smoke, "f64": True, "matmul_precision": "highest"},
        "bindings": bindings, "free_oracle": free, "cost_panel": cost,
        "decision": decision, "elapsed_s": float(time.perf_counter() - started),
        "npz": {"basename": os.path.basename(args.output_npz), "sha256": c.sha256(args.output_npz)},
    }
    c.save_json(args.output_json, report)
    c.log({"status": status, "decision": decision, "elapsed_s": report["elapsed_s"]})


if __name__ == "__main__":
    main()
