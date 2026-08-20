#!/usr/bin/env python
"""One-cell Phase-6 G1 Cox-weak/K3-full diagnostic."""
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
import b10_phase4_d as p4d
import b10_phase5 as p5
import b10_phase6 as p6
import b10_s0_spline as base


TIME_REPS = 24
TIME_WARM = 1
BURN_SECONDS = 3.0
IDENTITY_TOL = 2e-14
MEMORY_LIMIT = 20_000_000_000
CI_SEED = 20266100
METHODS = ("fom", "R0_polynomial_weak", "R1_cox_weak_k3_full")
EXPECTED_P5 = {
    "json": "97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96",
    "npz": "5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54",
    "audit": "c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff",
    "manifest": "6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6",
    "commit": "e18edda9b124be8f7fa21c07ef21804c8dbecc48",
    "job": "2669249",
}


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


def validate_p5_chain(json_path, npz_path, audit_path, manifest_path):
    paths = {
        "json": json_path, "npz": npz_path,
        "audit": audit_path, "manifest": manifest_path,
    }
    for name, path in paths.items():
        if c.sha256(path) != EXPECTED_P5[name]:
            raise SystemExit(f"immutable P5-D {name} hash mismatch")
    report, audit = load_json(json_path), load_json(audit_path)
    if report.get("status") != "complete" or audit.get("status") != "pass":
        raise SystemExit("P5-D chain is incomplete")
    expected_decision = {
        "target_integrity_pass": True,
        "arm_training_licenses": {"G1": False, "G2": False},
        "next_seed11_arm": None,
        "phase5_hard_stop": True,
        "scientific_promotion_allowed": True,
    }
    provenance = report["provenance"]
    if (
        report.get("decision") != expected_decision
        or audit.get("decision") != expected_decision
        or audit.get("negative_aware") is not True
        or audit.get("source_json_sha256") != EXPECTED_P5["json"]
        or audit.get("source_npz_sha256") != EXPECTED_P5["npz"]
        or report["npz"]["sha256"] != EXPECTED_P5["npz"]
        or provenance.get("commit") != EXPECTED_P5["commit"]
        or provenance.get("slurm_job_id") != EXPECTED_P5["job"]
        or provenance.get("jax_backend") != "gpu"
        or provenance.get("gpu_kind") != "NVIDIA H200"
        or provenance.get("x64") is not True
        or provenance.get("matmul_precision") != "highest"
        or report["train_targets"].get("snapshot_count") != 35_904
        or report["train_targets"].get("integrity_pass") is not True
        or report["config"].get("model_validation_touched") is not False
        or report["config"].get("confirmation_touched") is not False
    ):
        raise SystemExit("immutable P5-D report/audit/provenance mismatch")
    manifest = manifest_rows(manifest_path)
    for name, digest in provenance["source_sha256"].items():
        if manifest.get(f"code/{name}") != digest:
            raise SystemExit(f"P5-D staged source mismatch: {name}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        mean = np.asarray(arrays["coefficient_mean"], np.float64)
        scales = np.asarray(arrays["head_scales"], np.float64)
    normalization = report["train_targets"]["normalization"]
    if (
        mean.shape != (3328,) or scales.shape != (2,)
        or not np.array_equal(mean, np.asarray(normalization["coefficient_mean"]))
        or not np.array_equal(scales, np.asarray(normalization["head_scales"]))
        or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scales))
        or not np.all(scales > 0.0)
    ):
        raise SystemExit("P5-D immutable normalization mismatch")
    binding = {
        name: {"basename": os.path.basename(path), "sha256": c.sha256(path)}
        for name, path in paths.items()
    }
    binding.update({"commit": provenance["commit"], "job_id": provenance["slurm_job_id"],
                    "decision": expected_decision})
    return report, mean, scales, binding


def relative_record(candidate, control, case):
    names = ("full_fields", "current_stencils", "previous_centers", "weak_residual", "rho")
    values = {
        name: p6.relative_l2(value, baseline)
        for name, value, baseline in zip(names, candidate, control)
    }
    arrays_finite = bool(all(np.all(np.isfinite(np.asarray(value))) for value in candidate))
    n2 = np.asarray(candidate[0]).shape[1]
    n = int(round(np.sqrt(n2)))
    boundary = bool(np.all(
        np.asarray(candidate[0])[:, c.binary_boundary_mask(n) == 0.0] == 0.0
    ))
    passed = bool(arrays_finite and boundary and max(values.values()) <= IDENTITY_TOL)
    return {"case_index": int(case), "relative_l2": values,
            "max_relative_l2": max(values.values()), "all_finite": arrays_finite,
            "exact_boundary": boundary, "pass": passed}


def work_record(method, case, output):
    fields, residual, rho = map(np.asarray, (output[0], output[3], output[4]))
    return {
        "method": method, "case_index": int(case),
        "finite": bool(np.all(np.isfinite(fields)) and np.all(np.isfinite(residual))),
        "output_shape": list(fields.shape),
        "rho_all": rho.tolist(),
        "weak_residual_norm_all": np.linalg.norm(residual, axis=1).tolist(),
        "weak_objective_evaluations": int(rho.size),
        "weak_jacobian_evaluations": 0,
        "trial_residual_evaluations": 0,
        "coefficient_grid_evaluations": int(rho.size + 1),
        "zero_failures": bool(np.all(np.isfinite(rho))),
    }


def canonical_work_pass(row, n, steps):
    return bool(
        row["finite"] and row["zero_failures"]
        and row["output_shape"] == [steps + 1, n * n]
        and row["weak_objective_evaluations"] == steps
        and row["weak_jacobian_evaluations"] == 0
        and row["trial_residual_evaluations"] == 0
        and row["coefficient_grid_evaluations"] == steps + 1
        and len(row["rho_all"]) == steps
        and len(row["weak_residual_norm_all"]) == steps
        and np.all(np.isfinite(row["rho_all"]))
        and np.all(np.isfinite(row["weak_residual_norm_all"]))
    )


def expected_orders(methods, repetitions):
    orders = []
    for repetition in range(repetitions):
        offset = repetition % len(methods)
        order = list(methods[offset:] + methods[:offset])
        if (repetition // len(methods)) % 2:
            order.reverse()
        orders.append(order)
    return orders


def run_cost(mean, scales, smoke):
    n, steps = (32, 1) if smoke else (1024, c.NUM_STEPS)
    case_count = 1 if smoke else 4
    if smoke:
        parameters = {
            "cx": np.asarray((0.42,)), "cy": np.asarray((0.58,)),
            "width": np.asarray((0.12,)), "amplitude": np.asarray((1.2,)),
            "nu": np.asarray((0.01,)),
        }
        truth = np.asarray(c.bf.blob_ic(n, 0.42, 0.58, 0.12, 1.2))[None, None, :]
        reference, dummy = None, None
    else:
        truth, parameters, reference, dummy = base.generate_live_reference(n)
    features_by_case, representative_inputs = [], []
    for case in range(case_count):
        recovered, sample_indices = c.recover_blob_parameters_fixed_sample(
            truth[case, 0], n
        )
        expected = np.asarray((parameters["cx"][case], parameters["cy"][case],
                               parameters["width"][case], parameters["amplitude"][case]))
        error = np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
        if (sample_indices.size > 4096 or not np.all(np.isfinite(recovered))
                or error > 1e-9):
            raise SystemExit("Phase6 cold recovery gate failed")
        one = {"cx": recovered[0:1], "cy": recovered[1:2],
               "width": recovered[2:3], "amplitude": recovered[3:4],
               "nu": parameters["nu"][case:case + 1]}
        features = c.trajectory_features(one, n)[0, :steps + 1]
        features_by_case.append(features)
        representative_inputs.append({
            "case_index": case, "source_draw_index": case,
            "sample_count": int(sample_indices.size),
            "recovery_relative_error": float(error),
            "features": features.tolist(),
        })

    basis = p4d.basis_identity(p5.H1, smoke)
    pallas = p4d.pallas_basis_identity(p5.H1, smoke)
    if not basis["pass"] or not pallas["pass"]:
        raise SystemExit("Phase6 basis/support identity failed")
    compiled, setup, geometry, arguments = p6.compile_online_kernels(
        n, features_by_case[0], parameters["nu"][0], mean, scales, steps
    )
    identity = {"R0_polynomial_weak": [], "R1_cox_weak_k3_full": []}
    actual_route_consistency = []
    route_args = []
    for case in range(case_count):
        args = list(arguments)
        args[4] = jnp.asarray(features_by_case[case])
        args[7] = jnp.asarray(parameters["nu"][case])
        args = tuple(args)
        route_args.append(args)
        control, r0, r1 = compiled["identity_all"](*args)
        actual = compiled["R1_cox_weak_k3_full"](*args)
        jax.block_until_ready((control, r0, r1, actual))
        identity["R0_polynomial_weak"].append(relative_record(r0, control, case))
        identity["R1_cox_weak_k3_full"].append(relative_record(r1, control, case))
        route_relative = {
            "full_fields": p6.relative_l2(actual[0], r1[0]),
            "weak_residual": p6.relative_l2(actual[3], r1[3]),
            "rho": p6.relative_l2(actual[4], r1[4]),
        }
        actual_route_consistency.append({
            "case_index": case, "relative_l2": route_relative,
            "max_relative_l2": max(route_relative.values()),
            "all_finite": bool(all(np.all(np.isfinite(np.asarray(actual[index])))
                                      for index in (0, 3, 4))),
            "pass": bool(max(route_relative.values()) <= IDENTITY_TOL),
        })

    methods = METHODS if not smoke else METHODS[1:]
    fom = None if smoke else base.bc.make_chain(
        n, base.FOM_OUTER, lin_tol=base.FOM_INNER, preconditioner="helmholtz"
    )[0]

    def invoke(method, case):
        started = time.perf_counter()
        if method == "fom":
            output = fom(jnp.asarray(truth[case, 0]), parameters["nu"][case], dummy, jnp.int32(5))
        else:
            # The actual charged online route repeats the fixed <=9-point cold
            # recovery and raw-feature construction on every invocation.
            recovered, sample_indices = c.recover_blob_parameters_fixed_sample(
                truth[case, 0], n
            )
            if sample_indices.size > 4096 or not np.all(np.isfinite(recovered)):
                raise SystemExit("Phase6 charged cold recovery failed")
            one = {"cx": recovered[0:1], "cy": recovered[1:2],
                   "width": recovered[2:3], "amplitude": recovered[3:4],
                   "nu": parameters["nu"][case:case + 1]}
            args = list(arguments)
            args[4] = jnp.asarray(c.trajectory_features(one, n)[0, :steps + 1])
            args[7] = jnp.asarray(parameters["nu"][case])
            output = compiled[method](*tuple(args))
        jax.block_until_ready(output)
        return output, float(time.perf_counter() - started)

    first_execution = {}
    for method in methods:
        _, first_execution[method] = invoke(method, 0)
    canonical = {method: [work_record(method, case, invoke(method, case)[0])
                          for case in range(case_count)]
                 for method in methods if method != "fom"}
    for _ in range(TIME_WARM):
        for case in range(case_count):
            for method in methods:
                invoke(method, case)
    burn_count = c.gpu_burn(0.1 if smoke else BURN_SECONDS)
    repetitions = 1 if smoke else TIME_REPS
    orders = expected_orders(methods, repetitions)
    records = {method: [] for method in methods}
    for repetition, order in enumerate(orders):
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
    positions = {method: [sum(order[position] == method for order in orders)
                          for position in range(len(methods))] for method in methods}
    exact_balance = bool(smoke or all(value == 8 for counts in positions.values() for value in counts))
    if not exact_balance:
        raise SystemExit("Phase6 exact timing balance failed")
    summaries = {method: base.summarize_timing(rows, case_count)
                 for method, rows in records.items()}
    if smoke:
        fom_accuracy = None
    else:
        first_rows = [row for row in records["fom"] if row["repetition"] == 0]
        mean_error = float(np.mean([row["trajectory_relative_l2"] for row in first_rows]))
        worst_error = float(np.max([row["trajectory_relative_l2"] for row in first_rows]))
        healthy = bool(all(row["finite"] and row["breakdowns"] == 0
                           and row["flags_nonzero"] == 0
                           and row["max_returned_relative_residual"] <= base.FOM_OUTER
                           for row in records["fom"]))
        fom_accuracy = {"mean": mean_error, "worst": worst_error, "healthy": healthy,
                        "eligible": bool(healthy and mean_error <= 1e-3 and worst_error <= 3e-3)}
    identity_pass = bool(
        basis["pass"] and pallas["pass"]
        and all(row["pass"] for row in identity["R1_cox_weak_k3_full"])
        and all(row["pass"] for row in actual_route_consistency)
    )
    work_pass = bool(all(canonical_work_pass(row, n, steps)
                         for row in canonical["R1_cox_weak_k3_full"]))
    memory = setup["R1_cox_weak_k3_full"]["memory_analysis"]["eligibility_device_bytes"]
    if smoke:
        gate = {"scientific_promotion_allowed": False, "fom_eligible": False,
                "identity_pass": identity_pass, "canonical_work_pass": work_pass,
                "compiled_device_bytes": int(memory), "memory_pass": memory <= MEMORY_LIMIT,
                "paired_median_speedup": None, "clustered_speedup_ci": None,
                "pass": False}
    else:
        speed = summaries["fom"]["median_elapsed_s"] / summaries["R1_cox_weak_k3_full"]["median_elapsed_s"]
        ci = base.clustered_speedup_ci(
            summaries["fom"]["per_case_median_elapsed_s"],
            summaries["R1_cox_weak_k3_full"]["per_case_median_elapsed_s"], CI_SEED,
        )
        gate = {"scientific_promotion_allowed": True,
                "fom_eligible": fom_accuracy["eligible"], "identity_pass": identity_pass,
                "canonical_work_pass": work_pass, "compiled_device_bytes": int(memory),
                "memory_pass": memory <= MEMORY_LIMIT,
                "paired_median_speedup": float(speed), "clustered_speedup_ci": ci,
                "pass": bool(fom_accuracy["eligible"] and identity_pass and work_pass
                             and memory <= MEMORY_LIMIT and speed >= 10.0 and ci[0] >= 8.0)}
    return {"status": "excluded_execution_smoke_pass" if smoke else "complete",
            "reference_health": reference, "basis_identity": basis,
            "pallas_basis_identity": pallas, "weak_geometry": {"M": p5.H1["M"],
            "m": p5.H1["m"], "rule": geometry["rule"], "max_support": 32},
            "representative_inputs": representative_inputs,
            "setup": setup, "parameter_count": p6.parameter_count(),
            "identity": identity, "actual_route_consistency": actual_route_consistency,
            "canonical_work": canonical,
            "first_execution_after_compile_s": first_execution,
            "burn_count": burn_count, "timing_orders": orders,
            "position_counts": positions, "exact_position_balance": exact_balance,
            "records": records, "summaries": summaries,
            "fom_accuracy": fom_accuracy, "gate": gate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--p5-json")
    parser.add_argument("--p5-npz")
    parser.add_argument("--p5-audit")
    parser.add_argument("--p5-manifest")
    parser.add_argument("--prereg")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    if args.smoke:
        mean = np.zeros(3328, np.float64)
        scales = np.ones(2, np.float64)
        binding = None
    else:
        required = (args.p5_json, args.p5_npz, args.p5_audit,
                    args.p5_manifest, args.prereg)
        if any(value is None for value in required):
            raise SystemExit("scientific P6-D requires complete P5/prereg chain")
        _, mean, scales, binding = validate_p5_chain(
            args.p5_json, args.p5_npz, args.p5_audit, args.p5_manifest
        )
        binding["phase6_preregistration"] = {
            "basename": os.path.basename(args.prereg), "sha256": c.sha256(args.prereg)
        }
    panel = run_cost(mean, scales, args.smoke)
    gate_pass = bool(panel["gate"]["pass"])
    decision = {
        "repair_licensed": bool(not args.smoke and gate_pass),
        "phase6_hard_stop": bool(not args.smoke and not gate_pass),
        "training_authorized": False,
        "next_action": ("excluded smoke only" if args.smoke else
                        "separate training proposal/audit" if gate_pass else "hard stop"),
        "scientific_promotion_allowed": not args.smoke,
    }
    timing_arrays = {}
    if not args.smoke:
        for method, rows in panel["records"].items():
            timing_arrays[f"timing_{method}"] = np.asarray(
                [[row["case_index"], row["repetition"], row["elapsed_s"]] for row in rows],
                np.float64,
            )
    np.savez(args.output_npz, coefficient_mean=mean, head_scales=scales, **timing_arrays)
    report = {
        "status": panel["status"], "provenance": c.provenance(),
        "config": {"candidate": p6.G1, "H1": p5.H1,
                   "methods": list(METHODS), "time_repetitions": TIME_REPS,
                   "time_warmups": TIME_WARM, "burn_seconds": BURN_SECONDS,
                   "identity_tolerance": IDENTITY_TOL, "memory_limit_bytes": MEMORY_LIMIT,
                   "ci_seed": CI_SEED, "fom_seed": base.FOM_SEED, "fom_cases": 4,
                   "reference_outer": base.REFERENCE_OUTER, "reference_inner": base.REFERENCE_INNER,
                   "audit_outer": base.AUDIT_OUTER, "audit_inner": base.AUDIT_INNER,
                   "fom_outer": base.FOM_OUTER, "fom_inner": base.FOM_INNER,
                   "p5_targets_regenerated": False, "training_touched": False,
                   "model_validation_touched": False, "confirmation_touched": False,
                   "smoke": args.smoke, "f64": True, "matmul_precision": "highest"},
        "bindings": binding, "cost_panel": panel, "decision": decision,
        "elapsed_s": float(time.perf_counter() - started),
    }
    npz_sha = c.sha256(args.output_npz)
    report["npz"] = {"basename": os.path.basename(args.output_npz), "sha256": npz_sha}
    c.save_json(args.output_json, report)
    c.log({"status": report["status"], "decision": decision,
           "elapsed_s": report["elapsed_s"]})


if __name__ == "__main__":
    main()
