#!/usr/bin/env python
"""Independent negative-aware audit for the one-cell Phase-6 diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import b10_common as c
import b10_phase5 as p5
import b10_phase6 as p6
import b10_phase6_d as d
import b10_s0_spline as base


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            rows[relative.removeprefix("./")] = digest
    return rows


def close(value, expected, rtol=2e-12, atol=1e-15):
    return math.isclose(float(value), float(expected), rel_tol=rtol, abs_tol=atol)


def compare_summary(stored, expected, method):
    for key in ("median_elapsed_s", "elapsed_all_s", "per_case_median_elapsed_s",
                "outliers_gt_1p5_within_trajectory_all",
                "outliers_gt_1p5_within_trajectory_total"):
        if key not in stored:
            raise SystemExit(f"{method} missing timing summary {key}")
        if isinstance(expected[key], list):
            okay = np.allclose(stored[key], expected[key], rtol=2e-12, atol=1e-15)
        elif isinstance(expected[key], int):
            okay = stored[key] == expected[key]
        else:
            okay = close(stored[key], expected[key])
        if not okay:
            raise SystemExit(f"{method} timing summary mismatch: {key}")


def audit_work(row, method, n, steps):
    passed = bool(
        row.get("method") == method and row.get("finite")
        and row.get("zero_failures")
        and row.get("output_shape") == [steps + 1, n * n]
        and row.get("weak_objective_evaluations") == steps
        and row.get("weak_jacobian_evaluations") == 0
        and row.get("trial_residual_evaluations") == 0
        and row.get("coefficient_grid_evaluations") == steps + 1
        and len(row.get("rho_all", ())) == steps
        and len(row.get("weak_residual_norm_all", ())) == steps
        and np.all(np.isfinite(row.get("rho_all", ())))
        and np.all(np.isfinite(row.get("weak_residual_norm_all", ())))
    )
    return passed


def audit_p5_chain(args, report, main, smoke):
    if smoke:
        if report.get("bindings") is not None:
            raise SystemExit("smoke unexpectedly binds scientific P5 data")
        return
    required = (args.p5_json, args.p5_npz, args.p5_audit,
                args.p5_manifest, args.prereg)
    if any(value is None for value in required):
        raise SystemExit("scientific audit requires complete P5/prereg chain")
    _, mean, scales, binding = d.validate_p5_chain(
        args.p5_json, args.p5_npz, args.p5_audit, args.p5_manifest
    )
    expected = dict(binding)
    expected["phase6_preregistration"] = {
        "basename": os.path.basename(args.prereg), "sha256": c.sha256(args.prereg)
    }
    if report.get("bindings") != expected:
        raise SystemExit("P6-D immutable P5/prereg binding mismatch")
    if not np.array_equal(main["coefficient_mean"], mean):
        raise SystemExit("P6-D NPZ coefficient mean mismatch")
    if not np.array_equal(main["head_scales"], scales):
        raise SystemExit("P6-D NPZ head scales mismatch")


def audit_report(report, main, smoke):
    expected_status = "excluded_execution_smoke_pass" if smoke else "complete"
    if report.get("status") != expected_status:
        raise SystemExit("P6-D status mismatch")
    config = report["config"]
    exact_config = {
        "candidate": p6.G1, "H1": p5.H1,
        "methods": list(d.METHODS), "time_repetitions": d.TIME_REPS,
        "time_warmups": d.TIME_WARM, "burn_seconds": d.BURN_SECONDS,
        "identity_tolerance": d.IDENTITY_TOL,
        "memory_limit_bytes": d.MEMORY_LIMIT, "ci_seed": d.CI_SEED,
        "fom_seed": base.FOM_SEED, "fom_cases": 4,
        "reference_outer": base.REFERENCE_OUTER,
        "reference_inner": base.REFERENCE_INNER,
        "audit_outer": base.AUDIT_OUTER, "audit_inner": base.AUDIT_INNER,
        "fom_outer": base.FOM_OUTER, "fom_inner": base.FOM_INNER,
        "p5_targets_regenerated": False, "training_touched": False,
        "model_validation_touched": False, "confirmation_touched": False,
        "smoke": smoke, "f64": True, "matmul_precision": "highest",
    }
    if config != exact_config:
        raise SystemExit("P6-D exact configuration mismatch")
    if p6.parameter_count() != p6.G1["parameter_count"]:
        raise SystemExit("G1 parameter count mismatch")
    panel = report["cost_panel"]
    if panel["parameter_count"] != p6.G1["parameter_count"]:
        raise SystemExit("stored G1 parameter count mismatch")
    if not (panel["basis_identity"]["pass"]
            and panel["pallas_basis_identity"]["pass"]
            and panel["weak_geometry"]["M"] == 96
            and panel["weak_geometry"]["m"] == 384
            and panel["weak_geometry"]["max_support"] == 32):
        raise SystemExit("basis/support/weak geometry mismatch")
    case_count = 1 if smoke else 4
    representatives = panel.get("representative_inputs", [])
    if (
        len(representatives) != case_count
        or [row.get("case_index") for row in representatives] != list(range(case_count))
        or [row.get("source_draw_index") for row in representatives] != list(range(case_count))
        or not all(0 < row.get("sample_count", 0) <= 4096
                   and np.isfinite(row.get("recovery_relative_error", np.nan))
                   and row["recovery_relative_error"] <= 1e-9
                   and np.asarray(row.get("features", [])).shape
                   == (2 if smoke else 51, 7)
                   and np.all(np.isfinite(row.get("features", [])))
                   for row in representatives)
    ):
        raise SystemExit("cold-recovery/representative-input record mismatch")

    expected_methods = d.METHODS[1:] if smoke else d.METHODS
    repetitions = 1 if smoke else d.TIME_REPS
    orders = d.expected_orders(expected_methods, repetitions)
    if panel["timing_orders"] != orders:
        raise SystemExit("exact timing order mismatch")
    positions = {method: [sum(order[position] == method for order in orders)
                          for position in range(len(expected_methods))]
                 for method in expected_methods}
    if panel["position_counts"] != positions:
        raise SystemExit("timing position counts mismatch")
    if not smoke and not all(value == 8 for counts in positions.values() for value in counts):
        raise SystemExit("scientific timing position balance mismatch")
    if panel["exact_position_balance"] is not True:
        raise SystemExit("exact position balance flag mismatch")
    burn = panel["burn_count"]
    if not isinstance(burn, (int, float)) or not np.isfinite(burn) or burn <= 0:
        raise SystemExit("GPU burn record is not positive/finite")
    if set(panel["first_execution_after_compile_s"]) != set(expected_methods):
        raise SystemExit("first execution method coverage mismatch")
    if not all(np.isfinite(value) and value >= 0.0
               for value in panel["first_execution_after_compile_s"].values()):
        raise SystemExit("first execution timing mismatch")
    if set(panel["setup"]) != {"R0_polynomial_weak", "R1_cox_weak_k3_full", "identity_all"}:
        raise SystemExit("compiled setup route mismatch")
    for route, item in panel["setup"].items():
        values = (item["lower_s"], item["compile_s"], item["lower_plus_compile_s"])
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise SystemExit(f"{route} invalid setup timing")
        if not close(values[2], values[0] + values[1]):
            raise SystemExit(f"{route} setup timing sum mismatch")
        memory = item["memory_analysis"]["eligibility_device_bytes"]
        if not isinstance(memory, int) or memory <= 0:
            raise SystemExit(f"{route} invalid memory analysis")

    case_count, expected_coverage = (1, {(0, 0)}) if smoke else (
        4, {(case, repetition) for case in range(4) for repetition in range(d.TIME_REPS)}
    )
    work = {}
    for method in expected_methods:
        rows = panel["records"][method]
        coverage = {(int(row["case_index"]), int(row["repetition"])) for row in rows}
        if coverage != expected_coverage or len(rows) != len(coverage):
            raise SystemExit(f"{method} timing coverage mismatch")
        recomputed = base.summarize_timing(rows, case_count)
        compare_summary(panel["summaries"][method], recomputed, method)
        if not smoke:
            stored = np.asarray(main[f"timing_{method}"])
            expected = np.asarray([[row["case_index"], row["repetition"], row["elapsed_s"]]
                                   for row in rows], np.float64)
            if not np.array_equal(stored, expected):
                raise SystemExit(f"{method} timing NPZ mismatch")
        if method != "fom":
            canonical = panel["canonical_work"][method]
            work[method] = bool(
                len(canonical) == case_count
                and sorted(row["case_index"] for row in canonical) == list(range(case_count))
                and all(audit_work(row, method, 32 if smoke else 1024,
                                   1 if smoke else 50) for row in canonical)
            )

    identity = {}
    for route in ("R0_polynomial_weak", "R1_cox_weak_k3_full"):
        rows = panel["identity"][route]
        if len(rows) != case_count or sorted(row["case_index"] for row in rows) != list(range(case_count)):
            raise SystemExit(f"{route} identity case coverage mismatch")
        passed = True
        for row in rows:
            values = row["relative_l2"]
            if set(values) != {"full_fields", "current_stencils", "previous_centers", "weak_residual", "rho"}:
                raise SystemExit(f"{route} identity component mismatch")
            recomputed = bool(row["all_finite"] and row["exact_boundary"]
                              and max(values.values()) <= d.IDENTITY_TOL)
            if row["max_relative_l2"] != max(values.values()) or row["pass"] != recomputed:
                raise SystemExit(f"{route} identity decision mismatch")
            passed &= recomputed
            if route == "R1_cox_weak_k3_full" and any(
                values[name] != 0.0 for name in
                ("current_stencils", "previous_centers", "weak_residual", "rho")
            ):
                raise SystemExit("R1 common Cox weak outputs are not bitwise identical")
        identity[route] = bool(passed)
    consistency = panel.get("actual_route_consistency", [])
    if len(consistency) != case_count or sorted(row["case_index"] for row in consistency) != list(range(case_count)):
        raise SystemExit("R1 actual-route consistency coverage mismatch")
    actual_consistency = True
    for row in consistency:
        values = row.get("relative_l2", {})
        if set(values) != {"full_fields", "weak_residual", "rho"}:
            raise SystemExit("R1 actual-route consistency component mismatch")
        passed = bool(row.get("all_finite") and max(values.values()) <= d.IDENTITY_TOL)
        if row.get("max_relative_l2") != max(values.values()) or row.get("pass") != passed:
            raise SystemExit("R1 actual-route consistency decision mismatch")
        actual_consistency &= passed

    gate = panel["gate"]
    memory = panel["setup"]["R1_cox_weak_k3_full"]["memory_analysis"]["eligibility_device_bytes"]
    r1_identity_gate = bool(identity["R1_cox_weak_k3_full"] and actual_consistency)
    common = bool(panel["basis_identity"]["pass"] and panel["pallas_basis_identity"]["pass"]
                  and r1_identity_gate
                  and work["R1_cox_weak_k3_full"] and memory <= d.MEMORY_LIMIT)
    if smoke:
        if gate != {"scientific_promotion_allowed": False, "fom_eligible": False,
                    "identity_pass": r1_identity_gate,
                    "canonical_work_pass": work["R1_cox_weak_k3_full"],
                    "compiled_device_bytes": memory, "memory_pass": memory <= d.MEMORY_LIMIT,
                    "paired_median_speedup": None, "clustered_speedup_ci": None,
                    "pass": False}:
            raise SystemExit("smoke gate mismatch")
        expected_decision = {"repair_licensed": False, "phase6_hard_stop": False,
                             "training_authorized": False,
                             "next_action": "excluded smoke only",
                             "scientific_promotion_allowed": False}
    else:
        health = panel["reference_health"]
        reference_records = health.get("reference_records", [])
        audit_records = health.get("audit_records", [])
        differences = health.get("cross_chain_trajectory_relative_l2_all", [])
        reference_healthy = bool(
            len(reference_records) == 4
            and all(row["finite"] and row["breakdowns"] == 0
                    and row["flags_nonzero"] == 0
                    and np.isfinite(row["max_returned_relative_residual"])
                    and row["max_returned_relative_residual"] <= base.REFERENCE_OUTER
                    for row in reference_records)
        )
        audit_healthy = bool(
            len(audit_records) == 4
            and all(row["finite"] and row["breakdowns"] == 0
                    and row["flags_nonzero"] == 0
                    and np.isfinite(row["max_returned_relative_residual"])
                    and row["max_returned_relative_residual"] <= base.AUDIT_OUTER
                    for row in audit_records)
        )
        expected_differences = [row["trajectory_relative_l2"] for row in audit_records]
        if not (
            health.get("reference_outer") == base.REFERENCE_OUTER
            and health.get("reference_inner") == base.REFERENCE_INNER
            and health.get("audit_outer") == base.AUDIT_OUTER
            and health.get("audit_inner") == base.AUDIT_INNER
            and reference_healthy and audit_healthy
            and len(differences) == 4
            and np.array_equal(np.asarray(differences), np.asarray(expected_differences))
            and close(health["cross_chain_worst"], max(expected_differences))
            and max(expected_differences) <= 1e-4
            and health["all_finite_zero_flags_breakdowns"] is True
        ):
            raise SystemExit("tight/tighter reference health mismatch")
        first_fom = [row for row in panel["records"]["fom"] if row["repetition"] == 0]
        mean = float(np.mean([row["trajectory_relative_l2"] for row in first_fom]))
        worst = float(np.max([row["trajectory_relative_l2"] for row in first_fom]))
        healthy = bool(all(row["finite"] and row["breakdowns"] == 0
                           and row["flags_nonzero"] == 0
                           and row["max_returned_relative_residual"] <= base.FOM_OUTER
                           for row in panel["records"]["fom"]))
        eligible = bool(healthy and mean <= 1e-3 and worst <= 3e-3)
        expected_fom = {"mean": mean, "worst": worst, "healthy": healthy, "eligible": eligible}
        if panel["fom_accuracy"] != expected_fom:
            raise SystemExit("live FOM accuracy/health mismatch")
        speed = panel["summaries"]["fom"]["median_elapsed_s"] / panel["summaries"]["R1_cox_weak_k3_full"]["median_elapsed_s"]
        ci = base.clustered_speedup_ci(
            panel["summaries"]["fom"]["per_case_median_elapsed_s"],
            panel["summaries"]["R1_cox_weak_k3_full"]["per_case_median_elapsed_s"], d.CI_SEED,
        )
        passed = bool(common and eligible and speed >= 10.0 and ci[0] >= 8.0)
        if not (gate["scientific_promotion_allowed"] is True
                and gate["fom_eligible"] == eligible
                and gate["identity_pass"] == r1_identity_gate
                and gate["canonical_work_pass"] == work["R1_cox_weak_k3_full"]
                and gate["compiled_device_bytes"] == memory
                and gate["memory_pass"] == (memory <= d.MEMORY_LIMIT)
                and close(gate["paired_median_speedup"], speed)
                and np.allclose(gate["clustered_speedup_ci"], ci, rtol=2e-12, atol=1e-15)
                and gate["pass"] == passed):
            raise SystemExit("scientific R1 gate mismatch")
        expected_decision = {"repair_licensed": passed,
                             "phase6_hard_stop": not passed,
                             "training_authorized": False,
                             "next_action": "separate training proposal/audit" if passed else "hard stop",
                             "scientific_promotion_allowed": True}
    if report["decision"] != expected_decision:
        raise SystemExit("Phase6 immutable decision mismatch")
    return {"identity": identity, "work": work,
            "actual_route_consistency": bool(actual_consistency),
            "r1_pass": bool(report["decision"]["repair_licensed"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("npz")
    parser.add_argument("audit")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-job")
    parser.add_argument("--manifest")
    parser.add_argument("--p5-json")
    parser.add_argument("--p5-npz")
    parser.add_argument("--p5-audit")
    parser.add_argument("--p5-manifest")
    parser.add_argument("--prereg")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = load(args.json)
    with np.load(args.npz, allow_pickle=False) as handle:
        main_npz = {key: np.asarray(handle[key]) for key in handle.files}
    if report["npz"] != {"basename": os.path.basename(args.npz), "sha256": c.sha256(args.npz)}:
        raise SystemExit("P6-D main NPZ binding mismatch")
    provenance = report["provenance"]
    if not args.smoke:
        if any(value is None for value in (args.expected_commit, args.expected_job, args.manifest)):
            raise SystemExit("scientific provenance arguments required")
        if (provenance.get("commit") != args.expected_commit
                or provenance.get("slurm_job_id") != args.expected_job
                or provenance.get("jax_backend") != "gpu"
                or provenance.get("gpu_kind") != "NVIDIA H200"
                or provenance.get("x64") is not True
                or provenance.get("matmul_precision") != "highest"):
            raise SystemExit("P6-D runtime provenance mismatch")
        manifest = manifest_rows(args.manifest)
        for name, digest in provenance["source_sha256"].items():
            if manifest.get(f"code/{name}") != digest:
                raise SystemExit(f"P6-D staged source mismatch: {name}")
        if manifest.get("code/deps/p5/MANIFEST.sha256") != d.EXPECTED_P5["manifest"]:
            raise SystemExit("nested P5 root manifest is not staged/bound")
    audit_p5_chain(args, report, main_npz, args.smoke)
    checks = audit_report(report, main_npz, args.smoke)
    result = {"status": "pass", "negative_aware": True,
              "source_json": os.path.basename(args.json),
              "source_json_sha256": c.sha256(args.json),
              "source_npz_sha256": c.sha256(args.npz),
              "decision": report["decision"], "checks": checks,
              "model_validation_touched": False, "confirmation_touched": False}
    if not args.smoke:
        result.update({"expected_commit": args.expected_commit,
                       "expected_job": args.expected_job,
                       "manifest_sha256": c.sha256(args.manifest)})
    c.save_json(args.audit, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
