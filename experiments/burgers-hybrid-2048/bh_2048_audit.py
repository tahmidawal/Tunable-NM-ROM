"""Validate an N=2048 smoke or authoritative paired result."""
from __future__ import annotations

import hashlib
import json
import math
import sys

SOURCE, SELECTION, PAIR_AUDIT, EXPECTED_COMMIT, EXPECTED_SOURCE_SHA, OUT = sys.argv[1:7]
EXPECTED_CONDITIONS = ((1e-6, 1e-2), (1e-8, 1e-4), (1e-10, 1e-5))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


with open(SOURCE) as handle:
    report = json.load(handle)
if not report.get("complete"):
    raise SystemExit("N=2048 report is incomplete")
config = report["config"]
smoke = bool(config.get("smoke"))
if smoke:
    raise SystemExit("final3 audit does not accept smoke output")
expected_conditions = EXPECTED_CONDITIONS
expected = {
    "ns": [2048],
    "conditions": [
        {"fom_tau": tau, "linear_tol": linear_tol}
        for tau, linear_tol in expected_conditions
    ],
    "resolution_policy": "coarse_n=N at every mesh (full-grid fraction 1.0)",
    "relaxation": 1.0,
    "preconditioner": "exact target-grid Helmholtz for both FOM finishes",
    "dynamic_correction_steps_per_trajectory": 50,
    "dynamic_extra_full_grid_residual_evaluations": 50,
    "dynamic_extra_exact_helmholtz_inverses": 50,
    "test_seed": 20260830,
    "canonical_draw_count": 16,
    "selected_test_indices": [0, 1, 2, 3],
    "pair_blocks": 6,
    "repetitions_per_arm_per_trajectory": 12,
    "burn_before_every_AB_and_BA_block": True,
    "burn_seconds": 3.0,
    "accuracy_work_residual_from_every_timed_invocation": True,
    "reference_residual_gate": 1e-11,
    "reference_field_agreement_gate": 1e-10,
    "reference_max_newton_iterations_per_step": 25,
    "reference_routes": [
        {
            "name": "helmholtz_candidate",
            "outer_tol": 1e-12,
            "linear_tol": 1e-8,
        },
        {
            "name": "helmholtz_inner_strict",
            "outer_tol": 1e-12,
            "linear_tol": 1e-10,
        },
    ],
    "reference_candidate_fields_used_only_after_all_route_gates": True,
    "reference_gate_before_online_compile_or_timing": True,
    "reference_design_informed_by_excluded_development_seed": 20260827,
    "reference_design_diagnostic_job": "2677878",
    "excluded_implementation_smoke_seed": 20260829,
    "reference_cohort_was_untouched_before_final3": True,
    "fixed_public_jax_reference_used": False,
    "fresh_seed_touched": True,
    "smoke": False,
    "f64": True,
    "reference_solver_scope": (
        "offline dual exact-Helmholtz truth generation and direct counting "
        "equivalence only; online methods/tolerances/schedule remain unchanged"
    ),
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"N=2048 {key} drifted: {config.get(key)!r} != {value!r}")
if config["selection_sha256"] != sha256(SELECTION):
    raise SystemExit("N=2048 selection artifact hash mismatch")
if config["pair_audit_sha256"] != sha256(PAIR_AUDIT):
    raise SystemExit("N=2048 pair-audit artifact hash mismatch")

provenance = report["provenance"]
for key, value in {
    "jax_backend": "gpu",
    "gpu_kind": "NVIDIA H200",
    "x64": True,
    "matmul_precision": "highest",
}.items():
    if provenance.get(key) != value:
        raise SystemExit(f"N=2048 provenance {key} drifted")
if not provenance.get("slurm_job_id"):
    raise SystemExit("N=2048 result is missing GPU/Slurm provenance")
if provenance.get("commit") != EXPECTED_COMMIT:
    raise SystemExit("N=2048 execution commit drifted")
if provenance.get("source_sha256", {}).get("bh_2048_final.py") != EXPECTED_SOURCE_SHA:
    raise SystemExit("N=2048 staged driver source hash drifted")
if set(report["reference_health"]) != {"2048"}:
    raise SystemExit("N=2048 reference-health grid drifted")
reference_health = report["reference_health"]["2048"]
if (
    reference_health.get("kind") != "prospective_dual_exact_helmholtz"
    or not reference_health.get("gate_evaluated_before_online_compile_or_timing")
    or not reference_health.get("all_pass")
    or reference_health.get("candidate_route") != "helmholtz_candidate"
    or reference_health.get("routes") != expected["reference_routes"]
    or reference_health.get("reference_residual_gate") != 1e-11
    or reference_health.get("field_agreement_gate") != 1e-10
):
    raise SystemExit("N=2048 reference pre-timing gate failed")
reference_trajectories = reference_health.get("trajectories", [])
if (
    len(reference_trajectories) != 4
    or [item.get("trajectory_index") for item in reference_trajectories]
    != [0, 1, 2, 3]
):
    raise SystemExit("N=2048 reference cohort drifted")
expected_routes = {
    "helmholtz_candidate": (1e-12, 1e-8),
    "helmholtz_inner_strict": (1e-12, 1e-10),
}
for trajectory in reference_trajectories:
    routes = {item.get("name"): item for item in trajectory.get("routes", [])}
    if set(routes) != set(expected_routes) or not trajectory.get("trajectory_pass"):
        raise SystemExit("N=2048 reference route grid failed")
    for name, (outer_tol, linear_tol) in expected_routes.items():
        route = routes[name]
        if (
            route.get("outer_tol") != outer_tol
            or route.get("linear_tol") != linear_tol
            or route.get("preconditioner") != "exact_helmholtz"
            or route.get("max_newton_iterations_per_step") != 25
            or len(route.get("per_step_newton_iterations", [])) != 50
            or len(route.get("per_step_linear_iterations", [])) != 50
            or len(route.get("per_step_breakdowns_or_linear_max", [])) != 50
            or len(route.get("per_step_flags", [])) != 50
            or len(route.get("per_step_actual_outer_relative_residual", [])) != 50
            or any(value < 0 or value > 25 for value in route["per_step_newton_iterations"])
            or any(value != 0 for value in route["per_step_breakdowns_or_linear_max"])
            or any(value != 0 for value in route["per_step_flags"])
            or any(
                value is None or not math.isfinite(value) or value > 1e-11
                for value in route["per_step_actual_outer_relative_residual"]
            )
            or not route.get("all_fields_and_residuals_finite")
            or route.get("breakdowns_or_linear_max_total") != 0
            or route.get("flags_nonzero") != 0
            or route.get("max_actual_outer_relative_residual") is None
            or route["max_actual_outer_relative_residual"] > 1e-11
            or route["max_actual_outer_relative_residual"]
            != max(route["per_step_actual_outer_relative_residual"])
            or route.get("newton_total")
            != sum(route["per_step_newton_iterations"])
            or route.get("linear_total")
            != sum(route["per_step_linear_iterations"])
            or not route.get("route_pass")
        ):
            raise SystemExit(f"N=2048 reference route {name} failed")
    agreement = trajectory.get("candidate_vs_inner_strict", {})
    if (
        len(agreement.get("per_step_relative_field_difference", [])) != 50
        or agreement.get("max_step_relative_field_difference") is None
        or agreement["max_step_relative_field_difference"] > 1e-10
        or agreement.get("trajectory_relative_field_difference") is None
        or agreement["trajectory_relative_field_difference"] > 1e-10
        or not agreement.get("agreement_pass")
    ):
        raise SystemExit("N=2048 reference route agreement failed")
if (
    reference_health.get("max_actual_outer_relative_residual") is None
    or reference_health["max_actual_outer_relative_residual"] > 1e-11
    or reference_health.get("max_step_relative_field_difference") is None
    or reference_health["max_step_relative_field_difference"] > 1e-10
    or reference_health.get("max_trajectory_relative_field_difference") is None
    or reference_health["max_trajectory_relative_field_difference"] > 1e-10
):
    raise SystemExit("N=2048 reference summary gate failed")

rows = {(row["N"], row["fom_tau"]): row for row in report["rows"]}
expected_grid = {(2048, tau) for tau, _ in expected_conditions}
if set(rows) != expected_grid or len(report["rows"]) != len(expected_grid):
    raise SystemExit("N=2048 condition grid is incomplete or duplicated")

rendered = []
trajectory_indices = [0, 1, 2, 3]
repetitions = 12
for tau, linear_tol in expected_conditions:
    row = rows[(2048, tau)]
    if (
        not row.get("complete")
        or row.get("linear_tol") != linear_tol
        or row.get("coarse_n") != 2048
        or row.get("relaxation") != 1.0
    ):
        raise SystemExit(f"N=2048 tau={tau}: row metadata drifted")
    expected_records = {
        (trajectory, sample)
        for trajectory in trajectory_indices
        for sample in range(repetitions)
    }
    for arm in ("cubic", "dynamic"):
        records = row["records"].get(arm, [])
        record_grid = {
            (record["trajectory_index"], record["sample_index"])
            for record in records
        }
        if len(records) != len(expected_records) or record_grid != expected_records:
            raise SystemExit(f"N=2048 tau={tau}, {arm}: record grid drifted")
        for trajectory in trajectory_indices:
            case = [
                record for record in records
                if record["trajectory_index"] == trajectory
            ]
            if (
                len(case) != repetitions
                or sum(record["order"] == "cubic/dynamic" for record in case)
                != repetitions // 2
                or sum(record["order"] == "dynamic/cubic" for record in case)
                != repetitions // 2
            ):
                raise SystemExit(f"N=2048 tau={tau}, {arm}: AB/BA imbalance")
        expected_extra = 50 if arm == "dynamic" else 0
        if any(
            not math.isfinite(record["elapsed_s"])
            or record["elapsed_s"] <= 0
            or record["breakdowns"]
            or record["flags_nonzero"]
            or record["max_returned_residual"] > tau
            or not math.isfinite(record["trajectory_rel_l2"])
            or record["extra_full_grid_residual_evaluations"] != expected_extra
            or record["extra_exact_helmholtz_inverses"] != expected_extra
            for record in records
        ):
            raise SystemExit(f"N=2048 tau={tau}, {arm}: timed invocation failed")
    if len(row["burn_records"]) != len(trajectory_indices) * repetitions:
        raise SystemExit(f"N=2048 tau={tau}: burn grid drifted")
    burn_grid = {
        (record["trajectory_index"], record["sample_index"])
        for record in row["burn_records"]
    }
    if (
        burn_grid != expected_records
        or any(record.get("iterations", 0) <= 0 for record in row["burn_records"])
    ):
        raise SystemExit(f"N=2048 tau={tau}: burn balance failed")
    summary = row["summary"]
    if (
        summary["dynamic_extra_full_grid_residual_evaluations"] != 50
        or summary["dynamic_extra_exact_helmholtz_inverses"] != 50
        or summary["max_returned_residual"] > tau
        or summary["max_trajectory_rel_l2"] >= 1e-4
    ):
        raise SystemExit(f"N=2048 tau={tau}: summary health failed")
    equivalence = row["equivalence"]
    if (
        equivalence.get("reference_kind")
        != "prospective_dual_exact_helmholtz_candidate"
        or equivalence.get("comparison_start") != "previous_state"
        or equivalence.get("fom_tau") != tau
        or len(equivalence.get("per_trajectory", [])) != 4
        or any(
            item["breakdowns"]
            or item["flags_nonzero"]
            or not item["all_fields_finite"]
            or item["counting_max_rel_newton_residual"] > tau
            for item in equivalence["per_trajectory"]
        )
        or equivalence["max_step_rel_difference"] > 10 * tau
        or equivalence["max_trajectory_rel_difference"] > 10 * tau
    ):
        raise SystemExit(f"N=2048 tau={tau}: solver equivalence failed")
    ci = summary["paired_saving_trajectory_cluster_95ci_ms"]
    rendered.append({
        "N": 2048,
        "fom_tau": tau,
        "linear_tol": linear_tol,
        "cubic_median_ms": summary["cubic_median_ms"],
        "dynamic_median_ms": summary["dynamic_median_ms"],
        "speedup_cubic_over_dynamic": summary["speedup_cubic_over_dynamic"],
        "paired_saving_median_ms": summary["paired_saving_vs_cubic_median_ms"],
        "paired_saving_trajectory_cluster_95ci_ms": ci,
        "paired_saving_case_medians_ms": summary[
            "paired_saving_vs_cubic_case_medians_ms"
        ],
        "supported_speedup": ci[0] > 0,
        "finish_newton_median": summary["finish_newton_total_median"],
        "finish_bicgstab_median": summary["finish_bicgstab_total_median"],
        "dynamic_extra_full_grid_residual_evaluations": 50,
        "dynamic_extra_exact_helmholtz_inverses": 50,
        "max_returned_residual": summary["max_returned_residual"],
        "max_trajectory_rel_l2": summary["max_trajectory_rel_l2"],
        "outliers_retained": summary["tukey_1p5iqr_outliers_retained"],
    })

memory_points = [report.get("device_memory_at_start", {})]
for row in report["rows"]:
    memory_points.extend([
        row.get("device_memory_after_compile", {}),
        row.get("device_memory_after_condition", {}),
    ])
memory_points.append(report.get("device_memory_at_end", {}))
fractions = [
    point.get("peak_fraction_of_limit") for point in memory_points
    if point.get("peak_fraction_of_limit") is not None
]
peak_fraction = max(fractions) if fractions else None
if peak_fraction is None:
    raise SystemExit("N=2048 device peak/limit telemetry is missing")
audit = {
    "status": "complete",
    "source": SOURCE,
    "source_sha256": sha256(SOURCE),
    "execution_commit": EXPECTED_COMMIT,
    "execution_source_sha256": EXPECTED_SOURCE_SHA,
    "classification": "classical FOM warm start; not learned and not NM-ROM",
    "timing_scope": config["timing_scope"],
    "smoke_excluded_from_scientific_claims": False,
    "peak_device_fraction_of_limit": peak_fraction,
    "reference_health": {
        "max_actual_outer_relative_residual": reference_health[
            "max_actual_outer_relative_residual"
        ],
        "max_step_relative_field_difference": reference_health[
            "max_step_relative_field_difference"
        ],
        "max_trajectory_relative_field_difference": reference_health[
            "max_trajectory_relative_field_difference"
        ],
        "trajectory_count": len(reference_trajectories),
        "all_pass": reference_health["all_pass"],
    },
    "primary_final_panel_licensed": None,
    "learned_sensitivity_decision": None,
    "rows": rendered,
    "supported_speedup_cells": sum(row["supported_speedup"] for row in rendered),
    "total_cells": len(rendered),
}
with open(OUT, "w") as handle:
    json.dump(audit, handle, indent=1, allow_nan=False)
print(json.dumps({
    "status": audit["status"],
    "peak_device_fraction_of_limit": peak_fraction,
    "primary_final_panel_licensed": audit["primary_final_panel_licensed"],
    "learned_sensitivity_decision": audit["learned_sensitivity_decision"],
    "supported_speedup_cells": audit["supported_speedup_cells"],
}, indent=1))
