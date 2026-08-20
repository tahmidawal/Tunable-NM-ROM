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
expected_conditions = ((1e-6, 1e-2),) if smoke else EXPECTED_CONDITIONS
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
    "test_seed": 20260818 if smoke else 20260827,
    "canonical_draw_count": 16,
    "selected_test_indices": [12] if smoke else [0, 1, 2, 3],
    "pair_blocks": 1 if smoke else 6,
    "repetitions_per_arm_per_trajectory": 2 if smoke else 12,
    "burn_before_every_AB_and_BA_block": True,
    "burn_seconds": 0.1 if smoke else 3.0,
    "accuracy_work_residual_from_every_timed_invocation": True,
    "reference_residual_gate": 1e-11,
    "fresh_seed_touched": not smoke,
    "f64": True,
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
if report["reference_health"]["2048"]["max_reference_newton_residual"] > 1e-11:
    raise SystemExit("N=2048 reference generation failed its residual gate")

rows = {(row["N"], row["fom_tau"]): row for row in report["rows"]}
expected_grid = {(2048, tau) for tau, _ in expected_conditions}
if set(rows) != expected_grid or len(report["rows"]) != len(expected_grid):
    raise SystemExit("N=2048 condition grid is incomplete or duplicated")

rendered = []
trajectory_indices = [12] if smoke else [0, 1, 2, 3]
repetitions = 2 if smoke else 12
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
        any(
            item["breakdowns"]
            or item["flags_nonzero"]
            or item["testbed_max_rel_newton_residual"] > 1e-11
            for item in equivalence["per_trajectory"]
        )
        or equivalence["linear_solver"]["ours_flag"]
        or equivalence["linear_solver"]["relative_solution_difference_vs_jax"] > 1e-12
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
        "supported_speedup": (not smoke) and ci[0] > 0,
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
primary_licensed = smoke and peak_fraction <= 0.90
learned_licensed = smoke and peak_fraction <= 0.75

audit = {
    "status": (
        "smoke_pass" if primary_licensed else "smoke_hard_stop"
    ) if smoke else "complete",
    "source": SOURCE,
    "source_sha256": sha256(SOURCE),
    "execution_commit": EXPECTED_COMMIT,
    "execution_source_sha256": EXPECTED_SOURCE_SHA,
    "classification": "classical FOM warm start; not learned and not NM-ROM",
    "timing_scope": config["timing_scope"],
    "smoke_excluded_from_scientific_claims": smoke,
    "peak_device_fraction_of_limit": peak_fraction,
    "primary_final_panel_licensed": primary_licensed if smoke else None,
    "learned_sensitivity_decision": (
        "shall_run" if learned_licensed else "hard_stop"
    ) if smoke else None,
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
