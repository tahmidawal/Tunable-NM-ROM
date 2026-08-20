"""Independent integrity and scientific audit for the N=2048 FiLM sensitivity."""
from __future__ import annotations

import hashlib
import json
import math
import sys

import numpy as np

SOURCE, SMOKE_AUDIT, EXPECTED_COMMIT, EXPECTED_SOURCE_SHA, OUT = sys.argv[1:6]
CONDITIONS = ((1e-6, 1e-2), (1e-8, 1e-4), (1e-10, 1e-5))
TRAJECTORIES = (0, 1, 2, 3)
PAIR_BLOCKS = 6
CHECKPOINT_SHA256 = (
    "aa07cd4a1471c59ad34741b41d620731ebd40d5bf445bd75416ce60358f7ecb9"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


with open(SOURCE) as handle:
    report = json.load(handle)
with open(SMOKE_AUDIT) as handle:
    smoke = json.load(handle)
if not report.get("complete"):
    raise SystemExit("learned sensitivity is incomplete")
if (
    smoke.get("status") != "smoke_pass"
    or smoke.get("learned_sensitivity_decision") != "shall_run"
    or smoke.get("peak_device_fraction_of_limit", 1.0) > 0.75
):
    raise SystemExit("learned sensitivity was not prospectively licensed")

config = report["config"]
expected = {
    "N": 2048,
    "conditions": [
        {"fom_tau": tau, "linear_tol": linear_tol}
        for tau, linear_tol in CONDITIONS
    ],
    "test_seed": 20260828,
    "canonical_draw_count": 16,
    "selected_test_indices": list(TRAJECTORIES),
    "pair_blocks_per_comparison": 6,
    "repetitions_per_arm_per_trajectory_per_comparison": 12,
    "pair_schedule": [
        ["cubic", "film_nmrom"],
        ["film_nmrom", "cubic"],
        ["dynamic", "film_nmrom"],
        ["film_nmrom", "dynamic"],
    ],
    "burn_before_every_pair": True,
    "burn_seconds": 3.0,
    "variant": "lspg:eq256:weak64",
    "weak_test_modes": 64,
    "eq_points": 256,
    "eq_fit_snapshots": "decoder-output snapshots",
    "exact_upwind_weak_contract": True,
    "latent_history_mode": "extrapolation",
    "max_step_jacobians": 2,
    "decode_resolution": 64,
    "decode_chunk": 2,
    "charged_exact_fom_residual_guard_vs_live_cubic": True,
    "same_invocation_nmrom_construction_finish_accuracy_work_residual": True,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "learned_license_fixed_before_primary_outcome": True,
    "reference_residual_gate": 1e-11,
    "f64": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"learned config drifted at {key}: {config.get(key)!r}")
if config.get("smoke_audit_sha256") != sha256(SMOKE_AUDIT):
    raise SystemExit("learned smoke-license hash drifted")

provenance = report["provenance"]
for key, value in {
    "jax_backend": "gpu",
    "gpu_kind": "NVIDIA H200",
    "x64": True,
    "matmul_precision": "highest",
    "commit": EXPECTED_COMMIT,
}.items():
    if provenance.get(key) != value:
        raise SystemExit(f"learned provenance drifted at {key}")
if not provenance.get("slurm_job_id"):
    raise SystemExit("learned Slurm provenance missing")
if provenance.get("source_sha256", {}).get("bh_2048_learned.py") != EXPECTED_SOURCE_SHA:
    raise SystemExit("learned driver source hash drifted")

if set(report["reference_health"]) != {"2048"}:
    raise SystemExit("learned reference grid drifted")
if report["reference_health"]["2048"]["max_reference_newton_residual"] > 1e-11:
    raise SystemExit("learned reference residual gate failed")

offline = report["offline"]
info = offline["eq_info"]
for key, value in {
    "M": 64,
    "m": 256,
    "kind": "weak",
    "candidate_strategy": "deterministic_uniform_tensor_target_grid",
    "exact_full_grid_projection_targets": True,
    "exact_upwind_candidate_stencils": True,
}.items():
    if info.get(key) != value:
        raise SystemExit(f"learned EQ metadata drifted at {key}")
indices = np.asarray(offline["eq_indices"], np.int64)
weights = np.asarray(offline["eq_weights"], np.float64)
if (
    indices.shape != (256,)
    or weights.shape != (256,)
    or np.unique(indices).size != 256
    or np.any(indices // 2048 <= 0)
    or np.any(indices // 2048 >= 2047)
    or np.any(indices % 2048 <= 0)
    or np.any(indices % 2048 >= 2047)
    or not np.all(np.isfinite(weights))
    or np.any(weights <= 0)
):
    raise SystemExit("learned EQ nodes/weights failed")
if (
    offline.get("decode_resolution") != 64
    or offline.get("max_step_jacobians") != 2
    or offline.get("rollout_attempt_budget") != 1
    or offline.get("latent_extrapolation_scale") != 1.0
):
    raise SystemExit("learned deployed construction drifted")

rows = {(row["N"], row["fom_tau"]): row for row in report["rows"]}
expected_rows = {(2048, tau) for tau, _ in CONDITIONS}
if set(rows) != expected_rows or len(report["rows"]) != 3:
    raise SystemExit("learned condition grid incomplete or duplicated")

rendered = []
expected_pair_orders = {
    ("cubic", "cubic/film_nmrom"),
    ("cubic", "film_nmrom/cubic"),
    ("dynamic", "dynamic/film_nmrom"),
    ("dynamic", "film_nmrom/dynamic"),
}
for tau, linear_tol in CONDITIONS:
    row = rows[(2048, tau)]
    if not row.get("complete") or row.get("linear_tol") != linear_tol:
        raise SystemExit(f"learned tau={tau}: row metadata drifted")
    records = row.get("records", [])
    burns = row.get("burn_records", [])
    if len(records) != 192 or len(burns) != 96:
        raise SystemExit(f"learned tau={tau}: raw record/burn count drifted")
    if any(
        burn.get("iterations", 0) <= 0
        or (burn.get("comparison"), "/".join(burn.get("order", [])))
        not in expected_pair_orders
        for burn in burns
    ):
        raise SystemExit(f"learned tau={tau}: burn/order grid failed")
    burn_grid = {
        (
            burn["comparison"], burn["trajectory_index"],
            burn["sample_index"], "/".join(burn["order"]),
        )
        for burn in burns
    }
    if len(burn_grid) != 96:
        raise SystemExit(f"learned tau={tau}: burn grid duplicated")

    for comparison in ("cubic", "dynamic"):
        expected_grid = {
            (trajectory, sample, arm)
            for trajectory in TRAJECTORIES
            for sample in range(12)
            for arm in (comparison, "film_nmrom")
        }
        actual = {
            (record["trajectory_index"], record["sample_index"], record["arm"])
            for record in records
            if record["comparison"] == comparison
        }
        if actual != expected_grid:
            raise SystemExit(f"learned tau={tau} {comparison}: pair grid drifted")
    for record in records:
        arm = record["arm"]
        expected_residuals = 50 if arm in ("dynamic", "film_nmrom") else 0
        expected_inverses = 50 if arm == "dynamic" else 0
        if (
            not math.isfinite(record["elapsed_s"])
            or record["elapsed_s"] <= 0
            or record["breakdowns"]
            or record["flags_nonzero"]
            or record["max_returned_residual"] > tau
            or not math.isfinite(record["trajectory_rel_l2"])
            or record["extra_full_grid_residual_evaluations"] != expected_residuals
            or record["extra_exact_helmholtz_inverses"] != expected_inverses
        ):
            raise SystemExit(f"learned tau={tau}: timed invocation failed")
        if arm == "film_nmrom" and (
            record.get("weak_residual_steps") != 50
            or record.get("weak_test_modes") != 64
            or record.get("eq_points") != 256
            or len(record.get("reduced_jacobians_per_step", [])) != 50
            or max(record["reduced_jacobians_per_step"]) > 2
            or record.get("reduced_jacobians_total", 101) > 100
            or len(record.get("reduced_final_norm_per_step", [])) != 50
        ):
            raise SystemExit(f"learned tau={tau}: reduced-work telemetry failed")

    if max(record["trajectory_rel_l2"] for record in records) >= 1e-4:
        raise SystemExit(f"learned tau={tau}: final accuracy health failed")
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
        raise SystemExit(f"learned tau={tau}: solver equivalence failed")
    if set(row["summaries"]) != {"cubic", "dynamic"}:
        raise SystemExit(f"learned tau={tau}: summaries missing")
    rendered.append({
        "N": 2048,
        "fom_tau": tau,
        "linear_tol": linear_tol,
        "cubic_vs_film": row["summaries"]["cubic"],
        "dynamic_vs_film": row["summaries"]["dynamic"],
    })

memory_points = [report.get("device_memory_at_start", {})]
for row in report["rows"]:
    memory_points.extend((
        row.get("device_memory_after_compile", {}),
        row.get("device_memory_after_condition", {}),
    ))
memory_points.append(report.get("device_memory_at_end", {}))
fractions = [
    point.get("peak_fraction_of_limit")
    for point in memory_points
    if point.get("peak_fraction_of_limit") is not None
]
if not fractions:
    raise SystemExit("learned device-memory telemetry missing")

audit = {
    "status": "complete",
    "classification": "genuine weak FiLM NM-ROM sensitivity",
    "source": SOURCE,
    "source_sha256": sha256(SOURCE),
    "execution_commit": EXPECTED_COMMIT,
    "execution_source_sha256": EXPECTED_SOURCE_SHA,
    "license_smoke_sha256": sha256(SMOKE_AUDIT),
    "peak_device_fraction_of_limit": max(fractions),
    "rows": rendered,
    "film_supported_vs_cubic_cells": sum(
        row["cubic_vs_film"]["supported_film_speedup"] for row in rendered
    ),
    "film_supported_vs_dynamic_cells": sum(
        row["dynamic_vs_film"]["supported_film_speedup"] for row in rendered
    ),
    "total_cells_per_comparison": 3,
}
with open(OUT, "w") as handle:
    json.dump(audit, handle, indent=1, allow_nan=False)
print(json.dumps({
    "status": audit["status"],
    "peak_device_fraction_of_limit": audit["peak_device_fraction_of_limit"],
    "film_supported_vs_cubic_cells": audit["film_supported_vs_cubic_cells"],
    "film_supported_vs_dynamic_cells": audit["film_supported_vs_dynamic_cells"],
}, indent=1))
