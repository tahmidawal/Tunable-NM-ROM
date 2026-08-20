"""Producer-side structural check for the bounded N=2048 diagnostic.

This check deliberately does not promote a reference.  It verifies that the
diagnostic persisted the preregistered cohort, single-step instrumentation,
whole-trajectory counting routes, field artifact, and provenance.  An
independent read-only audit is still required after retrieval.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys

import numpy as np


RESULT = sys.argv[1]
AUDIT = sys.argv[2]
EXPECTED_ROUTES = {
    "unpreconditioned_control": (1e-12, 1e-10, "none"),
    "helmholtz_candidate": (1e-12, 1e-8, "helmholtz"),
    "helmholtz_strict_ladder": (1e-13, 1e-10, "helmholtz"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def contains_timing_key(value):
    forbidden = ("elapsed", "timing", "speedup", "wall_time", "burn_record")
    if isinstance(value, dict):
        return any(
            (
                str(key) != "scientific_timing_or_method_rows"
                and any(token in str(key).lower() for token in forbidden)
            )
            or contains_timing_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_timing_key(item) for item in value)
    return False


def trajectory_rank(item):
    value = item["max_final_nonlinear_relative_residual"]
    return (math.inf if value is None else value, -item["trajectory_index"])


def worst_step(values):
    nonfinite = [index for index, value in enumerate(values) if value is None]
    if nonfinite:
        return nonfinite[0] + 1
    return int(np.argmax(np.asarray(values, np.float64))) + 1


def main():
    with open(RESULT) as handle:
        report = json.load(handle)
    config = report["config"]
    provenance = report["provenance"]

    require(report["complete"] is True, "diagnostic is incomplete")
    require(config["scientific_timing_or_method_rows"] is False,
            "diagnostic contains scientific rows")
    require(config["candidate_can_self_promote"] is False,
            "diagnostic can self-promote")
    require(not contains_timing_key(report), "timing field found in diagnostic")
    require(config["N"] == 2048, "wrong mesh")
    require(config["test_seed"] == 20260827, "wrong seed")
    require(config["canonical_draw_count"] == 16, "wrong draw count")
    require(config["trajectory_indices"] == [0, 1, 2, 3], "wrong cohort")
    require(config["fixed_public_jax_newton_iterations"] == 25,
            "wrong fixed Newton count")
    require(config["public_jax_linear_tol"] == 1e-10,
            "wrong public linear tolerance")
    require(config["public_jax_linear_maxiter"] == 2000,
            "wrong public linear maximum")
    require(config["instrumented_jax_version"] == "0.10.2",
            "wrong instrumented JAX version")
    require(config["reference_gate"] == 1e-11, "wrong reference gate")
    require(config["instrumented_update_match_gate"] == 1e-12,
            "wrong update-match gate")
    require(config["tight_route_agreement_gate"] == 1e-10,
            "wrong tight-route agreement gate")

    require(provenance["jax_backend"] == "gpu", "non-GPU diagnostic")
    require(provenance["jax_version"] == "0.10.2", "wrong runtime JAX")
    require(provenance["x64"] is True, "x64 disabled")
    require(provenance["matmul_precision"] == "highest",
            "wrong matmul precision")
    require("H200" in provenance["gpu_kind"], "wrong GPU type")
    require(provenance["commit"] == os.environ["BH_COMMIT"],
            "commit provenance mismatch")
    staged_hashes = {
        os.path.basename(path): sha256(path)
        for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "bh_*.py")))
    }
    require(provenance["source_sha256"] == staged_hashes,
            "source provenance mismatch")

    phase1 = report["fixed25_all_trajectories"]
    require(len(phase1) == 4, "wrong phase-1 trajectory count")
    require([item["trajectory_index"] for item in phase1] == [0, 1, 2, 3],
            "phase-1 trajectory order drift")
    for item in phase1:
        residuals = item["per_step_final_nonlinear_relative_residual"]
        require(len(residuals) == 50, "phase-1 residual history is incomplete")
        require(item["worst_step"] == worst_step(residuals),
                "phase-1 worst-step selection mismatch")
        expected_finite = all(value is not None for value in residuals)
        require(item["all_finite"] is expected_finite,
                "phase-1 finite flag mismatch")
        if expected_finite:
            require(item["max_final_nonlinear_relative_residual"] == max(residuals),
                    "phase-1 maximum mismatch")
        else:
            require(item["max_final_nonlinear_relative_residual"] is None,
                    "nonfinite phase-1 maximum was not null")

    selected = max(phase1, key=trajectory_rank)
    require(config["selected_detailed_trajectory"] == selected["trajectory_index"],
            "detailed trajectory selection mismatch")
    details = report["fixed25_detailed_worst_trajectory"]
    require(details["trajectory_index"] == selected["trajectory_index"],
            "detailed trajectory mismatch")
    require(details["selected_worst_step"] == selected["worst_step"],
            "detailed step mismatch")
    step = details["step"]
    arrays_25 = (
        "pre_nonlinear_relative_residual",
        "post_nonlinear_relative_residual",
        "rhs_norm",
        "public_true_linear_relative_residual",
        "instrumented_true_linear_relative_residual",
        "instrumented_recursive_linear_relative_residual",
        "public_vs_instrumented_update_relative_difference",
        "public_update_norm",
        "public_update_finite",
        "nonlinear_correction_active",
        "public_update_accepted",
        "instrumented_jax_exit_code",
    )
    require(all(len(step[key]) == 25 for key in arrays_25),
            "single-step correction telemetry is incomplete")
    summary = details["summary"]
    require(summary["phase1_reproduction_abs_residual_difference"] is not None,
            "phase-1 residual reproduction is nonfinite")
    require(summary["phase1_reproduction_abs_residual_difference"] <= 1e-12,
            "phase-1 residual reproduction failed")
    require(summary["phase1_reproduction_field_relative_difference"] <= 1e-12,
            "phase-1 field reproduction failed")

    routes = report["counting_routes"]
    require(len(routes) == 3, "wrong counting-route count")
    require({item["name"] for item in routes} == set(EXPECTED_ROUTES),
            "counting-route names drifted")
    by_name = {item["name"]: item for item in routes}
    for name, (outer, inner, preconditioner) in EXPECTED_ROUTES.items():
        item = by_name[name]
        require(item["outer_tol"] == outer and item["linear_tol"] == inner,
                f"{name} tolerance drift")
        require(item["preconditioner"] == preconditioner,
                f"{name} preconditioner drift")
        for key in (
            "per_step_newton_iterations",
            "per_step_linear_iterations",
            "per_step_breakdowns_or_linear_max",
            "per_step_flags",
            "per_step_final_nonlinear_relative_residual",
            "per_step_field_l2_norm",
        ):
            require(len(item[key]) == 50, f"{name} incomplete {key}")

    pairwise = report["pairwise_field_agreement"]
    require(len(pairwise) == 3, "wrong pairwise field-comparison count")
    for item in pairwise.values():
        require(len(item["per_step_relative_difference"]) == 50,
                "incomplete pairwise difference history")

    field_meta = report["persisted_terminal_fields"]
    field_path = os.path.join(os.path.dirname(RESULT), field_meta["path"])
    require(os.path.isfile(field_path), "terminal-field artifact missing")
    require(sha256(field_path) == field_meta["sha256"],
            "terminal-field checksum mismatch")
    with np.load(field_path) as fields:
        require(set(fields.files) == set(EXPECTED_ROUTES),
                "terminal-field arrays drifted")
        for name in EXPECTED_ROUTES:
            require(fields[name].shape == (2048 * 2048,),
                    f"{name} terminal-field shape mismatch")
            require(fields[name].dtype == np.float64,
                    f"{name} terminal-field dtype mismatch")

    update_difference = summary[
        "max_active_public_vs_instrumented_update_relative_difference"
    ]
    update_match_pass = (
        update_difference is not None
        and update_difference <= config["instrumented_update_match_gate"]
    )
    tight_names = ("helmholtz_candidate", "helmholtz_strict_ladder")
    tight_health_pass = all(
        by_name[name]["all_fields_finite"]
        and by_name[name]["breakdowns_or_linear_max_total"] == 0
        and by_name[name]["flags_nonzero"] == 0
        and by_name[name]["max_final_nonlinear_relative_residual"] is not None
        and by_name[name]["max_final_nonlinear_relative_residual"]
        <= config["reference_gate"]
        for name in tight_names
    )
    agreement = pairwise[
        "helmholtz_candidate_vs_helmholtz_strict_ladder"
    ]
    tight_agreement_pass = (
        agreement["max_step_relative_difference"] is not None
        and agreement["trajectory_relative_difference"] is not None
        and agreement["max_step_relative_difference"]
        <= config["tight_route_agreement_gate"]
        and agreement["trajectory_relative_difference"]
        <= config["tight_route_agreement_gate"]
    )
    output = {
        "producer_structural_check_pass": True,
        "independent_audit_still_required": True,
        "does_not_license_reference_replacement_or_timing": True,
        "result_sha256": sha256(RESULT),
        "terminal_fields_sha256": field_meta["sha256"],
        "commit": provenance["commit"],
        "job_id": provenance["slurm_job_id"],
        "gpu_kind": provenance["gpu_kind"],
        "fixed_diagnostic_gate_evaluation": {
            "public_vs_instrumented_update_match": update_match_pass,
            "two_tight_routes_health_and_residual": tight_health_pass,
            "two_tight_routes_field_agreement": tight_agreement_pass,
            "all_pass": bool(
                update_match_pass and tight_health_pass and tight_agreement_pass
            ),
        },
    }
    with open(AUDIT, "w") as handle:
        json.dump(output, handle, indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
