"""Audit saved fresh phase trajectories and their actual initial-fit evidence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from audit_2026_11_23_jcp_ns_initializer import objective_guard, precision_record_verdict


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eta(r, j):
    return float(np.max(np.abs(j.T @ r) /
                        (np.maximum(np.linalg.norm(j, axis=0), 1e-30) * max(np.linalg.norm(r), 1e-30))))


def audit(run):
    run = Path(run).resolve()
    summary = read(run / "summary.json")
    cfg = summary["config"]
    assert cfg["eval_seed"] == 2026112301 and cfg["eval_cases"] == 32
    assert (cfg["n"], cfg["rank"], cfg["k"], cfg["rows"], cfg["dt"], cfg["horizon"]) == (64, 64, 8, 584, .01, .2)
    assert cfg["reference"]["dt"] == .000125 and cfg["reference"]["check_dt"] == .0000625
    assert summary["backend"] == "gpu" and summary["x64"] and summary["matmul_precision"] == "highest"
    assert [c["index"] for c in summary["cases"]] == list(range(32))
    assert summary["source_commit"] == (run.parent.parent / "COMMIT.txt").read_text().strip()
    assert str(summary["job_id"]) == (run.parent.parent / "JOB_ID.txt").read_text().strip()
    registration = summary["registration"]
    assert registration["seed"] == cfg["eval_seed"] and registration["count"] == cfg["eval_cases"]
    registered_parameters = np.asarray(registration["parameters"], dtype=np.float64)
    assert hashlib.sha256(registered_parameters.tobytes()).hexdigest() == registration["parameter_sha256"]
    operator = summary["operator_checks"]
    operator_pass = (max(operator[name] for name in ("A_relative", "derivative_relative", "diffusion_relative",
                                                     "tensor_relative", "subset_test_orthogonality"))
                     <= cfg["qualification"]["operator"] and operator["shift_sign_relative"] <= 1e-5)
    assert summary["operator_pass"] == operator_pass
    expected_runs = set(itertools.product(cfg["arms"], cfg["solver_gtols"]))
    step_count, initial_count = 0, 0
    initial_failures, trajectory_failures, sources = [], [], []
    head_values, span_values, concentrations = [], [], []
    for case in summary["cases"]:
        ci = case["index"]
        assert np.array_equal(np.asarray(case["parameters"]), registered_parameters[ci])
        assert case["normalization"] > 0 and np.isfinite(case["normalization"])
        directory = run / f"case_{ci:03d}"
        assert len(case["runs"]) == len(expected_runs)
        assert {(r["arm"], r["gtol"]) for r in case["runs"]} == expected_runs
        for gtol in cfg["fit_gtols"]:
            file = directory / f"initial_g{gtol:g}.json"
            initial = read(file)
            p = directory / initial["audit_arrays"]
            assert sha(p) == initial["audit_array_sha256"]
            arrays = np.load(p, allow_pickle=False)
            fit = initial["fit"]
            assert initial["gtol"] == gtol and initial["normalization"] == case["normalization"]
            assert np.array_equal(arrays["returned_code"], np.asarray(initial["code"]))
            assert np.array_equal(arrays["target"], np.asarray(initial["target"]))
            anchor_index = min(range(len(fit["controls"])), key=lambda i: fit["controls"][i]["direct"]["cost"])
            assert np.array_equal(arrays["anchor_code"], np.asarray(fit["controls"][anchor_index]["code"]))
            assert np.array_equal(arrays["control_codes"], np.asarray([x["code"] for x in fit["controls"]]))
            for name, label in (("returned", "direct_returned_check"), ("returned_anchored", "anchored_returned_check")):
                residual, jacobian = arrays[name + "_residual"], arrays[name + "_jacobian"]
                assert residual.dtype == jacobian.dtype == np.float64
                expected = fit[label]
                assert np.isclose(.5 * np.dot(residual, residual), expected["cost"], rtol=8e-15, atol=1e-300)
                gradient = eta(residual, jacobian)
                assert abs(gradient - expected["gradient"]) <= 5e-15 + abs(expected["gradient"]) * 1e-10
                assert (gradient <= gtol) == (expected["gradient"] <= gtol)
            assert len(fit["controls"]) == 8
            for i, control in enumerate(fit["controls"]):
                for evaluation in ("direct", "anchored"):
                    residual = arrays[f"control_{evaluation}_residuals"][i]
                    jacobian = arrays[f"control_{evaluation}_jacobians"][i]
                    stored = control[evaluation]
                    assert np.isclose(.5 * np.dot(residual, residual), stored["cost"], rtol=8e-15, atol=1e-300)
                    assert abs(eta(residual, jacobian) - stored["gradient"]) <= 5e-15 + abs(stored["gradient"]) * 1e-10
            direct, anchored = fit["direct_returned_check"], fit["anchored_returned_check"]
            quality = objective_guard(anchored["cost"], min(c["anchored"]["cost"] for c in fit["controls"]))
            original_flag = objective_guard(direct["cost"], min(c["direct"]["cost"] for c in fit["controls"]))
            assert original_flag == initial["ordinary_direct_objective_guard"]
            assert quality == fit["global_original_objective_quality"]
            core_pass = bool(direct["finite"] and anchored["finite"] and direct["gradient"] <= gtol
                             and anchored["gradient"] <= gtol and quality)
            assert core_pass == fit["qualified"] == initial["qualification"]["core_direct_and_anchored"]
            arithmetic = precision_record_verdict(initial["arithmetic_audit"])
            assert arithmetic == initial["qualification"]["independent_objective_arithmetic"]
            assert initial["qualification"]["old_new_initial_field"] == (initial["old_new_initial_field_gap"] <= 1e-6)
            assert initial["qualification"]["increment_consistency"] == initial["increment_checks"]["passed"]
            assert initial["qualification"]["activation_range"] == initial["activation_ranges"]["passed"]
            assert initial["qualified"] == all(initial["qualification"].values())
            assert initial["qualified"] == case["initial_qualification"]["actual_invocations"][str(gtol)]
            if not initial["qualified"]:
                initial_failures.append({"case": ci, "gtol": gtol, "checks": initial["qualification"]})
            initial_count += 1
            sources.extend([{"path": str(file), "sha256": sha(file)}, {"path": str(p), "sha256": sha(p)}])
        arm_metrics = {}
        for result in case["runs"]:
            times = np.asarray(result["times"])
            assert len(times) == 21 and np.allclose(times, np.arange(21) * .01, rtol=0, atol=1e-15)
            records_path = directory / result["records_file"]
            records = read(records_path)["records"]
            assert len(records) == 20
            stationary = all(r["reason"] == "stationarity" and r["gradient"] <= result["gtol"] for r in records)
            assert stationary == result["all_steps_stationary"]
            if not stationary:
                trajectory_failures.append({"case": ci, "arm": result["arm"], "gtol": result["gtol"]})
            step_count += len(records)
            metrics = result["metrics"]
            assert len(metrics) == 21
            errors = np.asarray([x["relative_error"] for x in metrics])
            assert np.all(np.isfinite(errors)) and np.all(errors >= 0)
            assert result["initial_error"] == errors[0] and result["evolved_max_error"] == max(errors[1:])
            checks = result["jacobian_action_checks"]
            assert len(checks) == (20 if result["gtol"] == min(cfg["solver_gtols"]) else 0)
            for check in checks:
                assert check["minimum"] == min(check["relative_errors"])
            assert result["jacobian_action_checked"] == (result["gtol"] == min(cfg["solver_gtols"]))
            expected_action = all(x["minimum"] <= cfg["qualification"]["jacobian_action"] for x in checks) if checks else None
            assert result["jacobian_action_pass"] == expected_action
            p = directory / result["state_file"]
            states = np.load(p, allow_pickle=False)
            assert states["coefficients"].shape == (21, 64) and states["centers"].shape == (21, 3)
            assert states["states"].shape == (21, 11 if result["arm"] == "head_common" else 67)
            assert all(states[name].dtype == np.float64 for name in ("states", "centers", "coefficients", "times"))
            assert np.all(np.isfinite(states["states"]))
            if result["gtol"] == min(cfg["solver_gtols"]):
                arm_metrics[result["arm"]] = {
                    "field_max": float(max(errors[1:])),
                    "aligned_max": float(max(x["shape_aligned_relative_error"] for x in metrics[1:])),
                    "phase_max": float(max(np.linalg.norm(x["phase_displacement"]) for x in metrics[1:]))}
                for metric in metrics:
                    concentrations.extend(metric["pred_centroid_concentration"])
                    concentrations.extend(metric["truth_centroid_concentration"])
            sources.extend([{"path": str(records_path), "sha256": sha(records_path)}, {"path": str(p), "sha256": sha(p)}])
        for refinement in case["tight_tighter"]:
            result = next(x for x in case["runs"] if x["arm"] == refinement["arm"] and x["gtol"] == min(cfg["solver_gtols"]))
            bound = min(cfg["qualification"]["field_absolute"], cfg["qualification"]["field_fraction"] * result["evolved_max_error"])
            assert bound == refinement["bound"] and refinement["max_gap"] == max(refinement["per_time_gap"])
            assert refinement["passed"] == (refinement["max_gap"] <= bound)
        reference = case["reference"]
        bound = min(cfg["qualification"]["reference_absolute"], cfg["qualification"]["reference_fraction"] * min(x["evolved_max_error"] for x in case["runs"]))
        assert bound == reference["bound"] and reference["max_gap"] == max(reference["per_time_gap"])
        reference_pass = (reference["max_gap"] <= bound
                          and max(reference["initial_projection_gap"], reference["fine_initial_projection_gap"]) <= 1e-12)
        assert reference["passed"] == reference_pass
        actual_common = case["actual_common_initial_state_checks"]
        assert {x["gtol"] for x in actual_common} == set(cfg["solver_gtols"])
        common_pass = all(max(x[k] for k in ("head_vs_span", "head_vs_qualified_fit", "span_vs_qualified_fit"))
                          <= cfg["qualification"]["common_state"] for x in actual_common)
        common_pass = common_pass and case["shared_ic_parity"] <= cfg["qualification"]["common_state"]
        checks = case["qualification"]
        assert checks["passed"] == all(v for k, v in checks.items() if k != "passed")
        assert checks["rollout"] == all(x["all_steps_stationary"] for x in case["runs"])
        assert checks["reference"] == reference["passed"]
        assert checks["initialization"] == case["initial_qualification"]["passed"]
        assert checks["common_state"] == common_pass
        assert checks["refinement"] == all(x["passed"] for x in case["tight_tighter"])
        assert checks["jacobian"] == all(x["jacobian_action_pass"] for x in case["runs"] if x["jacobian_action_checked"])
        assert checks["operator"] == summary["operator_pass"]
        initial_checks = case["initial_qualification"]
        assert initial_checks["passed"] == (all(initial_checks["actual_invocations"].values())
                                            and initial_checks["physical_tolerance_gap"] <= initial_checks["bound"])
        head_values.append({"case": ci, **arm_metrics["head_common"]})
        span_values.append({"case": ci, **arm_metrics["span_common"]})
    ratios = {}
    for metric in ("field_max", "aligned_max", "phase_max"):
        h = np.asarray([x[metric] for x in head_values])
        s = np.asarray([x[metric] for x in span_values])
        ratios[metric] = {"median_paired_span_over_head": float(np.median(s / h)),
                          "head_lower_count": int(np.sum(h < s)),
                          "head_median": float(np.median(h)), "head_worst": float(max(h)),
                          "span_median": float(np.median(s)), "span_worst": float(max(s))}
    all_pass = all(c["qualification"]["passed"] for c in summary["cases"])
    assert all_pass == summary["qualification"]["all_cases_pass"]
    return {"integrity_passed": True, "scope": "retained arrays, raw solver records, exact coverage and metric aggregation; field/refinement norms source-reviewed",
            "case_count": 32, "actual_initial_fit_invocations": initial_count, "recorded_trajectory_steps": step_count,
            "scientific_phase_confirmation_passed": all_pass,
            "initial_failures": initial_failures, "trajectory_failures": trajectory_failures,
            "case_checks": [{"case": c["index"], **c["qualification"]} for c in summary["cases"]],
            "paired_metrics": ratios, "head_case_maxima": head_values, "span_case_maxima": span_values,
            "minimum_centroid_concentration": float(min(concentrations)),
            "sources": [{"path": str(run / "summary.json"), "sha256": sha(run / "summary.json")}, *sources]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "head_case_maxima", "span_case_maxima", "case_checks")}))
