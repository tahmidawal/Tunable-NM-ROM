"""Independent arithmetic audit of retained initializer development arrays.

This reads saved numerical arrays only. It does not evaluate trained models,
change qualification thresholds, or turn a failed experiment into a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gradient(residual, jacobian):
    scale = np.maximum(np.linalg.norm(jacobian, axis=0), 1e-30)
    scale *= max(float(np.linalg.norm(residual)), 1e-30)
    return float(np.max(np.abs(jacobian.T @ residual) / scale))


def objective_guard(value, baseline):
    return bool(value <= baseline + 1024 * np.finfo(np.float64).eps * max(baseline, 1e-300))


def audit_run(run):
    run = Path(run).resolve()
    records = read(run / "records.json")
    config = read(run / "provenance.json")["config"]
    if config.get("policy", {}).get("version", "").startswith("v2-"):
        return audit_anchored_run(run, records, config)
    expected = set(itertools.product(
        [f"{c['seed']}/{i}" for c in config["cohorts"] for i in range(c["count"])],
        config["families"], config["input_paths"], config["gtols"]))
    actual = [(r["case_key"], r["family"], r["input_path"], r["gtol"]) for r in records]
    assert len(actual) == len(set(actual)), "duplicate selected-fit records"
    assert set(actual) == expected, "incomplete or unexpected cohort coverage"
    groups = defaultdict(Counter)
    failures = []
    sources = []
    gradient_gaps = []
    cpu_gpu_classification_disagreements = []
    missing_phase_raw_arrays = []
    original_minimum_phase_selected_differ = []
    for row in records:
        detail_path = run / row["detail"]
        d = read(detail_path)
        fit = d["repair"]
        direct = fit["direct_returned_check"]
        arrays_path = run / d["audit_arrays"]
        assert sha(arrays_path) == d["audit_arrays_sha256"], arrays_path
        arrays = np.load(arrays_path, allow_pickle=False)
        r, j = arrays["returned_residual"], arrays["returned_jacobian"]
        assert r.dtype == j.dtype == np.float64
        assert r.ndim == 1 and j.shape[0] == r.size
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(j))
        assert np.array_equal(arrays["returned_code"], np.asarray(direct["code"]))
        cost = float(.5 * np.dot(r, r))
        eta = gradient(r, j)
        assert np.isclose(cost, direct["cost"], rtol=8e-15, atol=1e-300)
        gap = abs(eta - direct["gradient"])
        gradient_gaps.append(gap)
        assert gap <= 5e-15 + 1e-10 * abs(direct["gradient"]), (row["detail"], gap)
        cpu_pass = eta <= row["gtol"]
        gpu_pass = direct["gradient"] <= row["gtol"]
        if cpu_pass != gpu_pass:
            cpu_gpu_classification_disagreements.append({
                "detail": row["detail"], "numpy_gradient": eta,
                "gpu_gradient": direct["gradient"], "gtol": row["gtol"]})
        baseline = fit["baseline_starts"][fit["baseline_raw"]]
        baseline_cost = float(.5 * np.dot(arrays["baseline_raw_residual"],
                                          arrays["baseline_raw_residual"]))
        assert np.isclose(baseline_cost, baseline["cost"], rtol=8e-15, atol=1e-300)
        global_quality = objective_guard(direct["cost"], baseline["cost"])
        assert global_quality == fit["global_original_objective_quality"]
        expected_solver = bool(direct["finite"] and gpu_pass and global_quality)
        assert expected_solver == fit["qualified"] == row["repaired_pass"]
        phase_raw = d["original_phase_common_raw"]
        phase_checks = d["original_phase_all_common_checks"]
        assert phase_raw == min(range(len(phase_checks)), key=lambda i: phase_checks[i]["cost"])
        phase_quality = objective_guard(direct["cost"], phase_checks[phase_raw]["cost"])
        assert phase_quality == d["phase_objective_quality"] == row["phase_objective_quality"]
        old_selected = d["original_phase"]["best_index"]
        if phase_raw != old_selected:
            original_minimum_phase_selected_differ.append(row["detail"])
        raw_key = "phase_raw_residual"
        if raw_key in arrays:
            phase_cost = float(.5 * np.dot(arrays[raw_key], arrays[raw_key]))
            assert np.isclose(phase_cost, phase_checks[phase_raw]["cost"], rtol=8e-15, atol=1e-300)
        elif phase_raw == old_selected:
            phase_cost = float(.5 * np.dot(arrays["phase_old_residual"], arrays["phase_old_residual"]))
            assert np.isclose(phase_cost, phase_checks[phase_raw]["cost"], rtol=8e-15, atol=1e-300)
        else:
            missing_phase_raw_arrays.append(row["detail"])
        coefficient_gap = float(np.linalg.norm(r - arrays["phase_old_residual"]) /
                                d["original_initial_norm"])
        assert abs(coefficient_gap - d["old_new_coefficient_gap"]) <= 1e-12
        old_new_agreement = d["old_new_field_gap"] <= 1e-6
        assert old_new_agreement == row["old_new_initial_agreement_pass"]
        agreement_path = run / "agreement" / (
            f"{row['family']}_{row['input_path']}_{d['cohort']['seed']}_{d['case']:02d}.json")
        agreement = read(agreement_path)
        expected_pairs = {tuple(x) for x in itertools.combinations(config["gtols"], 2)}
        assert {tuple(x["gtols"]) for x in agreement["pairs"]} == expected_pairs
        assert all(x["passed"] == (x["physical_relative_gap"] <= 1e-6)
                   for x in agreement["pairs"])
        pair_pass = all(x["passed"] for x in agreement["pairs"])
        assert pair_pass == agreement["all_pass"] == row["initial_field_agreement_pass"]
        qualified = expected_solver and phase_quality and old_new_agreement and pair_pass
        assert qualified == row["initializer_qualified"]
        key = f"{row['family']}/{row['input_path']}/{row['gtol']:g}"
        groups[key]["selected_checks"] += 1
        for field in ("original_phase_pass", "original_compiled_pass", "common_baseline_pass",
                      "qr_only_pass", "repaired_pass", "phase_objective_quality",
                      "old_new_initial_agreement_pass", "initial_field_agreement_pass",
                      "initializer_qualified"):
            groups[key][field] += int(row[field])
        if not qualified:
            failures.append({"case_key": row["case_key"], "family": row["family"],
                "input_path": row["input_path"], "gtol": row["gtol"],
                "stationarity": gpu_pass, "compiled_objective_quality": global_quality,
                "phase_objective_quality": phase_quality,
                "old_new_field_agreement": old_new_agreement, "tolerance_pair_agreement": pair_pass})
        sources.append({"detail": str(detail_path), "detail_sha256": sha(detail_path),
                        "arrays": str(arrays_path), "arrays_sha256": sha(arrays_path)})
    return {"integrity_passed": not cpu_gpu_classification_disagreements,
        "scope": "saved-array arithmetic, exact coverage, qualification reconstruction; no new model evaluation",
        "run": str(run), "selected_checks": len(records),
        "qualified_selected_checks": sum(x["initializer_qualified"] for x in records),
        "scientific_initializer_qualification_passed": not failures,
        "groups": {k: dict(v) for k, v in sorted(groups.items())}, "failures": failures,
        "maximum_stationarity_reduction_difference": max(gradient_gaps, default=0.),
        "cpu_gpu_classification_disagreements": cpu_gpu_classification_disagreements,
        "phase_raw_differs_from_original_selected": original_minimum_phase_selected_differ,
        "phase_raw_array_missing_source_and_record_verified_only": missing_phase_raw_arrays,
        "sources": sources}


def audit_anchored_run(run, records, config):
    expected = set(itertools.product(
        [f"{c['seed']}/{i}" for c in config["cohorts"] for i in range(c["count"])],
        config["families"], config["input_paths"], config["gtols"]))
    actual = [(r["case_key"], r["family"], r["input_path"], r["gtol"]) for r in records]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    groups, flags = defaultdict(Counter), Counter()
    failures, sources, gradient_gaps, criterion_disagreements = [], [], [], []
    for row in records:
        detail_path = run / row["detail"]
        d = read(detail_path)
        fit = d["repair"]
        arrays_path = run / d["audit_arrays"]
        assert sha(arrays_path) == d["audit_arrays_sha256"]
        arrays = np.load(arrays_path, allow_pickle=False)
        controls = fit["controls"]
        assert len(controls) == 8 and len(set(fit["seed_indices"])) == 4
        assert [(c["seed_index"], c["kind"]) for c in controls] == [
            (i, k) for i in range(4) for k in ("phase", "core")]
        assert np.array_equal(arrays["control_codes"], np.asarray([c["code"] for c in controls]))
        assert fit["anchor_index"] == min(range(8), key=lambda i: controls[i]["direct"]["cost"])
        assert np.array_equal(arrays["anchor_code"], arrays["control_codes"][fit["anchor_index"]])
        assert np.array_equal(arrays["returned_code"], np.asarray(fit["direct_returned_check"]["code"]))
        states = []
        for label, info, residual, jacobian in (
            ("direct", fit["direct_returned_check"], arrays["returned_residual"], arrays["returned_jacobian"]),
            ("anchored", fit["anchored_returned_check"], arrays["returned_anchored_residual"], arrays["returned_anchored_jacobian"])):
            states.append((label, info, residual, jacobian))
        for i, control in enumerate(controls):
            for label in ("direct", "anchored"):
                states.append((f"control{i}/{label}", control[label],
                    arrays[f"control_{label}_residuals"][i], arrays[f"control_{label}_jacobians"][i]))
        for label, info, residual, jacobian in states:
            assert residual.dtype == jacobian.dtype == np.float64
            assert np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian))
            cost = float(.5 * np.dot(residual, residual))
            eta = gradient(residual, jacobian)
            assert np.isclose(cost, info["cost"], rtol=8e-15, atol=1e-300)
            gap = abs(eta - info["gradient"])
            gradient_gaps.append(gap)
            assert gap <= 5e-15 + 1e-10 * abs(info["gradient"]), (row["detail"], label, gap)
            if label in ("direct", "anchored") and (eta <= row["gtol"]) != (info["gradient"] <= row["gtol"]):
                criterion_disagreements.append({"detail": row["detail"], "evaluation": label,
                    "array_gradient": eta, "stored_gradient": info["gradient"], "gtol": row["gtol"]})
        direct = fit["direct_returned_check"]
        anchored = fit["anchored_returned_check"]
        original_anchor_minimum = min(c["anchored"]["cost"] for c in controls)
        assert original_anchor_minimum == fit["global_original_anchored_minimum"]
        anchored_guard = objective_guard(anchored["cost"], original_anchor_minimum)
        direct_guard = objective_guard(direct["cost"], min(c["direct"]["cost"] for c in controls))
        assert anchored_guard == fit["global_original_objective_quality"]
        assert direct_guard == fit["direct_original_objective_guard"] == row["ordinary_direct_objective_guard"]
        flags["ordinary_direct_objective_guard_failures"] += int(not direct_guard)
        phase_guard = objective_guard(anchored["cost"], min(x["cost"] for x in d["original_phase_all_anchored_checks"]))
        assert phase_guard == row["phase_objective_quality"] == d["phase_objective_quality"]
        phase_raw = d["original_phase_common_raw"]
        phase_direct = d["original_phase_all_common_checks"][phase_raw]
        r_phase = arrays["phase_common_raw_residual"]
        assert np.isclose(.5 * np.dot(r_phase, r_phase), phase_direct["cost"], rtol=8e-15, atol=1e-300)
        ordinary_phase_guard = objective_guard(direct["cost"], phase_direct["cost"])
        assert ordinary_phase_guard == d["ordinary_direct_phase_objective_quality"]
        flags["ordinary_direct_phase_objective_guard_failures"] += int(not ordinary_phase_guard)
        precision = d["independent_arithmetic_audit"]
        assert precision["longdouble_explicit_mantissa_bits"] >= 63
        assert precision["ordinary_direct_guard"] == direct_guard
        with localcontext() as context:
            context.prec = 50
            costs = list(map(Decimal, precision["longdouble_objective_strings"]))
            assert len(costs) == 9
            minimum = min(costs[:-1])
            band = Decimal(1024) * Decimal.from_float(float(np.finfo(np.float64).eps)) * max(minimum, Decimal.from_float(1e-300))
            long_pass = costs[-1] - minimum <= band
            assert long_pass == precision["longdouble_pass"]
            decimal_pass = True
            if precision["decimal_required"]:
                flags["decimal_audited_records"] += 1
                assert precision["decimal_digits"] == 50
                costs = list(map(Decimal, precision["decimal_objective_strings"]))
                minimum = min(costs[:-1])
                band = Decimal(1024) * Decimal.from_float(float(np.finfo(np.float64).eps)) * max(minimum, Decimal.from_float(1e-300))
                assert costs[-1] - minimum == Decimal(precision["decimal_delta"])
                assert band == Decimal(precision["decimal_original_band"])
                decimal_pass = costs[-1] - minimum <= band
                agreement = Decimal(precision["longdouble_decimal_delta_max_discrepancy"]) <= band * Decimal('.01')
                assert decimal_pass == precision["decimal_pass"]
                assert agreement == precision["longdouble_decimal_delta_agreement"]
                decimal_pass = decimal_pass and agreement
            assert direct_guard or precision["decimal_required"]
            precision_pass = bool(long_pass and decimal_pass)
        assert precision_pass == precision["passed"] == row["arithmetic_audit_pass"]
        solver = bool(direct["finite"] and anchored["finite"] and direct["gradient"] <= row["gtol"]
                      and anchored["gradient"] <= row["gtol"] and anchored_guard)
        assert solver == fit["qualified"] == row["repaired_pass"]
        agreement_path = run / "agreement" / f"{row['family']}_{row['input_path']}_{d['cohort']['seed']}_{d['case']:02d}.json"
        agreement = read(agreement_path)
        assert {tuple(x["gtols"]) for x in agreement["pairs"]} == set(itertools.combinations(config["gtols"], 2))
        assert all(x["passed"] == (x["physical_relative_gap"] <= 1e-6) for x in agreement["pairs"])
        pair_pass = all(x["passed"] for x in agreement["pairs"])
        assert pair_pass == agreement["all_pass"] == row["initial_field_agreement_pass"]
        field_pass = d["old_new_field_gap"] <= 1e-6
        assert field_pass == row["old_new_initial_agreement_pass"]
        increment_pass = d["increment_checks"]["passed"]
        assert increment_pass == row["increment_checks_pass"]
        qualified = solver and phase_guard and field_pass and pair_pass and precision_pass and increment_pass
        assert qualified == row["initializer_qualified"]
        key = f"{row['family']}/{row['input_path']}/{row['gtol']:g}"
        groups[key]["selected_checks"] += 1
        for field in ("original_phase_pass", "original_compiled_pass", "repaired_pass", "initializer_qualified"):
            groups[key][field] += int(row[field])
        if not qualified:
            failures.append({"case_key": row["case_key"], "family": row["family"], "input_path": row["input_path"],
                "gtol": row["gtol"], "direct_stationarity": direct["gradient"] <= row["gtol"],
                "anchored_stationarity": anchored["gradient"] <= row["gtol"],
                "anchored_objective": anchored_guard, "phase_objective": phase_guard,
                "physical_agreement": field_pass and pair_pass, "arithmetic_audit": precision_pass,
                "increment_checks": increment_pass})
        sources.append({"detail": str(detail_path), "detail_sha256": sha(detail_path),
                        "arrays": str(arrays_path), "arrays_sha256": sha(arrays_path)})
    return {"integrity_passed": not criterion_disagreements, "run": str(run),
        "scope": "saved direct/anchored array arithmetic, exact coverage and independent precision-record checks; no local model evaluation",
        "selected_checks": len(records), "qualified_selected_checks": sum(x["initializer_qualified"] for x in records),
        "scientific_initializer_qualification_passed": not failures,
        "groups": {k: dict(v) for k, v in sorted(groups.items())}, "failures": failures,
        "maximum_stationarity_reduction_difference": max(gradient_gaps, default=0.),
        "cpu_gpu_classification_disagreements": criterion_disagreements,
        "preserved_arithmetic_flags": dict(flags), "sources": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in (
        "integrity_passed", "selected_checks", "qualified_selected_checks",
        "scientific_initializer_qualification_passed",
        "maximum_stationarity_reduction_difference")}))


if __name__ == "__main__":
    main()
