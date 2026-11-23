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
