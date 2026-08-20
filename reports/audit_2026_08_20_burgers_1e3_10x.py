"""Rerunnable integrity/numerical audit for the Burgers Phase-1 negative result."""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "experiments", "burgers-1e3-10x")
RUNS = os.path.join(EXP, "runs")
OUTPUT = os.path.join(ROOT, "reports", "generated", "burgers_1e3_10x_audit.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with open(path) as handle:
        return json.load(handle)


def check_local_manifest(run):
    directory = os.path.join(RUNS, run)
    checked = []
    with open(os.path.join(directory, "LOCAL.sha256")) as handle:
        for line in handle:
            expected, relative = line.rstrip().split("  ", 1)
            relative = relative.removeprefix("./")
            actual = sha256(os.path.join(directory, relative))
            assert actual == expected, (run, relative, expected, actual)
            checked.append(relative)
    return checked


def git_blob_sha(commit, relative):
    value = subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{commit}:{relative}"]
    )
    return hashlib.sha256(value).hexdigest()


def audit_source_hashes(report):
    commit = report["provenance"]["commit"]
    checked = {}
    for name, expected in report["provenance"]["source_sha256"].items():
        relative = f"experiments/burgers-1e3-10x/{name}"
        actual = git_blob_sha(commit, relative)
        assert actual == expected, (commit, name, expected, actual)
        checked[name] = actual
    return checked


def audit_d0():
    directory = os.path.join(RUNS, "d0_r2")
    report = load(os.path.join(directory, "out", "d0.json"))
    decision = load(os.path.join(directory, "d0_decision.json"))
    log = open(os.path.join(directory, "logs", "2667377.out")).read()
    error = open(os.path.join(directory, "logs", "2667377.err")).read()
    provenance = report["provenance"]
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2667377"
    assert provenance["gpu_kind"] == "NVIDIA H200"
    assert error == "" and "jax_backend=gpu" in log and "ALL-DONE" in log
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert max(
        report["reference_health"][part]["independent_max_relative_residual"]
        for part in ("train", "selection")
    ) <= 1e-8
    for name in ("HG4", "HG5"):
        concept = report["concepts"][name]
        expected = (
            concept["representation_oracle"]["trajectory_mean"] <= 2e-4
            and concept["representation_oracle"]["trajectory_worst"] <= 7e-4
        )
        assert expected == concept["representation_gate_pass"]
        assert expected == decision["concepts"][name]["representation_gate_pass"]
        fraction = decision["concepts"][name][
            "nearest_wall_squared_representation_oracle_error_fraction"
        ]
        assert (fraction >= 0.5) == decision["concepts"][name]["wall_license_pass"]
    assert decision["hard_stop_now"]
    assert not any(
        decision["promotions"][name]
        for name in ("train_HG4", "train_HG5", "run_conditional_D1_HG5S")
    )
    return {
        "manifest_files": check_local_manifest("d0_r2"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"],
        "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"],
        "hard_stop": True,
        "decision_sha256": sha256(os.path.join(directory, "d0_decision.json")),
    }


def audit_fom():
    directory = os.path.join(RUNS, "fom_cal_r4")
    report = load(os.path.join(directory, "out", "fom_cal.json"))
    log = open(os.path.join(directory, "logs", "2667536.out")).read()
    errors = open(os.path.join(directory, "logs", "2667536.err")).read().splitlines()
    known = re.compile(
        r"^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] "
        r"Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$"
    )
    provenance = report["provenance"]
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2667536"
    assert provenance["gpu_kind"] == "NVIDIA A100-PCIE-40GB"
    assert "jax_backend=gpu" in log and "ALL-DONE" in log
    assert len(errors) == 2 and all(known.fullmatch(line) for line in errors)
    assert not report["config"]["confirmation_touched"]
    meshes = {}
    for n in (256, 512, 1024):
        mesh = report["meshes"][str(n)]
        reference = mesh["reference_health"]
        audit = mesh["independent_tighter_reference_difference"]
        assert reference["all_finite"] and reference["zero_breakdowns"] and reference["zero_flags"]
        assert reference["max_returned_relative_residual"] <= reference["outer_tolerance"]
        assert audit["health_gate_pass"] and audit["gate_pass"]
        assert audit["worst"] <= 1e-4
        eligible = []
        for label, row in mesh["rows"].items():
            records = row["timed_records"]
            assert len(records) == 4 * report["config"]["time_reps"]
            for case in range(4):
                assert sum(item["case_index"] == case for item in records) == report["config"]["time_reps"]
            elapsed = [item["elapsed_s"] for item in records]
            assert statistics.median(elapsed) == row["summary"]["median_elapsed_s"]
            if row["summary"]["accuracy_eligible"]:
                eligible.append((row["summary"]["median_elapsed_s"], label))
        selected = min(eligible)[1]
        assert selected == mesh["selected_fastest_eligible"]
        selected_row = mesh["rows"][selected]
        assert selected_row["summary"]["healthy"]
        meshes[str(n)] = {
            "selected": selected,
            "timing_records": len(selected_row["timed_records"]),
            "median_elapsed_s": selected_row["summary"]["median_elapsed_s"],
        }
    return {
        "manifest_files": check_local_manifest("fom_cal_r4"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"],
        "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"],
        "classified_hwloc_lines": len(errors),
        "meshes": meshes,
    }


def audit_s0():
    directory = os.path.join(RUNS, "s0_spline_r1")
    report = load(os.path.join(directory, "out", "s0.json"))
    independent = load(os.path.join(directory, "out", "AUDIT.json"))
    log = open(os.path.join(directory, "logs", "2667808.out")).read()
    error = open(os.path.join(directory, "logs", "2667808.err")).read()
    provenance = report["provenance"]
    assert report["status"] == "complete" and report["decision"]["phase2_hard_stop"]
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2667808" and provenance["gpu_kind"] == "NVIDIA H200"
    assert error == "" and "jax_backend=gpu" in log and "ALL-DONE" in log
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert independent["status"] == "pass"
    assert independent["expected_commit"] == provenance["commit"]
    assert independent["job_id"] == provenance["slurm_job_id"]
    assert independent["source_json_sha256"] == sha256(os.path.join(directory, "out", "s0.json"))
    assert independent["source_npz_sha256"] == sha256(os.path.join(directory, "out", "s0.npz"))
    for arm in ("A", "B", "C"):
        oracle = report["free_oracle"][arm]
        expected_accuracy = all(
            row["trajectory_error_mean"] <= 2e-4 and row["trajectory_error_worst"] <= 7e-4
            for row in oracle["meshes"].values()
        ) and oracle["summary"]["trajectory_error_mean"] <= 2e-4 and oracle["summary"]["trajectory_error_worst"] <= 7e-4
        expected_health = all(row["zero_unhealthy_fits"] for row in oracle["meshes"].values())
        assert oracle["summary"]["every_mesh_and_pooled_accuracy_pass"] == expected_accuracy
        assert oracle["summary"]["zero_unhealthy_fits"] == expected_health
        gate = report["cost_panel"]["gates"][arm]
        assert gate["pass"] == (
            gate["point_ge_10"] and gate["clustered_lower_ge_8"]
            and gate["compiled_peak_memory_le_20GB"] and gate["paired_fom_accuracy_eligible"]
        )
        for suffix in ("direct", "mandatory", "maximum_one"):
            method = f"{arm}_{suffix}"
            records = report["cost_panel"]["records"][method]
            summary = report["cost_panel"]["summaries"][method]
            assert len(records) == 40
            assert statistics.median(row["elapsed_s"] for row in records) == summary["median_elapsed_s"]
    return {
        "manifest_files": check_local_manifest("s0_spline_r1"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"], "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"], "hard_stop": True,
        "independent_audit_sha256": sha256(os.path.join(directory, "out", "AUDIT.json")),
    }


def audit_phase3():
    directory = os.path.join(RUNS, "p3_d_r1")
    report = load(os.path.join(directory, "out", "phase3_d.json"))
    independent = load(os.path.join(directory, "out", "AUDIT.json"))
    log = open(os.path.join(directory, "logs", "2668417.out")).read()
    errors = open(os.path.join(directory, "logs", "2668417.err")).read().splitlines()
    known = re.compile(
        r"^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] "
        r"Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$"
    )
    provenance = report["provenance"]
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2668417" and provenance["gpu_kind"] == "NVIDIA H200"
    assert "jax_backend=gpu" in log and "ALL-DONE" in log
    assert len(errors) == 2 and all(known.fullmatch(line) for line in errors)
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert independent["status"] == "pass"
    assert independent["expected_commit"] == provenance["commit"]
    assert independent["job_id"] == provenance["slurm_job_id"]
    assert independent["source_json_sha256"] == sha256(os.path.join(directory, "out", "phase3_d.json"))
    assert independent["source_npz_sha256"] == sha256(os.path.join(directory, "out", "phase3_d.npz"))
    assert independent["manifest_file_sha256"] == sha256(os.path.join(directory, "MANIFEST.sha256"))
    assert report["decision"] == {
        "selected_solver": None, "selected_kernel": "K3",
        "run_P3_F": False, "phase3_hard_stop": True,
    }
    for solver in ("S1", "S2"):
        summary = report["solver_diagnostic"]["summaries"][solver]
        assert summary["fit_count"] == 128 and summary["all_fit_health_pass"]
        assert summary["worst_relative_normal_residual"] <= 1e-8
        assert summary["no_snapshot_regression_pass"]
        assert summary["target_N128_draw530_trajectory_relative_l2"] > 7e-4
        assert not summary["target_N128_draw530_pass"] and not summary["pass"]
    kernel = report["kernel_diagnostic"]
    assert kernel["fom_accuracy"]["eligible"] and kernel["exact_position_balance"]
    assert kernel["basis_identity"]["pass"] and kernel["pallas_basis_identity"]["pass"]
    for index, route in enumerate(("K0", "K1", "K2", "K3")):
        records, summary, gate = kernel["records"][route], kernel["summaries"][route], kernel["gates"][route]
        assert len(records) == 80
        assert statistics.median(row["elapsed_s"] for row in records) == summary["median_elapsed_s"]
        assert summary["outliers_gt_1p5_within_trajectory_total"] == 0
        assert gate["canonical_work_pass"] and gate["memory_pass"] and gate["fom_eligible"]
        assert gate["pass"] == (
            gate["identity_pass"] and gate["paired_median_speedup"] >= 10
            and gate["clustered_speedup_ci"][0] >= 8
        )
        assert kernel["position_counts"][route] == [4] * 5
        if index:
            assert all(row["pass"] and row["exact_boundary"] for row in kernel["identity"][route])
    assert kernel["selected_kernel"] == "K3"
    return {
        "manifest_files": check_local_manifest("p3_d_r1"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"], "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"], "classified_hwloc_lines": len(errors),
        "selected_solver": None, "selected_kernel": "K3", "hard_stop": True,
        "independent_audit_sha256": sha256(os.path.join(directory, "out", "AUDIT.json")),
    }


def main():
    result = {
        "status": "pass",
        "scope": "Burgers finite Phase 1-3 pure-NMROM negative result",
        "d0": audit_d0(),
        "fom_calibration": audit_fom(),
        "phase2_s0": audit_s0(),
        "phase3_d": audit_phase3(),
        "excluded": {
            "fom_cal_r2": "partial N256/N512 output; driver failed before N1024 reference audit",
            "local_smokes": "execution-only and excluded from scientific claims",
            "synthetic_modelval_and_stopped_p3f_drafts": "uncommitted; no scientific/model-validation data opened",
        },
        "model_validation_opened": False,
        "untouched_confirmation_opened": False,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as handle:
        json.dump(result, handle, indent=1, allow_nan=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
