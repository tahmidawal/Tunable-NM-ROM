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


def main():
    result = {
        "status": "pass",
        "scope": "Burgers Phase 1 preregistered representation falsification and FOM calibration",
        "d0": audit_d0(),
        "fom_calibration": audit_fom(),
        "excluded": {
            "fom_cal_r2": "partial N256/N512 output; driver failed before N1024 reference audit",
            "local_smokes": "execution-only and excluded from scientific claims",
        },
        "untouched_confirmation_opened": False,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as handle:
        json.dump(result, handle, indent=1, allow_nan=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
