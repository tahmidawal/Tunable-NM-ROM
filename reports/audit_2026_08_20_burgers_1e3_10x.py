"""Rerunnable integrity/numerical audit for the Burgers Phase-1--6 search."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import statistics
import subprocess

import numpy as np

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


def audit_phase4():
    directory = os.path.join(RUNS, "p4_d_r3")
    report_path = os.path.join(directory, "out", "phase4_d.json")
    npz_path = os.path.join(directory, "out", "phase4_d.npz")
    report = load(report_path)
    independent = load(os.path.join(directory, "out", "AUDIT.json"))
    log = open(os.path.join(directory, "logs", "2668794.out")).read()
    errors = open(os.path.join(directory, "logs", "2668794.err")).read().splitlines()
    known = re.compile(
        r"^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] "
        r"Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$"
    )
    provenance = report["provenance"]
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2668794"
    assert provenance["gpu_kind"] == "NVIDIA H200"
    assert "jax_backend=gpu" in log and "ALL-DONE" in log
    assert len(errors) == 2 and all(known.fullmatch(line) for line in errors)
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert independent["status"] == "pass"
    assert independent["expected_commit"] == provenance["commit"]
    assert independent["expected_job"] == provenance["slurm_job_id"]
    assert independent["source_json_sha256"] == sha256(report_path)
    assert independent["source_npz_sha256"] == sha256(npz_path)
    assert independent["manifest_sha256"] == sha256(os.path.join(directory, "MANIFEST.sha256"))
    expected_decision = {
        "selected_spatial_arm": "H1",
        "training_seed11_k24_licensed": True,
        "correction_classification": "conditional-zero-or-occasional-attempt",
        "phase4_hard_stop": False,
        "scientific_promotion_allowed": True,
    }
    assert report["decision"] == expected_decision
    assert independent["decision"] == expected_decision
    for arm in ("H1", "H2"):
        oracle = report["free_oracle"][arm]
        assert oracle["summary"]["pass"]
        assert oracle["summary"]["trajectory_mean"] <= 2e-4
        assert oracle["summary"]["trajectory_worst"] <= 7e-4
        for mesh in oracle["meshes"].values():
            assert mesh["healthy_count"] == mesh["fit_count"]
            assert mesh["normal_worst"] <= 1e-8
            assert mesh["trajectory_mean"] <= 2e-4 and mesh["trajectory_worst"] <= 7e-4
            assert mesh["exact_boundary"] and mesh["max_support"] <= 32
            assert mesh["s0_no_regression"] and mesh["p3_fixed_subset_no_regression"]
        gate = report["cost_panel"]["gates"][arm]
        assert gate["training_cost_license"] and gate["mandatory"]["pass"]
        assert not gate["maximum_one"]["pass"]
    panel = report["cost_panel"]
    assert panel["fom_accuracy"]["eligible"] and panel["exact_position_balance"]
    for method in ("fom", "H1_mandatory", "H1_maximum_one", "H2_mandatory", "H2_maximum_one"):
        records, summary = panel["records"][method], panel["summaries"][method]
        assert len(records) == 80
        assert statistics.median(row["elapsed_s"] for row in records) == summary["median_elapsed_s"]
        assert summary["outliers_gt_1p5_within_trajectory_total"] == 0
        assert panel["position_counts"][method] == [4] * 5
    for arm in ("H1", "H2"):
        assert all(row["pass"] and row["exact_boundary"] for row in panel["identity"][arm])
    return {
        "manifest_files": check_local_manifest("p4_d_r3"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"], "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"], "classified_hwloc_lines": len(errors),
        "selected_spatial_arm": "H1", "training_seed11_k24_licensed": True,
        "correction_classification": expected_decision["correction_classification"],
        "independent_audit_sha256": sha256(os.path.join(directory, "out", "AUDIT.json")),
    }


def audit_phase4_train():
    directory = os.path.join(RUNS, "p4_h1_s11_r1")
    report_path = os.path.join(directory, "out", "train.json")
    npz_path = os.path.join(directory, "out", "train.npz")
    checkpoint_path = os.path.join(directory, "out", "checkpoint.pkl")
    report = load(report_path)
    independent = load(os.path.join(directory, "out", "AUDIT.json"))
    log = open(os.path.join(directory, "logs", "2668956.out")).read()
    errors = open(os.path.join(directory, "logs", "2668956.err")).read().splitlines()
    known = re.compile(
        r"^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] "
        r"Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$"
    )
    provenance = report["provenance"]
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2668956"
    assert provenance["gpu_kind"] == "NVIDIA H200"
    assert "jax_backend=gpu" in log and "ALL-DONE" in log
    assert len(errors) == 2 and all(known.fullmatch(line) for line in errors)
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert report["config"]["training_seed"] == 11
    assert report["config"]["candidate"] == {
        "arm": "H1", "R": 48, "P": 32, "k": 24, "M": 96, "m": 384,
        "q": 19, "hyperdecoder_parameters": 111520, "predictor_parameters": 2104,
    }
    assert independent["status"] == "pass" and independent["negative_aware"]
    assert independent["expected_commit"] == provenance["commit"]
    assert independent["job_id"] == provenance["slurm_job_id"]
    assert independent["source_json_sha256"] == sha256(report_path)
    assert independent["source_npz_sha256"] == sha256(npz_path)
    assert independent["source_checkpoint_sha256"] == sha256(checkpoint_path)
    assert independent["source_manifest_sha256"] == sha256(
        os.path.join(directory, "MANIFEST.sha256")
    )
    assert independent["state_consistency_recomputed"]
    assert independent["deterministic_schedules_recomputed"]
    assert independent["metadata_and_history_recomputed"]
    assert independent["predictor_fold_identity_max_abs"] <= 1e-12
    assert report["npz"]["sha256"] == sha256(npz_path)
    assert report["checkpoint"]["sha256"] == sha256(checkpoint_path)
    assert audit_source_hashes(report)
    with open(checkpoint_path, "rb") as handle:
        checkpoint = pickle.load(handle)
    with np.load(npz_path, allow_pickle=False) as arrays:
        assert np.array_equal(
            arrays["training_states"],
            np.concatenate((arrays["training_affine"], np.tanh(arrays["training_q_raw"])), axis=1),
        )
        assert np.array_equal(checkpoint["training_autolatent_raw"], arrays["training_q_raw"])
        chosen = report["selection_oracle"]["chosen_start"]
        chosen_raw = arrays[f"selection_oracle_start{chosen}_q_raw"]
        assert np.array_equal(arrays["selection_oracle_chosen_q_raw"], chosen_raw)
        assert np.array_equal(
            arrays["selection_oracle_states"],
            np.concatenate((arrays["selection_affine"], np.tanh(chosen_raw)), axis=1),
        )
        assert np.all(np.isfinite(arrays["selection_predictor_states"]))

    def validate_metrics(metrics):
        pooled = []
        for n, expected_count in ((64, 64), (128, 32), (256, 16)):
            row = metrics["meshes"][str(n)]
            values = np.asarray(row["trajectory_error_all"], np.float64)
            assert values.shape == (expected_count,) and np.all(np.isfinite(values))
            assert np.isclose(np.mean(values), row["trajectory_error_mean"], rtol=1e-12)
            assert np.isclose(np.max(values), row["trajectory_error_worst"], rtol=1e-12)
            assert row["all_finite"] and row["exact_binary_boundary"]
            pooled.extend(values.tolist())
        pooled = np.asarray(pooled)
        assert np.array_equal(pooled, np.asarray(metrics["pooled"]["trajectory_error_all"]))
        assert np.isclose(np.mean(pooled), metrics["pooled"]["trajectory_error_mean"], rtol=1e-12)
        assert np.isclose(np.max(pooled), metrics["pooled"]["trajectory_error_worst"], rtol=1e-12)
        assert metrics["pooled"]["all_finite"] and metrics["pooled"]["exact_binary_boundary"]

    oracle, direct = report["selection_oracle"]["metrics"], report["selection_direct"]["metrics"]
    validate_metrics(oracle)
    validate_metrics(direct)
    oracle_pass = all(
        row["trajectory_error_mean"] <= 2e-4 and row["trajectory_error_worst"] <= 7e-4
        for row in list(oracle["meshes"].values()) + [oracle["pooled"]]
    )
    direct_pass = all(
        row["trajectory_error_mean"] <= 3e-4 and row["trajectory_error_worst"] <= 1e-3
        and row["trajectory_error_mean"] / max(base["trajectory_error_mean"], 1e-300) <= 1.5
        for row, base in [
            *( (direct["meshes"][key], oracle["meshes"][key]) for key in ("64", "128", "256") ),
            (direct["pooled"], oracle["pooled"]),
        ]
    )
    k32_near_miss = (not oracle_pass) and all(
        row["trajectory_error_mean"] <= 4e-4 and row["trajectory_error_worst"] <= 1.4e-3
        for row in list(oracle["meshes"].values()) + [oracle["pooled"]]
    )
    gates = report["gates"]
    assert gates["representation_oracle_all_N_and_pooled"] == oracle_pass
    assert gates["direct_all_N_and_pooled"] == direct_pass
    assert gates["promote_seed"] == (oracle_pass and direct_pass)
    assert gates["k32_retraining_near_miss_condition"] == k32_near_miss
    assert not gates["k32_retraining_licensed"] and not gates["loss_revision_licensed"]
    assert gates["phase4_next_decision"] == "hard stop: learned manifold misses the 2x bracket"
    assert independent["gates"] == gates
    assert report["training"]["manifold_history"][-1]["step"] == 30000
    assert report["training"]["predictor_history"][-1]["step"] == 20000
    return {
        "manifest_files": check_local_manifest("p4_h1_s11_r1"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"], "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"], "classified_hwloc_lines": len(errors),
        "oracle_pooled_mean": oracle["pooled"]["trajectory_error_mean"],
        "oracle_pooled_worst": oracle["pooled"]["trajectory_error_worst"],
        "direct_pooled_mean": direct["pooled"]["trajectory_error_mean"],
        "direct_pooled_worst": direct["pooled"]["trajectory_error_worst"],
        "hard_stop": True, "next_decision": gates["phase4_next_decision"],
        "independent_audit_sha256": sha256(os.path.join(directory, "out", "AUDIT.json")),
    }


def audit_phase5_d():
    directory = os.path.join(RUNS, "p5_d_r1")
    report_path = os.path.join(directory, "out", "phase5_d.json")
    npz_path = os.path.join(directory, "out", "phase5_d.npz")
    report = load(report_path)
    independent = load(os.path.join(directory, "out", "AUDIT.json"))
    log = open(os.path.join(directory, "logs", "2669249.out")).read()
    errors = open(os.path.join(directory, "logs", "2669249.err")).read().splitlines()
    known = re.compile(
        r"^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] "
        r"Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$"
    )
    provenance = report["provenance"]
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2669249"
    assert provenance["gpu_kind"] == "NVIDIA H200"
    assert "jax_backend=gpu" in log and "ALL-DONE" in log
    assert len(errors) == 3 and all(known.fullmatch(line) for line in errors)
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert independent["status"] == "pass" and independent["negative_aware"]
    assert independent["source_json_sha256"] == sha256(report_path)
    assert independent["source_npz_sha256"] == sha256(npz_path)
    assert report["npz"]["sha256"] == sha256(npz_path)
    assert audit_source_hashes(report)

    targets = report["train_targets"]
    assert targets["integrity_pass"]
    assert targets["snapshot_count"] == targets["expected_snapshot_count"] == 35904
    assert targets["chunk_count"] == len(targets["chunks"]) == 16
    assert sum(row["healthy_count"] for row in targets["chunks"]) == 35904
    assert max(row["normal_worst"] for row in targets["chunks"]) <= 1e-8
    assert all(
        row["boundary_all"] and row["pou_worst"] <= 1e-12
        and row["support_max"] <= 32 and row["rhs_finite_all"]
        and row["prediction_finite_all"] and row["coefficient_finite_all"]
        for row in targets["chunks"]
    )

    panel = report["cost_panel"]
    assert panel["fom_accuracy"]["eligible"] and panel["exact_position_balance"]
    for method in ("fom", "G1_mandatory", "G1_maximum_one", "G2_mandatory", "G2_maximum_one"):
        records, summary = panel["records"][method], panel["summaries"][method]
        assert len(records) == 80
        assert statistics.median(row["elapsed_s"] for row in records) == summary["median_elapsed_s"]
        assert summary["outliers_gt_1p5_within_trajectory_total"] == 0
        assert panel["position_counts"][method] == [4] * 5
    for arm in ("G1", "G2"):
        identity_pass = all(row["pass"] for row in panel["identity"][arm])
        gate = panel["gates"][arm]
        assert not identity_pass and not gate["identity_pass"]
        assert panel["geometry"][arm]["pass"] and gate["noncollapse_pass"]
        assert gate["mandatory"]["paired_median_speedup"] >= 10
        assert gate["mandatory"]["clustered_speedup_ci"][0] >= 8
        assert not gate["mandatory"]["pass"] and not gate["training_cost_license"]
        assert all(
            row["relative_l2"]["full_fields"] < 1e-14
            and row["relative_l2"]["weak_residual"] > report["config"]["identity_tolerance"]
            and row["exact_boundary"]
            for row in panel["identity"][arm]
        )
    expected = {
        "target_integrity_pass": True,
        "arm_training_licenses": {"G1": False, "G2": False},
        "next_seed11_arm": None,
        "phase5_hard_stop": True,
        "scientific_promotion_allowed": True,
    }
    assert report["decision"] == independent["decision"] == expected
    return {
        "manifest_files": check_local_manifest("p5_d_r1"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"], "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"], "classified_hwloc_lines": len(errors),
        "target_fits": targets["snapshot_count"],
        "target_normal_worst": max(row["normal_worst"] for row in targets["chunks"]),
        "hard_stop": True,
        "independent_audit_sha256": sha256(os.path.join(directory, "out", "AUDIT.json")),
    }


def audit_phase6_d():
    directory = os.path.join(RUNS, "p6_d_r1")
    report_path = os.path.join(directory, "out", "phase6_d.json")
    npz_path = os.path.join(directory, "out", "phase6_d.npz")
    audit_path = os.path.join(directory, "out", "AUDIT.json")
    report, independent = load(report_path), load(audit_path)
    log = open(os.path.join(directory, "logs", "2669652.out")).read()
    errors = open(os.path.join(directory, "logs", "2669652.err")).read().splitlines()
    known = re.compile(
        r"^E[0-9]{4} [0-9:.]+ [0-9]+ numa_hwloc\.cc:121\] "
        r"Call to hwloc_set_cpubind\(\) failed: Invalid argument \[22\]$"
    )
    provenance, panel = report["provenance"], report["cost_panel"]
    expected_decision = {
        "repair_licensed": True,
        "phase6_hard_stop": False,
        "training_authorized": False,
        "next_action": "separate training proposal/audit",
        "scientific_promotion_allowed": True,
    }
    assert report["status"] == "complete"
    assert provenance["jax_backend"] == "gpu" and provenance["x64"]
    assert provenance["matmul_precision"] == "highest"
    assert provenance["slurm_job_id"] == "2669652"
    assert provenance["gpu_kind"] == "NVIDIA H200"
    assert "jax_backend=gpu" in log and "ALL-DONE" in log
    assert len(errors) == 3 and all(known.fullmatch(line) for line in errors)
    assert report["decision"] == independent["decision"] == expected_decision
    assert independent["status"] == "pass" and independent["negative_aware"]
    assert independent["expected_commit"] == provenance["commit"]
    assert independent["expected_job"] == provenance["slurm_job_id"]
    assert independent["source_json_sha256"] == sha256(report_path)
    assert independent["source_npz_sha256"] == sha256(npz_path)
    assert report["npz"]["sha256"] == sha256(npz_path)
    assert independent["manifest_sha256"] == sha256(os.path.join(directory, "MANIFEST.sha256"))
    assert not report["config"]["p5_targets_regenerated"]
    assert not report["config"]["training_touched"]
    assert not report["config"]["model_validation_touched"]
    assert not report["config"]["confirmation_touched"]
    assert audit_source_hashes(report)

    health = panel["reference_health"]
    assert health["reference_outer"] == 1e-12 and health["audit_outer"] == 3e-13
    assert len(health["reference_records"]) == len(health["audit_records"]) == 4
    assert all(
        row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
        and row["max_returned_relative_residual"] <= tolerance
        for records, tolerance in (
            (health["reference_records"], health["reference_outer"]),
            (health["audit_records"], health["audit_outer"]),
        ) for row in records
    )
    differences = [row["trajectory_relative_l2"] for row in health["audit_records"]]
    assert np.array_equal(differences, health["cross_chain_trajectory_relative_l2_all"])
    assert max(differences) == health["cross_chain_worst"] <= 1e-4

    assert panel["exact_position_balance"]
    for method in ("fom", "R0_polynomial_weak", "R1_cox_weak_k3_full"):
        records, summary = panel["records"][method], panel["summaries"][method]
        assert len(records) == 4 * report["config"]["time_repetitions"] == 96
        assert len({(row["case_index"], row["repetition"]) for row in records}) == 96
        assert np.median([row["elapsed_s"] for row in records]) == summary["median_elapsed_s"]
        assert summary["outliers_gt_1p5_within_trajectory_total"] == 0
        assert panel["position_counts"][method] == [8, 8, 8]
    assert panel["fom_accuracy"]["eligible"] and panel["fom_accuracy"]["healthy"]
    assert panel["fom_accuracy"]["mean"] <= 1e-3
    assert panel["fom_accuracy"]["worst"] <= 3e-3

    r0 = panel["identity"]["R0_polynomial_weak"]
    r1 = panel["identity"]["R1_cox_weak_k3_full"]
    assert len(r0) == len(r1) == 4
    assert not all(row["pass"] for row in r0)
    assert all(row["pass"] and row["exact_boundary"] for row in r1)
    assert max(row["max_relative_l2"] for row in r1) <= report["config"]["identity_tolerance"]
    assert all(
        row["relative_l2"][name] == 0.0
        for row in r1
        for name in ("current_stencils", "previous_centers", "weak_residual", "rho")
    )
    consistency = panel["actual_route_consistency"]
    assert len(consistency) == 4 and all(row["pass"] for row in consistency)
    assert max(row["max_relative_l2"] for row in consistency) <= report["config"]["identity_tolerance"]
    for row in panel["canonical_work"]["R1_cox_weak_k3_full"]:
        assert row["finite"] and row["zero_failures"]
        assert row["output_shape"] == [51, 1024 * 1024]
        assert row["weak_objective_evaluations"] == 50
        assert row["weak_jacobian_evaluations"] == row["trial_residual_evaluations"] == 0
        assert row["coefficient_grid_evaluations"] == 51
        assert np.all(np.isfinite(row["rho_all"]))
        assert np.all(np.isfinite(row["weak_residual_norm_all"]))
    gate = panel["gate"]
    assert gate["pass"] and gate["identity_pass"] and gate["canonical_work_pass"]
    assert gate["fom_eligible"] and gate["memory_pass"]
    assert gate["compiled_device_bytes"] <= report["config"]["memory_limit_bytes"]
    assert gate["paired_median_speedup"] >= 10.0
    assert gate["clustered_speedup_ci"][0] >= 8.0
    return {
        "manifest_files": check_local_manifest("p6_d_r1"),
        "source_hashes": audit_source_hashes(report),
        "backend": provenance["jax_backend"], "gpu": provenance["gpu_kind"],
        "job_id": provenance["slurm_job_id"], "classified_hwloc_lines": len(errors),
        "repair_licensed": True,
        "paired_median_speedup": gate["paired_median_speedup"],
        "clustered_speedup_ci": gate["clustered_speedup_ci"],
        "identity_worst": max(row["max_relative_l2"] for row in r1),
        "actual_route_identity_worst": max(row["max_relative_l2"] for row in consistency),
        "independent_audit_sha256": sha256(audit_path),
    }


def main():
    result = {
        "status": "pass",
        "scope": "Burgers finite Phase 1-6 pure-NMROM search checkpoint",
        "d0": audit_d0(),
        "fom_calibration": audit_fom(),
        "phase2_s0": audit_s0(),
        "phase3_d": audit_phase3(),
        "phase4_d": audit_phase4(),
        "phase4_h1_seed11": audit_phase4_train(),
        "phase5_d": audit_phase5_d(),
        "phase6_d": audit_phase6_d(),
        "excluded": {
            "fom_cal_r2": "partial N256/N512 output; driver failed before N1024 reference audit",
            "local_smokes": "execution-only and excluded from scientific claims",
            "phase4_r1": "pre-science dependency-schema failure; zero scientific output",
            "phase4_r2": "completed output excluded without scientific inspection because the root manifest omitted the nested P3 manifest",
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
