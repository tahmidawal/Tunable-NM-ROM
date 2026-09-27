"""Independent audit for the joint all-seed Phase-2 spline model-validation cell."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np


if len(sys.argv) != 11:
    raise SystemExit(
        "usage: b10_audit_spline_modelval.py RESULT.json RESULT.npz AUDIT.json "
        "COMMIT JOB_ID MANIFEST EXPECTED_MANIFEST TRAIN11 TRAIN29 TRAIN47"
    )
(
    REPORT_PATH, NPZ_PATH, AUDIT_PATH, EXPECTED_COMMIT, JOB_ID, MANIFEST_PATH,
    EXPECTED_MANIFEST, TRAIN11, TRAIN29, TRAIN47,
) = sys.argv[1:]
TRAIN_PATHS = (TRAIN11, TRAIN29, TRAIN47)
SEEDS = (11, 29, 47)
MODELVAL = {64: 64, 128: 32, 256: 16}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"SPLINE MODELVAL AUDIT FAILED: {message}")


def manifest(path):
    result = {}
    with open(path) as handle:
        for line in handle:
            digest, name = line.rstrip().split(maxsplit=1)
            name = name.lstrip("*")
            require(name not in result, f"duplicate manifest path {name}")
            result[name] = digest
    return result


def load_trainer(path, seed):
    with open(path) as handle:
        report = json.load(handle)
    directory = os.path.dirname(os.path.abspath(path))
    npz_path = os.path.join(directory, report["npz"]["path"])
    checkpoint_path = os.path.join(directory, report["checkpoint"]["path"])
    audit_path = os.path.join(directory, "AUDIT.json")
    with open(audit_path) as handle:
        audit = json.load(handle)
    require(report["status"] == "complete" and report["config"]["training_seed"] == seed,
            f"trainer {seed} identity")
    require(sha256(npz_path) == report["npz"]["sha256"]
            and sha256(checkpoint_path) == report["checkpoint"]["sha256"],
            f"trainer {seed} companion hashes")
    require(audit["status"] == "pass" and audit["source_json_sha256"] == sha256(path)
            and audit["source_npz_sha256"] == sha256(npz_path)
            and audit["source_checkpoint_sha256"] == sha256(checkpoint_path)
            and audit["seed"] == seed, f"trainer {seed} independent audit")
    require(report["selection_oracle"]["gate_pass"] is True
            and report["selection_direct"]["gate_pass"] is True
            and report["gates"]["promote_seed"] is True, f"trainer {seed} gates")
    return report, {
        "json": sha256(path), "npz": sha256(npz_path),
        "checkpoint": sha256(checkpoint_path), "audit": sha256(audit_path),
        "candidate": report["config"]["candidate"],
        "s0": audit["s0_json_sha256"],
    }


def check_metrics(metrics, counts, mean_gate, worst_gate, name):
    pooled = []
    for n, count in counts.items():
        row = metrics["meshes"][str(n)]
        errors = np.asarray(row["trajectory_error_all"], np.float64)
        require(errors.shape == (count,) and np.all(np.isfinite(errors)),
                f"{name} N={n} errors")
        require(np.isclose(np.mean(errors), row["trajectory_error_mean"], rtol=1e-12)
                and np.isclose(np.max(errors), row["trajectory_error_worst"], rtol=1e-12),
                f"{name} N={n} summary")
        require(row["all_finite"] is True and row["exact_binary_boundary"] is True,
                f"{name} N={n} health")
        pooled.extend(errors.tolist())
    pooled = np.asarray(pooled)
    row = metrics["pooled"]
    require(np.allclose(pooled, row["trajectory_error_all"], rtol=0, atol=0)
            and np.isclose(np.mean(pooled), row["trajectory_error_mean"], rtol=1e-12)
            and np.isclose(np.max(pooled), row["trajectory_error_worst"], rtol=1e-12),
            f"{name} pooled summary")
    return bool(all(
        metrics["meshes"][str(n)]["trajectory_error_mean"] <= mean_gate
        and metrics["meshes"][str(n)]["trajectory_error_worst"] <= worst_gate
        for n in counts
    ) and row["trajectory_error_mean"] <= mean_gate
        and row["trajectory_error_worst"] <= worst_gate)


def summarize(records, cases):
    require(len(records) == cases * 6, "timing repetition count")
    canonical = [next(row for row in records if row["case_index"] == case
                      and row["repetition"] == 0) for case in range(cases)]
    errors = np.asarray([row["trajectory_relative_l2"] for row in canonical])
    elapsed = np.asarray([row["elapsed_s"] for row in records])
    require(np.all(np.isfinite(errors)) and np.all(np.isfinite(elapsed))
            and np.all(elapsed > 0), "nonfinite/nonpositive method values")
    case_medians, outliers = [], []
    for case in range(cases):
        values = np.asarray([row["elapsed_s"] for row in records
                             if row["case_index"] == case])
        require(values.size == 6, "per-case timing count")
        median = float(np.median(values)); case_medians.append(median)
        outliers.append(int(np.sum(values > 1.5 * median)))
    return errors, elapsed, case_medians, outliers


with open(REPORT_PATH) as handle:
    report = json.load(handle)
require(report.get("status") == "complete", "result not complete")
require(JOB_ID.isdigit(), "job id nonnumeric")
provenance = report["provenance"]
require(provenance["commit"] == EXPECTED_COMMIT
        and str(provenance["slurm_job_id"]) == JOB_ID, "commit/job provenance")
require(provenance["jax_backend"] == "gpu" and provenance["x64"] is True
        and provenance["matmul_precision"] == "highest", "GPU/f64/highest provenance")
require(sha256(MANIFEST_PATH) == EXPECTED_MANIFEST, "manifest-file hash")
source_manifest = manifest(MANIFEST_PATH)
reported_sources = provenance["source_sha256"]
manifest_sources = {
    os.path.basename(path): digest for path, digest in source_manifest.items()
    if path.startswith("./code/b10_") and path.endswith(".py")
}
require(reported_sources == manifest_sources, "staged b10 source hash mismatch")
config = report["config"]
require(config["seeds"] == list(SEEDS) and config["execution_seeds"] == list(SEEDS),
        "all three seeds not executed")
require(config["N_full_eq"] == 256 and config["model_validation_mix"]
        == [[64, 576, 64], [128, 576, 32], [256, 576, 16]], "split lock")
require(config["model_validation_touched"] is True
        and config["confirmation_touched"] is False
        and config["synthetic_smoke_only"] is False, "data touch flags")
require(config["num_steps"] == 50 and config["time_repetitions"] == 6,
        "step/timing protocol")
candidate = config["candidate"]
require(config["M"] >= max(64, 4 * candidate["k"])
        and config["m"] == 4 * config["M"], "M/m gate")

trainers = [load_trainer(path, seed) for path, seed in zip(TRAIN_PATHS, SEEDS)]
require(all(item[1]["candidate"] == candidate for item in trainers), "trainer candidate")
require(len({item[1]["s0"] for item in trainers}) == 1, "trainer S0 parent")
recorded = report["trainer_gate"]["artifacts"]
require(len(recorded) == 3 and report["trainer_gate"]["deployable_seed_policy"] == 11,
        "trainer gate count/policy")
for index, (seed, (_, hashes)) in enumerate(zip(SEEDS, trainers)):
    row = recorded[index]
    require(row["seed"] == seed and row["json_sha256"] == hashes["json"]
            and row["npz_sha256"] == hashes["npz"]
            and row["checkpoint_sha256"] == hashes["checkpoint"]
            and row["audit_sha256"] == hashes["audit"], f"recorded trainer {seed} hashes")
    expected_paths = {
        f"./code/deps/seed{seed}/train.json": hashes["json"],
        f"./code/deps/seed{seed}/train.npz": hashes["npz"],
        f"./code/deps/seed{seed}/checkpoint.pkl": hashes["checkpoint"],
        f"./code/deps/seed{seed}/AUDIT.json": hashes["audit"],
    }
    for path, digest in expected_paths.items():
        require(source_manifest.get(path) == digest, f"exact trainer manifest path {path}")

for n in MODELVAL:
    health = report["reference_health"][str(n)]
    require(health["all_finite_zero_flags_breakdowns"] is True
            and health["cross_chain_worst"] <= 1e-4, f"N={n} reference aggregate")
    require(len(health["reference_records"]) == MODELVAL[n]
            and len(health["audit_records"]) == MODELVAL[n], f"N={n} reference cases")
    require(all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
                and row["max_returned_relative_residual"] <= health["reference_outer"]
                for row in health["reference_records"]), f"N={n} reference records")
    require(all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
                and row["max_returned_relative_residual"] <= health["audit_outer"]
                for row in health["audit_records"]), f"N={n} audit records")

require(report["npz"]["sha256"] == sha256(NPZ_PATH), "NPZ checksum")
all_seed_pass, full_all, eq_zero_all, base_miss, conditional = True, True, True, False, True
with np.load(NPZ_PATH, allow_pickle=False) as arrays:
    for name in arrays.files:
        value = arrays[name]
        if np.issubdtype(value.dtype, np.number):
            require(np.all(np.isfinite(value)), f"nonfinite NPZ {name}")
    for seed in SEEDS:
        seed_report = report["seeds"][str(seed)]
        reconstruction_pass = check_metrics(
            seed_report["reconstruction"], MODELVAL, 3e-4, 1e-3,
            f"seed{seed} reconstruction",
        )
        require(seed_report["reconstruction"]["gate_pass"] == reconstruction_pass,
                f"seed{seed} reconstruction gate")
        eq_info = seed_report["eq_fit"]
        require(eq_info["training_mesh"] == 256
                and eq_info["training_seed"] == 0
                and eq_info["training_indices"] == [0, 63]
                and eq_info["snapshot_selection_seed"] == 20260824
                and eq_info["snapshot_count"] == 256
                and len(eq_info["snapshot_flat_indices"]) == 256
                and eq_info["M"] == config["M"] and eq_info["m"] == config["m"],
                f"seed{seed} EQ fit protocol")
        indices = arrays[f"seed{seed}_eq_indices"]
        weights = arrays[f"seed{seed}_eq_weights"]
        require(indices.shape == (config["m"],) and weights.shape == (config["m"],)
                and np.all(weights > 0) and np.unique(indices).size == config["m"],
                f"seed{seed} EQ support")
        orders = seed_report["timing_orders"]
        methods = {"direct", "full", "eq"}
        require(len(orders) == 6 and all(set(order) == methods for order in orders),
                f"seed{seed} balanced orders")
        for position in range(3):
            require({order[position] for order in orders} == methods,
                    f"seed{seed} method position coverage")
        recomputed = {}
        for method in methods:
            records = seed_report["records"][method]
            errors, elapsed, case_medians, outliers = summarize(records, 16)
            stored = seed_report["methods"][method]
            require(np.isclose(np.mean(errors), stored["trajectory_error_mean"], rtol=1e-12)
                    and np.isclose(np.max(errors), stored["trajectory_error_worst"], rtol=1e-12)
                    and np.isclose(np.median(elapsed), stored["median_elapsed_s"], rtol=1e-12)
                    and np.allclose(case_medians, stored["per_case_median_elapsed_s"],
                                    rtol=1e-12)
                    and outliers == stored["outliers_gt_1p5_within_trajectory_all"],
                    f"seed{seed} {method} summaries")
            require(stored["zero_failures"] is True, f"seed{seed} {method} failures")
            recomputed[method] = float(np.mean(errors))
        for method in ("full", "eq"):
            for case in range(16):
                prefix = f"seed{seed}_{method}_case{case}_"
                rho_before = arrays[prefix + "rho_before"]
                rho_after = arrays[prefix + "rho_after"]
                attempted = arrays[prefix + "attempted"].astype(bool)
                evaluations = arrays[prefix + "residual_evaluations"]
                accepted = arrays[prefix + "accepted"].astype(bool)
                factor = arrays[prefix + "factor"]
                step_norm = arrays[prefix + "step_norm"]
                stopping = arrays[prefix + "stopping_code"]
                residual_before = arrays[prefix + "residual_before"]
                residual_after = arrays[prefix + "residual_after"]
                jacobian = arrays[prefix + "jacobian"]
                require(rho_before.shape == (50,) and residual_before.shape
                        == (50, config["M"]) and jacobian.shape
                        == (50, config["M"], candidate["k"]), "work array shapes")
                should_attempt = ~(np.isfinite(rho_before) & (rho_before <= 1e-3))
                require(np.array_equal(attempted, should_attempt),
                        "zero/one weak stopping rule")
                require(np.array_equal(evaluations, np.where(attempted, 5, 1)),
                        "residual evaluation accounting")
                require(np.all(step_norm <= .25 + 1e-12), "trust radius")
                require(np.all(np.isin(factor, (0., .25, .5, 1.))), "trial factor")
                require(np.all(stopping[~attempted] == 0)
                        and np.all(stopping[attempted & accepted] == 1)
                        and np.all(stopping[attempted & ~accepted] == 2),
                        "stopping reason codes")
                require(np.all(rho_after[accepted] < rho_before[accepted]),
                        "accepted update did not reduce rho")
                require(np.allclose(residual_after[~accepted], residual_before[~accepted]),
                        "rejected/stopped residual changed")
        degradation = recomputed["eq"] / max(recomputed["full"], 1e-300)
        gates = {
            "reconstruction_mean_le_3e-4_worst_le_1e-3": reconstruction_pass,
            "full_mean_le_7e-4": recomputed["full"] <= 7e-4,
            "eq_mean_le_1e-3": recomputed["eq"] <= 1e-3,
            "eq_over_full_le_1p05": degradation <= 1.05,
            "zero_failures": True,
        }
        require(seed_report["gates"] == gates
                and seed_report["all_gates_pass"] == all(gates.values()),
                f"seed{seed} gates")
        all_seed_pass &= all(gates.values())
        full_all &= gates["full_mean_le_7e-4"]
        eq_zero_all &= seed_report["methods"]["eq"]["zero_failures"]
        base_miss |= not gates["eq_mean_le_1e-3"] or not gates["eq_over_full_le_1p05"]
        conditional &= recomputed["eq"] <= 1.2e-3 and degradation <= 1.20

next_m = bool(full_all and eq_zero_all and base_miss and conditional)
joint = report["joint_gates"]
require(joint["all_three_seeds_pass"] == all_seed_pass
        and joint["next_M_cell_licensed"] == next_m
        and joint["hard_stop"] == bool(not all_seed_pass and not next_m), "joint gates")
audit = {
    "status": "pass", "source_json_sha256": sha256(REPORT_PATH),
    "source_npz_sha256": sha256(NPZ_PATH),
    "source_manifest_sha256": sha256(MANIFEST_PATH),
    "expected_commit": EXPECTED_COMMIT, "job_id": JOB_ID,
    "candidate": candidate, "trainer_json_sha256": [item[1]["json"] for item in trainers],
    "joint_gates": joint,
}
os.makedirs(os.path.dirname(os.path.abspath(AUDIT_PATH)), exist_ok=True)
with open(AUDIT_PATH, "w") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps({"status": "pass", "joint_gates": joint}, indent=2))
