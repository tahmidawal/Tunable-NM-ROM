"""Independent negative-aware audit for the Phase-4 H1 seed-11 training cell."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

import numpy as np

if len(sys.argv) != 12:
    raise SystemExit(
        "usage: b10_audit_phase4_train.py TRAIN.json TRAIN.npz CHECKPOINT.pkl "
        "AUDIT.json EXPECTED_COMMIT JOB_ID P4.json P4_AUDIT.json "
        "P4_MANIFEST.sha256 MANIFEST.sha256 EXPECTED_MANIFEST_SHA256"
    )
(REPORT_PATH, NPZ_PATH, CHECKPOINT_PATH, AUDIT_PATH, EXPECTED_COMMIT, JOB_ID,
 P4_PATH, P4_AUDIT_PATH, P4_MANIFEST_PATH, MANIFEST_PATH,
 EXPECTED_MANIFEST_SHA256) = sys.argv[1:]
CANDIDATE = {
    "arm": "H1", "R": 48, "P": 32, "k": 24, "M": 96, "m": 384,
    "q": 19, "hyperdecoder_parameters": 111520, "predictor_parameters": 2104,
}
TRAIN_MIX = [[64, 0, 512], [128, 0, 128], [256, 0, 64]]
SELECTION_MIX = [[64, 512, 64], [128, 512, 32], [256, 512, 16]]
TRAIN_SNAPSHOTS, SELECTION_SNAPSHOTS = 704 * 51, 112 * 51


def require(condition, message):
    if not condition:
        raise SystemExit(f"PHASE4 TRAIN AUDIT FAILED: {message}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with open(path) as handle:
        return json.load(handle)


def finite(tree):
    if isinstance(tree, dict):
        return all(finite(value) for value in tree.values())
    if isinstance(tree, (tuple, list)):
        return all(finite(value) for value in tree)
    if isinstance(tree, np.ndarray) and np.issubdtype(tree.dtype, np.number):
        return bool(np.all(np.isfinite(tree)))
    if isinstance(tree, (float, np.floating)):
        return bool(np.isfinite(tree))
    return True


def manifest(path):
    result = {}
    with open(path) as handle:
        for line in handle:
            digest, name = line.rstrip().split(maxsplit=1)
            name = name.lstrip("*")
            require(len(digest) == 64 and name not in result, "malformed manifest")
            result[name] = digest
    return result


def mlp(parameters, inputs):
    hidden = np.asarray(inputs, np.float64)
    for layer in parameters[:-1]:
        value = hidden @ np.asarray(layer["W"]) + np.asarray(layer["b"])
        sigmoid = np.empty_like(value)
        positive = value >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exponential = np.exp(value[~positive])
        sigmoid[~positive] = exponential / (1.0 + exponential)
        hidden = value * sigmoid
    return hidden @ np.asarray(parameters[-1]["W"]) + np.asarray(parameters[-1]["b"])


def layers(parameters, dimensions, name):
    require(len(parameters) == len(dimensions) - 1, f"{name} depth")
    for index, (layer, n_in, n_out) in enumerate(zip(parameters, dimensions[:-1], dimensions[1:])):
        require(np.asarray(layer["W"]).shape == (n_in, n_out), f"{name} W{index}")
        require(np.asarray(layer["b"]).shape == (n_out,), f"{name} b{index}")
        require(finite(layer), f"{name} finite")


def validate_metrics(metrics, counts, name):
    pooled = []
    for n, count in zip((64, 128, 256), counts):
        row = metrics["meshes"][str(n)]
        values = np.asarray(row["trajectory_error_all"], np.float64)
        require(values.shape == (count,) and np.all(np.isfinite(values)), f"{name} N{n} values")
        require(np.isclose(np.mean(values), row["trajectory_error_mean"], rtol=1e-12), f"{name} N{n} mean")
        require(np.isclose(np.max(values), row["trajectory_error_worst"], rtol=1e-12), f"{name} N{n} worst")
        require(row["all_finite"] and row["exact_binary_boundary"], f"{name} N{n} health")
        pooled.extend(values.tolist())
    pooled = np.asarray(pooled)
    row = metrics["pooled"]
    require(np.array_equal(pooled, np.asarray(row["trajectory_error_all"])), f"{name} pooled vector")
    require(np.isclose(np.mean(pooled), row["trajectory_error_mean"], rtol=1e-12), f"{name} pooled mean")
    require(np.isclose(np.max(pooled), row["trajectory_error_worst"], rtol=1e-12), f"{name} pooled worst")
    require(row["all_finite"] and row["exact_binary_boundary"], f"{name} pooled health")


def oracle_gate(metrics):
    return all(
        row["trajectory_error_mean"] <= 2e-4 and row["trajectory_error_worst"] <= 7e-4
        and row["all_finite"] and row["exact_binary_boundary"]
        for row in list(metrics["meshes"].values()) + [metrics["pooled"]]
    )


def direct_gate(direct, oracle):
    result, ratios = True, {}
    for key in ("64", "128", "256", "pooled"):
        row = direct["pooled"] if key == "pooled" else direct["meshes"][key]
        base = oracle["pooled"] if key == "pooled" else oracle["meshes"][key]
        ratios[key] = row["trajectory_error_mean"] / max(base["trajectory_error_mean"], 1e-300)
        result = result and row["trajectory_error_mean"] <= 3e-4 and row["trajectory_error_worst"] <= 1e-3
        result = result and row["all_finite"] and row["exact_binary_boundary"] and ratios[key] <= 1.5
    return bool(result), ratios


require(JOB_ID.isdigit() and len(EXPECTED_COMMIT) == 40, "identity arguments")
require(sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256, "root manifest-file hash")
report, p4, p4_audit = load(REPORT_PATH), load(P4_PATH), load(P4_AUDIT_PATH)
root_manifest = manifest(MANIFEST_PATH)
source_hashes = report["provenance"]["source_sha256"]
staged_sources = {
    os.path.basename(path): digest for path, digest in root_manifest.items()
    if path.startswith("./code/b10_") and path.endswith(".py")
}
require(source_hashes == staged_sources, "staged source hash binding")
require(report["status"] == "complete", "report status")
provenance = report["provenance"]
require(provenance["commit"] == EXPECTED_COMMIT and str(provenance["slurm_job_id"]) == JOB_ID,
        "commit/job provenance")
require(provenance["jax_backend"] == "gpu" and provenance["x64"] is True
        and provenance["matmul_precision"] == "highest", "GPU/f64/highest")
config = report["config"]
require(config["candidate"] == CANDIDATE and config["arm"] == "H1"
        and config["training_seed"] == 11, "candidate/seed")
require(config["training_mix"] == TRAIN_MIX and config["selection_mix"] == SELECTION_MIX
        and config["all_51_times"] is True, "data split/time lock")
require(config["manifold_steps"] == 30000 and config["predictor_steps"] == 20000
        and config["selection_oracle_steps"] == 10000, "step protocol")
require(config["field_batch"] == 32 and config["field_points"] == 512
        and config["selection_oracle_batch"] == 64, "batch protocol")
require(config["model_validation_touched"] is False and config["confirmation_touched"] is False,
        "locked split touched")

p4_json_sha, p4_npz_sha = sha256(P4_PATH), p4["npz"]["sha256"]
decision = {
    "selected_spatial_arm": "H1", "training_seed11_k24_licensed": True,
    "correction_classification": "conditional-zero-or-occasional-attempt",
    "phase4_hard_stop": False, "scientific_promotion_allowed": True,
}
require(p4["decision"] == decision and p4_audit["decision"] == decision, "P4 decision")
require(p4_audit["status"] == "pass" and p4_audit["source_json_sha256"] == p4_json_sha
        and p4_audit["source_npz_sha256"] == p4_npz_sha
        and p4_audit["manifest_sha256"] == sha256(P4_MANIFEST_PATH), "P4 audit chain")
expected_dependencies = {
    "./code/deps/p4/phase4_d.json": p4_json_sha,
    "./code/deps/p4/phase4_d.npz": p4_npz_sha,
    "./code/deps/p4/AUDIT.json": sha256(P4_AUDIT_PATH),
    "./code/deps/p4/MANIFEST.sha256": sha256(P4_MANIFEST_PATH),
}
for path, digest in expected_dependencies.items():
    require(root_manifest.get(path) == digest, f"P4 staged dependency {path}")
gate = report["phase4_gate"]
require(gate["sha256"] == p4_json_sha and gate["npz_sha256"] == p4_npz_sha
        and gate["audit_sha256"] == sha256(P4_AUDIT_PATH)
        and gate["manifest_sha256"] == sha256(P4_MANIFEST_PATH)
        and gate["decision"] == decision
        and gate["source_commit"] == p4["provenance"]["commit"]
        and gate["source_job_id"] == str(p4["provenance"]["slurm_job_id"]),
        "trainer P4 gate")

require(report["npz"]["sha256"] == sha256(NPZ_PATH), "output NPZ checksum")
require(report["checkpoint"]["sha256"] == sha256(CHECKPOINT_PATH), "checkpoint checksum")
with open(CHECKPOINT_PATH, "rb") as handle:
    checkpoint = pickle.load(handle)
require(checkpoint["status"] == "complete" and checkpoint["candidate"] == CANDIDATE
        and checkpoint["training_seed"] == 11 and finite(checkpoint), "checkpoint identity/health")
layers(checkpoint["hyperdecoder"], (19, 32, 32, 48 * 48 + 32 * 32), "hyperdecoder")
layers(checkpoint["direct_predictor_standardized"], (7, 32, 32, 24), "predictor standardized")
layers(checkpoint["direct_predictor_folded_raw"], (7, 32, 32, 24), "predictor folded")
require(np.asarray(checkpoint["training_autolatent_raw"]).shape == (TRAIN_SNAPSHOTS, 19),
        "autolatent shape")
require(len(checkpoint["selection_oracle_optimizer_states"]) == 3,
        "three selection-oracle optimizer states")
require(report["training"]["manifold_history"][-1]["step"] == 30000
        and report["training"]["predictor_history"][-1]["step"] == 20000
        and finite(report["training"]), "complete finite histories")

with np.load(NPZ_PATH, allow_pickle=False) as arrays:
    require(all(finite(arrays[name]) for name in arrays.files), "NPZ finite")
    require(arrays["training_states"].shape == (TRAIN_SNAPSHOTS, 24), "training states")
    require(arrays["selection_oracle_states"].shape == (SELECTION_SNAPSHOTS, 24), "oracle states")
    require(arrays["selection_predictor_states"].shape == (SELECTION_SNAPSHOTS, 24), "direct states")
    require(arrays["manifold_snapshot_schedule"].shape == (30000, 32), "manifold schedule")
    require(arrays["predictor_snapshot_schedule"].shape == (20000, 32), "predictor schedule")
    require(arrays["selection_oracle_snapshot_schedule"].shape == (10000, 64), "oracle schedule")
    train_features = np.asarray(arrays["training_features"])
    select_features = np.asarray(arrays["selection_features"])
    mean, empirical = np.mean(train_features, axis=0), np.std(train_features, axis=0)
    scale = np.where(empirical < 1e-12, 1.0, empirical)
    normalized = mlp(checkpoint["direct_predictor_standardized"], (train_features - mean) / scale)
    folded = mlp(checkpoint["direct_predictor_folded_raw"], train_features)
    select_normalized = mlp(checkpoint["direct_predictor_standardized"], (select_features - mean) / scale)
    select_folded = mlp(checkpoint["direct_predictor_folded_raw"], select_features)
    fold_error = max(float(np.max(np.abs(normalized - folded))),
                     float(np.max(np.abs(select_normalized - select_folded))))
    require(fold_error <= 1e-12, "folded predictor identity")

for row in report["data"]["training"] + report["data"]["selection"]:
    health = row["reference_health"]
    require(row["num_times"] == 51 and health["reported_max_relative_residual"] <= 1e-8
            and health["independent_max_relative_residual"] <= 1e-8, "reference health")
starts = report["selection_oracle"]["starts"]
require([row["start_index"] for row in starts] == [0, 1, 2], "three oracle starts")
for row in starts:
    validate_metrics(row["metrics"], (64, 32, 16), f"oracle start {row['start_index']}")
losses = [row["metrics"]["pooled"]["mean_snapshot_relative_l2_squared"] for row in starts]
chosen = int(np.argmin(losses))
require(report["selection_oracle"]["chosen_start"] == chosen, "oracle chosen start")
oracle = report["selection_oracle"]["metrics"]
direct = report["selection_direct"]["metrics"]
validate_metrics(direct, (64, 32, 16), "direct")
oracle_pass, (direct_pass, ratios) = oracle_gate(oracle), direct_gate(direct, oracle)
gates = report["gates"]
require(gates["representation_oracle_all_N_and_pooled"] == oracle_pass
        and gates["direct_all_N_and_pooled"] == direct_pass
        and gates["promote_seed"] == bool(oracle_pass and direct_pass), "scientific gates")
require(all(np.isclose(ratios[key], report["selection_direct"]["degradation_direct_over_oracle"][key])
            for key in ratios), "degradation ratios")
oracle_rows = list(oracle["meshes"].values()) + [oracle["pooled"]]
k32_near_miss = bool(not oracle_pass and all(
    row["trajectory_error_mean"] <= 4e-4 and row["trajectory_error_worst"] <= 1.4e-3
    and row["all_finite"] and row["exact_binary_boundary"] for row in oracle_rows
))
require(gates["k32_retraining_near_miss_condition"] == k32_near_miss
        and gates["k32_retraining_licensed"] == k32_near_miss, "k32 near-miss license")
loss_near_miss = bool(oracle_pass and not direct_pass and all(
    row["trajectory_error_mean"] <= 6e-4 and row["trajectory_error_worst"] <= 2e-3
    and row["all_finite"] and row["exact_binary_boundary"] and ratios[key] <= 3.0
    for key, row in list(direct["meshes"].items()) + [("pooled", direct["pooled"])]
))
require(gates["local_loss_revision_near_miss_condition"] == loss_near_miss
        and gates["loss_revision_licensed"] is False, "loss-revision gate")
expected_next = (
    "promote H1/k24 seed11" if gates["promote_seed"]
    else "license same-H1 k32 complete retraining" if k32_near_miss
    else "hard stop: direct predictor misses locked gate" if oracle_pass
    else "hard stop: learned manifold misses the 2x bracket"
)
require(gates["phase4_next_decision"] == expected_next, "next-cell decision")

audit = {
    "status": "pass", "negative_aware": True,
    "source_json_sha256": sha256(REPORT_PATH), "source_npz_sha256": sha256(NPZ_PATH),
    "source_checkpoint_sha256": sha256(CHECKPOINT_PATH),
    "source_manifest_sha256": sha256(MANIFEST_PATH),
    "expected_commit": EXPECTED_COMMIT, "job_id": JOB_ID, "arm": "H1", "seed": 11,
    "p4_json_sha256": p4_json_sha, "p4_audit_sha256": sha256(P4_AUDIT_PATH),
    "p4_manifest_sha256": sha256(P4_MANIFEST_PATH),
    "predictor_fold_identity_max_abs": fold_error,
    "oracle_start_selection_losses": losses, "chosen_start": chosen, "gates": gates,
}
os.makedirs(os.path.dirname(os.path.abspath(AUDIT_PATH)), exist_ok=True)
with open(AUDIT_PATH, "w") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
print(json.dumps({"status": "pass", "gates": gates}, indent=2))
