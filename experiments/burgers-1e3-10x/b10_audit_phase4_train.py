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
AUDIT_SMOKE = os.environ.get("B10_AUDIT_SMOKE", "0") == "1"
CANDIDATE = {
    "arm": "H1", "R": 48, "P": 32, "k": 24, "M": 96, "m": 384,
    "q": 19, "hyperdecoder_parameters": 111520, "predictor_parameters": 2104,
}
TRAIN_MIX = [[64, 0, 512], [128, 0, 128], [256, 0, 64]]
SELECTION_MIX = [[64, 512, 64], [128, 512, 32], [256, 512, 16]]
TRAIN_SNAPSHOTS, SELECTION_SNAPSHOTS = 704 * 51, 112 * 51
if AUDIT_SMOKE:
    EXPECTED_STATUS = "excluded_execution_smoke"
    TRAIN_SPEC, SELECTION_SPEC = ((24, 0, 4, 3),), ((28, 4, 2, 3),)
    TRAIN_SNAPSHOTS, SELECTION_SNAPSHOTS = 12, 6
    MANIFOLD_STEPS = PREDICTOR_STEPS = ORACLE_STEPS = 1
    FIELD_BATCH = ORACLE_BATCH = 4
    FIELD_POINTS = 32
else:
    EXPECTED_STATUS = "complete"
    TRAIN_SPEC = ((64, 0, 512, 51), (128, 0, 128, 51), (256, 0, 64, 51))
    SELECTION_SPEC = ((64, 512, 64, 51), (128, 512, 32, 51), (256, 512, 16, 51))
    MANIFOLD_STEPS, PREDICTOR_STEPS, ORACLE_STEPS = 30_000, 20_000, 10_000
    FIELD_BATCH, ORACLE_BATCH, FIELD_POINTS = 32, 64, 512


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


def validate_metrics(metrics, specification, name):
    pooled = []
    require(set(metrics["meshes"]) == {str(row[0]) for row in specification},
            f"{name} mesh set")
    for n, _start, count, _times in specification:
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
    for key in tuple(direct["meshes"]) + ("pooled",):
        row = direct["pooled"] if key == "pooled" else direct["meshes"][key]
        base = oracle["pooled"] if key == "pooled" else oracle["meshes"][key]
        ratios[key] = row["trajectory_error_mean"] / max(base["trajectory_error_mean"], 1e-300)
        result = result and row["trajectory_error_mean"] <= 3e-4 and row["trajectory_error_worst"] <= 1e-3
        result = result and row["all_finite"] and row["exact_binary_boundary"] and ratios[key] <= 1.5
    return bool(result), ratios


def deterministic_schedule(total_snapshots, steps, batch_size, seed):
    rng = np.random.default_rng(int(seed))
    needed = int(steps) * int(batch_size)
    pieces = []
    while sum(item.size for item in pieces) < needed:
        pieces.append(rng.permutation(total_snapshots).astype(np.int32))
    ids = np.concatenate(pieces)[:needed].reshape(steps, batch_size)
    point_seeds = rng.integers(
        0, np.iinfo(np.uint64).max, size=(steps, batch_size), dtype=np.uint64
    )
    return ids, point_seeds


def validate_metadata(rows, specification, name):
    require(len(rows) == len(specification), f"{name} metadata count")
    global_start = 0
    for row, (n, start, count, times) in zip(rows, specification):
        snapshots = count * times
        expected = {
            "N": n, "source_start": start, "source_stop": start + count,
            "case_count": count, "num_times": times,
            "snapshot_count": snapshots, "global_start": global_start,
            "global_stop": global_start + snapshots,
        }
        require(all(row[key] == value for key, value in expected.items()),
                f"{name} metadata/global ranges N={n}")
        health = row["reference_health"]
        require(health["reported_max_relative_residual"] <= 1e-8
                and health["independent_max_relative_residual"] <= 1e-8,
                f"{name} reference health N={n}")
        if AUDIT_SMOKE:
            require(health["status"] == "synthetic excluded smoke",
                    f"{name} smoke health classification N={n}")
        else:
            require(health["seed"] == 0 and health["draw_count"] == 704
                    and health["indices"] == list(range(start, start + count)),
                    f"{name} reference provenance N={n}")
        global_start += snapshots


def validate_history(history, final_step, name):
    steps = [row["step"] for row in history]
    if AUDIT_SMOKE:
        require(steps == [1], f"{name} smoke history structure")
    else:
        require(steps == [1] + list(range(1000, final_step + 1, 1000)),
                f"{name} history structure")
    require(finite(history), f"{name} finite history")


def validate_oracle_history(row):
    history = row["history"]
    steps = [item["step"] for item in history]
    if AUDIT_SMOKE:
        require(steps == [1] and "full_metrics" in history[-1],
                "oracle smoke history structure")
        return
    require(steps[0] == 1 and steps[-1] in (4000, 6000, 8000, 10000),
            "oracle history terminal checkpoint")
    expected = [1] + [step for step in (4000, 6000, 8000, 10000) if step <= steps[-1]]
    require(steps == expected, "oracle checkpoint structure")
    require(all("full_metrics" in item and "gate_pass" in item
                for item in history[1:]), "oracle checkpoint metrics")
    if steps[-1] < 10000:
        require(history[-1]["gate_pass"] is True,
                "oracle early stop without passing gate")
    require(finite(history), "oracle finite history")


require((JOB_ID == "local" if AUDIT_SMOKE else JOB_ID.isdigit())
        and len(EXPECTED_COMMIT) == 40, "identity arguments")
require(sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256, "root manifest-file hash")
report, p4, p4_audit = load(REPORT_PATH), load(P4_PATH), load(P4_AUDIT_PATH)
root_manifest = manifest(MANIFEST_PATH)
source_hashes = report["provenance"]["source_sha256"]
staged_sources = {
    os.path.basename(path): digest for path, digest in root_manifest.items()
    if path.startswith("./code/b10_") and path.endswith(".py")
}
require(source_hashes == staged_sources, "staged source hash binding")
require(report["status"] == EXPECTED_STATUS, "report status")
provenance = report["provenance"]
require(provenance["commit"] == EXPECTED_COMMIT and str(provenance["slurm_job_id"]) == JOB_ID,
        "commit/job provenance")
require(provenance["jax_backend"] == "gpu" and provenance["x64"] is True
        and provenance["matmul_precision"] == "highest", "GPU/f64/highest")
config = report["config"]
require(config["candidate"] == CANDIDATE and config["arm"] == "H1"
        and config["training_seed"] == 11, "candidate/seed")
require(config["training_mix"] == TRAIN_MIX and config["selection_mix"] == SELECTION_MIX
        and config["all_51_times"] is (not AUDIT_SMOKE), "data split/time lock")
require(config["manifold_steps"] == MANIFOLD_STEPS
        and config["predictor_steps"] == PREDICTOR_STEPS
        and config["selection_oracle_steps"] == ORACLE_STEPS, "step protocol")
require(config["field_batch"] == FIELD_BATCH and config["field_points"] == FIELD_POINTS
        and config["selection_oracle_batch"] == ORACLE_BATCH, "batch protocol")
require(config["smoke"] is AUDIT_SMOKE, "smoke classification")
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
require(checkpoint["status"] == EXPECTED_STATUS and checkpoint["candidate"] == CANDIDATE
        and checkpoint["training_seed"] == 11 and finite(checkpoint), "checkpoint identity/health")
require(checkpoint["phase4_gate"] == gate, "checkpoint P4 gate binding")
layers(checkpoint["hyperdecoder"], (19, 32, 32, 48 * 48 + 32 * 32), "hyperdecoder")
layers(checkpoint["direct_predictor_standardized"], (7, 32, 32, 24), "predictor standardized")
layers(checkpoint["direct_predictor_folded_raw"], (7, 32, 32, 24), "predictor folded")
require(np.asarray(checkpoint["training_autolatent_raw"]).shape == (TRAIN_SNAPSHOTS, 19),
        "autolatent shape")
require(len(checkpoint["selection_oracle_optimizer_states"]) == 3,
        "three selection-oracle optimizer states")
validate_history(report["training"]["manifold_history"], MANIFOLD_STEPS, "manifold")
validate_history(report["training"]["predictor_history"], PREDICTOR_STEPS, "predictor")
require(finite(report["training"]), "finite training record")

with np.load(NPZ_PATH, allow_pickle=False) as arrays:
    require(all(finite(arrays[name]) for name in arrays.files), "NPZ finite")
    require(arrays["training_states"].shape == (TRAIN_SNAPSHOTS, 24), "training states")
    require(arrays["selection_oracle_states"].shape == (SELECTION_SNAPSHOTS, 24), "oracle states")
    require(arrays["selection_predictor_states"].shape == (SELECTION_SNAPSHOTS, 24), "direct states")
    for start_index in range(3):
        require(arrays[f"selection_oracle_start{start_index}_q_raw"].shape
                == (SELECTION_SNAPSHOTS, 19), f"oracle start {start_index} q shape")
    schedule_specs = (
        ("manifold", TRAIN_SNAPSHOTS, MANIFOLD_STEPS, FIELD_BATCH, 11),
        ("predictor", TRAIN_SNAPSHOTS, PREDICTOR_STEPS, FIELD_BATCH, 100_011),
        ("selection_oracle", SELECTION_SNAPSHOTS, ORACLE_STEPS, ORACLE_BATCH, 20_260_825),
    )
    for name, total, steps, batch, seed in schedule_specs:
        expected_ids, expected_seeds = deterministic_schedule(total, steps, batch, seed)
        require(np.array_equal(arrays[f"{name}_snapshot_schedule"], expected_ids),
                f"{name} deterministic snapshot schedule")
        require(np.array_equal(arrays[f"{name}_point_draw_seeds"], expected_seeds),
                f"{name} deterministic point seeds")
    training_affine = np.asarray(arrays["training_affine"], np.float64)
    training_q_raw = np.asarray(arrays["training_q_raw"], np.float64)
    expected_training_states = np.concatenate(
        (training_affine, np.tanh(training_q_raw)), axis=1
    )
    require(np.array_equal(arrays["training_states"], expected_training_states),
            "training state/autolatent consistency")
    require(np.array_equal(checkpoint["training_autolatent_raw"], training_q_raw),
            "checkpoint/NPZ training autolatent consistency")
    train_features = np.asarray(arrays["training_features"])
    select_features = np.asarray(arrays["selection_features"])
    mean, empirical = np.mean(train_features, axis=0), np.std(train_features, axis=0)
    scale = np.where(empirical < 1e-12, 1.0, empirical)
    require(np.allclose(mean, checkpoint["predictor_feature_mean"], rtol=1e-14, atol=1e-15)
            and np.allclose(scale, checkpoint["predictor_feature_scale"], rtol=1e-14,
                            atol=1e-15), "train-only feature statistics")
    normalized = mlp(checkpoint["direct_predictor_standardized"], (train_features - mean) / scale)
    folded = mlp(checkpoint["direct_predictor_folded_raw"], train_features)
    select_normalized = mlp(checkpoint["direct_predictor_standardized"], (select_features - mean) / scale)
    select_folded = mlp(checkpoint["direct_predictor_folded_raw"], select_features)
    fold_error = max(float(np.max(np.abs(normalized - folded))),
                     float(np.max(np.abs(select_normalized - select_folded))))
    require(fold_error <= 1e-12, "folded predictor identity")
    expected_predictor_states = np.tanh(select_folded)
    require(np.allclose(arrays["selection_predictor_states"], expected_predictor_states,
                        rtol=2e-13, atol=2e-14),
            "selection predictor-state consistency")
    chosen_for_states = int(report["selection_oracle"]["chosen_start"])
    chosen_q_raw = np.asarray(
        arrays[f"selection_oracle_start{chosen_for_states}_q_raw"], np.float64
    )
    require(np.array_equal(arrays["selection_oracle_chosen_q_raw"], chosen_q_raw),
            "chosen oracle q consistency")
    expected_oracle_states = np.concatenate(
        (np.asarray(arrays["selection_affine"]), np.tanh(chosen_q_raw)), axis=1
    )
    require(np.array_equal(arrays["selection_oracle_states"], expected_oracle_states),
            "selection oracle-state consistency")

validate_metadata(report["data"]["training"], TRAIN_SPEC, "training")
validate_metadata(report["data"]["selection"], SELECTION_SPEC, "selection")
starts = report["selection_oracle"]["starts"]
require([row["start_index"] for row in starts] == [0, 1, 2], "three oracle starts")
for row in starts:
    validate_oracle_history(row)
    validate_metrics(row["metrics"], SELECTION_SPEC, f"oracle start {row['start_index']}")
    require(row["gate_pass"] == oracle_gate(row["metrics"]), "oracle start gate")
losses = [row["metrics"]["pooled"]["mean_snapshot_relative_l2_squared"] for row in starts]
chosen = int(np.argmin(losses))
require(report["selection_oracle"]["chosen_start"] == chosen, "oracle chosen start")
require(report["selection_oracle"]["start_selection_metric"]
        == "pooled_mean_snapshot_relative_l2_squared"
        and np.isclose(report["selection_oracle"]["chosen_start_selection_loss"],
                       losses[chosen], rtol=1e-14), "oracle start-selection loss")
oracle = report["selection_oracle"]["metrics"]
require(oracle == starts[chosen]["metrics"], "chosen oracle metrics")
direct = report["selection_direct"]["metrics"]
validate_metrics(direct, SELECTION_SPEC, "direct")
oracle_pass, (direct_pass, ratios) = oracle_gate(oracle), direct_gate(direct, oracle)
gates = report["gates"]
require(report["selection_oracle"]["gate_pass"] == oracle_pass
        and report["selection_direct"]["gate_pass"] == direct_pass,
        "reported oracle/direct gates")
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
    "status": "pass", "negative_aware": True, "smoke": AUDIT_SMOKE,
    "source_json_sha256": sha256(REPORT_PATH), "source_npz_sha256": sha256(NPZ_PATH),
    "source_checkpoint_sha256": sha256(CHECKPOINT_PATH),
    "source_manifest_sha256": sha256(MANIFEST_PATH),
    "expected_commit": EXPECTED_COMMIT, "job_id": JOB_ID, "arm": "H1", "seed": 11,
    "p4_json_sha256": p4_json_sha, "p4_audit_sha256": sha256(P4_AUDIT_PATH),
    "p4_manifest_sha256": sha256(P4_MANIFEST_PATH),
    "predictor_fold_identity_max_abs": fold_error,
    "state_consistency_recomputed": True,
    "deterministic_schedules_recomputed": True,
    "metadata_and_history_recomputed": True,
    "oracle_start_selection_losses": losses, "chosen_start": chosen, "gates": gates,
}
os.makedirs(os.path.dirname(os.path.abspath(AUDIT_PATH)), exist_ok=True)
with open(AUDIT_PATH, "w") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
print(json.dumps({"status": "pass", "smoke": AUDIT_SMOKE, "gates": gates}, indent=2))
