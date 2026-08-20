"""Independent structural/numerical audit of a Phase-2 spline training artifact."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

import numpy as np


if len(sys.argv) < 13:
    raise SystemExit(
        "usage: b10_audit_spline_train.py TRAIN.json TRAIN.npz CHECKPOINT.pkl "
        "AUDIT.json EXPECTED_COMMIT JOB_ID ARM SEED S0.json S0_AUDIT.json "
        "MANIFEST.sha256 EXPECTED_MANIFEST_SHA256 [PRIOR_TRAIN.json ...]"
    )
(
    REPORT_PATH, NPZ_PATH, CHECKPOINT_PATH, AUDIT_PATH, EXPECTED_COMMIT,
    JOB_ID, ARM, SEED_TEXT, S0_PATH, S0_AUDIT_PATH, MANIFEST_PATH,
    EXPECTED_MANIFEST_SHA256,
) = sys.argv[1:13]
PRIOR_PATHS = tuple(sys.argv[13:])
SEED = int(SEED_TEXT)
CANDIDATES = {
    "A": {"arm": "A", "R": 24, "k": 12, "M": 64, "m": 256},
    "B": {"arm": "B", "R": 32, "k": 16, "M": 64, "m": 256},
    "C": {"arm": "C", "R": 48, "k": 24, "M": 96, "m": 384},
}
TRAIN_MIX = [[64, 0, 512], [128, 0, 128], [256, 0, 64]]
SELECTION_MIX = [[64, 512, 64], [128, 512, 32], [256, 512, 16]]
TRAIN_SNAPSHOTS = 704 * 51
SELECTION_SNAPSHOTS = 112 * 51


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"SPLINE TRAIN AUDIT FAILED: {message}")


def all_finite(tree):
    if isinstance(tree, dict):
        return all(all_finite(value) for value in tree.values())
    if isinstance(tree, (tuple, list)):
        return all(all_finite(value) for value in tree)
    if isinstance(tree, np.ndarray):
        return bool(np.all(np.isfinite(tree))) if np.issubdtype(tree.dtype, np.number) else True
    if isinstance(tree, (float, np.floating)):
        return bool(np.isfinite(tree))
    return True


def manifest_hashes(path):
    hashes = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split(maxsplit=1)
            require(len(fields) == 2 and len(fields[0]) == 64, "malformed source manifest")
            name = fields[1].lstrip("*")
            require(name not in hashes, f"duplicate manifest path: {name}")
            hashes[name] = fields[0]
    return hashes


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


def validate_layers(parameters, dimensions, name):
    require(len(parameters) == len(dimensions) - 1, f"{name} layer count")
    for index, (layer, n_in, n_out) in enumerate(
        zip(parameters, dimensions[:-1], dimensions[1:])
    ):
        require(np.asarray(layer["W"]).shape == (n_in, n_out), f"{name} W{index} shape")
        require(np.asarray(layer["b"]).shape == (n_out,), f"{name} b{index} shape")
        require(all_finite(layer), f"{name} layer {index} nonfinite")


def validate_metrics(metrics, expected_cases, name):
    require(set(metrics["meshes"]) == {"64", "128", "256"}, f"{name} mesh set")
    pooled = []
    for n, count in zip((64, 128, 256), expected_cases):
        row = metrics["meshes"][str(n)]
        values = np.asarray(row["trajectory_error_all"], np.float64)
        require(values.shape == (count,), f"{name} N={n} trajectory count")
        require(np.all(np.isfinite(values)), f"{name} N={n} nonfinite errors")
        require(np.isclose(np.mean(values), row["trajectory_error_mean"], rtol=1e-12),
                f"{name} N={n} mean mismatch")
        require(np.isclose(np.max(values), row["trajectory_error_worst"], rtol=1e-12),
                f"{name} N={n} worst mismatch")
        require(row["all_finite"] is True and row["exact_binary_boundary"] is True,
                f"{name} N={n} health/boundary")
        pooled.extend(values.tolist())
    pooled = np.asarray(pooled, np.float64)
    row = metrics["pooled"]
    require(np.allclose(pooled, row["trajectory_error_all"], rtol=0, atol=0),
            f"{name} pooled trajectory vector mismatch")
    require(np.isclose(np.mean(pooled), row["trajectory_error_mean"], rtol=1e-12),
            f"{name} pooled mean mismatch")
    require(np.isclose(np.max(pooled), row["trajectory_error_worst"], rtol=1e-12),
            f"{name} pooled worst mismatch")
    require(row["all_finite"] is True and row["exact_binary_boundary"] is True,
            f"{name} pooled health/boundary")
    require(np.isfinite(row["mean_snapshot_relative_l2_squared"])
            and row["mean_snapshot_relative_l2_squared"] >= 0, f"{name} selection loss")


def representation_gate(metrics):
    rows = list(metrics["meshes"].values()) + [metrics["pooled"]]
    return bool(all(
        row["trajectory_error_mean"] <= 2e-4
        and row["trajectory_error_worst"] <= 7e-4
        and row["all_finite"] and row["exact_binary_boundary"] for row in rows
    ))


def direct_gate(direct, oracle):
    result = True
    ratios = {}
    for key in ("64", "128", "256", "pooled"):
        direct_row = direct["pooled"] if key == "pooled" else direct["meshes"][key]
        oracle_row = oracle["pooled"] if key == "pooled" else oracle["meshes"][key]
        ratio = direct_row["trajectory_error_mean"] / max(
            oracle_row["trajectory_error_mean"], 1e-300
        )
        ratios[key] = ratio
        result = result and bool(
            direct_row["trajectory_error_mean"] <= 3e-4
            and direct_row["trajectory_error_worst"] <= 1e-3
            and direct_row["all_finite"] and direct_row["exact_binary_boundary"]
            and ratio <= 1.5
        )
    return result, ratios


def load_prior(path, s0_sha):
    with open(path) as handle:
        artifact = json.load(handle)
    npz_path = os.path.join(os.path.dirname(os.path.abspath(path)), artifact["npz"]["path"])
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(path)), artifact["checkpoint"]["path"]
    )
    audit_path = os.path.join(os.path.dirname(os.path.abspath(path)), "AUDIT.json")
    require(artifact.get("status") == "complete", f"prior incomplete: {path}")
    require(os.path.isfile(npz_path) and sha256(npz_path) == artifact["npz"]["sha256"],
            f"prior NPZ invalid: {path}")
    require(os.path.isfile(checkpoint_path)
            and sha256(checkpoint_path) == artifact["checkpoint"]["sha256"],
            f"prior checkpoint invalid: {path}")
    require(os.path.isfile(audit_path), f"prior audit missing: {path}")
    with open(audit_path) as handle:
        audit = json.load(handle)
    require(
        audit.get("status") == "pass"
        and audit.get("source_json_sha256") == sha256(path)
        and audit.get("source_npz_sha256") == sha256(npz_path)
        and audit.get("source_checkpoint_sha256") == sha256(checkpoint_path)
        and audit.get("s0_json_sha256") == s0_sha,
        f"prior independent audit invalid: {path}",
    )
    return artifact, {
        "json_sha256": sha256(path), "npz_sha256": sha256(npz_path),
        "checkpoint_sha256": sha256(checkpoint_path), "audit_sha256": sha256(audit_path),
        "arm": artifact["config"]["arm"], "seed": artifact["config"]["training_seed"],
        "oracle": artifact["selection_oracle"]["gate_pass"],
        "direct": artifact["selection_direct"]["gate_pass"],
        "promote": artifact["gates"]["promote_seed"],
    }


require(ARM in CANDIDATES and SEED in (11, 29, 47), "invalid arm/seed")
require(JOB_ID.isdigit(), "job id must be numeric")
with open(REPORT_PATH) as handle:
    report = json.load(handle)
with open(S0_PATH) as handle:
    s0 = json.load(handle)
with open(S0_AUDIT_PATH) as handle:
    s0_audit = json.load(handle)
manifest = manifest_hashes(MANIFEST_PATH)
require(sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256,
        "manifest-file checksum differs from launch record")

require(report.get("status") == "complete", "report is not complete")
provenance = report.get("provenance") or {}
require(provenance.get("commit") == EXPECTED_COMMIT, "commit mismatch")
require(str(provenance.get("slurm_job_id")) == JOB_ID, "job id mismatch")
require(provenance.get("jax_backend") == "gpu" and provenance.get("x64") is True,
        "GPU/f64 provenance failed")
require(provenance.get("matmul_precision") == "highest", "matmul precision failed")
source_hashes = provenance.get("source_sha256") or {}
manifest_b10 = {
    os.path.basename(path): digest for path, digest in manifest.items()
    if path.startswith("./code/b10_") and path.endswith(".py")
}
require(source_hashes == manifest_b10,
        "reported b10 source hashes differ from exact staged manifest paths")
config = report["config"]
candidate = CANDIDATES[ARM]
require(config["arm"] == ARM and config["candidate"] == candidate, "candidate mismatch")
require(config["training_seed"] == SEED, "seed mismatch")
require(config["training_mix"] == TRAIN_MIX and config["selection_mix"] == SELECTION_MIX,
        "mixed-resolution split mismatch")
require(config["all_51_times"] is True and config["manifold_steps"] == 30_000
        and config["predictor_steps"] == 20_000
        and config["selection_oracle_steps"] == 10_000, "training protocol mismatch")
require(config["field_batch"] == 32 and config["field_points"] == 512
        and config["selection_oracle_batch"] == 64, "batch protocol mismatch")
require(config["model_validation_touched"] is False
        and config["confirmation_touched"] is False, "locked split touched")
manifold_history = report["training"]["manifold_history"]
predictor_history = report["training"]["predictor_history"]
require(manifold_history[-1]["step"] == 30_000
        and all(all_finite(row) for row in manifold_history),
        "manifold history did not end at 30000 with finite losses")
require(predictor_history[-1]["step"] == 20_000
        and all(all_finite(row) for row in predictor_history),
        "predictor history did not end at 20000 with finite losses")

s0_sha = sha256(S0_PATH)
s0_npz_sha = s0["npz"]["sha256"]
require(
    s0_audit.get("status") == "pass"
    and s0_audit.get("source_json_sha256") == s0_sha
    and s0_audit.get("source_npz_sha256") == s0_npz_sha
    and s0_audit.get("decision") == s0.get("decision"),
    "S0 independent audit binding failed",
)
gate = report["s0_gate"]
require(gate["sha256"] == s0_sha and gate["npz_sha256"] == s0_npz_sha
        and gate["audit_sha256"] == sha256(S0_AUDIT_PATH)
        and gate["audit_status"] == "pass" and gate["audit_decision"] == s0["decision"],
        "trainer S0 gate binding mismatch")
expected_s0_manifest = {
    "./code/deps/s0/s0.json": s0_sha,
    "./code/deps/s0/s0.npz": s0_npz_sha,
    "./code/deps/s0/AUDIT.json": sha256(S0_AUDIT_PATH),
}
for path, digest in expected_s0_manifest.items():
    require(manifest.get(path) == digest, f"S0 artifact manifest mismatch: {path}")

prior_pairs = [load_prior(path, s0_sha) for path in PRIOR_PATHS]
prior_records = [item[1] for item in prior_pairs]
recorded_priors = gate["artifact_chain"]["prior_artifacts"]
require(len(recorded_priors) == len(prior_records), "recorded prior count mismatch")
for prior_index, prior in enumerate(prior_records):
    matches = [row for row in recorded_priors
               if row["arm"] == prior["arm"] and row["seed"] == prior["seed"]]
    require(len(matches) == 1, "recorded prior identity mismatch")
    row = matches[0]
    require(row["json_sha256"] == prior["json_sha256"]
            and row["npz"]["sha256"] == prior["npz_sha256"]
            and row["checkpoint"]["sha256"] == prior["checkpoint_sha256"]
            and row["audit"]["sha256"] == prior["audit_sha256"],
            "recorded prior hashes mismatch")
    expected_prior_manifest = {
        f"./code/deps/prior_{prior_index}/train.json": prior["json_sha256"],
        f"./code/deps/prior_{prior_index}/train.npz": prior["npz_sha256"],
        f"./code/deps/prior_{prior_index}/checkpoint.pkl": prior["checkpoint_sha256"],
        f"./code/deps/prior_{prior_index}/AUDIT.json": prior["audit_sha256"],
    }
    for path, digest in expected_prior_manifest.items():
        require(manifest.get(path) == digest, f"prior artifact manifest mismatch: {path}")

promoted = s0["decision"]["promoted_arms"]
require(ARM in promoted, "arm was not S0-promoted")
if SEED == 11:
    smaller = promoted[:promoted.index(ARM)]
    for one in smaller:
        matches = [row for row in prior_records if row["arm"] == one and row["seed"] == 11]
        require(len(matches) == 1 and matches[0]["oracle"] is False,
                f"seed11 escalation lacks learned-oracle failure for {one}")
else:
    matches = [row for row in prior_records if row["arm"] == ARM and row["seed"] == 11]
    require(len(matches) == 1 and matches[0]["oracle"] is True
            and matches[0]["direct"] is True and matches[0]["promote"] is True,
            "robustness seed lacks passing seed11 finalist")

require(report["npz"]["sha256"] == sha256(NPZ_PATH), "output NPZ checksum")
require(report["checkpoint"]["sha256"] == sha256(CHECKPOINT_PATH),
        "output checkpoint checksum")
with open(CHECKPOINT_PATH, "rb") as handle:
    checkpoint = pickle.load(handle)
require(checkpoint["status"] == "complete" and checkpoint["candidate"] == candidate
        and checkpoint["training_seed"] == SEED, "checkpoint identity")
require(all_finite(checkpoint), "checkpoint contains nonfinite numerical values")
k, q, r = candidate["k"], candidate["k"] - 5, candidate["R"]
validate_layers(checkpoint["hyperdecoder"], (q, 32, 32, r * r), "hyperdecoder")
validate_layers(checkpoint["direct_predictor_standardized"], (7, 32, 32, k),
                "standardized predictor")
validate_layers(checkpoint["direct_predictor_folded_raw"], (7, 32, 32, k),
                "folded predictor")
require(np.asarray(checkpoint["training_autolatent_raw"]).shape == (TRAIN_SNAPSHOTS, q),
        "training autolatent shape")

with np.load(NPZ_PATH, allow_pickle=False) as arrays:
    for name in arrays.files:
        require(all_finite(arrays[name]), f"NPZ nonfinite: {name}")
    raw_features = np.asarray(arrays["training_features"], np.float64)
    require(raw_features.shape == (TRAIN_SNAPSHOTS, 7), "training feature shape")
    require(np.asarray(arrays["training_states"]).shape == (TRAIN_SNAPSHOTS, k),
            "training state shape")
    require(np.asarray(arrays["selection_oracle_states"]).shape
            == (SELECTION_SNAPSHOTS, k), "selection oracle state shape")
    require(np.asarray(arrays["selection_predictor_states"]).shape
            == (SELECTION_SNAPSHOTS, k), "selection predictor state shape")
    require(arrays["manifold_snapshot_schedule"].shape == (30_000, 32)
            and arrays["manifold_point_draw_seeds"].shape == (30_000, 32),
            "manifold schedule shape")
    require(arrays["predictor_snapshot_schedule"].shape == (20_000, 32)
            and arrays["predictor_point_draw_seeds"].shape == (20_000, 32),
            "predictor schedule shape")
    require(arrays["selection_oracle_snapshot_schedule"].shape == (10_000, 64)
            and arrays["selection_oracle_point_draw_seeds"].shape == (10_000, 64),
            "selection oracle schedule shape")
    mean = np.mean(raw_features, axis=0, dtype=np.float64)
    empirical = np.std(raw_features, axis=0, dtype=np.float64)
    scale = np.where(empirical < 1e-12, 1.0, empirical)
    require(np.allclose(mean, checkpoint["predictor_feature_mean"], rtol=1e-14, atol=1e-15)
            and np.allclose(scale, checkpoint["predictor_feature_scale"], rtol=1e-14,
                            atol=1e-15), "train-only feature statistics mismatch")
    normalized_output = mlp(
        checkpoint["direct_predictor_standardized"], (raw_features - mean) / scale
    )
    folded_output = mlp(checkpoint["direct_predictor_folded_raw"], raw_features)
    fold_error_train = float(np.max(np.abs(normalized_output - folded_output)))
    selection_features = np.asarray(arrays["selection_features"], np.float64)
    require(selection_features.shape == (SELECTION_SNAPSHOTS, 7),
            "selection feature shape")
    selection_normalized = mlp(
        checkpoint["direct_predictor_standardized"],
        (selection_features - mean) / scale,
    )
    selection_folded = mlp(
        checkpoint["direct_predictor_folded_raw"], selection_features
    )
    fold_error_selection = float(np.max(np.abs(selection_normalized - selection_folded)))
    fold_error = max(fold_error_train, fold_error_selection)
    require(fold_error <= 1e-12, "folded predictor identity gate")
    require(np.isclose(fold_error_train, checkpoint["predictor_fold_identity_train_max_abs"],
                       rtol=1e-10, atol=1e-15), "recorded fold identity mismatch")
    require(np.isclose(
        fold_error_selection, checkpoint["predictor_fold_identity_selection_max_abs"],
        rtol=1e-10, atol=1e-15,
    ), "recorded selection fold identity mismatch")
    require(np.isclose(fold_error, checkpoint["predictor_fold_identity_max_abs"],
                       rtol=1e-10, atol=1e-15), "recorded maximum fold identity mismatch")
    recorded_fold = report["training"]["predictor_feature_standardization"]
    require(np.allclose(recorded_fold["mean"], mean, rtol=1e-14, atol=1e-15)
            and np.allclose(recorded_fold["scale"], scale, rtol=1e-14, atol=1e-15),
            "reported feature statistics mismatch")
    require(np.isclose(recorded_fold["folded_raw_identity_train_max_abs"],
                       fold_error_train, rtol=1e-10, atol=1e-15)
            and np.isclose(recorded_fold["folded_raw_identity_selection_max_abs"],
                           fold_error_selection, rtol=1e-10, atol=1e-15)
            and np.isclose(recorded_fold["folded_raw_identity_max_abs"], fold_error,
                           rtol=1e-10, atol=1e-15),
            "reported train/selection/maximum fold identities mismatch")

for row in report["data"]["training"] + report["data"]["selection"]:
    health = row["reference_health"]
    require(row["num_times"] == 51
            and health["reported_max_relative_residual"] <= 1e-8
            and health["independent_max_relative_residual"] <= 1e-8,
            "reference/data health failed")

starts = report["selection_oracle"]["starts"]
require([row["start_index"] for row in starts] == [0, 1, 2], "oracle start set")
for row in starts:
    validate_metrics(row["metrics"], (64, 32, 16), f"oracle start {row['start_index']}")
    require(row["gate_pass"] == representation_gate(row["metrics"]),
            "oracle start gate mismatch")
losses = [row["metrics"]["pooled"]["mean_snapshot_relative_l2_squared"] for row in starts]
chosen = int(np.argmin(losses))
require(report["selection_oracle"]["start_selection_metric"]
        == "pooled_mean_snapshot_relative_l2_squared", "oracle selection metric")
require(report["selection_oracle"]["chosen_start"] == chosen, "chosen oracle start")
require(np.isclose(report["selection_oracle"]["chosen_start_selection_loss"],
                   losses[chosen], rtol=1e-14), "chosen oracle loss")
oracle_metrics = report["selection_oracle"]["metrics"]
require(oracle_metrics == starts[chosen]["metrics"], "chosen oracle metrics mismatch")
validate_metrics(report["selection_direct"]["metrics"], (64, 32, 16), "direct")
oracle_pass = representation_gate(oracle_metrics)
direct_pass, ratios = direct_gate(report["selection_direct"]["metrics"], oracle_metrics)
require(report["selection_oracle"]["gate_pass"] == oracle_pass, "oracle gate")
require(report["selection_direct"]["gate_pass"] == direct_pass, "direct gate")
require(all(np.isclose(ratios[key], report["selection_direct"][
        "degradation_direct_over_oracle"][key], rtol=1e-12) for key in ratios),
        "direct/oracle degradation mismatch")
gates = report["gates"]
require(gates["representation_oracle_all_N_and_pooled"] == oracle_pass
        and gates["direct_all_N_and_pooled"] == direct_pass
        and gates["promote_seed"] == bool(oracle_pass and direct_pass), "final gates")
near_miss = bool(
    oracle_pass and not direct_pass and all(
        row["trajectory_error_mean"] <= 6e-4
        and row["trajectory_error_worst"] <= 2e-3
        and row["all_finite"] and row["exact_binary_boundary"]
        and ratios[key] <= 3.0
        for key, row in list(
            report["selection_direct"]["metrics"]["meshes"].items()
        ) + [("pooled", report["selection_direct"]["metrics"]["pooled"])]
    )
)
require(gates["local_loss_revision_near_miss_condition"] == near_miss,
        "local loss near-miss condition")
require(gates["loss_revision_licensed"] is False, "single seed licensed loss revision")

audit = {
    "status": "pass", "source_json_sha256": sha256(REPORT_PATH),
    "source_npz_sha256": sha256(NPZ_PATH),
    "source_checkpoint_sha256": sha256(CHECKPOINT_PATH),
    "source_manifest_sha256": sha256(MANIFEST_PATH),
    "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "expected_commit": EXPECTED_COMMIT, "job_id": JOB_ID,
    "arm": ARM, "seed": SEED, "s0_json_sha256": s0_sha,
    "s0_audit_sha256": sha256(S0_AUDIT_PATH),
    "prior_json_sha256": [row["json_sha256"] for row in prior_records],
    "predictor_fold_identity_train_max_abs": fold_error_train,
    "predictor_fold_identity_selection_max_abs": fold_error_selection,
    "predictor_fold_identity_max_abs": fold_error,
    "oracle_start_selection_losses": losses, "chosen_start": chosen,
    "gates": gates,
}
os.makedirs(os.path.dirname(os.path.abspath(AUDIT_PATH)), exist_ok=True)
with open(AUDIT_PATH, "w") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps({"status": "pass", "arm": ARM, "seed": SEED, "gates": gates}, indent=2))
