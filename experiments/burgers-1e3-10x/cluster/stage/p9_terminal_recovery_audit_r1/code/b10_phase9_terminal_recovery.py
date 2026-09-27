#!/usr/bin/env python
"""Zero-update terminal recovery for the consumed Phase-9 T1 cell.

Real execution is cluster-only.  ``--smoke`` creates only a small synthetic
contract and never reads a locked target or work checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import b10_common as c
import b10_phase8_d as p8
import b10_phase9_train as p9
import b10_spline as spline
import b10_spline_train as legacy


CONFIG = dict(p9.ARMS["T1"])
AFFINE_ATOL = 2e-15
STATE_ATOL = 2e-15
EXPECTED_FAILED = {
    "FAILURE.json": "0810269919ab1d4c9dd315f64e666cb258620b11faaea1cf302c5cd9ba88e9db",
    "LOCAL.sha256": "46b3d681fb0927368b8696acc800736691ef8f3ddade65363689b14bfb436ac9",
    "MANIFEST.sha256": "a403b5facfa4e6cd3b6a14abdb3503c1e85721408182419e2e197a05a51bfa1e",
    "REMOTE.sha256": "2426d43463506065706ca1a76b0ccc064df6a971a02be4063d1c64e72487eaf1",
    "SACCT.txt": "98292764147d97708b47ebf50d18f16df29fb6db3cc166a09b74ae9c1a474cf7",
    "logs/2702357.out": "ea0ad35af5d8b5bfd94176e40ee41b8a91fcba50cf2733d57ade352ddcc73fbb",
    "logs/2702357.err": "d5428105ef196548ad927f13e1cb1e6ad8dc7e1fe176dde8ebc938df743ea5ce",
    "out/PROGRESS.json": "1aca12633096dd0dd75fcf5da353b3500b2dac7d77055b3112a8a32f1cae53b5",
    "out/work_checkpoint.pkl": "dd7b5fecc07e9021c6264febf5c7187004daaa73a50381921d5190eede639ce0",
}
EXPECTED_ORIGINAL_COMMIT = "4a87fe325ce07ce88cdb0995375a614fa864bf7a"
EXPECTED_ORIGINAL_JOB = "2702357"


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def tree_count(tree):
    return int(sum(np.prod(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(tree)))


def tree_finite(tree):
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]
    return bool(leaves and all(np.all(np.isfinite(x)) for x in leaves))


def tree_exact(left, right):
    a = jax.tree_util.tree_leaves(left); b = jax.tree_util.tree_leaves(right)
    return bool(len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)))


def tree_sha(tree):
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        value = np.ascontiguousarray(np.asarray(leaf))
        digest.update(np.asarray(index, np.int64).tobytes())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify_checksum_file(root, name):
    path = os.path.join(root, name)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            target = os.path.join(root, relative.removeprefix("./"))
            if not os.path.isfile(target) or c.sha256(target) != digest:
                raise SystemExit(f"failed bundle checksum mismatch: {relative}")


def validate_failed(root):
    bindings = {}
    for relative, expected in EXPECTED_FAILED.items():
        path = os.path.join(root, relative)
        if not os.path.isfile(path) or c.sha256(path) != expected:
            raise SystemExit(f"immutable failed-run artifact mismatch: {relative}")
        bindings[relative] = expected
    verify_checksum_file(root, "LOCAL.sha256")
    failure = read_json(os.path.join(root, "FAILURE.json"))
    progress = read_json(os.path.join(root, "out/PROGRESS.json"))
    stdout = open(os.path.join(root, "logs/2702357.out"), encoding="utf-8").read()
    stderr = open(os.path.join(root, "logs/2702357.err"), encoding="utf-8").read()
    sacct = open(os.path.join(root, "SACCT.txt"), encoding="utf-8").read()
    if not (failure["status"] == "excluded_post_update_terminal_finalization_failure"
            and failure["remote_cleaned"] is True
            and failure["scientific_cell_consumed"] is True
            and progress == {"epoch": 18, "global_update": 528768, "phase": "predictor",
                             "scientific_metrics_exposed": False, "status": "in_progress"}
            and "commit=" + EXPECTED_ORIGINAL_COMMIT in stdout
            and "jax_backend=gpu" in stdout and "host=pax008 gpu=NVIDIA H200" in stdout
            and "KeyError: 'mean_snapshot_relative_l2_squared'" in stderr
            and "2702357|p9_t1_s11_r1|FAILED|1:0|00:47:20" in sacct):
        raise SystemExit("failed-run status/provenance mismatch")
    return bindings


def validate_target_order(args, p5_report):
    expected = 0
    for row in p5_report["train_targets"]["chunks"]:
        path = os.path.join(args.target_dir, row["basename"])
        with np.load(path, allow_pickle=False) as data:
            count = int(row["snapshot_count"])
            indices = np.asarray(data["global_snapshot_index"], np.int64).reshape(-1)
            if not (indices.size == count
                    and np.array_equal(indices, np.arange(expected, expected + count))):
                raise SystemExit("immutable target order mismatch")
            expected += count
    if expected != 35904:
        raise SystemExit("immutable target count mismatch")


def bind_train_regeneration(train, physical, target_features, p7_arrays):
    affine = legacy.concatenate(train, "affine")
    mapped = np.stack([spline.normalized_state_from_affine(row) for row in physical])
    regenerated_features = legacy.concatenate(train, "features")
    exact_columns = (0, 1, 2, 3, 5, 6)
    viscosity_delta = np.abs(regenerated_features[:, 4] - target_features[:, 4])
    viscosity_tolerance = np.abs(np.spacing(np.maximum(
        np.abs(regenerated_features[:, 4]), np.abs(target_features[:, 4]))))
    deltas = {
        "regenerated_vs_physical_mapping": float(np.max(np.abs(affine - mapped))),
        "p7_vs_physical_mapping": float(np.max(np.abs(p7_arrays["training_affine"] - mapped))),
    }
    checks = {
        "absolute_tolerance": AFFINE_ATOL,
        "relative_tolerance": 0.0,
        **deltas,
        "feature_exact_columns_bitwise": bool(np.array_equal(
            regenerated_features[:, exact_columns], target_features[:, exact_columns])),
        "viscosity_feature_max_abs": float(np.max(viscosity_delta)),
        "viscosity_feature_within_ulp": bool(np.all(viscosity_delta <= viscosity_tolerance)),
        "p7_training_features_exact": bool(np.array_equal(
            p7_arrays["training_features"], target_features)),
    }
    if not (max(deltas.values()) <= AFFINE_ATOL
            and checks["feature_exact_columns_bitwise"]
            and checks["viscosity_feature_within_ulp"]
            and checks["p7_training_features_exact"]):
        raise SystemExit("train affine/feature provenance mismatch")
    return checks


def load_train(args, smoke):
    legacy.SMOKE = smoke
    if smoke:
        train = p9.p7train.smoke_datasets("train")
        coefficients = np.random.default_rng(92011).normal(0, .05, (4, 3328))
        features = legacy.concatenate(train, "features")
        return train, coefficients, features, [{"basename": "synthetic-only", "N": 16,
                                                "snapshot_count": 4, "sha256": None}], {
            "mode": "synthetic", "pass": True}
    train = legacy.load_mix(p9.TRAIN_MIX, "train")
    p5_report = read_json(args.p5_json)
    validate_target_order(args, p5_report)
    coefficients, physical, features, records = p8.load_train_targets(args, p5_report)
    with np.load(args.p7_npz, allow_pickle=False) as source:
        provenance = bind_train_regeneration(train, physical, features, source)
    return train, coefficients, features, records, provenance


def synthetic_work(train, norm):
    generator = p9.init_generator(CONFIG)
    encoder = p9.init_encoder(CONFIG)
    predictor = p9.init_predictor(CONFIG)
    q_raw = p9.encode_batches(encoder, norm["normalized"], CONFIG)
    final = np.concatenate((legacy.concatenate(train, "affine"), np.tanh(q_raw)), axis=1)
    folded = legacy.fold_predictor_standardization(predictor, norm["feature_mean"], norm["feature_scale"])
    evaluated = p9.predictor_batches(folded, legacy.concatenate(train, "features"))
    optimizer = p9.optimizer(lambda _: 1e-4).init(predictor)
    return {"phase": "predictor", "epoch": 18, "global_update": 528768,
            "generator": generator, "encoder": encoder, "predictor": predictor,
            "q_raw": q_raw, "final_target_states": final,
            "evaluation_states": evaluated, "predictor_optimizer_state": optimizer,
            "optimizer_state": optimizer}


def validate_work(work, train, norm, scientific):
    expected_keys = {"phase", "epoch", "global_update", "generator", "encoder", "predictor",
                     "q_raw", "final_target_states", "evaluation_states",
                     "predictor_optimizer_state", "optimizer_state"}
    if set(work) != expected_keys:
        raise SystemExit("unexpected work-checkpoint schema")
    q = np.asarray(work["q_raw"], np.float64)
    final = np.asarray(work["final_target_states"], np.float64)
    evaluated = np.asarray(work["evaluation_states"], np.float64)
    count = sum(len(item["flat"]) for item in train)
    affine = legacy.concatenate(train, "affine")
    folded = legacy.fold_predictor_standardization(
        work["predictor"], norm["feature_mean"], norm["feature_scale"])
    regenerated_evaluation = p9.predictor_batches(
        folded, legacy.concatenate(train, "features"))
    evaluation_delta = np.abs(regenerated_evaluation - evaluated)
    fold_identity = legacy.predictor_fold_identity(
        work["predictor"], folded, legacy.concatenate(train, "features"),
        norm["feature_mean"], norm["feature_scale"])
    counts = {name: tree_count(work[name]) for name in ("generator", "encoder", "predictor")}
    optimizer_duplicate = tree_exact(work["optimizer_state"], work["predictor_optimizer_state"])
    checks = {
        "phase_epoch_update_exact": bool(work["phase"] == "predictor" and work["epoch"] == 18
                                          and work["global_update"] == 528768),
        "parameter_counts": counts,
        "parameter_counts_exact": counts == {"generator": 30594, "encoder": 29811, "predictor": 2104},
        "all_trees_finite": bool(all(tree_finite(work[name]) for name in
                                      ("generator", "encoder", "predictor",
                                       "predictor_optimizer_state", "optimizer_state"))),
        "q_shape_dtype_finite": bool(q.shape == (count, 19) and q.dtype == np.float64
                                      and np.all(np.isfinite(q))),
        "final_state_shape_dtype_finite": bool(final.shape == (count, 24)
                                                and final.dtype == np.float64
                                                and np.all(np.isfinite(final))),
        "evaluation_state_shape_dtype_finite": bool(evaluated.shape == (count, 24)
                                                     and evaluated.dtype == np.float64
                                                     and np.all(np.isfinite(evaluated))),
        "final_affine_exact": bool(np.array_equal(final[:, :5], affine)),
        "final_latent_exact": bool(np.array_equal(final[:, 5:], np.tanh(q))),
        "optimizer_duplicate_exact": optimizer_duplicate,
        "evaluation_state_exact": bool(np.array_equal(regenerated_evaluation, evaluated)),
        "evaluation_state_max_abs": float(np.max(evaluation_delta)),
        "evaluation_state_portable": bool(np.all(evaluation_delta <= STATE_ATOL)),
        "fold_identity_max_abs": float(fold_identity),
        "fold_identity_pass": bool(fold_identity <= 1e-12),
        "scientific_source": scientific,
    }
    required = [checks[key] for key in (
        "phase_epoch_update_exact", "parameter_counts_exact", "all_trees_finite",
        "q_shape_dtype_finite", "final_state_shape_dtype_finite",
        "evaluation_state_shape_dtype_finite", "final_affine_exact", "final_latent_exact",
        "optimizer_duplicate_exact", "evaluation_state_portable", "fold_identity_pass")]
    if not all(required):
        raise SystemExit("terminal work-checkpoint binding failed")
    return folded, regenerated_evaluation, checks


def add_snapshot_losses(metrics, arrays, prefix, train):
    for item in train:
        n = item["N"]
        num = np.asarray(arrays[f"{prefix}_N{n}_numerator"])
        den = np.asarray(arrays[f"{prefix}_N{n}_denominator"])
        metrics["meshes"][str(n)]["mean_snapshot_relative_l2_squared"] = float(
            np.mean(num / np.maximum(den, 1e-300)))


def source_tree_records(work):
    return {name: {"sha256": tree_sha(work[name]), "parameter_count": tree_count(work[name]),
                   "finite": tree_finite(work[name])}
            for name in ("generator", "encoder", "predictor", "predictor_optimizer_state",
                         "optimizer_state")}


def parse_args():
    parser = argparse.ArgumentParser()
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            parser.add_argument(f"--{phase}-{kind}", dest=f"{phase}_{kind}")
    for kind in ("json", "npz", "checkpoint", "audit", "manifest"):
        parser.add_argument(f"--p7-{kind}", dest=f"p7_{kind}")
    for kind in ("json", "npz", "audit", "manifest"):
        parser.add_argument(f"--p8-{kind}", dest=f"p8_{kind}")
    parser.add_argument("--target-dir"); parser.add_argument("--failed-dir")
    parser.add_argument("--prereg"); parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--progress-json", required=True); parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); c.require_gpu_highest(); smoke = args.smoke
    if smoke:
        bindings = {"mode": "synthetic"}; failed_bindings = {"mode": "synthetic"}
    else:
        required = [getattr(args, f"{phase}_{kind}") for phase in ("p4", "p5", "p6")
                    for kind in ("json", "npz", "audit", "manifest")]
        required += [getattr(args, f"p7_{kind}") for kind in ("json", "npz", "checkpoint", "audit", "manifest")]
        required += [getattr(args, f"p8_{kind}") for kind in ("json", "npz", "audit", "manifest")]
        required += [args.target_dir, args.failed_dir, args.prereg]
        if any(value is None for value in required):
            raise SystemExit("scientific terminal recovery requires exact P4-P8 and failed bundle")
        _, bindings = p9.validate_chains(args, "T1")
        failed_bindings = validate_failed(args.failed_dir)
    train, coefficients, target_features, target_records, regeneration = load_train(args, smoke)
    features = legacy.concatenate(train, "features")
    norm = p9.train_normalization(coefficients, features)
    source_path = None if smoke else os.path.join(args.failed_dir, "out/work_checkpoint.pkl")
    source_before = "synthetic" if smoke else c.sha256(source_path)
    work = synthetic_work(train, norm) if smoke else pickle.load(open(source_path, "rb"))
    before_trees = source_tree_records(work)
    folded, evaluated, work_checks = validate_work(work, train, norm, not smoke)
    arrays = {
        "coefficient_mean": norm["mean"], "head_scales": norm["scales"],
        "predictor_feature_mean": norm["feature_mean"],
        "predictor_feature_scale": norm["feature_scale"],
        "predictor_feature_empirical_scale": norm["feature_empirical"],
        "normalization_source_indices": np.arange(len(coefficients), dtype=np.int32),
        "training_features": features,
        "source_q_raw": np.asarray(work["q_raw"]),
        "source_final_target_states": np.asarray(work["final_target_states"]),
        "source_evaluation_states": np.asarray(work["evaluation_states"]),
        "recomputed_evaluation_states": evaluated,
        "optimizer_updates": np.asarray((0,), np.int64),
    }
    q = np.tanh(np.asarray(work["q_raw"])); states = np.concatenate((legacy.concatenate(train, "affine"), q), axis=1)
    terminal = p9.evaluate_full(train, work["generator"], states, norm["mean"], norm["scales"],
                                CONFIG, arrays, "terminal_train", True)
    add_snapshot_losses(terminal, arrays, "terminal_train", train)
    train_health = bool(all(row["all_finite"] and row["boundary_violation_count"] == 0
                            and row["k3_cox_identity_worst"] <= p9.IDENTITY_TOL
                            for row in [*terminal["meshes"].values(), terminal["pooled"]]))
    train_pass = p9.gate(terminal, 2e-4, 7e-4, True)
    capacity = p9.capacity_metrics(train, work["generator"], q, norm["mean"], norm["scales"], arrays)
    control_flow_match = bool(train_health and not train_pass)
    after_trees = source_tree_records(work)
    source_after = "synthetic" if smoke else c.sha256(source_path)
    zero_update_binding = bool(before_trees == after_trees and source_before == source_after)
    recovered_checkpoint = {
        "status": "excluded_execution_smoke" if smoke else "terminal_recovery_from_failed_work_checkpoint",
        "complete_original_phase9_result": False,
        "source_work_checkpoint_sha256": source_before,
        "optimizer_updates": 0,
        "generator": jax.tree_util.tree_map(np.asarray, work["generator"]),
        "encoder": jax.tree_util.tree_map(np.asarray, work["encoder"]),
        "predictor": jax.tree_util.tree_map(np.asarray, work["predictor"]),
        "folded_predictor": jax.tree_util.tree_map(np.asarray, folded),
        "q_raw": np.asarray(work["q_raw"]),
        "final_target_states": np.asarray(work["final_target_states"]),
        "evaluation_states": np.asarray(work["evaluation_states"]),
        "predictor_optimizer_state": jax.tree_util.tree_map(np.asarray, work["predictor_optimizer_state"]),
        "normalization": {key: np.asarray(norm[key]) for key in ("mean", "scales", "feature_mean", "feature_scale")},
        "missing_not_fabricated": ["update_resolution_order", "epoch_history",
                                   "encoder_optimizer_state", "joint_optimizer_state",
                                   "preflight", "encoder_handoff_telemetry"],
    }
    p9.atomic_pickle(args.checkpoint, recovered_checkpoint)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    information_boundary = {"train_only_data_regenerated": True, "selection_touched": False,
                            "model_validation_touched": False, "confirmation_touched": False,
                            "weak_eq_touched": False, "scaling_touched": False}
    decision = {"recovery_valid": bool(zero_update_binding and control_flow_match),
                "complete_original_phase9_result": False, "optimizer_updates": 0,
                "train_health": train_health, "train_pass": train_pass,
                "selection_evaluated": False, "capacity_license_complete": False,
                "g2_licensed": False, "t2_licensed": False,
                "scientific_promotion_allowed": False, "next_action": "root audit; hard stop"}
    report = {
        "status": "excluded_execution_smoke" if smoke else "terminal_recovery_complete",
        "classification": "audited terminal metrics from exact final work checkpoint; not a complete original Phase9 result",
        "provenance": c.provenance(), "original_job": EXPECTED_ORIGINAL_JOB,
        "original_commit": EXPECTED_ORIGINAL_COMMIT, "bindings": bindings,
        "failed_run_bindings": failed_bindings, "config": CONFIG,
        "parameter_counts": {name: tree_count(work[name]) for name in ("generator", "encoder", "predictor")},
        "source_work": {"sha256_before": source_before, "sha256_after": source_after,
                        "trees_before": before_trees, "trees_after": after_trees,
                        "zero_update_binding": zero_update_binding},
        "work_checkpoint_checks": work_checks,
        "data": {"training": legacy.metadata(train), "target_chunks": target_records,
                 "snapshot_count": int(len(coefficients)), "regeneration": regeneration},
        "normalization": {"definition": "train-only coefficient mean/two head RMS and predictor feature statistics",
                          "mean_sha256": p9.array_sha(norm["mean"]), "scales_sha256": p9.array_sha(norm["scales"]),
                          "normalized_coefficients_sha256": p9.array_sha(norm["normalized"]),
                          "feature_mean_sha256": p9.array_sha(norm["feature_mean"]),
                          "feature_scale_sha256": p9.array_sha(norm["feature_scale"]),
                          "features_sha256": p9.array_sha(features)},
        "updates": {"optimizer_updates": 0, "schedule_steps": 0, "permutations_used": 0},
        "terminal_train": terminal, "capacity_final_only": capacity,
        "capacity_license": {"complete": False, "g2_licensed": False,
                             "reason": "joint epochs 24 and 27 are absent; late-improvement evidence is unrecoverable",
                             "terminal_capacity_descriptive_only": True},
        "control_flow_match": control_flow_match, "information_boundary": information_boundary,
        "decision": decision,
        "npz": {"basename": os.path.basename(args.output_npz), "sha256": c.sha256(args.output_npz)},
        "checkpoint": {"basename": os.path.basename(args.checkpoint), "sha256": c.sha256(args.checkpoint)},
    }
    p9.atomic_json(args.output_json, report)
    p9.atomic_json(args.progress_json, {"status": "complete", "optimizer_updates": 0,
                                       "scientific_metrics_exposed": False})
    print(json.dumps({"status": report["status"], "decision": decision}, sort_keys=True), flush=True)
    print("ALL-DONE", flush=True)


if __name__ == "__main__":
    main()
