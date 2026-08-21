#!/usr/bin/env python
"""Independent negative-aware audit for Phase-9 terminal recovery."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import b10_common as c
import b10_phase9_terminal_recovery as recovery
import b10_phase9_train as p9
import b10_spline_train as legacy


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, name = line.rstrip().split("  ", 1)
            rows[name.removeprefix("./")] = digest
    return rows


def close(left, right, rtol=2e-13, atol=2e-14):
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol,
                            atol=atol, equal_nan=False))


def capacity_close(left, right):
    return close(left, right, rtol=5e-13, atol=5e-13)


def capacity_delta_close(left, right):
    a = np.asarray(left); b = np.asarray(right)
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and float(np.max(np.abs(a-b))) <= 1e-12 * scale)


def metrics_from_arrays(arrays, prefix, n):
    num = np.asarray(arrays[f"{prefix}_N{n}_numerator"], np.float64)
    den = np.asarray(arrays[f"{prefix}_N{n}_denominator"], np.float64)
    trajectory = np.asarray(arrays[f"{prefix}_N{n}_trajectory"], np.float64)
    boundary = np.asarray(arrays[f"{prefix}_N{n}_boundary_count"], np.int64)
    identity = np.asarray(arrays[f"{prefix}_N{n}_identity"], np.float64)
    return {"mean": float(np.mean(trajectory)), "worst": float(np.max(trajectory)),
            "loss": float(np.mean(num / np.maximum(den, 1e-300))),
            "boundary": int(np.sum(boundary)), "identity": float(np.max(identity)),
            "finite": bool(all(np.all(np.isfinite(x)) for x in (num, den, trajectory, identity)))}


def metric_contract(report, arrays, prefix="terminal_train"):
    pooled_trajectory = []; pooled_num = []; pooled_den = []; pooled_identity = []; pooled_boundary = []
    checks = []
    for key, row in report["meshes"].items():
        n = int(key); one = metrics_from_arrays(arrays, prefix, n)
        checks.extend((close(one["mean"], row["trajectory_error_mean"]),
                       close(one["worst"], row["trajectory_error_worst"]),
                       close(one["loss"], row["mean_snapshot_relative_l2_squared"]),
                       one["boundary"] == row["boundary_violation_count"],
                       close(one["identity"], row["k3_cox_identity_worst"]),
                       one["finite"] == row["all_finite"]))
        pooled_trajectory.extend(np.asarray(arrays[f"{prefix}_N{n}_trajectory"]))
        pooled_num.extend(np.asarray(arrays[f"{prefix}_N{n}_numerator"]))
        pooled_den.extend(np.asarray(arrays[f"{prefix}_N{n}_denominator"]))
        pooled_identity.extend(np.asarray(arrays[f"{prefix}_N{n}_identity"]))
        pooled_boundary.extend(np.asarray(arrays[f"{prefix}_N{n}_boundary_count"]))
    row = report["pooled"]
    checks.extend((close(np.mean(pooled_trajectory), row["trajectory_error_mean"]),
                   close(np.max(pooled_trajectory), row["trajectory_error_worst"]),
                   close(np.mean(np.asarray(pooled_num) / np.maximum(pooled_den, 1e-300)),
                         row["mean_snapshot_relative_l2_squared"]),
                   int(np.sum(pooled_boundary)) == row["boundary_violation_count"],
                   close(np.max(pooled_identity), row["k3_cox_identity_worst"])))
    return bool(all(checks))


def capacity_contract(report, arrays):
    summaries = {"meshes": {}}; classifications = []; finite_work = []
    for n in [int(x) for x in report["meshes"]]:
        values = [np.asarray(arrays[f"capacity_N{n}_{name}"]) for name in
                  ("gamma", "eta", "residual", "tangent", "bound_fraction",
                   "cg_breakdown", "cg_nonconverged")]
        summaries["meshes"][str(n)] = p9.capacity_summary(*values)
        relative = np.asarray(arrays[f"capacity_N{n}_cg_relative"])
        breakdown = np.asarray(arrays[f"capacity_N{n}_cg_breakdown"], bool)
        converged = np.asarray(arrays[f"capacity_N{n}_cg_converged"], bool)
        nonconverged = np.asarray(arrays[f"capacity_N{n}_cg_nonconverged"], bool)
        expected_converged = np.isfinite(relative) & (relative <= p9.CAPACITY_CG_RELATIVE_TOL) & ~breakdown
        expected_nonconverged = ~breakdown & ~expected_converged
        classifications.append(np.array_equal(converged, expected_converged)
                               and np.array_equal(nonconverged, expected_nonconverged))
        delta = np.asarray(arrays[f"capacity_N{n}_delta"])
        iters = np.asarray(arrays[f"capacity_N{n}_cg_iterations"])
        finite_work.append(delta.ndim == 2 and delta.shape[1] == 19
                           and np.all(np.isfinite(delta)) and np.all(np.isfinite(relative))
                           and np.all(relative >= 0) and np.all(iters >= 0) and np.all(iters <= 38))
    pooled = [np.concatenate([np.asarray(arrays[f"capacity_N{n}_{name}"])
                              for n in map(int, report["meshes"])])
              for name in ("gamma", "eta", "residual", "tangent", "bound_fraction")]
    breakdown = np.concatenate([np.asarray(arrays[f"capacity_N{n}_cg_breakdown"], bool)
                                for n in map(int, report["meshes"])])
    nonconverged = np.concatenate([np.asarray(arrays[f"capacity_N{n}_cg_nonconverged"], bool)
                                   for n in map(int, report["meshes"])])
    summaries["pooled"] = p9.capacity_summary(*pooled, breakdown, nonconverged)
    return bool(summaries == report and all(classifications) and all(finite_work))


def exact_array_sets(left, right, prefix, suffixes, meshes):
    return bool(all(np.array_equal(np.asarray(left[f"{prefix}_N{n}_{suffix}"]),
                                      np.asarray(right[f"{prefix}_N{n}_{suffix}"]))
                    for n in meshes for suffix in suffixes))


def capacity_array_checks(left, right, meshes):
    exact = ("cg_breakdown", "cg_converged", "cg_nonconverged")
    floating = ("gamma", "eta", "residual", "tangent", "delta", "cg_relative", "bound_fraction")
    checks = {f"N{n}_{suffix}": {"pass": bool(np.array_equal(np.asarray(left[f"capacity_N{n}_{suffix}"]),
                                                                np.asarray(right[f"capacity_N{n}_{suffix}"])))}
              for n in meshes for suffix in exact}
    for n in meshes:
        a = np.asarray(left[f"capacity_N{n}_cg_iterations"], np.int64)
        b = np.asarray(right[f"capacity_N{n}_cg_iterations"], np.int64)
        checks[f"N{n}_cg_iterations"] = {"pass": bool(np.all(a >= 0) and np.all(a <= 38)
                                                       and np.all(b >= 0) and np.all(b <= 38)
                                                       and np.max(np.abs(a-b)) <= 1),
                                              "max_abs": int(np.max(np.abs(a-b)))}
    for n in meshes:
        for suffix in floating:
            a = np.asarray(left[f"capacity_N{n}_{suffix}"]); b = np.asarray(right[f"capacity_N{n}_{suffix}"])
            one_pass = (capacity_delta_close(a, b) if suffix == "delta" else
                        close(a, b, rtol=5e-13, atol=1e-12) if suffix == "cg_relative" else
                        capacity_close(a, b))
            checks[f"N{n}_{suffix}"] = {"pass": one_pass,
                                         "max_abs": float(np.max(np.abs(a-b))),
                                         "max_magnitude": float(max(np.max(np.abs(a)), np.max(np.abs(b))))}
    checks["pass"] = bool(all(value["pass"] for value in checks.values()))
    return checks


def nested_close(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return bool(set(left) == set(right) and all(nested_close(left[key], right[key]) for key in left))
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        return close(left, right)
    return left == right


def decision_contract(report, arrays, checkpoint, source_work, folded):
    decision = report["decision"]
    expected_boundary = {"train_only_data_regenerated": True, "selection_touched": False,
                         "model_validation_touched": False, "confirmation_touched": False,
                         "weak_eq_touched": False, "scaling_touched": False}
    zero = bool(np.array_equal(np.asarray(arrays["optimizer_updates"]), np.asarray((0,), np.int64))
                and checkpoint["optimizer_updates"] == 0
                and report["updates"] == {"optimizer_updates": 0, "schedule_steps": 0,
                                           "permutations_used": 0})
    state = bool(recovery.tree_exact(checkpoint["generator"], source_work["generator"])
                 and recovery.tree_exact(checkpoint["encoder"], source_work["encoder"])
                 and recovery.tree_exact(checkpoint["predictor"], source_work["predictor"])
                 and recovery.tree_exact(checkpoint["predictor_optimizer_state"],
                                         source_work["predictor_optimizer_state"])
                 and recovery.tree_exact(checkpoint["folded_predictor"], folded)
                 and np.array_equal(checkpoint["q_raw"], source_work["q_raw"])
                 and np.array_equal(checkpoint["final_target_states"], source_work["final_target_states"])
                 and np.array_equal(checkpoint["evaluation_states"], source_work["evaluation_states"]))
    forced = bool(report["capacity_license"] == {
        "complete": False, "g2_licensed": False,
        "reason": "joint epochs 24 and 27 are absent; late-improvement evidence is unrecoverable",
        "terminal_capacity_descriptive_only": True}
        and decision["complete_original_phase9_result"] is False
        and decision["selection_evaluated"] is False
        and decision["capacity_license_complete"] is False
        and decision["g2_licensed"] is False and decision["t2_licensed"] is False
        and decision["scientific_promotion_allowed"] is False)
    metrics = metric_contract(report["terminal_train"], arrays)
    capacity = capacity_contract(report["capacity_final_only"], arrays)
    return bool(zero and state and forced and metrics and capacity
                and report["information_boundary"] == expected_boundary
                and report["control_flow_match"] is True
                and decision["train_health"] is True and decision["train_pass"] is False
                and decision["recovery_valid"] is True)


def corruption_tests(report, arrays, checkpoint, source_work, folded):
    cases = []
    one = copy.deepcopy(report); one["updates"]["optimizer_updates"] = 1; cases.append((one, dict(arrays), copy.deepcopy(checkpoint)))
    one = copy.deepcopy(report); one["information_boundary"]["selection_touched"] = True; cases.append((one, dict(arrays), copy.deepcopy(checkpoint)))
    one = copy.deepcopy(report); one["capacity_license"]["g2_licensed"] = True; cases.append((one, dict(arrays), copy.deepcopy(checkpoint)))
    one_arrays = dict(arrays); one_arrays["optimizer_updates"] = np.asarray((1,), np.int64); cases.append((copy.deepcopy(report), one_arrays, copy.deepcopy(checkpoint)))
    one_checkpoint = copy.deepcopy(checkpoint); one_checkpoint["q_raw"] = np.asarray(one_checkpoint["q_raw"]).copy(); one_checkpoint["q_raw"].flat[0] += 1e-6; cases.append((copy.deepcopy(report), dict(arrays), one_checkpoint))
    one_checkpoint = copy.deepcopy(checkpoint); leaves = jax.tree_util.tree_leaves(one_checkpoint["folded_predictor"]); leaves[0].flat[0] += 1e-6; cases.append((copy.deepcopy(report), dict(arrays), one_checkpoint))
    n = int(next(iter(report["terminal_train"]["meshes"])))
    one_arrays = dict(arrays); one_arrays[f"terminal_train_N{n}_boundary_count"] = np.asarray(one_arrays[f"terminal_train_N{n}_boundary_count"]).copy(); one_arrays[f"terminal_train_N{n}_boundary_count"].flat[0] = 1; cases.append((copy.deepcopy(report), one_arrays, copy.deepcopy(checkpoint)))
    one_arrays = dict(arrays); one_arrays[f"capacity_N{n}_cg_converged"] = ~np.asarray(one_arrays[f"capacity_N{n}_cg_converged"], bool); cases.append((copy.deepcopy(report), one_arrays, copy.deepcopy(checkpoint)))
    one_arrays = dict(arrays); one_arrays[f"capacity_N{n}_cg_iterations"] = np.asarray(one_arrays[f"capacity_N{n}_cg_iterations"]).copy(); one_arrays[f"capacity_N{n}_cg_iterations"].flat[0] = 39; cases.append((copy.deepcopy(report), one_arrays, copy.deepcopy(checkpoint)))
    rejected = [not decision_contract(one_report, one_arrays, one_checkpoint, source_work, folded)
                for one_report, one_arrays, one_checkpoint in cases]
    return {"positive_contract_pass": decision_contract(report, arrays, checkpoint, source_work, folded),
            "corruption_count": len(rejected), "corruptions_rejected": bool(all(rejected)),
            "pass": bool(decision_contract(report, arrays, checkpoint, source_work, folded) and all(rejected))}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True); parser.add_argument("--source-npz", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--prereg", required=True); parser.add_argument("--failed-dir")
    parser.add_argument("--p5-json"); parser.add_argument("--p7-npz"); parser.add_argument("--target-dir")
    parser.add_argument("--expected-commit", required=True); parser.add_argument("--expected-job", required=True)
    parser.add_argument("--slurm-out", required=True); parser.add_argument("--slurm-err", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); report = load_json(args.source_json)
    arrays_file = np.load(args.source_npz, allow_pickle=False); arrays = {key: arrays_file[key] for key in arrays_file.files}
    checkpoint = pickle.load(open(args.checkpoint, "rb")); rows = manifest(args.manifest)
    stdout = open(args.slurm_out, encoding="utf-8").read(); stderr = open(args.slurm_err, encoding="utf-8").read()
    provenance = {"commit": args.smoke or report["provenance"].get("commit") == args.expected_commit,
                  "job": args.smoke or str(report["provenance"].get("slurm_job_id")) == str(args.expected_job),
                  "gpu": report["provenance"].get("jax_backend") == "gpu"
                         and (args.smoke or report["provenance"].get("gpu_kind") == "NVIDIA H200"),
                  "precision": report["provenance"].get("x64") is True
                               and report["provenance"].get("matmul_precision") == "highest",
                  "logs": "jax_backend=gpu" in stdout and "ALL-DONE" in stdout
                          and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)", stdout + stderr)}
    if args.smoke:
        failed = {"mode": "synthetic"}; dependency_binding = True
    else:
        failed = recovery.validate_failed(args.failed_dir)
        dependency_binding = all(rows.get("code/deps/failed/" + relative) == digest
                                 for relative, digest in recovery.EXPECTED_FAILED.items())
        phase_stems = {"p4": "phase4_d", "p5": "phase5_d", "p6": "phase6_d",
                       "p7": "phase7_train", "p8": "phase8_d"}
        for phase, stem in phase_stems.items():
            kinds = ("json", "npz", "checkpoint", "audit", "manifest") if phase == "p7" else ("json", "npz", "audit", "manifest")
            for kind in kinds:
                basename = ("AUDIT.json" if kind == "audit" else "MANIFEST.sha256" if kind == "manifest"
                            else "checkpoint.pkl" if kind == "checkpoint" else f"{stem}.{kind}")
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}") == report["bindings"][f"{phase}_{kind}"]["sha256"]
    source_binding = bool(args.smoke or (rows.get("code/b10_phase9_terminal_recovery.py") is not None
                          and rows.get("code/b10_audit_phase9_terminal_recovery.py") is not None
                          and rows.get("code/PHASE-9-TERMINAL-RECOVERY-PRE-REGISTRATION.md") == c.sha256(args.prereg)))
    artifact_binding = bool(report["npz"]["sha256"] == c.sha256(args.source_npz)
                            and report["checkpoint"]["sha256"] == c.sha256(args.checkpoint))
    train, coefficients, target_features, _, regeneration = recovery.load_train(args, args.smoke)
    features = legacy.concatenate(train, "features"); norm = p9.train_normalization(coefficients, features)
    source_work = ({"generator": checkpoint["generator"], "encoder": checkpoint["encoder"],
                    "predictor": checkpoint["predictor"], "q_raw": checkpoint["q_raw"],
                    "final_target_states": checkpoint["final_target_states"],
                    "evaluation_states": checkpoint["evaluation_states"],
                    "predictor_optimizer_state": checkpoint["predictor_optimizer_state"]}
                   if args.smoke else
                   pickle.load(open(os.path.join(args.failed_dir, "out/work_checkpoint.pkl"), "rb")))
    folded = legacy.fold_predictor_standardization(source_work["predictor"], norm["feature_mean"], norm["feature_scale"])
    state_delta = np.abs(p9.predictor_batches(folded, features) - np.asarray(source_work["evaluation_states"]))
    state_binding = bool(recovery.tree_exact(checkpoint["generator"], source_work["generator"])
                         and recovery.tree_exact(checkpoint["encoder"], source_work["encoder"])
                         and recovery.tree_exact(checkpoint["predictor"], source_work["predictor"])
                         and recovery.tree_exact(checkpoint["folded_predictor"], folded)
                         and np.array_equal(checkpoint["q_raw"], source_work["q_raw"])
                         and np.array_equal(checkpoint["final_target_states"], source_work["final_target_states"])
                         and np.all(state_delta <= recovery.STATE_ATOL))
    normalization = bool(np.array_equal(arrays["coefficient_mean"], norm["mean"])
                         and np.array_equal(arrays["head_scales"], norm["scales"])
                         and np.array_equal(arrays["predictor_feature_mean"], norm["feature_mean"])
                         and np.array_equal(arrays["predictor_feature_scale"], norm["feature_scale"])
                         and np.array_equal(arrays["training_features"], features)
                         and np.array_equal(arrays["normalization_source_indices"], np.arange(len(coefficients))))
    fresh = {}; q = np.tanh(np.asarray(checkpoint["q_raw"])); states = np.concatenate((legacy.concatenate(train, "affine"), q), axis=1)
    fresh_metrics = p9.evaluate_full(train, checkpoint["generator"], states, norm["mean"], norm["scales"],
                                     recovery.CONFIG, fresh, "terminal_train", True)
    recovery.add_snapshot_losses(fresh_metrics, fresh, "terminal_train", train)
    meshes = [item["N"] for item in train]
    full_field_arrays = exact_array_sets(fresh, arrays, "terminal_train",
                                         ("numerator", "denominator", "trajectory", "boundary_count", "identity"), meshes)
    full_field = bool(full_field_arrays and fresh_metrics == report["terminal_train"])
    fresh_capacity_arrays = {}; fresh_capacity = p9.capacity_metrics(
        train, checkpoint["generator"], q, norm["mean"], norm["scales"], fresh_capacity_arrays)
    capacity_array_detail = capacity_array_checks(fresh_capacity_arrays, arrays, meshes)
    capacity_arrays = capacity_array_detail["pass"]
    capacity_summary = nested_close(fresh_capacity, report["capacity_final_only"])
    capacity_persisted = capacity_contract(report["capacity_final_only"], arrays)
    capacity = bool(capacity_arrays and capacity_summary and capacity_persisted)
    contract = decision_contract(report, arrays, checkpoint, source_work, folded)
    negative = corruption_tests(report, arrays, checkpoint, source_work, folded)
    metadata_expected = ([(16, 0, 2, 2, 2, 0, 4)] if args.smoke else
                         [(64, 0, 512, 512, 51, 0, 26112), (128, 0, 128, 128, 51, 26112, 32640),
                          (256, 0, 64, 64, 51, 32640, 35904)])
    metadata_actual = [(row["N"], row["source_start"], row["source_stop"], row["case_count"],
                        row["num_times"], row["global_start"], row["global_stop"])
                       for row in legacy.metadata(train)]
    metadata = bool(metadata_actual == metadata_expected and report["data"]["training"] == legacy.metadata(train)
                    and all(row["reference_health"]["reported_max_relative_residual"] <= 1e-8
                            and row["reference_health"]["independent_max_relative_residual"] <= 1e-8
                            for row in legacy.metadata(train)))
    health = bool(all(provenance.values()) and failed == report["failed_run_bindings"]
                  and dependency_binding and source_binding and artifact_binding and state_binding
                  and normalization and metadata and full_field and capacity and contract and negative["pass"])
    result = {"status": "pass" if health else "fail", "negative_aware": True,
              "classification": report["classification"], "expected_commit": args.expected_commit,
              "expected_job": str(args.expected_job), "manifest_sha256": c.sha256(args.manifest),
              "source_json_sha256": c.sha256(args.source_json), "source_npz_sha256": c.sha256(args.source_npz),
              "checkpoint_sha256": c.sha256(args.checkpoint),
              "checks": {"provenance": provenance, "failed_bundle": failed == report["failed_run_bindings"],
                         "dependency_binding": dependency_binding, "source_binding": source_binding,
                         "artifact_binding": artifact_binding, "state_fold_binding": state_binding,
                         "state_fold_max_abs": float(np.max(state_delta)),
                         "normalization": normalization, "metadata_reference_health": metadata,
                         "full_field_independent": full_field, "capacity_independent": capacity,
                         "capacity_array_match": capacity_arrays,
                         "capacity_array_detail": capacity_array_detail,
                         "capacity_summary_match": capacity_summary,
                         "capacity_persisted_contract": capacity_persisted,
                         "decision_contract": contract, "negative_self_test": negative,
                         "selection_sealed": report["information_boundary"]["selection_touched"] is False,
                         "capacity_license_forced_incomplete": report["capacity_license"]["complete"] is False
                                                               and report["decision"]["g2_licensed"] is False},
              "decision": report["decision"]}
    p9.atomic_json(args.output, result)
    if not health:
        raise SystemExit("Phase9 terminal-recovery independent audit failed")
    print(json.dumps({"status": "pass", "decision": report["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
