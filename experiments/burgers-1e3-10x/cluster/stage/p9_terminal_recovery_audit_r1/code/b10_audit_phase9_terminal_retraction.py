#!/usr/bin/env python
"""Audit-only acceptance of the Phase-9 terminal field with capacity retracted."""
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
import b10_audit_phase9_terminal_recovery as prior_audit
import b10_phase9_terminal_recovery as recovery
import b10_phase9_train as p9
import b10_spline_train as legacy


EXPECTED_RECOVERY = {
    "FAILURE.json": "22821e4366011d5d45627c9a7aa8d9daf46d5ed5324fceede0b5738a301ba083",
    "LOCAL.sha256": "2ea4a09470eb36d7b1ea5d45fc1ff2144be1a23dac55789281b05ce7b33f94fc",
    "MANIFEST.sha256": "3b001ee45df4d9889dec99052dee9489623da2ca158b9db9d9006e35739279dd",
    "REMOTE.sha256": "84e5664b6abe7099430a8734def6aaaff41e9a8ae87f86879efa75c1b427264f",
    "SACCT.txt": "08badfd4376b5bcfbad71ada7365fc8b2a03c292a19225f1d38444e16d014ead",
    "logs/2735251.out": "8bfbb5eb23004652bf159307829d55a4b1bb4f2fad5d4af572f39022fd936dfa",
    "logs/2735251.err": "96de81be0ba1091dd1fb18d6cf6c58ea1d768537536e68f08d671524f4bbd212",
    "out/PROGRESS.json": "21c60d69237d8c3f728ab6c646088d7bc288f0b7d1d368a9b8e6f9e127ea1391",
    "out/phase9_terminal_recovery.json": "8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d",
    "out/phase9_terminal_recovery.npz": "d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e",
    "out/terminal_checkpoint.pkl": "90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9",
    "out/AUDIT.json": "a1c6c0ee23e16da4aab1e4320aedcab4bd7039ec650429908feb15a471f99cec",
}


def validate_recovery_bundle(root):
    bindings = {}
    for relative, expected in EXPECTED_RECOVERY.items():
        path = os.path.join(root, relative)
        if not os.path.isfile(path) or c.sha256(path) != expected:
            raise SystemExit(f"immutable failed recovery mismatch: {relative}")
        bindings[relative] = expected
    recovery.verify_checksum_file(root, "LOCAL.sha256")
    failure = prior_audit.load_json(os.path.join(root, "FAILURE.json"))
    old = prior_audit.load_json(os.path.join(root, "out/AUDIT.json"))
    if not (failure["status"] == "terminal_recovery_artifacts_complete_independent_audit_failed"
            and failure["remote_preserved"] is True and failure["remote_cleaned"] is False
            and old["status"] == "fail" and old["checks"]["full_field_independent"] is True
            and old["checks"]["capacity_independent"] is False
            and old["decision"]["g2_licensed"] is False):
        raise SystemExit("failed recovery classification mismatch")
    return bindings, old


def internal_capacity_health(summary, arrays, meshes):
    contract = prior_audit.capacity_contract(summary, arrays)
    finite = bool(all(row["all_finite"] for row in [*summary["meshes"].values(), summary["pooled"]]))
    breakdown_free = bool(all(row["cg_breakdown_count"] == 0
                              for row in [*summary["meshes"].values(), summary["pooled"]]))
    return {"internal_classification_and_summary": contract, "finite": finite,
            "breakdown_free": breakdown_free, "pass": bool(contract and finite and breakdown_free)}


def capacity_disagreements(original, repeat, meshes):
    report = {"meshes": {}}
    for n in meshes:
        oit = np.asarray(original[f"capacity_N{n}_cg_iterations"], np.int64)
        rit = np.asarray(repeat[f"capacity_N{n}_cg_iterations"], np.int64)
        oconv = np.asarray(original[f"capacity_N{n}_cg_converged"], bool)
        rconv = np.asarray(repeat[f"capacity_N{n}_cg_converged"], bool)
        onon = np.asarray(original[f"capacity_N{n}_cg_nonconverged"], bool)
        rnon = np.asarray(repeat[f"capacity_N{n}_cg_nonconverged"], bool)
        iwhere = np.flatnonzero(oit != rit); cwhere = np.flatnonzero(oconv != rconv)
        nwhere = np.flatnonzero(onon != rnon)
        report["meshes"][str(n)] = {
            "iteration_disagreement_count": int(iwhere.size),
            "iteration_max_abs_difference": int(np.max(np.abs(oit-rit))),
            "iteration_disagreement_indices_sha256": p9.array_sha(iwhere.astype(np.int32)),
            "convergence_disagreement_count": int(cwhere.size),
            "convergence_disagreement_indices_sha256": p9.array_sha(cwhere.astype(np.int32)),
            "nonconvergence_disagreement_count": int(nwhere.size),
            "nonconvergence_disagreement_indices_sha256": p9.array_sha(nwhere.astype(np.int32)),
            "cg_relative_max_abs_difference": float(np.max(np.abs(
                np.asarray(original[f"capacity_N{n}_cg_relative"])
                - np.asarray(repeat[f"capacity_N{n}_cg_relative"])))),
        }
    report["historical_or_current_disagreement_present"] = bool(any(
        row["iteration_disagreement_count"] or row["convergence_disagreement_count"]
        for row in report["meshes"].values()))
    return report


def retraction_contract(value):
    return bool(value["terminal_full_field_accepted"] is True
                and value["capacity_retracted"] is True
                and value["capacity_reproducible"] is False
                and value["capacity_accepted"] is False
                and value["capacity_license_complete"] is False
                and value["g2_licensed"] is False
                and value["optimizer_updates"] == 0
                and value["selection_evaluated"] is False
                and value["historical_disagreement_bound"] is True
                and value["original_capacity_internal_health"] is True
                and value["repeat_capacity_internal_health"] is True)


def retraction_negative_test(value):
    corruptions = []
    for key, replacement in (("terminal_full_field_accepted", False),
                             ("capacity_retracted", False),
                             ("capacity_reproducible", True),
                             ("capacity_accepted", True),
                             ("capacity_license_complete", True),
                             ("g2_licensed", True),
                             ("optimizer_updates", 1),
                             ("selection_evaluated", True),
                             ("historical_disagreement_bound", False),
                             ("original_capacity_internal_health", False),
                             ("repeat_capacity_internal_health", False)):
        one = copy.deepcopy(value); one[key] = replacement
        corruptions.append(not retraction_contract(one))
    return {"positive_contract_pass": retraction_contract(value),
            "corruption_count": len(corruptions), "corruptions_rejected": bool(all(corruptions)),
            "pass": bool(retraction_contract(value) and all(corruptions))}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True); parser.add_argument("--source-npz", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--prior-failed-audit", required=True)
    parser.add_argument("--recovery-dir"); parser.add_argument("--failed-t1-dir")
    parser.add_argument("--manifest", required=True); parser.add_argument("--prereg", required=True)
    parser.add_argument("--p5-json"); parser.add_argument("--p7-npz"); parser.add_argument("--target-dir")
    parser.add_argument("--expected-commit", required=True); parser.add_argument("--expected-job", required=True)
    parser.add_argument("--slurm-out", required=True); parser.add_argument("--slurm-err", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--audit-work-npz", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); c.require_gpu_highest()
    report = prior_audit.load_json(args.source_json)
    source_file = np.load(args.source_npz, allow_pickle=False)
    arrays = {key: source_file[key] for key in source_file.files}
    checkpoint = pickle.load(open(args.checkpoint, "rb"))
    rows = prior_audit.manifest(args.manifest)
    stdout = open(args.slurm_out, encoding="utf-8").read()
    stderr = open(args.slurm_err, encoding="utf-8").read()
    runtime = c.provenance()
    provenance = {
        "commit": runtime["commit"] == args.expected_commit,
        "job": str(runtime["slurm_job_id"]) == str(args.expected_job),
        "gpu": runtime["jax_backend"] == "gpu" and (args.smoke or runtime["gpu_kind"] == "NVIDIA H200"),
        "precision": runtime["x64"] is True and runtime["matmul_precision"] == "highest",
        "logs": "jax_backend=gpu" in stdout and "audit_only=true optimizer_updates=0" in stdout
                and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)", stdout+stderr),
    }
    if args.smoke:
        recovery_bindings = {"mode": "synthetic"}
        old_audit = prior_audit.load_json(args.prior_failed_audit)
        dependency_binding = source_binding = target_binding = failed_t1_binding = True
    else:
        recovery_bindings, old_audit = validate_recovery_bundle(args.recovery_dir)
        failed_t1_binding = recovery.validate_failed(args.failed_t1_dir) == report["failed_run_bindings"]
        dependency_binding = bool(all(rows.get("code/deps/recovery/" + relative) == digest
                                      for relative, digest in EXPECTED_RECOVERY.items()))
        for phase, stem in {"p4":"phase4_d", "p5":"phase5_d", "p6":"phase6_d",
                            "p7":"phase7_train", "p8":"phase8_d"}.items():
            kinds = ("json","npz","checkpoint","audit","manifest") if phase == "p7" else ("json","npz","audit","manifest")
            for kind in kinds:
                basename = ("AUDIT.json" if kind == "audit" else "MANIFEST.sha256" if kind == "manifest"
                            else "checkpoint.pkl" if kind == "checkpoint" else f"{stem}.{kind}")
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}") == report["bindings"][f"{phase}_{kind}"]["sha256"]
        p5_report = prior_audit.load_json(args.p5_json)
        target_binding = bool(len(p5_report["train_targets"]["chunks"]) == 16
                              and all(rows.get("code/deps/p5/targets/"+row["basename"]) == row["sha256"]
                                      for row in p5_report["train_targets"]["chunks"]))
        source_binding = bool(all(rows.get("code/"+name) == digest
                                  for name, digest in runtime["source_sha256"].items())
                              and rows.get("code/PHASE-9-TERMINAL-RECOVERY-PRE-REGISTRATION.md") == c.sha256(args.prereg))
    immutable_artifacts = bool(args.smoke or (
        c.sha256(args.source_json) == EXPECTED_RECOVERY["out/phase9_terminal_recovery.json"]
        and c.sha256(args.source_npz) == EXPECTED_RECOVERY["out/phase9_terminal_recovery.npz"]
        and c.sha256(args.checkpoint) == EXPECTED_RECOVERY["out/terminal_checkpoint.pkl"]
        and c.sha256(args.prior_failed_audit) == EXPECTED_RECOVERY["out/AUDIT.json"]))
    train, coefficients, _, target_records, regeneration = recovery.load_train(args, args.smoke)
    features = legacy.concatenate(train, "features"); norm = p9.train_normalization(coefficients, features)
    source_work = ({"generator":checkpoint["generator"], "encoder":checkpoint["encoder"],
                    "predictor":checkpoint["predictor"], "q_raw":checkpoint["q_raw"],
                    "final_target_states":checkpoint["final_target_states"],
                    "evaluation_states":checkpoint["evaluation_states"],
                    "predictor_optimizer_state":checkpoint["predictor_optimizer_state"],
                    "optimizer_state":checkpoint["predictor_optimizer_state"]}
                   if args.smoke else pickle.load(open(os.path.join(args.failed_t1_dir,"out/work_checkpoint.pkl"),"rb")))
    folded = legacy.fold_predictor_standardization(source_work["predictor"], norm["feature_mean"], norm["feature_scale"])
    state_binding = bool(recovery.tree_exact(checkpoint["generator"],source_work["generator"])
        and recovery.tree_exact(checkpoint["encoder"],source_work["encoder"])
        and recovery.tree_exact(checkpoint["predictor"],source_work["predictor"])
        and prior_audit.tree_max_abs(checkpoint["folded_predictor"],folded) <= recovery.STATE_ATOL
        and np.array_equal(checkpoint["q_raw"],source_work["q_raw"])
        and np.array_equal(checkpoint["final_target_states"],source_work["final_target_states"])
        and np.max(np.abs(p9.predictor_batches(folded,features)-source_work["evaluation_states"])) <= recovery.STATE_ATOL)
    normalization = bool(np.array_equal(arrays["coefficient_mean"],norm["mean"])
        and np.array_equal(arrays["head_scales"],norm["scales"])
        and np.array_equal(arrays["training_features"],features)
        and report["normalization"]["normalized_coefficients_sha256"] == p9.array_sha(norm["normalized"]))
    q = np.tanh(np.asarray(checkpoint["q_raw"])); states = np.concatenate((legacy.concatenate(train,"affine"),q),axis=1)
    fresh_field = {}; metrics = p9.evaluate_full(train,checkpoint["generator"],states,norm["mean"],norm["scales"],
                                                  recovery.CONFIG,fresh_field,"terminal_train",True)
    recovery.add_snapshot_losses(metrics,fresh_field,"terminal_train",train)
    meshes = [item["N"] for item in train]
    full_field = bool(prior_audit.exact_array_sets(fresh_field,arrays,"terminal_train",
        ("numerator","denominator","trajectory","boundary_count","identity"),meshes)
        and metrics == report["terminal_train"] and prior_audit.metric_contract(metrics,arrays))
    repeat_arrays = {}; repeat_summary = p9.capacity_metrics(train,checkpoint["generator"],q,norm["mean"],norm["scales"],repeat_arrays)
    original_health = internal_capacity_health(report["capacity_final_only"],arrays,meshes)
    repeat_health = internal_capacity_health(repeat_summary,repeat_arrays,meshes)
    disagreement = capacity_disagreements(arrays,repeat_arrays,meshes)
    historical = bool(old_audit["status"] == "fail" and old_audit["checks"]["capacity_independent"] is False
                      and old_audit["checks"]["full_field_independent"] is True)
    work = {}
    for n in meshes:
        for suffix in ("gamma","eta","residual","tangent","delta","cg_iterations","cg_relative",
                       "cg_breakdown","cg_converged","cg_nonconverged","bound_fraction"):
            work[f"original_capacity_N{n}_{suffix}"] = arrays[f"capacity_N{n}_{suffix}"]
            work[f"repeat_capacity_N{n}_{suffix}"] = repeat_arrays[f"capacity_N{n}_{suffix}"]
        work[f"iteration_disagreement_indices_N{n}"] = np.flatnonzero(
            arrays[f"capacity_N{n}_cg_iterations"] != repeat_arrays[f"capacity_N{n}_cg_iterations"]).astype(np.int32)
        work[f"convergence_disagreement_indices_N{n}"] = np.flatnonzero(
            arrays[f"capacity_N{n}_cg_converged"] != repeat_arrays[f"capacity_N{n}_cg_converged"]).astype(np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(args.audit_work_npz)),exist_ok=True)
    np.savez_compressed(args.audit_work_npz,**work)
    decision = {"terminal_full_field_accepted": full_field, "train_health": report["decision"]["train_health"],
                "train_pass": report["decision"]["train_pass"], "capacity_retracted": True,
                "capacity_reproducible": False, "capacity_accepted": False,
                "capacity_license_complete": False, "g2_licensed": False,
                "optimizer_updates": 0, "selection_evaluated": False,
                "historical_disagreement_bound": historical,
                "original_capacity_internal_health": original_health["pass"],
                "repeat_capacity_internal_health": repeat_health["pass"],
                "complete_original_phase9_result": False, "scientific_promotion_allowed": False,
                "next_action": "Phase10 proposal root audit; no T2"}
    negative = retraction_negative_test(decision)
    old_negative = prior_audit.corruption_tests(report,arrays,checkpoint,source_work,folded)
    information = bool(report["information_boundary"] == {"train_only_data_regenerated":True,"selection_touched":False,
        "model_validation_touched":False,"confirmation_touched":False,"weak_eq_touched":False,"scaling_touched":False})
    health = bool(all(provenance.values()) and immutable_artifacts and dependency_binding and source_binding
                  and target_binding and failed_t1_binding and state_binding and normalization and full_field
                  and original_health["pass"] and repeat_health["pass"] and historical
                  and retraction_contract(decision) and negative["pass"] and old_negative["pass"] and information)
    result = {"status":"pass" if health else "fail", "negative_aware":True,
              "classification":"terminal full-field accepted; Phase9 recovery capacity retracted as nonportable descriptive evidence",
              "expected_commit":args.expected_commit,"expected_job":str(args.expected_job),
              "manifest_sha256":c.sha256(args.manifest),"source_json_sha256":c.sha256(args.source_json),
              "source_npz_sha256":c.sha256(args.source_npz),"checkpoint_sha256":c.sha256(args.checkpoint),
              "prior_failed_audit_sha256":c.sha256(args.prior_failed_audit),
              "audit_work_npz_sha256":c.sha256(args.audit_work_npz),
              "runtime_provenance":runtime,"recovery_bindings":recovery_bindings,
              "checks":{"provenance":provenance,"immutable_artifacts":immutable_artifacts,
                        "dependency_binding":dependency_binding,"source_binding":source_binding,
                        "target_manifest_binding":target_binding,"failed_t1_binding":failed_t1_binding,
                        "state_fold_binding":state_binding,"normalization":normalization,
                        "terminal_full_field_independent":full_field,
                        "original_capacity_internal_health":original_health,
                        "repeat_capacity_internal_health":repeat_health,
                        "historical_capacity_disagreement_bound":historical,
                        "capacity_reproducible":False,"capacity_accepted":False,
                        "information_boundary":information,"retraction_negative_test":negative,
                        "base_corruption_test":old_negative},
              "terminal_train":report["terminal_train"],
              "capacity_retraction":{"capacity_reproducible":False,"capacity_accepted":False,
                    "original_summary_unaccepted":report["capacity_final_only"],
                    "repeat_summary_unaccepted":repeat_summary,"disagreements":disagreement,
                    "work_npz_basename":os.path.basename(args.audit_work_npz),
                    "work_npz_sha256":c.sha256(args.audit_work_npz)},
              "data":{"training":legacy.metadata(train),"target_chunks":target_records,"regeneration":regeneration},
              "decision":decision}
    p9.atomic_json(args.output,result)
    if not health: raise SystemExit("Phase9 terminal retraction audit failed")
    print(json.dumps({"status":"pass","decision":decision},sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
