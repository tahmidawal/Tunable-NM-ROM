#!/usr/bin/env python
"""Independent negative-aware audit for the immutable Phase-8 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import numpy as np


EXPECTED = {
    "p4_json": "ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617",
    "p4_npz": "720c5890b22709c83858f46e43f18fa3d28bb1305544f02ad9aea71625a2228f",
    "p4_audit": "f41010b72ad9ddae43409a1c1d2073dc2839edea22e57a7d5bafc8e2359e08a3",
    "p4_manifest": "67a3bf8d0d8c755ec39cd4056a0bca0e852b4ba17493e8f052a0b288e63a201a",
    "p5_json": "97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96",
    "p5_npz": "5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54",
    "p5_audit": "c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff",
    "p5_manifest": "6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6",
    "p6_json": "9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a",
    "p6_npz": "9f0372daba8c12e3aff86efde3201d3ae0612aba8cd297aa37559aa286967381",
    "p6_audit": "9e017b37709bf37fc8c8b87bbbb70461cba8afb47603a5b65dd2618c901ad2b4",
    "p6_manifest": "f8932a6a4304a14b93bfdf6783e900a47a9a1d45ba03f9915941bb00770bfb40",
    "p7_json": "a59e92640aae787003d5753d4614de1b6fe90c68fdba7f2daa81a125a3c0956c",
    "p7_npz": "fd40d339c0746b48c07408d5d017dff8819595350c365f7af5fc0d40673946ae",
    "p7_checkpoint": "113101637ef2b4fb75fe2ca0ba0dabe5a90c35d03a5c563b765438385c62db8c",
    "p7_audit": "35c95ee38dea7622f40b6c199f4164b6c27ec3f37ad5561a51de6cd3b0a79322",
    "p7_manifest": "a95fc4621a90cef13071df1ad1db363187deac0f7fc7c8c774c0fedd7c9c3a19",
}
STARTS = ("predictor_q", "free_target_encoder_q")
CONTROL_PREFIX = {
    "train_free_h1": ("train_free_h1", "training"),
    "train_encoder_handoff": ("train_encoder_handoff", "training"),
    "train_final_autolatent": ("train_final_autolatent", "training"),
    "selection_free_h1": ("selection_free_h1", "selection"),
    "selection_free_target_encoder": ("selection_free_target_encoder", "selection"),
    "selection_predictor_q_exact_affine": ("selection_predictor_q_exact_affine", "selection"),
    "selection_deployed_predictor": ("selection_deployed_predictor", "selection"),
    "selection_locked_p7_oracle": ("selection_locked_p7_oracle", "selection"),
    "selection_trust_two_start": ("trust_two_start", "selection"),
}


def require(value, message):
    if not value:
        raise SystemExit(f"AUDIT FAIL: {message}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, name = line.rstrip().split("  ", 1)
            require(len(digest) == 64 and name.startswith("./"), "manifest row")
            rows[name] = digest
    return rows


def close(left, right, message, rtol=2e-13, atol=2e-14):
    require(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False), message)


def normalized_affine(physical):
    physical = np.asarray(physical, np.float64)
    l11, l22 = np.exp(physical[:, 2]), np.exp(physical[:, 4])
    return np.column_stack(((physical[:, 0] - 0.5) / 0.75,
        (physical[:, 1] - 0.5) / 0.75,
        2.0 * (np.log(l11) - np.log(0.015)) / np.log(1.0 / 0.015) - 1.0,
        physical[:, 3] / (2.0 * np.sqrt(l11 * l22)),
        2.0 * (np.log(l22) - np.log(0.015)) / np.log(1.0 / 0.015) - 1.0))


def metric_from_arrays(arrays, prefix, metadata):
    meshes, pooled_trajectory, identities = {}, [], []
    total_squared, count = 0.0, 0
    for row in metadata:
        n = int(row["N"])
        numerator = arrays[f"{prefix}_N{n}_snapshot_numerator_sq"]
        denominator = arrays[f"{prefix}_N{n}_truth_norm_sq"]
        identity = arrays[f"{prefix}_N{n}_identity"]
        boundary_violations = int(arrays[f"{prefix}_N{n}_boundary_violation_count"])
        require(numerator.shape == denominator.shape == (row["case_count"], row["num_times"]),
                f"{prefix} N{n} norm shapes")
        trajectory = np.sqrt(np.sum(numerator, axis=1) /
                             np.maximum(np.sum(denominator, axis=1), 1e-300))
        snapshot = np.sqrt(numerator / np.maximum(denominator, 1e-300))
        pooled_trajectory.extend(trajectory.tolist()); identities.extend(identity.tolist())
        total_squared += float(np.sum(numerator / np.maximum(denominator, 1e-300)))
        count += int(numerator.size)
        meshes[str(n)] = {
            "trajectory_error_mean": float(np.mean(trajectory)),
            "trajectory_error_worst": float(np.max(trajectory)),
            "trajectory_error_all": trajectory.tolist(),
            "snapshot_error_mean": float(np.mean(snapshot)),
            "snapshot_error_worst": float(np.max(snapshot)),
            "k3_cox_identity_worst": float(np.max(identity)),
            "all_finite": True,
            "exact_binary_boundary": boundary_violations == 0,
            "boundary_violation_count": boundary_violations,
        }
    pooled = np.asarray(pooled_trajectory)
    return {"meshes": meshes, "pooled": {
        "trajectory_error_mean": float(np.mean(pooled)),
        "trajectory_error_worst": float(np.max(pooled)),
        "trajectory_error_all": pooled.tolist(),
        "mean_snapshot_relative_l2_squared": total_squared / count,
        "k3_cox_identity_worst": float(np.max(identities)),
        "all_finite": True,
        "exact_binary_boundary": bool(all(
            row["exact_binary_boundary"] for row in meshes.values())),
        "boundary_violation_count": int(sum(
            row["boundary_violation_count"] for row in meshes.values())),
    }}


def audit_traces(arrays, attempts, total):
    shape = (2, total)
    q = arrays["trust_q"]; objective = arrays["trust_objective"]
    delta = arrays["trust_delta"]; damping = arrays["trust_damping"]
    step = arrays["trust_step"]; trial = arrays["trust_trial_objective"]
    predicted = arrays["trust_predicted"]; actual = arrays["trust_actual"]
    rho = arrays["trust_rho"]; accepted = arrays["trust_accepted"]
    terminated = arrays["trust_terminated"]; attempted = arrays["trust_attempted"]
    finite = arrays["trust_finite"]; breakdown = arrays["trust_cg_breakdown"]
    cg_iters = arrays["trust_cg_iterations"]
    jvp = arrays["trust_jvp_count"]; vjp = arrays["trust_vjp_count"]
    bound = arrays["trust_bound_active_count"]
    require(q.shape == shape + (attempts + 1, 19), "q trace shape")
    require(objective.shape == shape + (attempts + 1,), "objective trace shape")
    require(step.shape == shape + (attempts, 19), "step trace shape")
    for value, name in ((delta,"delta"),(damping,"damping"),(trial,"trial"),
                        (predicted,"predicted"),(actual,"actual"),(rho,"rho"),
                        (accepted,"accepted"),(terminated,"terminated"),
                        (attempted,"attempted"),(finite,"finite"),(breakdown,"breakdown"),
                        (cg_iters,"cg iterations"),(jvp,"JVP count"),(vjp,"VJP count"),
                        (bound,"bound count")):
        require(value.shape == shape + (attempts,), f"{name} shape")
    require(np.all(np.abs(q) <= 1.0), "bounded q")
    require(np.all(delta >= 2.0**-20) and np.all(delta <= 1.0), "radius bounds")
    require(np.all(damping >= 1e-12) and np.all(damping <= 1e12), "damping bounds")
    require(np.all((cg_iters >= 0) & (cg_iters <= 19)), "CG iteration cap")
    require(np.array_equal(jvp, np.where(attempted, cg_iters + 1, 0))
            and np.array_equal(vjp, jvp), "matrix-free work counts")
    require(np.all((bound >= 0) & (bound <= 19)), "bound activity counts")
    require(np.all(finite | ~attempted), "attempt finiteness")
    require(not np.any(breakdown & attempted), "no CG breakdown")
    require(np.all(attempted[:,:,0]), "all starts attempted once")
    for index in range(attempts):
        active = attempted[:,:,index]
        expected_trial_q = np.clip(q[:,:,index] + step[:,:,index], -1.0, 1.0)
        expected_next_q = np.where(accepted[:,:,index,None], expected_trial_q, q[:,:,index])
        close(q[:,:,index+1], expected_next_q, f"q transition {index}")
        expected_next_objective = np.where(accepted[:,:,index], trial[:,:,index],
                                           objective[:,:,index])
        close(objective[:,:,index+1], expected_next_objective,
              f"objective transition {index}")
        close(actual[:,:,index], 0.5*(objective[:,:,index]-trial[:,:,index]),
              f"actual decrease {index}")
        positive = predicted[:,:,index] > 0.0
        close(rho[:,:,index][positive],
              (actual[:,:,index]/predicted[:,:,index])[positive], f"rho {index}")
        expected_accept = (active & finite[:,:,index] & ~breakdown[:,:,index]
                           & positive & (actual[:,:,index] > 0.0)
                           & (rho[:,:,index] >= 1e-4))
        require(np.array_equal(accepted[:,:,index], expected_accept), f"acceptance {index}")
        if index + 1 < attempts:
            shrink = active & (~accepted[:,:,index] | (rho[:,:,index] < 0.25))
            expand = (accepted[:,:,index] & (rho[:,:,index] > 0.75)
                      & (np.linalg.norm(step[:,:,index], axis=2) >= 0.9*delta[:,:,index]))
            improve = accepted[:,:,index] & (rho[:,:,index] > 0.75)
            expected_delta = np.where(shrink, np.maximum(delta[:,:,index]/4,2.0**-20),
                             np.where(expand, np.minimum(2*delta[:,:,index],1.0),delta[:,:,index]))
            expected_damping = np.where(shrink,np.minimum(10*damping[:,:,index],1e12),
                               np.where(improve,np.maximum(damping[:,:,index]/3,1e-12),damping[:,:,index]))
            close(delta[:,:,index+1], expected_delta, f"radius update {index}")
            close(damping[:,:,index+1], expected_damping, f"damping update {index}")
            require(np.array_equal(attempted[:,:,index+1], active & ~terminated[:,:,index]),
                    f"active update {index}")
    require(np.array_equal(arrays["trust_terminal_q"], q[:,:,-1]), "terminal q")
    chosen = np.argmin(objective[:,:,-1], axis=0).astype(np.int8)
    require(np.array_equal(arrays["trust_chosen_start"], chosen), "chosen starts")
    chosen_q = np.where(chosen[:,None] == 0, q[0,:,-1], q[1,:,-1])
    require(np.array_equal(arrays["trust_chosen_q"], chosen_q), "chosen q")
    exhausted = []
    for index in range(2):
        terminal_delta = arrays[f"trust_start{index}_terminal_delta"]
        terminal_damping = arrays[f"trust_start{index}_terminal_damping"]
        terminal_active = arrays[f"trust_start{index}_terminal_active"]
        never_accepted = ~np.any(accepted[index], axis=1)
        expected = (terminal_active & never_accepted & (terminal_delta <= 2.0**-20)
                    & (terminal_damping >= 1e12))
        exhausted.append(expected)
    exhausted = np.stack(exhausted)
    require(np.array_equal(arrays["trust_unhealthy_exhaustion"], exhausted),
            "unhealthy exhaustion classification")
    return {"attempted_total": int(np.sum(attempted)),
            "accepted_total": int(np.sum(accepted)),
            "terminated_total": int(np.sum(terminated)),
            "jvp_total": int(np.sum(jvp)), "vjp_total": int(np.sum(vjp)),
            "unhealthy_exhaustion_count": int(np.sum(exhausted)),
            "any_cg_breakdown": bool(np.any(breakdown & attempted)),
            "all_attempt_values_finite": bool(np.all(finite | ~attempted)),
            "max_attempts": attempts}


def checkpoint_summary(metadata, objective, attempted, accepted):
    final_attempt = objective.shape[2] - 1
    checkpoints = sorted(set((0, final_attempt) + tuple(
        value for value in (10, 20, 30, 40) if value <= final_attempt)))
    result = {}
    for start_index, name in enumerate(STARTS):
        rows = []
        for checkpoint in checkpoints:
            meshes, pooled = {}, []
            offset = 0
            for item in metadata:
                size = item["snapshot_count"]
                values = objective[start_index, offset:offset+size, checkpoint]
                offset += size; pooled.extend(values.tolist())
                meshes[str(item["N"])] = {
                    "mean_snapshot_relative_l2_squared": float(np.mean(values)),
                    "worst_snapshot_relative_l2": float(np.sqrt(np.max(values))),
                }
            pooled = np.asarray(pooled)
            rows.append({"attempt": checkpoint, "meshes": meshes,
                "pooled_mean_snapshot_relative_l2_squared": float(np.mean(pooled)),
                "pooled_worst_snapshot_relative_l2": float(np.sqrt(np.max(pooled))),
                "attempted_total_through_checkpoint": int(np.sum(
                    attempted[start_index,:,:checkpoint])),
                "accepted_total_through_checkpoint": int(np.sum(
                    accepted[start_index,:,:checkpoint]))})
        result[name] = rows
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True); parser.add_argument("--npz", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--work-checkpoint", required=True)
    parser.add_argument("--audit", required=True); parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--job-id", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest", required=True); parser.add_argument("--prereg", required=True)
    for phase in ("p4","p5","p6"):
        for kind in ("json","npz","audit","manifest"):
            parser.add_argument(f"--{phase}-{kind}", dest=f"{phase}_{kind}")
    for kind in ("json","npz","checkpoint","audit","manifest"):
        parser.add_argument(f"--p7-{kind}", dest=f"p7_{kind}")
    parser.add_argument("--target-dir"); parser.add_argument("--stdout")
    parser.add_argument("--stderr"); parser.add_argument("--sacct")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); smoke = args.smoke
    require(len(args.expected_commit) == 40, "commit syntax")
    require((args.job_id == "local") if smoke else args.job_id.isdigit(), "job identity")
    require(sha256(args.manifest) == args.expected_manifest, "manifest-file SHA")
    manifest = read_manifest(args.manifest); report = load(args.report)
    require(report["status"] == ("excluded_execution_smoke" if smoke else "complete"), "status")
    provenance = report["provenance"]
    require(provenance["commit"] == args.expected_commit
            and str(provenance["slurm_job_id"]) == args.job_id, "commit/job")
    require(provenance["jax_backend"] == "gpu" and provenance["x64"] is True
            and provenance["matmul_precision"] == "highest", "GPU/f64/highest")
    staged = {os.path.basename(name): digest for name,digest in manifest.items()
              if name.startswith("./code/b10_") and name.endswith(".py")}
    require(provenance["source_sha256"] == staged, "source manifest binding")
    require(manifest.get("./code/PHASE-8-PRE-REGISTRATION.md") == sha256(args.prereg),
            "prereg manifest binding")
    require(report["npz"]["sha256"] == sha256(args.npz), "NPZ SHA")
    require(report["progress"]["sha256"] == sha256(args.progress), "progress SHA")
    require(report["work_checkpoint"]["sha256"] == sha256(args.work_checkpoint),
            "work-checkpoint SHA")
    progress = load(args.progress)
    require(progress["status"] == "complete" and progress["stage"] == "complete"
            and progress["scientific_metrics_exposed"] is False, "final progress health")
    if not smoke:
        for phase in ("p4","p5","p6"):
            for kind in ("json","npz","audit","manifest"):
                key=f"{phase}_{kind}"; path=getattr(args,key)
                require(path and sha256(path)==EXPECTED[key],f"immutable {key}")
                require(report["bindings"][key]["sha256"]==EXPECTED[key],f"binding {key}")
                require(manifest.get(f"./code/deps/{phase}/{os.path.basename(path)}")
                        == EXPECTED[key],f"manifest {key}")
        for kind in ("json","npz","checkpoint","audit","manifest"):
            key=f"p7_{kind}"; path=getattr(args,key)
            require(path and sha256(path)==EXPECTED[key],f"immutable {key}")
            require(report["bindings"][key]["sha256"]==EXPECTED[key],f"binding {key}")
            require(manifest.get(f"./code/deps/p7/{os.path.basename(path)}")
                    == EXPECTED[key],f"manifest {key}")
        p7_report=load(args.p7_json); p7_audit=load(args.p7_audit)
        require(p7_report["decision"]["phase7_hard_stop"] is True
                and p7_audit["status"]=="pass" and p7_audit["negative_aware"] is True,
                "P7 negative chain")
        p5_report = load(args.p5_json)
        target_physical, target_features = [], []
        require(args.target_dir is not None, "target directory")
        target_rows = p5_report["train_targets"]["chunks"]
        require(len(target_rows) == 16, "exactly 16 ordered P5 target chunks")
        next_global = 0
        for row in target_rows:
            path = os.path.join(args.target_dir, row["basename"])
            require(sha256(path) == row["sha256"], f"target {row['basename']}")
            require(manifest.get(f"./code/deps/p5/targets/{row['basename']}")
                    == row["sha256"], f"manifest target {row['basename']}")
            with np.load(path, allow_pickle=False) as target:
                size = int(row["snapshot_count"])
                expected_draw = np.repeat(
                    np.asarray(row["indices"], np.int64)[:,None], 51, axis=1)
                expected_time = np.repeat(
                    np.arange(51,dtype=np.int64)[None,:], len(row["indices"]), axis=0)
                expected_global = np.arange(next_global,next_global+size,dtype=np.int64).reshape(
                    target["global_snapshot_index"].shape)
                require(row["global_snapshot_start"] == next_global
                        and row["global_snapshot_stop"] == next_global + size
                        and size == len(row["indices"])*51
                        and np.array_equal(target["source_draw_index"],expected_draw)
                        and np.array_equal(target["time_index"],expected_time)
                        and np.array_equal(target["global_snapshot_index"],expected_global)
                        and np.all(target["N"] == row["N"]),
                        f"target order/count/N {row['basename']}")
                require(np.all(target["healthy"]) and np.all(target["boundary"])
                        and np.all(target["rhs_finite"])
                        and np.all(target["prediction_finite"])
                        and np.all(target["coefficient_finite"])
                        and np.max(target["normal"]) <= 1e-8
                        and np.max(target["pou"]) == 0.0
                        and np.max(target["support"]) == 32,
                        f"target health {row['basename']}")
                target_physical.append(np.asarray(target["affine"],np.float64).reshape(-1,5))
                target_features.append(np.asarray(target["features"],np.float64).reshape(-1,7))
                next_global += size
        require(next_global == 35_904, "target global total")
        target_physical=np.concatenate(target_physical)
        target_features=np.concatenate(target_features)
        require(args.stdout and args.stderr and args.sacct, "scheduler evidence paths")
        with open(args.stdout,encoding="utf-8") as handle: stdout=handle.read()
        with open(args.stderr,encoding="utf-8") as handle: stderr=handle.read()
        with open(args.sacct,encoding="utf-8") as handle: sacct=handle.read()
        require("jax_backend=gpu" in stdout and "ALL-DONE" in stdout, "stdout health")
        require("COMPLETED" in sacct and "0:0" in sacct and args.job_id in sacct,
                "Slurm completion")
        require("Traceback" not in stdout+stderr and "captured large constant" not in (stdout+stderr).lower(),
                "log health")
    else:
        require(report["bindings"] is None, "smoke chain bypass")

    config=report["config"]
    require(config["objective"]=="exact discrete full-grid FOM relative-L2-squared"
            and config["starts"]==list(STARTS) and config["direct_q_optimization"] is True,
            "objective/starts")
    require(config["free_target_encoder_deployable"] is False
            and config["direct_predictor_deployable"] is True, "deployment classification")
    require(all(config[name] is False for name in
                ("model_validation_touched","confirmation_touched","training_touched",
                 "weak_eq_touched","scaling_touched","terminated_local_diagnostic_reused")),
            "forbidden access flags")
    attempts=1 if smoke else 40; total=4 if smoke else 5712
    require(config["trust"]=={"max_attempts":attempts,"cg_max":19,"cg_tolerance":1e-12,
            "delta0":0.25,"delta_min":2.0**-20,"delta_max":1.0,"lambda0":1e-6,
            "lambda_min":1e-12,"lambda_max":1e12,"accept_rho":1e-4}, "trust protocol")
    if smoke:
        require([(row["N"],row["source_start"],row["source_stop"],row["case_count"],
                  row["num_times"],row["snapshot_count"],row["global_start"],row["global_stop"])
                 for row in report["data"]["training"]] == [(16,0,2,2,2,4,0,4)]
                and [(row["N"],row["source_start"],row["source_stop"],row["case_count"],
                      row["num_times"],row["snapshot_count"],row["global_start"],row["global_stop"])
                     for row in report["data"]["selection"]] == [(16,2,4,2,2,4,0,4)],
                "smoke metadata")
    else:
        require([(row["N"],row["source_start"],row["source_stop"],row["case_count"],
                  row["num_times"],row["snapshot_count"],row["global_start"],row["global_stop"])
                 for row in report["data"]["training"]]
                == [(64,0,512,512,51,26112,0,26112),
                    (128,0,128,128,51,6528,26112,32640),
                    (256,0,64,64,51,3264,32640,35904)], "training split")
        require([(row["N"],row["source_start"],row["source_stop"],row["case_count"],
                  row["num_times"],row["snapshot_count"],row["global_start"],row["global_stop"])
                 for row in report["data"]["selection"]]
                == [(64,512,576,64,51,3264,0,3264),
                    (128,512,544,32,51,1632,3264,4896),
                    (256,512,528,16,51,816,4896,5712)], "selection split")
        for row in report["data"]["training"]+report["data"]["selection"]:
            health_row=row["reference_health"]
            require(health_row["reported_max_relative_residual"]<=1e-8
                    and health_row["independent_max_relative_residual"]<=1e-8,
                    f"FOM health N{row['N']}")
        require(report["data"]["target_chunks"] == [
            {key:row[key] for key in ("basename","sha256","snapshot_count","N")}
            for row in p5_report["train_targets"]["chunks"]], "target records")
    with np.load(args.npz,allow_pickle=False) as arrays:
        require(all(np.all(np.isfinite(arrays[name])) for name in arrays.files
                    if arrays[name].dtype.kind in "fc" and name != "trust_rho"), "finite NPZ")
        require(arrays["selection_affine"].shape==(total,5)
                and arrays["selection_features"].shape==(total,7), "selection shapes")
        require(np.array_equal(arrays["trust_starts_q"][0],arrays["selection_predictor_q"])
                and np.array_equal(arrays["trust_starts_q"][1],arrays["selection_encoder_q"]),
                "fixed start bindings")
        if not smoke:
            with np.load(args.p7_npz,allow_pickle=False) as p7_arrays:
                for name in ("training_affine","training_features","selection_affine",
                             "selection_features","selection_predictor_states"):
                    require(np.array_equal(arrays[name],p7_arrays[name]),f"P7 array {name}")
                require(np.array_equal(arrays["training_encoder_handoff_q_raw"],
                                       p7_arrays["encoder_handoff_q_raw"]),"P7 encoder handoff")
                require(np.array_equal(arrays["training_final_q_raw"],
                                       p7_arrays["training_q_raw"]),"P7 final q")
                require(np.array_equal(arrays["training_final_states"],
                                       p7_arrays["training_states"]),"P7 train states")
                require(np.array_equal(arrays["selection_locked_p7_oracle_states"],
                                       p7_arrays["selection_oracle_states"]),"P7 oracle states")
            with np.load(args.p5_npz,allow_pickle=False) as p5_arrays:
                require(np.array_equal(arrays["coefficient_mean"],p5_arrays["coefficient_mean"])
                        and np.array_equal(arrays["head_scales"],p5_arrays["head_scales"]),
                        "P5 normalization")
            local_affine=normalized_affine(target_physical)
            require(np.max(np.abs(arrays["training_affine"]-local_affine))<=2e-15
                    and np.array_equal(arrays["training_features"],target_features),
                    "independent physical affine/features")
        trace_health=audit_traces(arrays,attempts,total)
        with open(args.work_checkpoint,"rb") as handle:
            work_checkpoint=pickle.load(handle)
        require(work_checkpoint["status"]=="complete"
                and work_checkpoint["attempt_cap"]==attempts
                and np.array_equal(work_checkpoint["terminal_q"],arrays["trust_terminal_q"])
                and np.array_equal(work_checkpoint["terminal_objective"],
                                   arrays["trust_objective"][:,:,-1]),
                "atomic work checkpoint")
        require(checkpoint_summary(report["data"]["selection"],arrays["trust_objective"],
                                   arrays["trust_attempted"],arrays["trust_accepted"])
                == report["trust_checkpoints"], "trust checkpoint summaries")
        for name,(prefix,which) in CONTROL_PREFIX.items():
            observed=metric_from_arrays(arrays,prefix,report["data"][which])
            require(observed==report["controls"][name],f"full-grid metric {name}")
        for index,name in enumerate(STARTS):
            observed=metric_from_arrays(arrays,f"trust_{name}",report["data"]["selection"])
            require(observed==report["controls"]["selection_trust_starts"][name],
                    f"trust metric {name}")
            terminal_objective=arrays["trust_objective"][index,:,-1]
            ratios=np.concatenate([
                (arrays[f"trust_{name}_N{row['N']}_snapshot_numerator_sq"] /
                 np.maximum(arrays[f"trust_{name}_N{row['N']}_truth_norm_sq"],1e-300)).reshape(-1)
                for row in report["data"]["selection"]])
            close(terminal_objective,ratios,f"terminal exact objective {name}")
        for index,prefix in enumerate(("selection_predictor_q_exact_affine",
                                       "selection_free_target_encoder")):
            ratios=np.concatenate([
                (arrays[f"{prefix}_N{row['N']}_snapshot_numerator_sq"] /
                 np.maximum(arrays[f"{prefix}_N{row['N']}_truth_norm_sq"],1e-300)).reshape(-1)
                for row in report["data"]["selection"]])
            close(arrays["trust_objective"][index,:,0],ratios,f"initial exact objective {index}")
        binding=report["data"]["affine_feature_binding"]
        require(binding["absolute_tolerance"]==2e-15 and binding["relative_tolerance"]==0.0
                and max(binding[k] for k in ("train_regenerated_vs_physical_mapping",
                    "train_r3_vs_physical_mapping","selection_regenerated_vs_r3"))<=2e-15
                and binding["train_feature_exact_columns_bitwise"]
                and binding["train_viscosity_feature_within_ulp"]
                and binding["selection_feature_exact_columns_bitwise"]
                and binding["selection_viscosity_feature_within_ulp"], "affine/features")
    require(trace_health==report["trust_health"], "trace health summary")
    identity=all(row["pooled"]["k3_cox_identity_worst"]<=2e-14
                 for name,row in report["controls"].items() if name not in
                 ("selection_trust_starts",)) and all(
                 row["pooled"]["k3_cox_identity_worst"]<=2e-14
                 for row in report["controls"]["selection_trust_starts"].values())
    health=bool(identity and trace_health["all_attempt_values_finite"]
                and not trace_health["any_cg_breakdown"]
                and trace_health["unhealthy_exhaustion_count"]==0)
    require(report["gates"]=={"identity":identity,"health":health}, "health gates")
    expected_decision={"p8_d_valid":bool(not smoke and health),
        "t1_implementation_authorized":False,"t1_submission_authorized":False,
        "t2_authorized":False,
        "next_action":"root audit of P8-D" if not smoke else "excluded smoke only",
        "scientific_promotion_allowed":False}
    require(report["decision"]==expected_decision,"negative-aware decision")
    output={"status":"pass","negative_aware":True,"smoke":smoke,
            "source_json_sha256":sha256(args.report),"source_npz_sha256":sha256(args.npz),
            "progress_sha256":sha256(args.progress),
            "work_checkpoint_sha256":sha256(args.work_checkpoint),
            "manifest_sha256":sha256(args.manifest),"expected_commit":args.expected_commit,
            "expected_job":args.job_id,"prereg_sha256":sha256(args.prereg),
            "immutable_bindings_verified":not smoke,"affine_features_verified":True,
            "full_grid_metrics_recomputed":True,"trust_traces_recomputed":True,
            "slurm_backend_checks_verified":not smoke,"trust_health":trace_health,
            "gates":report["gates"],"decision":expected_decision}
    os.makedirs(os.path.dirname(os.path.abspath(args.audit)),exist_ok=True)
    with open(args.audit,"w",encoding="utf-8") as handle:
        json.dump(output,handle,indent=1,sort_keys=True,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"pass","smoke":smoke,"gates":report["gates"],
                      "decision":expected_decision},indent=2))


if __name__=="__main__":
    main()
