#!/usr/bin/env python
"""Independent, negative-aware audit for the one-cell Phase-7 G1 trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import numpy as np


EXPECTED = {
    "affine_regression": "218e648d2a3e1ad53b6d556435320a222daf3b449140865d07c09ff18ae0cad5",
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
}
G1 = {
    "arm": "G1", "name": "DualUpConv16", "channels": 16,
    "residual": False, "parameter_count": 30_594,
    "encoder_parameter_count": 29_811, "k": 24, "q": 19,
    "R": 48, "P": 32, "M": 96, "m": 384,
    "mac_per_state": 10_132_928,
    "mac_per_51_state_trajectory": 516_779_328,
}
H1 = {
    "arm": "H1", "R": 48, "P": 32, "k": 24, "q": 19,
    "M": 96, "m": 384,
    "hyperdecoder_parameters": 111_520, "predictor_parameters": 2_104,
}
TRAIN_MIX = [[64, 0, 512], [128, 0, 128], [256, 0, 64]]
SELECT_MIX = [[64, 512, 64], [128, 512, 32], [256, 512, 16]]
TRAIN_SPEC = ((64, 0, 512, 51), (128, 0, 128, 51), (256, 0, 64, 51))
SELECT_SPEC = ((64, 512, 64, 51), (128, 512, 32, 51), (256, 512, 16, 51))
IDENTITY_TOL = 2e-14


def require(value, message):
    if not value:
        raise SystemExit(f"PHASE7 TRAIN AUDIT FAILED: {message}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def json_normalized(value):
    """Match the driver's intentional tuple-to-list JSON normalization."""
    return json.loads(json.dumps(value))


def manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, name = line.rstrip().split(maxsplit=1)
            name = name.lstrip("*")
            require(len(digest) == 64 and name not in rows, "manifest syntax")
            rows[name] = digest
    return rows


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


def parameter_count(tree):
    if isinstance(tree, dict):
        return sum(parameter_count(value) for value in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(parameter_count(value) for value in tree)
    return int(np.asarray(tree).size)


def swish(value):
    positive = value >= 0
    sigmoid = np.empty_like(value)
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return value * sigmoid


def mlp(parameters, inputs):
    value = np.asarray(inputs, np.float64)
    for layer in parameters[:-1]:
        value = swish(value @ np.asarray(layer["W"]) + np.asarray(layer["b"]))
    return value @ np.asarray(parameters[-1]["W"]) + np.asarray(parameters[-1]["b"])


def predictor_shapes(parameters):
    require(len(parameters) == 3, "predictor depth")
    for index, (n_in, n_out) in enumerate(((7, 32), (32, 32), (32, 24))):
        require(np.asarray(parameters[index]["W"]).shape == (n_in, n_out),
                f"predictor W{index}")
        require(np.asarray(parameters[index]["b"]).shape == (n_out,),
                f"predictor b{index}")
    require(parameter_count(parameters) == 2_104 and finite(parameters),
            "predictor count/finite")


def generator_shapes(generator):
    require(np.asarray(generator["coarse_seed"]["W"]).shape == (19, 576),
            "G1 coarse seed W")
    require(np.asarray(generator["fine_seed"]["W"]).shape == (19, 256),
            "G1 fine seed W")
    for name, count in (("coarse_seed", 576), ("fine_seed", 256)):
        require(np.asarray(generator[name]["b"]).shape == (count,), f"G1 {name} b")
    for head in ("coarse_blocks", "fine_blocks"):
        require(len(generator[head]) == 3, f"G1 {head} depth")
        for block in generator[head]:
            require(len(block) == 1, f"G1 {head} nonresidual depth")
            require(np.asarray(block[0]["W"]).shape == (3, 3, 16, 16),
                    f"G1 {head} W")
            require(np.asarray(block[0]["b"]).shape == (16,), f"G1 {head} b")
    for name in ("coarse_out", "fine_out"):
        require(np.asarray(generator[name]["W"]).shape == (1, 1, 16, 1),
                f"G1 {name} W")
        require(np.asarray(generator[name]["b"]).shape == (1,), f"G1 {name} b")
    require(parameter_count(generator) == 30_594 and finite(generator),
            "G1 count/finite")


def encoder_shapes(encoder):
    for head in ("coarse", "fine"):
        require(np.asarray(encoder[head]["input"]["W"]).shape == (1, 1, 1, 16),
                f"encoder {head} input W")
        require(np.asarray(encoder[head]["input"]["b"]).shape == (16,),
                f"encoder {head} input b")
        require(len(encoder[head]["blocks"]) == 3, f"encoder {head} depth")
        for block in encoder[head]["blocks"]:
            require(np.asarray(block["W"]).shape == (3, 3, 16, 16),
                    f"encoder {head} block W")
            require(np.asarray(block["b"]).shape == (16,), f"encoder {head} block b")
    require(np.asarray(encoder["output"]["W"]).shape == (832, 19),
            "encoder output W")
    require(np.asarray(encoder["output"]["b"]).shape == (19,), "encoder output b")
    require(parameter_count(encoder) == 29_811 and finite(encoder),
            "encoder count/finite")


def schedule(total, steps, batch, seed):
    rng = np.random.default_rng(seed)
    needed, pieces = steps * batch, []
    while sum(value.size for value in pieces) < needed:
        pieces.append(rng.permutation(total).astype(np.int32))
    ids = np.concatenate(pieces)[:needed].reshape(steps, batch)
    seeds = rng.integers(0, np.iinfo(np.uint64).max,
                         size=(steps, batch), dtype=np.uint64)
    return ids, seeds


def normalized_affine_from_physical(physical):
    """Independent NumPy form of the locked decoder-state normalization."""
    value = np.asarray(physical, np.float64)
    l11, l22 = np.exp(value[:, 2]), np.exp(value[:, 4])
    log_ratio = np.log(1.0 / 0.015)
    result = np.column_stack((
        (value[:, 0] - 0.5) / 0.75,
        (value[:, 1] - 0.5) / 0.75,
        2.0 * (np.log(l11) - np.log(0.015)) / log_ratio - 1.0,
        value[:, 3] / (2.0 * np.sqrt(l11 * l22)),
        2.0 * (np.log(l22) - np.log(0.015)) / log_ratio - 1.0,
    ))
    require(np.all(np.isfinite(result)) and np.all(np.abs(result) <= 1.0),
            "normalized physical affine range")
    return result


def validate_metadata(rows, spec, smoke, name):
    require(len(rows) == len(spec), f"{name} metadata count")
    offset = 0
    for row, (n, start, count, times) in zip(rows, spec):
        size = count * times
        require(all((
            row["N"] == n, row["source_start"] == start,
            row["source_stop"] == start + count, row["case_count"] == count,
            row["num_times"] == times, row["snapshot_count"] == size,
            row["global_start"] == offset, row["global_stop"] == offset + size,
        )), f"{name} metadata N{n}")
        health = row["reference_health"]
        require(health["reported_max_relative_residual"] <= 1e-8
                and health["independent_max_relative_residual"] <= 1e-8,
                f"{name} health N{n}")
        if smoke:
            require(health["status"] == "synthetic excluded smoke", f"{name} smoke health")
        else:
            require(health["seed"] == 0 and health["draw_count"] == 704
                    and health["indices"] == list(range(start, start + count)),
                    f"{name} source indices N{n}")
        offset += size


def validate_history(rows, final, smoke, name):
    expected = [1] if smoke else [1] + list(range(1000, final + 1, 1000))
    require([row["step"] for row in rows] == expected, f"{name} history steps")
    require(finite(rows), f"{name} finite history")


def validate_metrics(metrics, spec, name):
    require(set(metrics["meshes"]) == {str(row[0]) for row in spec}, f"{name} meshes")
    pooled = []
    identity = []
    for n, _start, count, _times in spec:
        row = metrics["meshes"][str(n)]
        values = np.asarray(row["trajectory_error_all"], np.float64)
        require(values.shape == (count,) and np.all(np.isfinite(values)), f"{name} N{n} vector")
        require(np.isclose(np.mean(values), row["trajectory_error_mean"], rtol=1e-13)
                and np.isclose(np.max(values), row["trajectory_error_worst"], rtol=1e-13),
                f"{name} N{n} summaries")
        require(row["all_finite"] and row["exact_binary_boundary"], f"{name} N{n} output health")
        snapshot = np.asarray(row["snapshot_error_all"], np.float64)
        route_identity = np.asarray(row["k3_cox_identity_all"], np.float64)
        require(snapshot.shape == (count, _times) and np.all(np.isfinite(snapshot))
                and np.isclose(np.mean(snapshot), row["snapshot_error_mean"], rtol=1e-13)
                and np.isclose(np.max(snapshot), row["snapshot_error_worst"], rtol=1e-13)
                and route_identity.shape == (count * _times,)
                and np.all(np.isfinite(route_identity))
                and np.max(route_identity) == row["k3_cox_identity_worst"]
                and row["k3_cox_identity_worst"] <= IDENTITY_TOL,
                f"{name} N{n} snapshot/identity")
        pooled.extend(values.tolist())
        identity.append(row["k3_cox_identity_worst"])
    pooled = np.asarray(pooled, np.float64)
    row = metrics["pooled"]
    require(np.array_equal(pooled, np.asarray(row["trajectory_error_all"])), f"{name} pooled vector")
    require(np.isclose(np.mean(pooled), row["trajectory_error_mean"], rtol=1e-13)
            and np.isclose(np.max(pooled), row["trajectory_error_worst"], rtol=1e-13),
            f"{name} pooled summaries")
    require(row["all_finite"] and row["exact_binary_boundary"]
            and np.isfinite(row["mean_snapshot_relative_l2_squared"])
            and row["k3_cox_identity_worst"] == max(identity), f"{name} pooled health")


def oracle_gate(metrics):
    return bool(all(row["trajectory_error_mean"] <= 2e-4
                    and row["trajectory_error_worst"] <= 7e-4
                    and row["k3_cox_identity_worst"] <= IDENTITY_TOL
                    and row["all_finite"] and row["exact_binary_boundary"]
                    for row in list(metrics["meshes"].values()) + [metrics["pooled"]]))


def direct_gate(metrics, oracle):
    passed, ratios = True, {}
    for key in list(metrics["meshes"]) + ["pooled"]:
        row = metrics["pooled"] if key == "pooled" else metrics["meshes"][key]
        base = oracle["pooled"] if key == "pooled" else oracle["meshes"][key]
        ratios[key] = row["trajectory_error_mean"] / max(base["trajectory_error_mean"], 1e-300)
        passed &= bool(row["trajectory_error_mean"] <= 3e-4
                       and row["trajectory_error_worst"] <= 1e-3
                       and row["k3_cox_identity_worst"] <= IDENTITY_TOL
                       and row["all_finite"] and row["exact_binary_boundary"]
                       and ratios[key] <= 1.5)
    return bool(passed), ratios


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--affine-regression")
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            parser.add_argument(f"--{phase}-{kind}", dest=f"{phase}_{kind}")
    parser.add_argument("--target-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    smoke = args.smoke
    require(len(args.expected_commit) == 40, "expected commit syntax")
    require((args.job_id == "local") if smoke else args.job_id.isdigit(), "job identity")
    require(sha256(args.manifest) == args.expected_manifest, "manifest-file hash")
    root_manifest = manifest(args.manifest)
    report = load(args.report)
    require(report["status"] == ("excluded_execution_smoke" if smoke else "complete"),
            "status")
    provenance = report["provenance"]
    require(provenance["commit"] == args.expected_commit
            and str(provenance["slurm_job_id"]) == args.job_id, "commit/job")
    require(provenance["jax_backend"] == "gpu" and provenance["x64"] is True
            and provenance["matmul_precision"] == "highest", "gpu/f64/highest")
    staged_sources = {os.path.basename(path): digest for path, digest in root_manifest.items()
                      if path.startswith("./code/b10_") and path.endswith(".py")}
    require(provenance["source_sha256"] == staged_sources, "source-manifest binding")
    require(root_manifest.get("./code/PHASE-7-PRE-REGISTRATION.md") == sha256(args.prereg),
            "prereg manifest binding")

    bindings = None
    if not smoke:
        require(args.target_dir is not None, "target directory")
        require(args.affine_regression is not None
                and sha256(args.affine_regression) == EXPECTED["affine_regression"],
                "affine regression hash")
        regression = load(args.affine_regression)
        require(regression["status"] == "pass"
                and regression["snapshot_count"] == 35_904
                and regression["locked_vs_independent_max_abs"] == 0.0
                and regression["actual_phase7_loader_mapping_match"] is True
                and regression["regenerated_N64_draw0_normalized_affine_max_abs"]
                <= regression["regenerated_N64_draw0_tolerance_max_abs"],
                "affine regression result")
        require(root_manifest.get("./code/phase7_affine_schema_regression.json")
                == EXPECTED["affine_regression"], "manifest affine regression")
        bindings = report["bindings"]
        require(bindings["affine_regression"]["sha256"]
                == EXPECTED["affine_regression"], "report affine regression")
        for phase in ("p4", "p5", "p6"):
            for kind in ("json", "npz", "audit", "manifest"):
                key = f"{phase}_{kind}"
                path = getattr(args, key)
                require(path and sha256(path) == EXPECTED[key], f"immutable {key}")
                require(bindings[key]["sha256"] == EXPECTED[key], f"report {key} binding")
                staged = f"./code/deps/{phase}/{os.path.basename(path)}"
                require(root_manifest.get(staged) == EXPECTED[key], f"manifest {key}")
        p4, p5, p6 = (load(args.p4_json), load(args.p5_json), load(args.p6_json))
        for phase, source in (("p4", p4), ("p5", p5), ("p6", p6)):
            audit = load(getattr(args, f"{phase}_audit"))
            require(audit["status"] == "pass" and audit["negative_aware"] is True
                    and audit["source_json_sha256"] == EXPECTED[f"{phase}_json"]
                    and audit["source_npz_sha256"] == EXPECTED[f"{phase}_npz"],
                    f"{phase} audit chain")
            if phase in ("p4", "p6"):
                require(audit["manifest_sha256"] == EXPECTED[f"{phase}_manifest"]
                        and audit["expected_commit"] == source["provenance"]["commit"]
                        and str(audit["expected_job"]) == str(source["provenance"]["slurm_job_id"]),
                        f"{phase} manifest/job chain")
        target_rows = p5["train_targets"]["chunks"]
        require(report["data"]["target_chunks"] == [{
            "basename": row["basename"], "sha256": row["sha256"],
            "snapshot_count": row["snapshot_count"], "N": row["N"],
            "global_snapshot_start": row["global_snapshot_start"],
            "global_snapshot_stop": row["global_snapshot_stop"],
        } for row in target_rows], "target chunk records")
        target_affine, target_features, next_global = [], [], 0
        for row in target_rows:
            path = os.path.join(args.target_dir, row["basename"])
            require(sha256(path) == row["sha256"], f"target {row['basename']}")
            staged = f"./code/deps/p5/targets/{row['basename']}"
            require(root_manifest.get(staged) == row["sha256"], f"manifest target {row['basename']}")
            with np.load(path, allow_pickle=False) as target:
                size = row["snapshot_count"]
                expected_global = np.arange(next_global, next_global + size).reshape(
                    target["global_snapshot_index"].shape
                )
                expected_draw = np.repeat(
                    np.asarray(row["indices"], np.int64)[:, None], 51, axis=1
                )
                expected_time = np.repeat(np.arange(51, dtype=np.int64)[None, :],
                                          len(row["indices"]), axis=0)
                require(np.array_equal(target["global_snapshot_index"], expected_global)
                        and np.array_equal(target["source_draw_index"], expected_draw)
                        and np.array_equal(target["time_index"], expected_time)
                        and np.all(target["N"] == row["N"]),
                        f"target metadata {row['basename']}")
                require(np.all(target["healthy"]) and np.all(target["boundary"])
                        and np.all(target["rhs_finite"])
                        and np.all(target["prediction_finite"])
                        and np.all(target["coefficient_finite"])
                        and np.max(target["normal"]) <= 1e-8
                        and np.max(target["pou"]) == 0.0
                        and np.max(target["support"]) == 32,
                        f"target health {row['basename']}")
                target_affine.append(np.asarray(target["affine"], np.float64).reshape(-1, 5))
                target_features.append(np.asarray(target["features"], np.float64).reshape(-1, 7))
                next_global += size
        target_physical_affine = np.concatenate(target_affine)
        target_affine = normalized_affine_from_physical(target_physical_affine)
        target_features = np.concatenate(target_features)
        require(next_global == 35_904, "target global total")
        with np.load(args.p5_npz, allow_pickle=False) as p5_arrays:
            p5_mean = np.asarray(p5_arrays["coefficient_mean"], np.float64)
            p5_scales = np.asarray(p5_arrays["head_scales"], np.float64)
    else:
        require(report["bindings"] is None, "smoke chain bypass classification")

    config = report["config"]
    require(config["candidate"] == G1 and config["H1"] == H1
            and config["training_seed"] == 11, "candidate/H1/seed")
    require(config["model_validation_touched"] is False
            and config["confirmation_touched"] is False
            and config["weak_eq_touched"] is False
            and config["scaling_touched"] is False
            and config["p5_targets_regenerated"] is False
            and config["g2_fallback"] is False and config["retry_allowed"] is False,
            "forbidden work flags")
    require(config["smoke"] is smoke and config["f64"] is True
            and config["matmul_precision"] == "highest", "execution config")
    if smoke:
        train_spec = select_spec = ((16, 0, 2, 2),)
        select_spec = ((16, 2, 2, 2),)
        train_total = select_total = 4
        warm_steps = joint_steps = predictor_steps = oracle_steps = 1
        field_batch = oracle_batch = 2
        field_points = 32
    else:
        train_spec, select_spec = TRAIN_SPEC, SELECT_SPEC
        train_total, select_total = 35_904, 5_712
        warm_steps, joint_steps, predictor_steps, oracle_steps = 10_000, 30_000, 20_000, 10_000
        field_batch, oracle_batch, field_points = 32, 64, 512
        require(config["training_mix"] == TRAIN_MIX and config["selection_mix"] == SELECT_MIX
                and config["all_51_times"] is True, "split locks")
    require(config["warmup_steps"] == warm_steps and config["joint_steps"] == joint_steps
            and config["predictor_steps"] == predictor_steps
            and config["oracle_steps"] == oracle_steps
            and config["field_batch"] == field_batch
            and config["oracle_batch"] == oracle_batch
            and config["field_points"] == field_points, "training protocol")
    validate_metadata(report["data"]["training"], train_spec, smoke, "training")
    validate_metadata(report["data"]["selection"], select_spec, smoke, "selection")
    require(report["data"]["target_snapshot_count"] == train_total, "target total")
    require(report["data"]["target_affine_schema"] == (
        "immutable P5 chunk affine is physical moment transport; "
        "training_affine is its exact locked normalized_state_from_affine mapping"
    ), "target affine schema")

    require(report["npz"]["sha256"] == sha256(args.npz), "NPZ checksum")
    require(report["checkpoint"]["sha256"] == sha256(args.checkpoint), "checkpoint checksum")
    with open(args.checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    require(checkpoint["status"] == report["status"]
            and checkpoint["provenance"] == provenance
            and json_normalized(checkpoint["config"]) == config
            and json_normalized(checkpoint["bindings"]) == bindings and finite(checkpoint),
            "checkpoint identity/finite")
    encoder_shapes(checkpoint["encoder"])
    generator_shapes(checkpoint["warmup_generator"])
    generator_shapes(checkpoint["generator"])
    predictor_shapes(checkpoint["direct_predictor_standardized"])
    predictor_shapes(checkpoint["direct_predictor_folded_raw"])
    require(np.asarray(checkpoint["encoder_handoff_q_raw"]).shape == (train_total, 19),
            "encoder handoff shape")
    require(np.asarray(checkpoint["training_autolatent_raw"]).shape == (train_total, 19),
            "autolatent shape")
    require(len(checkpoint["oracle_optimizer_states"]) == 3, "three oracle states")

    training = report["training"]
    validate_history(training["warmup_history"], warm_steps, smoke, "warmup")
    validate_history(training["joint_history"], joint_steps, smoke, "joint")
    validate_history(training["predictor_history"], predictor_steps, smoke, "predictor")
    require(training["encoder_parameters"] == 29_811
            and training["generator_parameters"] == 30_594
            and training["predictor_parameters"] == 2_104
            and training["autolatent_count"] == train_total
            and training["handoff_bitwise_identity"] is True
            and finite(training), "training summary")

    with np.load(args.npz, allow_pickle=False) as arrays:
        require(all(finite(arrays[name]) for name in arrays.files), "NPZ finite")
        shape_specs = {
            "training_affine": (train_total, 5), "training_features": (train_total, 7),
            "selection_affine": (select_total, 5), "selection_features": (select_total, 7),
            "coefficient_mean": (3328,), "head_scales": (2,),
            "encoder_handoff_q_raw": (train_total, 19),
            "training_q_raw": (train_total, 19), "training_states": (train_total, 24),
            "selection_oracle_chosen_q_raw": (select_total, 19),
            "selection_oracle_states": (select_total, 24),
            "selection_predictor_states": (select_total, 24),
        }
        for name, shape in shape_specs.items():
            require(arrays[name].shape == shape, f"NPZ {name} shape")
        require(np.all(arrays["head_scales"] > 0.0), "positive head scales")
        if not smoke:
            require(np.array_equal(arrays["training_affine"], target_affine)
                    and np.array_equal(arrays["training_features"], target_features),
                    "P5 target metadata/output binding")
            require(np.array_equal(arrays["training_target_physical_affine"],
                                   target_physical_affine),
                    "immutable physical target affine binding")
            require(np.array_equal(checkpoint["training_target_physical_affine"],
                                   target_physical_affine),
                    "checkpoint physical target affine binding")
            require(np.array_equal(arrays["training_affine"],
                                   normalized_affine_from_physical(
                                       arrays["training_target_physical_affine"]
                                   )), "locked affine normalization mapping")
            require(np.array_equal(arrays["coefficient_mean"], p5_mean)
                    and np.array_equal(arrays["head_scales"], p5_scales),
                    "P5 normalization/output binding")
        require(np.array_equal(arrays["encoder_handoff_q_raw"],
                               checkpoint["encoder_handoff_q_raw"]), "handoff checkpoint")
        require(np.array_equal(arrays["training_q_raw"],
                               checkpoint["training_autolatent_raw"]), "autolatent checkpoint")
        require(np.array_equal(arrays["training_states"], np.concatenate((
            arrays["training_affine"], np.tanh(arrays["training_q_raw"])), axis=1)),
            "training state consistency")
        random_initial = np.random.default_rng(20_260_825).normal(
            0.0, 0.25, (select_total, 19)
        )
        initials = (np.zeros_like(random_initial), random_initial, -random_initial)
        for index, initial in enumerate(initials):
            require(np.array_equal(arrays[f"oracle_start{index}_initial_q_raw"], initial),
                    f"oracle initial {index}")
            require(arrays[f"oracle_start{index}_q_raw"].shape == (select_total, 19),
                    f"oracle result {index}")
        chosen = int(report["selection_oracle"]["chosen_start"])
        require(np.array_equal(arrays["selection_oracle_chosen_q_raw"],
                               arrays[f"oracle_start{chosen}_q_raw"]), "chosen q")
        require(np.array_equal(arrays["selection_oracle_states"], np.concatenate((
            arrays["selection_affine"], np.tanh(arrays[f"oracle_start{chosen}_q_raw"])
        ), axis=1)), "oracle states")
        mean = np.mean(arrays["training_features"], axis=0, dtype=np.float64)
        empirical = np.std(arrays["training_features"], axis=0, dtype=np.float64)
        scale = np.where(empirical < 1e-12, 1.0, empirical)
        require(np.allclose(mean, arrays["predictor_feature_mean"], rtol=1e-14, atol=1e-15)
                and np.allclose(scale, arrays["predictor_feature_scale"], rtol=1e-14, atol=1e-15)
                and np.array_equal(empirical, arrays["predictor_feature_empirical_scale"]),
                "train-only feature statistics")
        standard = checkpoint["direct_predictor_standardized"]
        folded = checkpoint["direct_predictor_folded_raw"]
        train_fold = float(np.max(np.abs(
            mlp(standard, (arrays["training_features"] - mean) / scale)
            - mlp(folded, arrays["training_features"])
        )))
        selection_fold = float(np.max(np.abs(
            mlp(standard, (arrays["selection_features"] - mean) / scale)
            - mlp(folded, arrays["selection_features"])
        )))
        require(max(train_fold, selection_fold) <= 1e-12, "folded predictor identity")
        require(np.allclose(arrays["selection_predictor_states"],
                            np.tanh(mlp(folded, arrays["selection_features"])),
                            rtol=2e-13, atol=2e-14), "predictor state consistency")
        schedule_specs = (
            ("warmup", train_total, warm_steps, field_batch, 300_011),
            ("joint", train_total, joint_steps, field_batch, 11),
            ("predictor", train_total, predictor_steps, field_batch, 100_011),
            ("oracle", select_total, oracle_steps, oracle_batch, 20_260_825),
        )
        for name, total, steps, batch, seed in schedule_specs:
            expected_ids, expected_seeds = schedule(total, steps, batch, seed)
            require(np.array_equal(arrays[f"{name}_snapshot_schedule"], expected_ids),
                    f"{name} schedule")
            require(np.array_equal(arrays[f"{name}_point_draw_seeds"], expected_seeds),
                    f"{name} point seeds")

    starts = report["selection_oracle"]["starts"]
    require([row["start_index"] for row in starts] == [0, 1, 2], "oracle starts")
    checkpoint_steps = [1] if smoke else [4000, 6000, 8000, 10000]
    for row in starts:
        steps = [item["step"] for item in row["history"]]
        if smoke:
            require(steps == [1] and "full_metrics" in row["history"][-1], "smoke oracle history")
        else:
            terminal = steps[-1]
            require(steps == [1] + [value for value in checkpoint_steps if value <= terminal]
                    and terminal in checkpoint_steps, "oracle checkpoint history")
            if terminal < 10_000:
                require(row["history"][-1]["gate_pass"] is True, "oracle early stop gate")
        validate_metrics(row["metrics"], select_spec, f"oracle start {row['start_index']}")
        require(row["gate_pass"] == oracle_gate(row["metrics"]), "oracle start gate")
    losses = [row["metrics"]["pooled"]["mean_snapshot_relative_l2_squared"] for row in starts]
    chosen = int(np.argmin(losses))
    require(report["selection_oracle"]["chosen_start"] == chosen
            and report["selection_oracle"]["metrics"] == starts[chosen]["metrics"]
            and report["selection_oracle"]["chosen_start_selection_loss"] == losses[chosen],
            "oracle selection")
    oracle = report["selection_oracle"]["metrics"]
    direct = report["selection_direct"]["metrics"]
    validate_metrics(direct, select_spec, "direct")
    oracle_pass = oracle_gate(oracle)
    direct_pass, ratios = direct_gate(direct, oracle)
    require(report["selection_oracle"]["gate_pass"] == oracle_pass
            and report["selection_direct"]["gate_pass"] == direct_pass, "reported gates")
    require(all(np.isclose(ratios[key], report["selection_direct"]
                           ["degradation_direct_over_oracle"][key], rtol=1e-14)
                for key in ratios), "direct/oracle ratios")
    identity_pass = bool(oracle["pooled"]["k3_cox_identity_worst"] <= IDENTITY_TOL
                         and direct["pooled"]["k3_cox_identity_worst"] <= IDENTITY_TOL)
    promote = bool(not smoke and oracle_pass and direct_pass and identity_pass)
    expected_gates = {
        "representation_oracle_all_N_and_pooled": oracle_pass,
        "direct_all_N_and_pooled": direct_pass,
        "route_identity_all_selection": identity_pass,
        "promote_seed": promote,
    }
    require(report["gates"] == expected_gates, "negative-aware gates")
    expected_decision = {
        "g1_seed11_pass": promote, "seeds29_47_proposal_licensed": promote,
        "phase7_hard_stop": bool(not smoke and not promote),
        "training_authorized": False,
        "next_action": ("excluded smoke only" if smoke else
                        "separate seeds29/47 proposal/audit" if promote else "hard stop"),
        "scientific_promotion_allowed": not smoke,
    }
    require(report["decision"] == expected_decision, "decision")
    output = {
        "status": "pass", "negative_aware": True, "smoke": smoke,
        "source_json_sha256": sha256(args.report), "source_npz_sha256": sha256(args.npz),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest), "expected_commit": args.expected_commit,
        "expected_job": args.job_id, "prereg_sha256": sha256(args.prereg),
        "immutable_bindings_verified": not smoke, "state_consistency_recomputed": True,
        "schedules_recomputed": True, "metadata_recomputed": True,
        "predictor_fold_train_max_abs": train_fold,
        "predictor_fold_selection_max_abs": selection_fold,
        "oracle_start_losses": losses, "chosen_start": chosen,
        "gates": expected_gates, "decision": expected_decision,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.audit)), exist_ok=True)
    with open(args.audit, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "pass", "smoke": smoke,
                      "gates": expected_gates, "decision": expected_decision}, indent=2))


if __name__ == "__main__":
    main()
