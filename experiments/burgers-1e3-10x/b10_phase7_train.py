#!/usr/bin/env python
"""One-cell Phase-7 G1 seed-11 trainer and exposed-selection evaluator."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

import b10_common as c
import b10_phase7 as p7
import b10_spline as spline
import b10_spline_train as legacy


TRAIN_MIX = ((64, 0, 512), (128, 0, 128), (256, 0, 64))
SELECTION_MIX = ((64, 512, 64), (128, 512, 32), (256, 512, 16))
DRAW_COUNT = 704
SEED = 11
FIELD_BATCH = 32
FIELD_POINTS = 512
WARMUP_STEPS = 10_000
JOINT_STEPS = 30_000
PREDICTOR_STEPS = 20_000
ORACLE_STEPS = 10_000
ORACLE_BATCH = 64
ORACLE_CHECKPOINTS = (4_000, 6_000, 8_000, 10_000)
HISTORY_EVERY = 1_000
IDENTITY_TOL = 2e-14
EVALUATION_BATCH = 8
_FULL_EVALUATORS = {}

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
}

warnings.filterwarnings(
    "error", message=r"(?i).*captured.*large.*constant.*", category=Warning
)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def path_record(path):
    return {"basename": os.path.basename(path), "sha256": c.sha256(path)}


def validate_exact_hash(path, key):
    digest = c.sha256(path)
    if digest != EXPECTED[key]:
        raise SystemExit(f"immutable Phase-7 input mismatch: {key}")
    return {"basename": os.path.basename(path), "sha256": digest}


def validate_chains(args):
    records = {}
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            key = f"{phase}_{kind}"
            records[key] = validate_exact_hash(getattr(args, key), key)
    p4, p4_audit = load_json(args.p4_json), load_json(args.p4_audit)
    p5, p5_audit = load_json(args.p5_json), load_json(args.p5_audit)
    p6, p6_audit = load_json(args.p6_json), load_json(args.p6_audit)
    p4_decision = {
        "selected_spatial_arm": "H1", "training_seed11_k24_licensed": True,
        "correction_classification": "conditional-zero-or-occasional-attempt",
        "phase4_hard_stop": False, "scientific_promotion_allowed": True,
    }
    p5_decision = {
        "target_integrity_pass": True,
        "arm_training_licenses": {"G1": False, "G2": False},
        "next_seed11_arm": None, "phase5_hard_stop": True,
        "scientific_promotion_allowed": True,
    }
    p6_decision = {
        "repair_licensed": True, "phase6_hard_stop": False,
        "training_authorized": False,
        "next_action": "separate training proposal/audit",
        "scientific_promotion_allowed": True,
    }
    checks = (
        p4.get("decision") == p4_audit.get("decision") == p4_decision,
        p5.get("decision") == p5_audit.get("decision") == p5_decision,
        p6.get("decision") == p6_audit.get("decision") == p6_decision,
        p4_audit.get("status") == p5_audit.get("status") == p6_audit.get("status") == "pass",
        p5.get("train_targets", {}).get("integrity_pass") is True,
        p5.get("train_targets", {}).get("snapshot_count") == 35_904,
        p6.get("config", {}).get("p5_targets_regenerated") is False,
        p6.get("config", {}).get("training_touched") is False,
        all(audit.get("negative_aware") is True
            for audit in (p4_audit, p5_audit, p6_audit)),
        p4_audit.get("source_json_sha256") == EXPECTED["p4_json"],
        p4_audit.get("source_npz_sha256") == EXPECTED["p4_npz"],
        p4_audit.get("manifest_sha256") == EXPECTED["p4_manifest"],
        p4_audit.get("expected_commit") == p4.get("provenance", {}).get("commit"),
        str(p4_audit.get("expected_job"))
        == str(p4.get("provenance", {}).get("slurm_job_id")),
        p5_audit.get("source_json_sha256") == EXPECTED["p5_json"],
        p5_audit.get("source_npz_sha256") == EXPECTED["p5_npz"],
        p6_audit.get("source_json_sha256") == EXPECTED["p6_json"],
        p6_audit.get("source_npz_sha256") == EXPECTED["p6_npz"],
        p6_audit.get("manifest_sha256") == EXPECTED["p6_manifest"],
        p6_audit.get("expected_commit") == p6.get("provenance", {}).get("commit"),
        str(p6_audit.get("expected_job"))
        == str(p6.get("provenance", {}).get("slurm_job_id")),
        all(report.get("provenance", {}).get("jax_backend") == "gpu"
            and report.get("provenance", {}).get("x64") is True
            and report.get("provenance", {}).get("matmul_precision") == "highest"
            for report in (p4, p5, p6)),
    )
    if not all(checks):
        raise SystemExit("immutable P4/P5/P6 decision/provenance chain mismatch")
    records["phase7_preregistration"] = path_record(args.prereg)
    records["decisions"] = {"P4": p4_decision, "P5": p5_decision, "P6": p6_decision}
    return p4, p5, p6, records


def smoke_datasets(kind):
    n, count, start, times = (16, 2, 0, 2) if kind == "train" else (16, 2, 2, 2)
    fields, parameters, health = legacy.synthetic_population(n, count, start, times)
    coords, mask = c.grid_coords(n), c.binary_boundary_mask(n)
    flat = fields.reshape(-1, n * n)
    affine = np.stack([
        spline.normalized_state_from_affine(
            spline.affine_state_from_field(field, coords)
        ) for field in flat
    ])
    tau = np.linspace(0.0, 1.0, times)
    base = np.column_stack((
        (parameters["cx"] - 0.5) / 0.35,
        (parameters["cy"] - 0.5) / 0.35,
        (parameters["width"] - 0.125) / 0.075,
        (parameters["amplitude"] - 1.25) / 0.75,
        np.log(parameters["nu"] / np.sqrt(0.001)) / (0.5 * np.log(10.0)),
    ))
    features = np.concatenate((
        np.repeat(base[:, None], times, axis=1),
        np.broadcast_to(tau[None, :, None], (count, times, 1)),
        np.full((count, times, 1), 1.0 / (n - 1), np.float64),
    ), axis=2).reshape(-1, 7)
    return [{
        "kind": kind, "N": n, "source_start": start, "case_count": count,
        "num_times": times, "fields": fields, "flat": flat,
        "coords": coords, "mask": mask, "affine": affine,
        "features": features, "parameters": parameters,
        "reference_health": health, "global_start": 0, "global_stop": flat.shape[0],
    }]


def load_target_coefficients(args, p5_report, smoke):
    if smoke:
        total = 4
        rng = np.random.default_rng(70011)
        normalized = rng.normal(0.0, 0.1, size=(total, 3328)).astype(np.float64)
        return {
            "normalized": normalized,
            "affine": None, "features": None,
            "mean": np.zeros(3328, np.float64), "scales": np.ones(2, np.float64),
            "chunks": [{"basename": "synthetic-only", "sha256": None,
                        "snapshot_count": total}],
        }
    with np.load(args.p5_npz, allow_pickle=False) as main:
        mean = np.asarray(main["coefficient_mean"], np.float64)
        scales = np.asarray(main["head_scales"], np.float64)
    coefficients, affine, features, chunks = [], [], [], []
    next_global = 0
    for row in p5_report["train_targets"]["chunks"]:
        path = os.path.join(args.target_dir, row["basename"])
        if c.sha256(path) != row["sha256"]:
            raise SystemExit(f"immutable target chunk mismatch: {row['basename']}")
        with np.load(path, allow_pickle=False) as data:
            values = np.asarray(data["coefficients"], np.float64)
            one_affine = np.asarray(data["affine"], np.float64)
            one_features = np.asarray(data["features"], np.float64)
            global_indices = np.asarray(data["global_snapshot_index"], np.int64).reshape(-1)
            healthy = np.asarray(data["healthy"], bool)
            if (
                not np.all(healthy) or not np.all(data["boundary"])
                or np.max(data["normal"]) > 1e-8 or np.max(data["pou"]) != 0.0
                or np.max(data["support"]) != 32
                or not np.array_equal(
                    global_indices,
                    np.arange(next_global, next_global + global_indices.size),
                )
            ):
                raise SystemExit(f"target health/order mismatch: {row['basename']}")
            next_global += global_indices.size
            coefficients.append(values.reshape(-1, 3328))
            affine.append(one_affine.reshape(-1, 5))
            features.append(one_features.reshape(-1, 7))
        chunks.append({
            "basename": row["basename"], "sha256": row["sha256"],
            "snapshot_count": row["snapshot_count"], "N": row["N"],
            "global_snapshot_start": row["global_snapshot_start"],
            "global_snapshot_stop": row["global_snapshot_stop"],
        })
    values = np.concatenate(coefficients)
    if values.shape != (35_904, 3328) or next_global != 35_904:
        raise SystemExit("immutable target total mismatch")
    normalized = np.concatenate((
        (values[:, :2304] - mean[:2304]) / scales[0],
        (values[:, 2304:] - mean[2304:]) / scales[1],
    ), axis=1)
    if not np.all(np.isfinite(normalized)):
        raise SystemExit("nonfinite normalized coefficient targets")
    return {
        "normalized": normalized,
        "affine": np.concatenate(affine), "features": np.concatenate(features),
        "mean": mean, "scales": scales, "chunks": chunks,
    }


def bind_training_metadata(datasets, targets, smoke):
    if smoke:
        return
    affine = legacy.concatenate(datasets, "affine")
    features = legacy.concatenate(datasets, "features")
    if not np.array_equal(affine, targets["affine"]):
        raise SystemExit("regenerated training affine/P5 target mismatch")
    exact = (0, 1, 2, 3, 5, 6)
    scale = np.maximum(np.abs(features[:, 4]), np.abs(targets["features"][:, 4]))
    tolerance = np.abs(np.spacing(scale))
    if (
        not np.array_equal(features[:, exact], targets["features"][:, exact])
        or not np.all(np.abs(features[:, 4] - targets["features"][:, 4]) <= tolerance)
    ):
        raise SystemExit("regenerated training features/P5 target mismatch")
    offset = 0
    for item in datasets:
        size = item["flat"].shape[0]
        item["affine"] = targets["affine"][offset:offset + size]
        item["features"] = targets["features"][offset:offset + size]
        offset += size


def history_item(step, loss, parts, schedule):
    return {
        "step": int(step), "loss": float(loss),
        "relative_field_loss": float(parts[0]),
        "coefficient_loss": float(parts[1]),
        "latent_regularization": float(parts[2]),
        "learning_rate": float(schedule(step - 1)),
    }


def train_warmup(datasets, targets, arrays, steps, batch, points):
    ids, seeds = legacy.make_schedule(len(targets["normalized"]), steps, batch, SEED + 300_000)
    arrays["warmup_snapshot_schedule"] = ids
    arrays["warmup_point_draw_seeds"] = seeds
    variables = {"generator": p7.init_generator(), "encoder": p7.init_encoder()}
    schedule = legacy.optimizer_schedule(1e-3, 1e-4, steps)
    optimizer = legacy.adamw(schedule)
    opt_state = optimizer.init(variables)

    def loss_fn(parameters, normalized, coords, masks, truth, affine, mean, scales):
        q_raw, q = p7.apply_encoder(parameters["encoder"], normalized)
        states = jnp.concatenate((affine, q), axis=1)
        prediction = p7.decode_sample_batch(
            parameters["generator"], states, coords, masks, mean, scales
        )
        field = jnp.mean(
            jnp.sum(jnp.square(prediction - truth), axis=1)
            / jnp.maximum(jnp.sum(jnp.square(truth), axis=1), 1e-300)
        )
        generated = p7.normalized_generated_coefficients(
            parameters["generator"], q, mean, scales
        )
        coefficient = 0.5 * (
            jnp.mean(jnp.square(generated[:, :2304] - normalized[:, :2304]))
            + jnp.mean(jnp.square(generated[:, 2304:] - normalized[:, 2304:]))
        )
        regularization = 1e-6 * jnp.mean(jnp.square(q_raw))
        return field + coefficient + regularization, (field, coefficient, regularization)

    @jax.jit
    def update(parameters, state, normalized, coords, masks, truth, affine, mean, scales):
        (loss, parts), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            parameters, normalized, coords, masks, truth, affine, mean, scales
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, parts

    history, started = [], time.perf_counter()
    for index in range(steps):
        coords, masks, truth, affine, _ = legacy.sample_field_batch(
            datasets, ids[index], seeds[index], points
        )
        variables, opt_state, loss, parts = update(
            variables, opt_state, jnp.asarray(targets["normalized"][ids[index]]),
            jnp.asarray(coords), jnp.asarray(masks), jnp.asarray(truth),
            jnp.asarray(affine), jnp.asarray(targets["mean"]),
            jnp.asarray(targets["scales"]),
        )
        if index == 0 or (index + 1) % (1 if steps < 1000 else HISTORY_EVERY) == 0:
            history.append(history_item(index + 1, loss, parts, schedule))
    return variables, opt_state, history, float(time.perf_counter() - started)


def encode_all(encoder, normalized, batch=128):
    evaluate = jax.jit(p7.apply_encoder)
    output = np.empty((normalized.shape[0], 19), np.float64)
    for start in range(0, normalized.shape[0], batch):
        output[start:start + batch] = np.asarray(
            evaluate(encoder, jnp.asarray(normalized[start:start + batch]))[0]
        )
    return output


def train_joint(datasets, targets, generator, q_raw, arrays, steps, batch, points):
    ids, seeds = legacy.make_schedule(len(q_raw), steps, batch, SEED)
    arrays["joint_snapshot_schedule"] = ids
    arrays["joint_point_draw_seeds"] = seeds
    variables = {"generator": generator, "q_raw": jnp.asarray(q_raw)}
    schedule = legacy.optimizer_schedule(1e-3, 1e-5, steps)
    optimizer = legacy.adamw(schedule)
    opt_state = optimizer.init(variables)

    def loss_fn(parameters, take, normalized, coords, masks, truth, affine, mean, scales):
        q_raw_batch = parameters["q_raw"][take]
        q = jnp.tanh(q_raw_batch)
        states = jnp.concatenate((affine, q), axis=1)
        prediction = p7.decode_sample_batch(
            parameters["generator"], states, coords, masks, mean, scales
        )
        field = jnp.mean(
            jnp.sum(jnp.square(prediction - truth), axis=1)
            / jnp.maximum(jnp.sum(jnp.square(truth), axis=1), 1e-300)
        )
        generated = p7.normalized_generated_coefficients(
            parameters["generator"], q, mean, scales
        )
        coefficient = 0.5 * (
            jnp.mean(jnp.square(generated[:, :2304] - normalized[:, :2304]))
            + jnp.mean(jnp.square(generated[:, 2304:] - normalized[:, 2304:]))
        )
        regularization = 1e-6 * jnp.mean(jnp.square(q_raw_batch))
        return field + 0.1 * coefficient + regularization, (
            field, coefficient, regularization
        )

    @jax.jit
    def update(parameters, state, take, normalized, coords, masks, truth, affine, mean, scales):
        (loss, parts), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            parameters, take, normalized, coords, masks, truth, affine, mean, scales
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, parts

    history, started = [], time.perf_counter()
    for index in range(steps):
        coords, masks, truth, affine, _ = legacy.sample_field_batch(
            datasets, ids[index], seeds[index], points
        )
        variables, opt_state, loss, parts = update(
            variables, opt_state, jnp.asarray(ids[index]),
            jnp.asarray(targets["normalized"][ids[index]]), jnp.asarray(coords),
            jnp.asarray(masks), jnp.asarray(truth), jnp.asarray(affine),
            jnp.asarray(targets["mean"]), jnp.asarray(targets["scales"]),
        )
        if index == 0 or (index + 1) % (1 if steps < 1000 else HISTORY_EVERY) == 0:
            history.append(history_item(index + 1, loss, parts, schedule))
    return variables, opt_state, history, float(time.perf_counter() - started)


def train_predictor(
    datasets, generator, target_states, mean, scales, feature_mean, feature_scale,
    arrays, steps, batch, points,
):
    ids, seeds = legacy.make_schedule(len(target_states), steps, batch, SEED + 100_000)
    arrays["predictor_snapshot_schedule"] = ids
    arrays["predictor_point_draw_seeds"] = seeds
    predictor = spline.init_mlp(
        jax.random.PRNGKey(SEED + 1), (7, 32, 32, 24)
    )
    p7.predictor_parameter_count(predictor)
    schedule = legacy.optimizer_schedule(5e-4, 5e-6, steps)
    optimizer = legacy.adamw(schedule)
    opt_state = optimizer.init(predictor)

    def loss_fn(parameters, generator_arg, coords, masks, truth, features, target, mean_arg, scales_arg):
        states = jnp.tanh(spline.apply_mlp(parameters, features))
        prediction = p7.decode_sample_batch(
            generator_arg, states, coords, masks, mean_arg, scales_arg
        )
        field = jnp.mean(
            jnp.sum(jnp.square(prediction - truth), axis=1)
            / jnp.maximum(jnp.sum(jnp.square(truth), axis=1), 1e-300)
        )
        state_loss = jnp.mean(jnp.square(states - target))
        return field + 0.1 * state_loss, (field, state_loss)

    @jax.jit
    def update(parameters, state, generator_arg, coords, masks, truth, features,
               target, mean_arg, scales_arg):
        (loss, parts), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            parameters, generator_arg, coords, masks, truth, features, target,
            mean_arg, scales_arg,
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, parts

    history, started = [], time.perf_counter()
    for index in range(steps):
        coords, masks, truth, _, features = legacy.sample_field_batch(
            datasets, ids[index], seeds[index], points
        )
        standardized = (features - feature_mean) / feature_scale
        predictor, opt_state, loss, parts = update(
            predictor, opt_state, generator, jnp.asarray(coords), jnp.asarray(masks),
            jnp.asarray(truth), jnp.asarray(standardized),
            jnp.asarray(target_states[ids[index]]), jnp.asarray(mean), jnp.asarray(scales),
        )
        if index == 0 or (index + 1) % (1 if steps < 1000 else HISTORY_EVERY) == 0:
            history.append({
                "step": index + 1, "loss": float(loss),
                "relative_field_loss": float(parts[0]),
                "normalized_state_loss": float(parts[1]),
                "learning_rate": float(schedule(index)),
            })
    return predictor, opt_state, history, float(time.perf_counter() - started)


def evaluate_states(datasets, generator, states, mean, scales):
    result, pooled, identity_all = {"meshes": {}}, [], []
    snapshot_squared_sum, snapshot_count = 0.0, 0
    all_finite, exact_boundary = True, True
    for item in datasets:
        start, stop = item["global_start"], item["global_stop"]
        one_states = np.asarray(states[start:stop], np.float64)
        point_count = item["coords"].shape[0]
        padded_points = ((point_count + 127) // 128) * 128
        coords = np.pad(item["coords"], ((0, padded_points - point_count), (0, 0)))
        mask = np.pad(item["mask"], (0, padded_points - point_count))
        evaluator_key = (EVALUATION_BATCH, padded_points)
        if evaluator_key not in _FULL_EVALUATORS:
            _FULL_EVALUATORS[evaluator_key] = (
                p7.full_decode_k3_factory(EVALUATION_BATCH, padded_points),
                jax.jit(p7.full_decode_cox),
            )
        k3, cox = _FULL_EVALUATORS[evaluator_key]
        prediction = np.empty_like(item["flat"])
        identity = np.empty(one_states.shape[0], np.float64)
        for batch_start in range(0, one_states.shape[0], EVALUATION_BATCH):
            take = min(EVALUATION_BATCH, one_states.shape[0] - batch_start)
            batch_states = one_states[batch_start:batch_start + take]
            if take < EVALUATION_BATCH:
                batch_states = np.concatenate((
                    batch_states,
                    np.repeat(batch_states[-1:], EVALUATION_BATCH - take, axis=0),
                ))
            arguments = (
                generator, jnp.asarray(batch_states), jnp.asarray(coords),
                jnp.asarray(mask), jnp.asarray(mean), jnp.asarray(scales),
            )
            actual = np.asarray(k3(*arguments))[:, :point_count]
            control = np.asarray(cox(*arguments))[:, :point_count]
            prediction[batch_start:batch_start + take] = actual[:take]
            numerator = np.linalg.norm(actual[:take] - control[:take], axis=1)
            denominator = np.maximum(np.linalg.norm(control[:take], axis=1), 1e-300)
            identity[batch_start:batch_start + take] = numerator / denominator
        truth = item["fields"]
        shaped = prediction.reshape(truth.shape)
        difference = shaped - truth
        trajectory = np.linalg.norm(
            difference.reshape(item["case_count"], -1), axis=1
        ) / np.maximum(
            np.linalg.norm(truth.reshape(item["case_count"], -1), axis=1), 1e-300
        )
        snapshot = np.linalg.norm(difference, axis=2) / np.maximum(
            np.linalg.norm(truth, axis=2), 1e-300
        )
        finite = bool(np.all(np.isfinite(prediction)))
        boundary = bool(np.all(prediction[:, item["mask"] == 0.0] == 0.0))
        pooled.extend(trajectory.tolist())
        identity_all.extend(identity.tolist())
        snapshot_squared_sum += float(np.sum(np.square(snapshot)))
        snapshot_count += int(snapshot.size)
        all_finite &= finite
        exact_boundary &= boundary
        result["meshes"][str(item["N"])] = {
            "trajectory_error_mean": float(np.mean(trajectory)),
            "trajectory_error_worst": float(np.max(trajectory)),
            "trajectory_error_all": trajectory.tolist(),
            "snapshot_error_mean": float(np.mean(snapshot)),
            "snapshot_error_worst": float(np.max(snapshot)),
            "k3_cox_identity_worst": float(np.max(identity)),
            "all_finite": finite, "exact_binary_boundary": boundary,
        }
    pooled = np.asarray(pooled, np.float64)
    result["pooled"] = {
        "trajectory_error_mean": float(np.mean(pooled)),
        "trajectory_error_worst": float(np.max(pooled)),
        "trajectory_error_all": pooled.tolist(),
        "mean_snapshot_relative_l2_squared": float(
            snapshot_squared_sum / max(snapshot_count, 1)
        ),
        "k3_cox_identity_worst": float(np.max(identity_all)),
        "all_finite": bool(all_finite),
        "exact_binary_boundary": bool(exact_boundary),
    }
    return result


def oracle_gate(metrics):
    return bool(all(
        row["trajectory_error_mean"] <= 2e-4
        and row["trajectory_error_worst"] <= 7e-4
        and row["k3_cox_identity_worst"] <= IDENTITY_TOL
        and row["all_finite"] and row["exact_binary_boundary"]
        for row in list(metrics["meshes"].values()) + [metrics["pooled"]]
    ))


def direct_gate(metrics, oracle):
    passed, degradation = True, {}
    for key in list(metrics["meshes"]) + ["pooled"]:
        row = metrics["pooled"] if key == "pooled" else metrics["meshes"][key]
        base = oracle["pooled"] if key == "pooled" else oracle["meshes"][key]
        ratio = row["trajectory_error_mean"] / max(base["trajectory_error_mean"], 1e-300)
        degradation[key] = float(ratio)
        passed &= bool(
            row["trajectory_error_mean"] <= 3e-4
            and row["trajectory_error_worst"] <= 1e-3
            and row["k3_cox_identity_worst"] <= IDENTITY_TOL
            and row["all_finite"] and row["exact_binary_boundary"] and ratio <= 1.5
        )
    return bool(passed), degradation


def optimize_oracle(datasets, generator, mean, scales, arrays, steps, batch, points):
    total = sum(item["flat"].shape[0] for item in datasets)
    affine = legacy.concatenate(datasets, "affine")
    ids, seeds = legacy.make_schedule(total, steps, batch, 20_260_825)
    arrays["oracle_snapshot_schedule"] = ids
    arrays["oracle_point_draw_seeds"] = seeds
    random_start = np.random.default_rng(20_260_825).normal(0.0, 0.25, (total, 19))
    starts = (np.zeros_like(random_start), random_start, -random_start)
    for start_index, initial in enumerate(starts):
        arrays[f"oracle_start{start_index}_initial_q_raw"] = initial
    schedule = legacy.optimizer_schedule(5e-2, 1e-3, steps)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-8)

    def loss_fn(q_raw, generator_arg, take, coords, masks, truth, affine_batch, mean_arg, scales_arg):
        states = jnp.concatenate((affine_batch, jnp.tanh(q_raw[take])), axis=1)
        prediction = p7.decode_sample_batch(
            generator_arg, states, coords, masks, mean_arg, scales_arg
        )
        relative = jnp.mean(
            jnp.sum(jnp.square(prediction - truth), axis=1)
            / jnp.maximum(jnp.sum(jnp.square(truth), axis=1), 1e-300)
        )
        return relative + 1e-8 * jnp.mean(jnp.square(q_raw[take]))

    @jax.jit
    def update(q_raw, state, generator_arg, take, coords, masks, truth, affine_batch,
               mean_arg, scales_arg):
        loss, gradient = jax.value_and_grad(loss_fn)(
            q_raw, generator_arg, take, coords, masks, truth, affine_batch,
            mean_arg, scales_arg,
        )
        updates, state = optimizer.update(gradient, state, q_raw)
        return optax.apply_updates(q_raw, updates), state, loss

    checkpoint_steps = (steps,) if steps < 4000 else ORACLE_CHECKPOINTS
    results, optimizer_states = [], []
    for start_index, initial in enumerate(starts):
        q_raw, state = jnp.asarray(initial), optimizer.init(jnp.asarray(initial))
        history, metrics, started = [], None, time.perf_counter()
        for index in range(steps):
            coords, masks, truth, affine_batch, _ = legacy.sample_field_batch(
                datasets, ids[index], seeds[index], points
            )
            q_raw, state, loss = update(
                q_raw, state, generator, jnp.asarray(ids[index]),
                jnp.asarray(coords), jnp.asarray(masks), jnp.asarray(truth),
                jnp.asarray(affine_batch), jnp.asarray(mean), jnp.asarray(scales),
            )
            if index == 0 or index + 1 in checkpoint_steps:
                record = {"step": index + 1, "sampled_loss": float(loss)}
                if index + 1 in checkpoint_steps:
                    states = np.concatenate((affine, np.tanh(np.asarray(q_raw))), axis=1)
                    metrics = evaluate_states(datasets, generator, states, mean, scales)
                    record["full_metrics"] = metrics
                    record["gate_pass"] = oracle_gate(metrics)
                history.append(record)
                if record.get("gate_pass"):
                    break
        if metrics is None:
            states = np.concatenate((affine, np.tanh(np.asarray(q_raw))), axis=1)
            metrics = evaluate_states(datasets, generator, states, mean, scales)
        q_value = np.asarray(q_raw)
        arrays[f"oracle_start{start_index}_q_raw"] = q_value
        results.append({
            "start_index": start_index, "history": history, "metrics": metrics,
            "gate_pass": oracle_gate(metrics),
            "elapsed_s": float(time.perf_counter() - started),
            "q_raw": q_value,
        })
        optimizer_states.append(state)
    chosen = min(range(3), key=lambda index: results[index]["metrics"]["pooled"][
        "mean_snapshot_relative_l2_squared"
    ])
    serializable = [{key: value for key, value in row.items() if key != "q_raw"}
                    for row in results]
    return results[chosen]["q_raw"], chosen, serializable, results[chosen]["metrics"], tuple(optimizer_states)


def main():
    parser = argparse.ArgumentParser()
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            parser.add_argument(f"--{phase}-{kind}", dest=f"{phase}_{kind}")
    parser.add_argument("--target-dir")
    parser.add_argument("--prereg")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    c.require_gpu_highest()
    provenance = c.provenance()
    started = time.perf_counter()
    legacy.SMOKE = args.smoke
    if args.smoke:
        bindings, p4_report, p5_report, p6_report = None, None, None, None
    else:
        required = [getattr(args, f"{phase}_{kind}")
                    for phase in ("p4", "p5", "p6")
                    for kind in ("json", "npz", "audit", "manifest")]
        required += [args.target_dir, args.prereg]
        if any(value is None for value in required):
            raise SystemExit("scientific Phase-7 requires complete immutable chains")
        p4_report, p5_report, p6_report, bindings = validate_chains(args)
    steps = 1 if args.smoke else None
    warmup_steps = steps or WARMUP_STEPS
    joint_steps = steps or JOINT_STEPS
    predictor_steps = steps or PREDICTOR_STEPS
    oracle_steps = steps or ORACLE_STEPS
    field_batch = 2 if args.smoke else FIELD_BATCH
    oracle_batch = 2 if args.smoke else ORACLE_BATCH
    field_points = 32 if args.smoke else FIELD_POINTS
    training = smoke_datasets("train") if args.smoke else legacy.load_mix(TRAIN_MIX, "train")
    selection = smoke_datasets("selection") if args.smoke else legacy.load_mix(SELECTION_MIX, "selection")
    targets = load_target_coefficients(args, p5_report, args.smoke)
    bind_training_metadata(training, targets, args.smoke)
    arrays = {
        "training_affine": legacy.concatenate(training, "affine"),
        "training_features": legacy.concatenate(training, "features"),
        "selection_affine": legacy.concatenate(selection, "affine"),
        "selection_features": legacy.concatenate(selection, "features"),
        "coefficient_mean": targets["mean"], "head_scales": targets["scales"],
    }
    feature_mean, feature_scale, feature_empirical = legacy.predictor_feature_statistics(
        arrays["training_features"]
    )
    arrays.update({
        "predictor_feature_mean": feature_mean,
        "predictor_feature_scale": feature_scale,
        "predictor_feature_empirical_scale": feature_empirical,
    })
    warm_variables, warm_state, warm_history, warm_elapsed = train_warmup(
        training, targets, arrays, warmup_steps, field_batch, field_points
    )
    encoder_q_raw = encode_all(warm_variables["encoder"], targets["normalized"])
    arrays["encoder_handoff_q_raw"] = encoder_q_raw
    copied_q_raw = encoder_q_raw.copy()
    handoff_identity = bool(np.array_equal(encoder_q_raw, copied_q_raw))
    joint_variables, joint_state, joint_history, joint_elapsed = train_joint(
        training, targets, warm_variables["generator"], copied_q_raw, arrays,
        joint_steps, field_batch, field_points,
    )
    generator = joint_variables["generator"]
    training_q_raw = np.asarray(joint_variables["q_raw"])
    training_states = np.concatenate((arrays["training_affine"], np.tanh(training_q_raw)), axis=1)
    predictor, predictor_state, predictor_history, predictor_elapsed = train_predictor(
        training, generator, training_states, targets["mean"], targets["scales"],
        feature_mean, feature_scale, arrays, predictor_steps, field_batch, field_points,
    )
    folded = legacy.fold_predictor_standardization(predictor, feature_mean, feature_scale)
    fold_train = legacy.predictor_fold_identity(
        predictor, folded, arrays["training_features"], feature_mean, feature_scale
    )
    fold_selection = legacy.predictor_fold_identity(
        predictor, folded, arrays["selection_features"], feature_mean, feature_scale
    )
    oracle_q_raw, chosen, starts, oracle_metrics, oracle_states = optimize_oracle(
        selection, generator, targets["mean"], targets["scales"], arrays,
        oracle_steps, oracle_batch, field_points,
    )
    selection_oracle_states = np.concatenate((
        arrays["selection_affine"], np.tanh(oracle_q_raw)
    ), axis=1)
    predictor_states = np.asarray(jnp.tanh(
        spline.apply_mlp(folded, jnp.asarray(arrays["selection_features"]))
    ))
    direct_metrics = evaluate_states(
        selection, generator, predictor_states, targets["mean"], targets["scales"]
    )
    oracle_pass = oracle_gate(oracle_metrics)
    direct_pass, degradation = direct_gate(direct_metrics, oracle_metrics)
    identity_pass = bool(
        oracle_metrics["pooled"]["k3_cox_identity_worst"] <= IDENTITY_TOL
        and direct_metrics["pooled"]["k3_cox_identity_worst"] <= IDENTITY_TOL
    )
    promote = bool(not args.smoke and oracle_pass and direct_pass and identity_pass)
    arrays.update({
        "training_q_raw": training_q_raw, "training_states": training_states,
        "selection_oracle_chosen_q_raw": oracle_q_raw,
        "selection_oracle_states": selection_oracle_states,
        "selection_predictor_states": predictor_states,
    })
    config = {
        "candidate": p7.G1, "H1": p7.H1, "training_seed": SEED,
        "training_mix": TRAIN_MIX, "selection_mix": SELECTION_MIX,
        "all_51_times": not args.smoke,
        "warmup_steps": warmup_steps, "joint_steps": joint_steps,
        "predictor_steps": predictor_steps, "oracle_steps": oracle_steps,
        "field_batch": field_batch, "oracle_batch": oracle_batch,
        "field_points": field_points, "identity_tolerance": IDENTITY_TOL,
        "oracle_starts": {
            "zero": "zeros", "random": "PCG64 Normal(0,0.25^2), seed 20260825",
            "reverse": "exact negative of random",
            "checkpoints": list((oracle_steps,) if oracle_steps < 4000
                                else ORACLE_CHECKPOINTS),
        },
        "model_validation_touched": False, "confirmation_touched": False,
        "weak_eq_touched": False, "scaling_touched": False,
        "p5_targets_regenerated": False, "g2_fallback": False,
        "retry_allowed": False, "smoke": args.smoke,
        "f64": True, "matmul_precision": "highest",
    }
    data = {
        "training": legacy.metadata(training), "selection": legacy.metadata(selection),
        "target_chunks": targets["chunks"],
        "target_snapshot_count": int(targets["normalized"].shape[0]),
    }
    checkpoint = {
        "status": "excluded_execution_smoke" if args.smoke else "complete",
        "provenance": provenance, "config": config, "bindings": bindings,
        "encoder": jax.tree_util.tree_map(np.asarray, warm_variables["encoder"]),
        "encoder_optimizer_state": jax.tree_util.tree_map(np.asarray, warm_state),
        "encoder_handoff_q_raw": encoder_q_raw,
        "warmup_generator": jax.tree_util.tree_map(
            np.asarray, warm_variables["generator"]
        ),
        "generator": jax.tree_util.tree_map(np.asarray, generator),
        "training_autolatent_raw": training_q_raw,
        "joint_optimizer_state": jax.tree_util.tree_map(np.asarray, joint_state),
        "direct_predictor_standardized": jax.tree_util.tree_map(np.asarray, predictor),
        "direct_predictor_folded_raw": jax.tree_util.tree_map(np.asarray, folded),
        "predictor_optimizer_state": jax.tree_util.tree_map(np.asarray, predictor_state),
        "oracle_optimizer_states": jax.tree_util.tree_map(np.asarray, oracle_states),
        "predictor_feature_mean": feature_mean,
        "predictor_feature_scale": feature_scale,
        "data_metadata": data,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
    with open(args.checkpoint, "wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    decision = {
        "g1_seed11_pass": promote,
        "seeds29_47_proposal_licensed": promote,
        "phase7_hard_stop": bool(not args.smoke and not promote),
        "training_authorized": False,
        "next_action": ("excluded smoke only" if args.smoke else
                        "separate seeds29/47 proposal/audit" if promote else "hard stop"),
        "scientific_promotion_allowed": not args.smoke,
    }
    report = {
        "status": "excluded_execution_smoke" if args.smoke else "complete",
        "provenance": provenance, "config": config, "bindings": bindings,
        "data": data,
        "training": {
            "warmup_history": warm_history, "warmup_elapsed_s": warm_elapsed,
            "joint_history": joint_history, "joint_elapsed_s": joint_elapsed,
            "predictor_history": predictor_history,
            "predictor_elapsed_s": predictor_elapsed,
            "encoder_parameters": p7.parameter_count(warm_variables["encoder"]),
            "generator_parameters": p7.parameter_count(generator),
            "predictor_parameters": p7.predictor_parameter_count(predictor),
            "autolatent_count": int(training_q_raw.shape[0]),
            "handoff_bitwise_identity": handoff_identity,
            "predictor_feature_standardization": {
                "mean": feature_mean.tolist(), "scale": feature_scale.tolist(),
                "empirical_scale": feature_empirical.tolist(),
                "folded_raw_identity_train_max_abs": fold_train,
                "folded_raw_identity_selection_max_abs": fold_selection,
                "identity_gate": 1e-12,
            },
        },
        "selection_oracle": {
            "starts": starts, "chosen_start": chosen, "metrics": oracle_metrics,
            "gate_pass": oracle_pass,
            "start_selection_metric": "pooled_mean_snapshot_relative_l2_squared",
            "chosen_start_selection_loss": oracle_metrics["pooled"][
                "mean_snapshot_relative_l2_squared"
            ],
        },
        "selection_direct": {
            "metrics": direct_metrics,
            "degradation_direct_over_oracle": degradation,
            "gate_pass": direct_pass,
        },
        "gates": {
            "representation_oracle_all_N_and_pooled": oracle_pass,
            "direct_all_N_and_pooled": direct_pass,
            "route_identity_all_selection": identity_pass,
            "promote_seed": promote,
        },
        "decision": decision,
        "checkpoint": {"basename": os.path.basename(args.checkpoint),
                       "sha256": c.sha256(args.checkpoint)},
        "npz": {"basename": os.path.basename(args.output_npz),
                "sha256": c.sha256(args.output_npz)},
        "elapsed_s": float(time.perf_counter() - started),
    }
    c.save_json(args.output_json, report)
    c.log({"status": report["status"], "gates": report["gates"],
           "decision": decision}, "\nALL-DONE")


if __name__ == "__main__":
    main()
