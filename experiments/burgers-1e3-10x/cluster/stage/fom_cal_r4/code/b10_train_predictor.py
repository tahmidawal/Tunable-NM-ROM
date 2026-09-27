"""Train a direct deployable predictor for a transported HG decoder.

The architecture is fixed by ``HG_DEGREE``.  It predicts the complete nonlinear
manifold state from recovered Gaussian-IC parameters, viscosity, time, and mesh
spacing.  No grid-sized learned parameter is present.

Usage: b10_train_predictor.py OUTPUT_CHECKPOINT OUTPUT_JSON
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import b10_common as c

OUTPUT_CHECKPOINT, OUTPUT_JSON = sys.argv[1:3]
DEGREE = int(os.environ.get("HG_DEGREE", "4"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "11"))
DATA_SEED = int(os.environ.get("DATA_SEED", "0"))
DRAW_COUNT = int(os.environ.get("DRAW_COUNT", "704"))
TRAIN_MIX = os.environ.get("TRAIN_MIX", "64:512,128:128,256:64")
SELECTION_MIX = os.environ.get("SELECTION_MIX", "64:64,128:32,256:16")
STATE_STEPS = int(os.environ.get("STATE_STEPS", "20000"))
FIELD_STEPS = int(os.environ.get("FIELD_STEPS", "20000"))
BATCH = int(os.environ.get("PRED_BATCH", "64"))
FIELD_POINTS = int(os.environ.get("FIELD_POINTS", "512"))
HIDDEN = int(os.environ.get("PRED_HIDDEN", "64"))
LAYERS = int(os.environ.get("PRED_LAYERS", "3"))
PEAK_LR = float(os.environ.get("PEAK_LR", "2e-3"))


def parse_mix(text):
    rows = []
    for item in text.split(","):
        n, count = item.split(":")
        rows.append((int(n), int(count)))
    if any(n < 8 or count < 1 for n, count in rows):
        raise ValueError(f"invalid resolution mix {text}")
    return tuple(rows)


def dense(key, d_in, d_out):
    return {
        "W": jax.random.normal(key, (d_in, d_out), jnp.float64)
             * np.sqrt(1.0 / d_in),
        "b": jnp.zeros((d_out,), jnp.float64),
    }


def init_predictor(key, d_in, d_out):
    keys = jax.random.split(key, LAYERS + 1)
    layers = [dense(keys[0], d_in, HIDDEN)]
    for index in range(1, LAYERS):
        layers.append(dense(keys[index], HIDDEN, HIDDEN))
    layers.append(dense(keys[-1], HIDDEN, d_out))
    # Begin near the standardized mean state.
    layers[-1]["W"] = layers[-1]["W"] * 0.01
    return layers


def apply_predictor(params, features):
    hidden = features
    for layer in params[:-1]:
        hidden = jax.nn.swish(hidden @ layer["W"] + layer["b"])
    return hidden @ params[-1]["W"] + params[-1]["b"]


def parameter_count(params):
    return sum(int(np.prod(value.shape)) for value in jax.tree_util.tree_leaves(params))


def dataset(n, count, start):
    fields, parameters, health = c.generate_population(
        n, DATA_SEED, DRAW_COUNT, np.arange(start, start + count),
        chunk=max(1, min(16, count))
    )
    flat = fields.reshape(-1, n * n)
    states, oracle, conditions = c.fit_hermite_states(
        flat, c.grid_coords(n), c.binary_boundary_mask(n), DEGREE,
        batch=max(1, min(32, flat.shape[0]))
    )
    features = c.trajectory_features(parameters, n).reshape(-1, 7)
    return {
        "n": n, "count": count, "fields": fields, "parameters": parameters,
        "features": features, "states": states,
        "oracle": oracle.reshape(fields.shape), "conditions": conditions,
        "health": health,
    }


def normalize_features(train_sets):
    features = np.concatenate([item["features"] for item in train_sets])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return mean, scale


def evaluate(params, feature_mean, feature_scale, state_mean, state_scale, item):
    normalized_features = (
        item["features"] - feature_mean[None]
    ) / feature_scale[None]
    predict = jax.jit(lambda values: apply_predictor(params, values))
    states_normalized = []
    for start in range(0, normalized_features.shape[0], 512):
        states_normalized.append(np.asarray(predict(jnp.asarray(
            normalized_features[start:start + 512], jnp.float64
        ))))
    states = np.concatenate(states_normalized) * state_scale + state_mean
    decoded = c.decode_hermite_states(
        states, c.grid_coords(item["n"]), c.binary_boundary_mask(item["n"]), DEGREE,
        batch=64
    ).reshape(item["fields"].shape)
    metrics = c.error_metrics(decoded, item["fields"])
    online = decoded.copy()
    online[:, 0] = item["fields"][:, 0]
    online_metrics = c.error_metrics(online, item["fields"])
    oracle_metrics = c.error_metrics(item["oracle"], item["fields"])
    state_relative = np.linalg.norm(states - item["states"], axis=1) / np.maximum(
        np.linalg.norm(item["states"], axis=1), 1e-300
    )
    return {
        "oracle": oracle_metrics,
        "decoder_including_t0": metrics,
        "deployable_trajectory_with_exact_known_t0": online_metrics,
        "state_relative_mean": float(np.mean(state_relative)),
        "state_relative_worst": float(np.max(state_relative)),
        "oracle_to_decoder_mean_ratio": float(
            metrics["trajectory_mean"] / max(oracle_metrics["trajectory_mean"], 1e-300)
        ),
        "basis_condition_median": float(np.median(item["conditions"])),
        "basis_condition_worst": float(np.max(item["conditions"])),
    }


def main():
    c.require_gpu_highest()
    if DEGREE not in (4, 5):
        raise SystemExit("only preregistered HG4/HG5 are allowed")
    train_mix = parse_mix(TRAIN_MIX)
    selection_mix = parse_mix(SELECTION_MIX)
    if any(count > 512 for _, count in train_mix):
        raise SystemExit("training indices must stay inside 0:512")
    if any(count > 64 for _, count in selection_mix):
        raise SystemExit("selection indices must stay inside 512:576")
    started = time.time()
    report = {
        "stage": f"HG{DEGREE} direct-predictor training",
        "status": "running",
        "config": {
            "degree": DEGREE, "latent_dimension": 4 + len(c.hermite_pairs(DEGREE)),
            "train_seed": TRAIN_SEED, "data_seed": DATA_SEED,
            "draw_count": DRAW_COUNT, "train_mix": list(train_mix),
            "selection_mix": list(selection_mix), "train_indices": [0, 511],
            "selection_indices": [512, 575], "model_validation_touched": False,
            "confirmation_touched": False, "predictor_hidden": HIDDEN,
            "predictor_layers": LAYERS, "state_steps": STATE_STEPS,
            "field_steps": FIELD_STEPS, "batch": BATCH,
            "field_points": FIELD_POINTS, "peak_lr": PEAK_LR,
            "boundary": "binary exact FOM grid boundary",
            "parameter_count_grid_independent": True, "f64": True,
        },
        "provenance": c.provenance(),
        "training": {}, "selection": {},
    }
    c.save_json(OUTPUT_JSON, report)
    c.log("build train datasets", train_mix)
    train_sets = [dataset(n, count, 0) for n, count in train_mix]
    c.log("build selection datasets", selection_mix)
    selection_sets = [dataset(n, count, 512) for n, count in selection_mix]
    worst_reference = max(
        item["health"]["independent_max_relative_residual"]
        for item in train_sets + selection_sets
    )
    if worst_reference > 1e-8:
        raise SystemExit(f"reference residual gate failed {worst_reference:.3e}")

    feature_mean, feature_scale = normalize_features(train_sets)
    all_states = np.concatenate([item["states"] for item in train_sets])
    state_mean = all_states.mean(axis=0)
    state_scale = all_states.std(axis=0)
    state_scale = np.where(state_scale > 1e-10, state_scale, 1.0)
    train_features = np.concatenate([
        (item["features"] - feature_mean) / feature_scale for item in train_sets
    ])
    train_targets = (all_states - state_mean) / state_scale

    params = init_predictor(
        jax.random.PRNGKey(TRAIN_SEED), train_features.shape[1], train_targets.shape[1]
    )
    total_steps = STATE_STEPS + FIELD_STEPS
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, max(1, total_steps // 20), total_steps, end_value=1e-6
    )
    optimizer = optax.adamw(schedule, weight_decay=1e-6)
    state = optimizer.init(params)

    @jax.jit
    def state_step(params_, state_, features_, targets_):
        def loss_fn(values):
            predicted = apply_predictor(values, features_)
            return jnp.mean(jnp.square(predicted - targets_))
        loss, gradients = jax.value_and_grad(loss_fn)(params_)
        updates, state_ = optimizer.update(gradients, state_, params_)
        return optax.apply_updates(params_, updates), state_, loss

    rng = np.random.default_rng(TRAIN_SEED + 20260819)
    history = []
    for step in range(STATE_STEPS):
        indices = rng.integers(0, train_features.shape[0], size=BATCH)
        params, state, loss = state_step(
            params, state, jnp.asarray(train_features[indices]),
            jnp.asarray(train_targets[indices])
        )
        if step % 2000 == 0 or step == STATE_STEPS - 1:
            value = float(loss)
            history.append({"step": step, "phase": "state", "loss": value})
            c.log("state", step, value)

    # Field-relative fine tuning. One compiled step per resolution avoids padded grids.
    field_kernels = []
    for item in train_sets:
        n = item["n"]
        coords = jnp.asarray(c.grid_coords(n), jnp.float64)
        mask = jnp.asarray(c.binary_boundary_mask(n), jnp.float64)

        @jax.jit
        def kernel(params_, state_, features_, targets_, state_targets_, point_indices,
                   coords_=coords, mask_=mask):
            coords_sub = coords_[point_indices]
            mask_sub = mask_[point_indices]

            def loss_fn(values):
                predicted_normalized = apply_predictor(values, features_)
                predicted_states = predicted_normalized * jnp.asarray(state_scale) + jnp.asarray(state_mean)
                predicted = jax.vmap(lambda latent: c.basis_from_warp(
                    coords_sub, mask_sub, latent[:2], jnp.exp(latent[2:4]), DEGREE
                ) @ latent[4:])(predicted_states)
                target = targets_[:, point_indices]
                relative = jnp.sum(jnp.square(predicted - target), axis=1) / jnp.maximum(
                    jnp.sum(jnp.square(target), axis=1), 1e-12
                )
                target_normalized = (
                    state_targets_ - jnp.asarray(state_mean)
                ) / jnp.asarray(state_scale)
                state_penalty = jnp.mean(jnp.square(predicted_normalized - target_normalized))
                return jnp.mean(relative) + 1e-3 * state_penalty

            loss, gradients = jax.value_and_grad(loss_fn)(params_)
            updates, state_ = optimizer.update(gradients, state_, params_)
            return optax.apply_updates(params_, updates), state_, loss

        field_kernels.append(kernel)

    for offset in range(FIELD_STEPS):
        dataset_index = offset % len(train_sets)
        item = train_sets[dataset_index]
        total_rows = item["features"].shape[0]
        rows = rng.integers(0, total_rows, size=BATCH)
        points = rng.choice(item["n"] ** 2, size=min(FIELD_POINTS, item["n"] ** 2), replace=False)
        features = (item["features"][rows] - feature_mean) / feature_scale
        truth = item["fields"].reshape(-1, item["n"] ** 2)[rows]
        params, state, loss = field_kernels[dataset_index](
            params, state, jnp.asarray(features), jnp.asarray(truth),
            jnp.asarray(item["states"][rows]), jnp.asarray(points)
        )
        step = STATE_STEPS + offset
        if offset % 2000 == 0 or offset == FIELD_STEPS - 1:
            value = float(loss)
            history.append({"step": step, "phase": "field", "loss": value})
            c.log("field", offset, "N", item["n"], value)

    for item in train_sets:
        report["training"][str(item["n"])] = evaluate(
            params, feature_mean, feature_scale, state_mean, state_scale, item
        )
    for item in selection_sets:
        report["selection"][str(item["n"])] = evaluate(
            params, feature_mean, feature_scale, state_mean, state_scale, item
        )
    report["reference_max_relative_residual"] = worst_reference
    report["training_history"] = history
    report["n_parameters"] = parameter_count(params)
    report["elapsed_seconds"] = float(time.time() - started)
    report["status"] = "complete"

    checkpoint = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "state_mean": state_mean, "state_scale": state_scale,
        "config": report["config"], "n_parameters": report["n_parameters"],
        "provenance": report["provenance"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CHECKPOINT)), exist_ok=True)
    with open(OUTPUT_CHECKPOINT, "wb") as handle:
        pickle.dump(checkpoint, handle)
    report["checkpoint"] = {
        "path": os.path.basename(OUTPUT_CHECKPOINT),
        "sha256": c.sha256(OUTPUT_CHECKPOINT),
    }
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps({
        "status": report["status"], "selection": report["selection"],
        "n_parameters": report["n_parameters"], "checkpoint": report["checkpoint"],
    }, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
