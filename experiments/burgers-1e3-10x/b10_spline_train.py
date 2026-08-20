"""Locked Phase-2 spline hyperdecoder/autolatent/direct-predictor trainer.

Scientific execution is impossible unless a pulled, checksummed S0 artifact and
the preregistered seed/arm artifact chain license the requested cell.  Model-
validation and confirmation draws are never generated here.

Usage: b10_spline_train.py S0_JSON ARM OUTPUT_JSON OUTPUT_NPZ CHECKPOINT.pkl \
       [PRIOR_TRAIN_JSON ...]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

import b10_common as c
import b10_spline as s

if len(sys.argv) < 6:
    raise SystemExit(__doc__)
S0_JSON, ARM, OUTPUT_JSON, OUTPUT_NPZ, CHECKPOINT = sys.argv[1:6]
PRIOR_TRAIN_JSONS = tuple(sys.argv[6:])
SMOKE = os.environ.get("SMOKE", "0") == "1"
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "11"))
SMOKE_UPDATES = int(os.environ.get("SMOKE_UPDATES", "2"))
SMOKE_SHAPE_FAITHFUL = os.environ.get("SMOKE_SHAPE_FAITHFUL", "0") == "1"
if SMOKE and not 1 <= SMOKE_UPDATES <= 100:
    raise SystemExit("excluded SMOKE_UPDATES must be in [1,100]")

# Treat this mandatory health condition as an exception if JAX emits it through
# Python warnings.  The cluster pull audit also rejects the log spelling.
warnings.filterwarnings(
    "error", message=r"(?i).*captured.*large.*constant.*", category=Warning
)

TRAIN_MIX = ((64, 0, 512), (128, 0, 128), (256, 0, 64))
SELECTION_MIX = ((64, 512, 64), (128, 512, 32), (256, 512, 16))
DRAW_COUNT = 704
FIELD_BATCH = 32
FIELD_POINTS = 512
MANIFOLD_STEPS = 30_000
PREDICTOR_STEPS = 20_000
ORACLE_STEPS = 10_000
ORACLE_BATCH = 64
ORACLE_CHECKPOINTS = (4_000, 6_000, 8_000, 10_000)
HISTORY_EVERY = 1_000
PARENT_GATE_NAME = "s0_gate"
STAGE_NAME = "Phase-2 spline hyperdecoder/autolatent/direct predictor training"
EXTRA_GATE_FN = None


def candidate_by_arm(arm):
    matches = [item for item in s.CANDIDATES if item["arm"] == arm]
    if len(matches) != 1:
        raise SystemExit(f"unknown Phase-2 arm {arm!r}")
    return dict(matches[0])


def _resolved_companion(json_path, entry):
    return os.path.join(
        os.path.dirname(os.path.abspath(json_path)), entry.get("path", "")
    )


def _validate_companion(json_path, artifact, key):
    entry = artifact.get(key) or {}
    resolved = _resolved_companion(json_path, entry)
    expected = entry.get("sha256")
    if not expected or not os.path.isfile(resolved) or c.sha256(resolved) != expected:
        raise SystemExit(f"invalid {key} companion/checksum for {json_path}")
    return {"path": resolved, "sha256": expected}


def _load_prior_artifact(path, s0_sha256):
    with open(path) as handle:
        artifact = json.load(handle)
    if artifact.get("status") != "complete":
        raise SystemExit(f"prior artifact is not complete: {path}")
    if (artifact.get("s0_gate") or {}).get("sha256") != s0_sha256:
        raise SystemExit(f"prior artifact has a different S0 parent: {path}")
    provenance = artifact.get("provenance") or {}
    if not (
        provenance.get("jax_backend") == "gpu"
        and provenance.get("x64") is True
        and provenance.get("matmul_precision") == "highest"
    ):
        raise SystemExit(f"prior artifact has invalid numerical provenance: {path}")
    config = artifact.get("config") or {}
    if config.get("model_validation_touched") is not False or config.get(
        "confirmation_touched"
    ) is not False:
        raise SystemExit(f"prior artifact touched a locked split: {path}")
    candidate = (artifact.get("config") or {}).get("candidate") or {}
    npz_record = _validate_companion(path, artifact, "npz")
    checkpoint_record = _validate_companion(path, artifact, "checkpoint")
    audit_path = os.path.join(os.path.dirname(os.path.abspath(path)), "AUDIT.json")
    if not os.path.isfile(audit_path):
        raise SystemExit(f"prior artifact lacks independent AUDIT.json: {path}")
    with open(audit_path) as handle:
        audit = json.load(handle)
    if not (
        audit.get("status") == "pass"
        and audit.get("source_json_sha256") == c.sha256(path)
        and audit.get("source_npz_sha256") == npz_record["sha256"]
        and audit.get("source_checkpoint_sha256") == checkpoint_record["sha256"]
        and audit.get("arm") == candidate.get("arm")
        and audit.get("seed") == config.get("training_seed")
        and audit.get("s0_json_sha256") == s0_sha256
    ):
        raise SystemExit(f"prior artifact independent audit mismatch: {path}")
    record = {
        "json_path": os.path.abspath(path), "json_sha256": c.sha256(path),
        "arm": candidate.get("arm"),
        "seed": (artifact.get("config") or {}).get("training_seed"),
        "oracle_gate_pass": (artifact.get("selection_oracle") or {}).get("gate_pass"),
        "direct_gate_pass": (artifact.get("selection_direct") or {}).get("gate_pass"),
        "promote_seed": (artifact.get("gates") or {}).get("promote_seed"),
        "npz": npz_record, "checkpoint": checkpoint_record,
        "audit": {"path": audit_path, "sha256": c.sha256(audit_path)},
    }
    if record["arm"] not in tuple(item["arm"] for item in s.CANDIDATES):
        raise SystemExit(f"invalid arm in prior artifact: {path}")
    if record["seed"] not in (11, 29, 47):
        raise SystemExit(f"invalid seed in prior artifact: {path}")
    return artifact, record


def load_s0_gate(path, arm):
    with open(path) as handle:
        artifact = json.load(handle)
    gate = {
        "path": os.path.abspath(path), "sha256": c.sha256(path),
        "status": artifact.get("status"), "decision": artifact.get("decision"),
        "source_commit": artifact.get("provenance", {}).get("commit"),
        "smoke_bypass": SMOKE,
    }
    if SMOKE:
        gate["artifact_chain"] = {
            "decision": "excluded smoke bypass; cannot license science",
            "prior_artifacts": [],
        }
        return artifact, gate
    if artifact.get("status") != "complete":
        raise SystemExit("S0 artifact is not complete")
    npz = artifact.get("npz", {})
    npz_path = os.path.join(os.path.dirname(os.path.abspath(path)), npz.get("path", ""))
    if not os.path.isfile(npz_path) or c.sha256(npz_path) != npz.get("sha256"):
        raise SystemExit("S0 NPZ checksum is missing or invalid")
    audit_path = os.path.join(os.path.dirname(os.path.abspath(path)), "AUDIT.json")
    if not os.path.isfile(audit_path):
        raise SystemExit("independent sibling S0 AUDIT.json is missing")
    with open(audit_path) as handle:
        audit = json.load(handle)
    if not (
        audit.get("status") == "pass"
        and audit.get("source_json_sha256") == gate["sha256"]
        and audit.get("source_npz_sha256") == npz["sha256"]
        and audit.get("expected_commit") == artifact.get("provenance", {}).get("commit")
        and str(audit.get("job_id"))
        == str(artifact.get("provenance", {}).get("slurm_job_id"))
        and audit.get("decision") == artifact.get("decision")
    ):
        raise SystemExit("independent S0 audit provenance/decision validation failed")
    promoted_arms = list((artifact.get("decision") or {}).get("promoted_arms") or [])
    promoted = (artifact.get("free_oracle") or {}).get(arm, {}).get("promote")
    if arm not in promoted_arms or promoted is not True:
        raise SystemExit(f"S0 does not promote arm {arm}: promoted={promoted_arms!r}")
    gate["npz_path"] = npz_path
    gate["npz_sha256"] = npz["sha256"]
    gate["audit_path"] = audit_path
    gate["audit_sha256"] = c.sha256(audit_path)
    gate["audit_status"] = audit["status"]
    gate["audit_decision"] = audit["decision"]
    priors = [_load_prior_artifact(item, gate["sha256"])[1]
              for item in PRIOR_TRAIN_JSONS]
    duplicates = [(item["arm"], item["seed"]) for item in priors]
    if len(duplicates) != len(set(duplicates)):
        raise SystemExit("duplicate prior arm/seed artifacts are ambiguous")
    if TRAIN_SEED == 11:
        smaller = promoted_arms[:promoted_arms.index(arm)]
        for smaller_arm in smaller:
            matches = [item for item in priors
                       if item["arm"] == smaller_arm and item["seed"] == 11]
            if len(matches) != 1 or matches[0]["oracle_gate_pass"] is not False:
                raise SystemExit(
                    f"arm {arm} requires one seed-11 learned-oracle failure for {smaller_arm}"
                )
        decision = (
            "smallest S0-promoted arm" if not smaller
            else "all smaller S0-promoted arms have seed-11 learned-oracle failures"
        )
    elif TRAIN_SEED in (29, 47):
        matches = [item for item in priors if item["arm"] == arm and item["seed"] == 11]
        if len(matches) != 1 or not (
            matches[0]["oracle_gate_pass"] is True
            and matches[0]["direct_gate_pass"] is True
            and matches[0]["promote_seed"] is True
        ):
            raise SystemExit(
                f"seed {TRAIN_SEED} requires one passing seed-11 finalist artifact for {arm}"
            )
        decision = "passing seed-11 finalist licenses full robustness retraining"
    else:
        raise SystemExit("scientific training seed must be 11, 29, or 47")
    gate["artifact_chain"] = {
        "decision": decision, "promoted_arms": promoted_arms,
        "required_seed": TRAIN_SEED, "required_arm": arm,
        "prior_artifacts": priors,
    }
    return artifact, gate


def synthetic_population(n, count, start, num_times):
    """Excluded smoke-only moving Gaussian fields."""
    indices = np.arange(start, start + count)
    cx = 0.35 + 0.05 * (indices % 3)
    cy = 0.55 - 0.04 * (indices % 2)
    width = 0.11 + 0.01 * (indices % 2)
    amplitude = 1.0 + 0.1 * (indices % 3)
    nu = np.full(count, 0.03, np.float64)
    fields = np.empty((count, num_times, n * n), np.float64)
    for case in range(count):
        for step in range(num_times):
            tau = step / max(num_times - 1, 1)
            fields[case, step] = c.bf.blob_ic(
                n, cx[case] + 0.02 * tau, cy[case] + 0.015 * tau,
                width[case] * (1.0 + 0.1 * tau), amplitude[case] * (1.0 - 0.1 * tau),
            )
    parameters = {
        "cx": cx, "cy": cy, "width": width, "amplitude": amplitude, "nu": nu,
    }
    health = {
        "status": "synthetic excluded smoke", "reported_max_relative_residual": 0.0,
        "independent_max_relative_residual": 0.0,
    }
    return fields, parameters, health


def load_mix(specification, kind):
    datasets = []
    global_start = 0
    for n, start, count in specification:
        if SMOKE:
            n = 24 if kind == "train" else 28
            count = 4 if kind == "train" else 2
            start = 0 if kind == "train" else 4
            num_times = 3
            fields, parameters, health = synthetic_population(n, count, start, num_times)
        else:
            num_times = c.NUM_STEPS + 1
            fields, parameters, health = c.generate_population(
                n, 0, DRAW_COUNT, np.arange(start, start + count),
                chunk=max(1, min(8, count)),
            )
            if (
                not np.isfinite(health["reported_max_relative_residual"])
                or not np.isfinite(health["independent_max_relative_residual"])
                or health["reported_max_relative_residual"] > 1e-8
                or health["independent_max_relative_residual"] > 1e-8
            ):
                raise SystemExit(f"{kind} N={n} legacy reference health failed")
        coords = c.grid_coords(n)
        mask = c.binary_boundary_mask(n)
        flat = fields.reshape(-1, n * n)
        affine = np.empty((flat.shape[0], 5), np.float64)
        for index, field in enumerate(flat):
            physical = s.affine_state_from_field(field, coords)
            affine[index] = s.normalized_state_from_affine(physical)
        if SMOKE:
            tau = np.linspace(0.0, 1.0, num_times)
            base = np.column_stack((
                (parameters["cx"] - 0.5) / 0.35,
                (parameters["cy"] - 0.5) / 0.35,
                (parameters["width"] - 0.125) / 0.075,
                (parameters["amplitude"] - 1.25) / 0.75,
                np.log(parameters["nu"] / np.sqrt(0.001)) / (0.5 * np.log(10.0)),
            ))
            features = np.concatenate((
                np.repeat(base[:, None], num_times, axis=1),
                np.broadcast_to(tau[None, :, None], (count, num_times, 1)),
                np.full((count, num_times, 1), 1.0 / (n - 1), np.float64),
            ), axis=2).reshape(-1, 7)
        else:
            features = c.trajectory_features(parameters, n).reshape(-1, 7)
        size = flat.shape[0]
        datasets.append({
            "kind": kind, "N": int(n), "source_start": int(start),
            "case_count": int(count), "num_times": int(num_times),
            "fields": fields, "flat": flat, "coords": coords, "mask": mask,
            "affine": affine, "features": features, "parameters": parameters,
            "reference_health": health, "global_start": global_start,
            "global_stop": global_start + size,
        })
        global_start += size
        if SMOKE:
            break
    return datasets


def metadata(datasets):
    return [{
        "N": item["N"], "source_start": item["source_start"],
        "source_stop": item["source_start"] + item["case_count"],
        "case_count": item["case_count"], "num_times": item["num_times"],
        "snapshot_count": item["flat"].shape[0],
        "global_start": item["global_start"], "global_stop": item["global_stop"],
        "reference_health": item["reference_health"],
    } for item in datasets]


def concatenate(datasets, key):
    return np.concatenate([item[key] for item in datasets], axis=0)


def make_schedule(total_snapshots, steps, batch_size, seed):
    rng = np.random.default_rng(int(seed))
    needed = int(steps) * int(batch_size)
    schedule = []
    while sum(item.size for item in schedule) < needed:
        schedule.append(rng.permutation(total_snapshots).astype(np.int32))
    snapshot_ids = np.concatenate(schedule)[:needed].reshape(steps, batch_size)
    point_seeds = rng.integers(
        0, np.iinfo(np.uint64).max, size=(steps, batch_size), dtype=np.uint64
    )
    return snapshot_ids, point_seeds


def locate_snapshot(datasets, global_index):
    for item in datasets:
        if item["global_start"] <= global_index < item["global_stop"]:
            return item, int(global_index - item["global_start"])
    raise IndexError(global_index)


def sample_field_batch(datasets, snapshot_ids, point_seeds, point_count):
    coords, masks, targets, affine, features = [], [], [], [], []
    for global_index, point_seed in zip(snapshot_ids, point_seeds):
        item, local = locate_snapshot(datasets, int(global_index))
        take = min(int(point_count), item["flat"].shape[1])
        rng = np.random.default_rng(int(point_seed))
        indices = rng.choice(item["flat"].shape[1], take, replace=False)
        if take < point_count:
            indices = np.resize(indices, point_count)
        coords.append(item["coords"][indices])
        masks.append(item["mask"][indices])
        targets.append(item["flat"][local, indices])
        affine.append(item["affine"][local])
        features.append(item["features"][local])
    return (
        np.stack(coords), np.stack(masks), np.stack(targets),
        np.stack(affine), np.stack(features),
    )


def make_decoder_batch(candidate):
    def decode(hyper, states, coords, masks):
        coefficients = jax.vmap(lambda state: s.apply_mlp(hyper, state[5:]))(states)
        return jax.vmap(
            lambda state, coeff, xy, mask: s.decode_candidate_jax(
                state, coeff, xy, mask, candidate
            )
        )(states, coefficients, coords, masks)
    return decode


def optimizer_schedule(initial, final, steps):
    return optax.cosine_decay_schedule(
        init_value=float(initial), decay_steps=max(int(steps), 1),
        alpha=float(final) / float(initial),
    )


def adamw(schedule):
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-6
        ),
    )


def predictor_feature_statistics(raw_features):
    raw = np.asarray(raw_features, np.float64)
    mean = np.mean(raw, axis=0, dtype=np.float64)
    empirical_scale = np.std(raw, axis=0, dtype=np.float64)
    scale = np.where(empirical_scale < 1e-12, 1.0, empirical_scale)
    if mean.shape != (7,) or scale.shape != (7,) or not (
        np.all(np.isfinite(mean)) and np.all(np.isfinite(scale))
        and np.all(scale > 0.0)
    ):
        raise ValueError("invalid train-only predictor feature statistics")
    return mean, scale, empirical_scale


def fold_predictor_standardization(predictor, mean, scale):
    """Fold x->(x-mean)/scale into the first locked affine layer."""
    layers = [dict(layer) for layer in predictor]
    first = dict(layers[0])
    weight = jnp.asarray(first["W"], jnp.float64)
    bias = jnp.asarray(first["b"], jnp.float64)
    mean_jax = jnp.asarray(mean, jnp.float64)
    scale_jax = jnp.asarray(scale, jnp.float64)
    first["W"] = weight / scale_jax[:, None]
    first["b"] = bias - (mean_jax / scale_jax) @ weight
    layers[0] = first
    return tuple(layers)


def predictor_fold_identity(predictor, folded, raw_features, mean, scale):
    raw = jnp.asarray(raw_features, jnp.float64)
    standardized = (raw - jnp.asarray(mean)) / jnp.asarray(scale)
    normalized_output = s.apply_mlp(predictor, standardized)
    folded_output = s.apply_mlp(folded, raw)
    error = float(jnp.max(jnp.abs(normalized_output - folded_output)))
    if not np.isfinite(error) or error > 1e-12:
        raise AssertionError(f"predictor standardization fold identity failed: {error:.3e}")
    return error


def train_manifold(candidate, datasets, arrays):
    k = candidate["k"]
    q_dimension = k - 5
    total = sum(item["flat"].shape[0] for item in datasets)
    steps = SMOKE_UPDATES if SMOKE else MANIFOLD_STEPS
    batch = (FIELD_BATCH if SMOKE_SHAPE_FAITHFUL else min(4, total)) if SMOKE else FIELD_BATCH
    points = (FIELD_POINTS if SMOKE_SHAPE_FAITHFUL else 32) if SMOKE else FIELD_POINTS
    schedule_ids, point_seeds = make_schedule(total, steps, batch, TRAIN_SEED)
    arrays["manifold_snapshot_schedule"] = schedule_ids
    arrays["manifold_point_draw_seeds"] = point_seeds
    hyper_key, q_key = jax.random.split(jax.random.PRNGKey(TRAIN_SEED))
    variables = {
        "hyper": s.init_mlp(
            hyper_key, (
                q_dimension, s.HYPER_WIDTH, s.HYPER_WIDTH,
                s.coefficient_count(candidate),
            )
        ),
        "q_raw": 0.01 * jax.random.normal(q_key, (total, q_dimension), dtype=jnp.float64),
    }
    schedule = optimizer_schedule(1e-3, 1e-5, steps)
    optimizer = adamw(schedule)
    opt_state = optimizer.init(variables)
    decode = make_decoder_batch(candidate)

    def loss_fn(parameters, ids, coords, masks, targets, affine):
        q = jnp.tanh(parameters["q_raw"][ids])
        states = jnp.concatenate((affine, q), axis=1)
        prediction = decode(parameters["hyper"], states, coords, masks)
        relative = jnp.sum(jnp.square(prediction - targets), axis=1) / jnp.maximum(
            jnp.sum(jnp.square(targets), axis=1), 1e-300
        )
        regularization = 1e-6 * jnp.mean(jnp.square(parameters["q_raw"][ids]))
        return jnp.mean(relative) + regularization, (jnp.mean(relative), regularization)

    @jax.jit
    def update(parameters, state, ids, coords, masks, targets, affine):
        (loss, auxiliary), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            parameters, ids, coords, masks, targets, affine
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, auxiliary

    history = []
    started = time.perf_counter()
    for step in range(steps):
        coords, masks, targets, affine, _ = sample_field_batch(
            datasets, schedule_ids[step], point_seeds[step], points
        )
        variables, opt_state, loss, auxiliary = update(
            variables, opt_state, jnp.asarray(schedule_ids[step]),
            jnp.asarray(coords), jnp.asarray(masks), jnp.asarray(targets),
            jnp.asarray(affine),
        )
        if step == 0 or (step + 1) % (1 if SMOKE else HISTORY_EVERY) == 0 or step + 1 == steps:
            history.append({
                "step": step + 1, "loss": float(loss),
                "relative_field_loss": float(auxiliary[0]),
                "autolatent_regularization": float(auxiliary[1]),
                "learning_rate": float(schedule(step)),
            })
    return variables, opt_state, history, float(time.perf_counter() - started)


def train_predictor(
    candidate, datasets, hyper, target_states, feature_mean, feature_scale, arrays
):
    k = candidate["k"]
    total = target_states.shape[0]
    steps = SMOKE_UPDATES if SMOKE else PREDICTOR_STEPS
    batch = (FIELD_BATCH if SMOKE_SHAPE_FAITHFUL else min(4, total)) if SMOKE else FIELD_BATCH
    points = (FIELD_POINTS if SMOKE_SHAPE_FAITHFUL else 32) if SMOKE else FIELD_POINTS
    schedule_ids, point_seeds = make_schedule(total, steps, batch, TRAIN_SEED + 100_000)
    arrays["predictor_snapshot_schedule"] = schedule_ids
    arrays["predictor_point_draw_seeds"] = point_seeds
    predictor = s.init_mlp(
        jax.random.PRNGKey(TRAIN_SEED + 1), (7, s.HYPER_WIDTH, s.HYPER_WIDTH, k)
    )
    schedule = optimizer_schedule(5e-4, 5e-6, steps)
    optimizer = adamw(schedule)
    opt_state = optimizer.init(predictor)
    decode = make_decoder_batch(candidate)

    def loss_fn(
        parameters, hyper_parameters, coords, masks, targets, features,
        target_state_batch,
    ):
        states = jnp.tanh(s.apply_mlp(parameters, features))
        prediction = decode(hyper_parameters, states, coords, masks)
        relative = jnp.sum(jnp.square(prediction - targets), axis=1) / jnp.maximum(
            jnp.sum(jnp.square(targets), axis=1), 1e-300
        )
        state_loss = jnp.mean(jnp.square(states - target_state_batch))
        return jnp.mean(relative) + 0.1 * state_loss, (jnp.mean(relative), state_loss)

    @jax.jit
    def update(
        parameters, state, hyper_parameters, coords, masks, targets, features,
        target_state_batch,
    ):
        (loss, auxiliary), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            parameters, hyper_parameters, coords, masks, targets, features,
            target_state_batch,
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, auxiliary

    history = []
    started = time.perf_counter()
    for step in range(steps):
        coords, masks, targets, _, features = sample_field_batch(
            datasets, schedule_ids[step], point_seeds[step], points
        )
        standardized_features = (
            features - feature_mean[None, :]
        ) / feature_scale[None, :]
        target_state_batch = np.asarray(target_states)[schedule_ids[step]]
        predictor, opt_state, loss, auxiliary = update(
            predictor, opt_state, hyper,
            jnp.asarray(coords), jnp.asarray(masks), jnp.asarray(targets),
            jnp.asarray(standardized_features), jnp.asarray(target_state_batch),
        )
        if step == 0 or (step + 1) % (1 if SMOKE else HISTORY_EVERY) == 0 or step + 1 == steps:
            history.append({
                "step": step + 1, "loss": float(loss),
                "relative_field_loss": float(auxiliary[0]),
                "normalized_state_loss": float(auxiliary[1]),
                "learning_rate": float(schedule(step)),
            })
    return predictor, opt_state, history, float(time.perf_counter() - started)


def evaluate_states(candidate, datasets, hyper, states):
    result = {"meshes": {}}
    pooled_trajectory = []
    total_error_sq = 0.0
    total_truth_sq = 0.0
    snapshot_relative_l2_squared_sum = 0.0
    snapshot_count = 0
    all_finite = True
    exact_boundary = True
    for item in datasets:
        start, stop = item["global_start"], item["global_stop"]
        one_states = np.asarray(states[start:stop], np.float64)
        coords = jnp.asarray(item["coords"], jnp.float64)
        mask = jnp.asarray(item["mask"], jnp.float64)

        @jax.jit
        def decode_batch(hyper_parameters, batch_states, coordinate_arg, mask_arg):
            coefficients = jax.vmap(
                lambda state: s.apply_mlp(hyper_parameters, state[5:])
            )(batch_states)
            return jax.vmap(
                lambda state, coeff: s.decode_candidate_jax(
                    state, coeff, coordinate_arg, mask_arg, candidate
                )
            )(batch_states, coefficients)

        prediction = np.empty_like(item["flat"])
        for batch_start in range(0, one_states.shape[0], 8):
            prediction[batch_start:batch_start + 8] = np.asarray(
                decode_batch(
                    hyper, jnp.asarray(one_states[batch_start:batch_start + 8]),
                    coords, mask,
                )
            )
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
        boundary = item["mask"] == 0.0
        one_finite = bool(np.all(np.isfinite(prediction)))
        one_boundary = bool(np.all(prediction[:, boundary] == 0.0))
        all_finite = all_finite and one_finite
        exact_boundary = exact_boundary and one_boundary
        pooled_trajectory.extend(trajectory.tolist())
        total_error_sq += float(np.sum(np.square(difference)))
        total_truth_sq += float(np.sum(np.square(truth)))
        snapshot_relative_l2_squared_sum += float(np.sum(np.square(snapshot)))
        snapshot_count += int(snapshot.size)
        result["meshes"][str(item["N"])] = {
            "trajectory_error_mean": float(np.mean(trajectory)),
            "trajectory_error_worst": float(np.max(trajectory)),
            "trajectory_error_all": trajectory.tolist(),
            "snapshot_error_mean": float(np.mean(snapshot)),
            "snapshot_error_worst": float(np.max(snapshot)),
            "all_finite": one_finite, "exact_binary_boundary": one_boundary,
        }
    pooled = np.asarray(pooled_trajectory, np.float64)
    result["pooled"] = {
        "trajectory_error_mean": float(np.mean(pooled)),
        "trajectory_error_worst": float(np.max(pooled)),
        "trajectory_error_all": pooled.tolist(),
        "relative_field_mse": float(total_error_sq / max(total_truth_sq, 1e-300)),
        "mean_snapshot_relative_l2_squared": float(
            snapshot_relative_l2_squared_sum / max(snapshot_count, 1)
        ),
        "all_finite": all_finite, "exact_binary_boundary": exact_boundary,
    }
    return result


def oracle_gate(metrics):
    rows = list(metrics["meshes"].values()) + [metrics["pooled"]]
    return bool(all(
        row["trajectory_error_mean"] <= 2e-4
        and row["trajectory_error_worst"] <= 7e-4
        and row["all_finite"] and row["exact_binary_boundary"]
        for row in rows
    ))


def direct_gate(metrics, oracle):
    passed = True
    degradation = {}
    for key in list(metrics["meshes"]) + ["pooled"]:
        direct_row = metrics["pooled"] if key == "pooled" else metrics["meshes"][key]
        oracle_row = oracle["pooled"] if key == "pooled" else oracle["meshes"][key]
        ratio = direct_row["trajectory_error_mean"] / max(
            oracle_row["trajectory_error_mean"], 1e-300
        )
        degradation[key] = float(ratio)
        passed = passed and bool(
            direct_row["trajectory_error_mean"] <= 3e-4
            and direct_row["trajectory_error_worst"] <= 1e-3
            and direct_row["all_finite"] and direct_row["exact_binary_boundary"]
            and ratio <= 1.5
        )
    return passed, degradation


def optimize_selection_oracle(candidate, datasets, hyper, arrays):
    total = sum(item["flat"].shape[0] for item in datasets)
    affine = concatenate(datasets, "affine")
    q_dimension = candidate["k"] - 5
    steps = SMOKE_UPDATES if SMOKE else ORACLE_STEPS
    batch = (ORACLE_BATCH if SMOKE_SHAPE_FAITHFUL else min(4, total)) if SMOKE else ORACLE_BATCH
    points = (FIELD_POINTS if SMOKE_SHAPE_FAITHFUL else 32) if SMOKE else FIELD_POINTS
    checkpoints = (steps,) if SMOKE else ORACLE_CHECKPOINTS
    schedule_ids, point_seeds = make_schedule(total, steps, batch, 20260825)
    arrays["selection_oracle_snapshot_schedule"] = schedule_ids
    arrays["selection_oracle_point_draw_seeds"] = point_seeds
    random_start = np.random.default_rng(20260825).normal(
        0.0, 0.25, size=(total, q_dimension)
    )
    starts = (np.zeros_like(random_start), random_start, -random_start)
    decode = make_decoder_batch(candidate)
    results = []
    schedule = optimizer_schedule(5e-2, 1e-3, steps)
    for start_index, initial in enumerate(starts):
        q_raw = jnp.asarray(initial, jnp.float64)
        optimizer = optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = optimizer.init(q_raw)

        def loss_fn(
            values, hyper_parameters, ids, coords, masks, targets, affine_batch
        ):
            states = jnp.concatenate((affine_batch, jnp.tanh(values[ids])), axis=1)
            prediction = decode(hyper_parameters, states, coords, masks)
            relative = jnp.sum(jnp.square(prediction - targets), axis=1) / jnp.maximum(
                jnp.sum(jnp.square(targets), axis=1), 1e-300
            )
            return jnp.mean(relative) + 1e-8 * jnp.mean(jnp.square(values[ids]))

        @jax.jit
        def update(
            values, state, hyper_parameters, ids, coords, masks, targets, affine_batch
        ):
            loss, gradient = jax.value_and_grad(loss_fn)(
                values, hyper_parameters, ids, coords, masks, targets, affine_batch
            )
            updates, state = optimizer.update(gradient, state, values)
            return optax.apply_updates(values, updates), state, loss

        history = []
        final_metrics = None
        started = time.perf_counter()
        for step in range(steps):
            coords, masks, targets, affine_batch, _ = sample_field_batch(
                datasets, schedule_ids[step], point_seeds[step], points
            )
            q_raw, opt_state, loss = update(
                q_raw, opt_state, hyper, jnp.asarray(schedule_ids[step]),
                jnp.asarray(coords), jnp.asarray(masks), jnp.asarray(targets),
                jnp.asarray(affine_batch),
            )
            if step == 0 or (step + 1) in checkpoints:
                item = {"step": step + 1, "sampled_loss": float(loss)}
                if step + 1 in checkpoints:
                    states = np.concatenate((affine, np.tanh(np.asarray(q_raw))), axis=1)
                    final_metrics = evaluate_states(candidate, datasets, hyper, states)
                    item["full_metrics"] = final_metrics
                    item["gate_pass"] = oracle_gate(final_metrics)
                history.append(item)
                if (step + 1) in checkpoints and item["gate_pass"]:
                    break
        if final_metrics is None:
            states = np.concatenate((affine, np.tanh(np.asarray(q_raw))), axis=1)
            final_metrics = evaluate_states(candidate, datasets, hyper, states)
        arrays[f"selection_oracle_start{start_index}_q_raw"] = np.asarray(q_raw)
        results.append({
            "start_index": start_index, "history": history, "metrics": final_metrics,
            "gate_pass": oracle_gate(final_metrics),
            "elapsed_s": float(time.perf_counter() - started),
            "q_raw": np.asarray(q_raw), "optimizer_state": opt_state,
        })
    chosen = min(
        range(len(results)),
        key=lambda index: results[index]["metrics"]["pooled"][
            "mean_snapshot_relative_l2_squared"
        ],
    )
    serializable = [{key: value for key, value in item.items()
                     if key not in ("q_raw", "optimizer_state")}
                    for item in results]
    optimizer_states = tuple(item["optimizer_state"] for item in results)
    return (
        results[chosen]["q_raw"], chosen, serializable,
        results[chosen]["metrics"], optimizer_states,
    )


def main():
    c.require_gpu_highest()
    candidate = candidate_by_arm(ARM)
    s0_artifact, s0_gate = load_s0_gate(S0_JSON, ARM)
    if not SMOKE and TRAIN_SEED not in (11, 29, 47):
        raise SystemExit("scientific training seed must be 11, 29, or 47")
    report = {
        "stage": STAGE_NAME,
        "status": "excluded_execution_smoke" if SMOKE else "running",
        "config": {
            "arm": ARM, "candidate": candidate, "training_seed": TRAIN_SEED,
            "training_mix": TRAIN_MIX, "selection_mix": SELECTION_MIX,
            "all_51_times": not SMOKE,
            "manifold_steps": SMOKE_UPDATES if SMOKE else MANIFOLD_STEPS,
            "predictor_steps": SMOKE_UPDATES if SMOKE else PREDICTOR_STEPS,
            "selection_oracle_steps": SMOKE_UPDATES if SMOKE else ORACLE_STEPS,
            "field_batch": (
                FIELD_BATCH if (not SMOKE or SMOKE_SHAPE_FAITHFUL) else 4
            ),
            "field_points": (
                FIELD_POINTS if (not SMOKE or SMOKE_SHAPE_FAITHFUL) else 32
            ),
            "selection_oracle_batch": (
                ORACLE_BATCH if (not SMOKE or SMOKE_SHAPE_FAITHFUL) else 4
            ),
            "smoke_shape_faithful_batches": SMOKE_SHAPE_FAITHFUL if SMOKE else None,
            "manifold_adamw": {
                "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
                "weight_decay": 1e-6, "gradient_clip": 1.0,
                "learning_rate": [1e-3, 1e-5],
            },
            "predictor_learning_rate": [5e-4, 5e-6],
            "selection_oracle_learning_rate": [5e-2, 1e-3],
            "model_validation_touched": False, "confirmation_touched": False,
            "f64": True, "smoke": SMOKE,
        },
        "provenance": c.provenance(), PARENT_GATE_NAME: s0_gate,
    }
    c.save_json(OUTPUT_JSON, report)
    arrays = {}
    load_started = time.perf_counter()
    training = load_mix(TRAIN_MIX, "train")
    selection = load_mix(SELECTION_MIX, "selection")
    report["data"] = {
        "training": metadata(training), "selection": metadata(selection),
        "elapsed_s": float(time.perf_counter() - load_started),
    }
    arrays["training_affine"] = concatenate(training, "affine")
    arrays["training_features"] = concatenate(training, "features")
    arrays["selection_affine"] = concatenate(selection, "affine")
    arrays["selection_features"] = concatenate(selection, "features")
    feature_mean, feature_scale, empirical_feature_scale = (
        predictor_feature_statistics(arrays["training_features"])
    )
    arrays["predictor_feature_mean"] = feature_mean
    arrays["predictor_feature_scale"] = feature_scale
    arrays["predictor_feature_empirical_scale"] = empirical_feature_scale
    report["config"]["predictor_feature_standardization"] = {
        "source": "locked training mix only",
        "mean": feature_mean.tolist(), "scale": feature_scale.tolist(),
        "empirical_scale": empirical_feature_scale.tolist(),
        "small_scale_replacement": 1.0, "small_scale_threshold": 1e-12,
        "deployment": "folded into first affine layer",
    }

    variables, manifold_opt_state, manifold_history, manifold_elapsed = train_manifold(
        candidate, training, arrays
    )
    hyper = variables["hyper"]
    training_q_raw = np.asarray(variables["q_raw"])
    training_states = np.concatenate((
        arrays["training_affine"], np.tanh(training_q_raw)
    ), axis=1)
    predictor, predictor_opt_state, predictor_history, predictor_elapsed = train_predictor(
        candidate, training, hyper, training_states, feature_mean, feature_scale, arrays
    )
    folded_predictor = fold_predictor_standardization(
        predictor, feature_mean, feature_scale
    )
    fold_identity_train = predictor_fold_identity(
        predictor, folded_predictor, arrays["training_features"],
        feature_mean, feature_scale,
    )
    fold_identity_selection = predictor_fold_identity(
        predictor, folded_predictor, arrays["selection_features"],
        feature_mean, feature_scale,
    )
    fold_identity_error = max(fold_identity_train, fold_identity_selection)
    selection_q_raw, chosen_start, oracle_starts, oracle_metrics, oracle_opt_states = (
        optimize_selection_oracle(candidate, selection, hyper, arrays)
    )
    selection_oracle_states = np.concatenate((
        arrays["selection_affine"], np.tanh(selection_q_raw)
    ), axis=1)
    selection_features = jnp.asarray(arrays["selection_features"])
    predictor_states = np.asarray(jnp.tanh(
        s.apply_mlp(folded_predictor, selection_features)
    ))
    direct_metrics = evaluate_states(candidate, selection, hyper, predictor_states)
    oracle_pass = oracle_gate(oracle_metrics)
    direct_pass, degradation = direct_gate(direct_metrics, oracle_metrics)
    arrays["training_q_raw"] = training_q_raw
    arrays["training_states"] = training_states
    arrays["selection_oracle_chosen_q_raw"] = selection_q_raw
    arrays["selection_oracle_states"] = selection_oracle_states
    arrays["selection_predictor_states"] = predictor_states

    checkpoint = {
        "status": "excluded_execution_smoke" if SMOKE else "complete",
        "candidate": candidate, "training_seed": TRAIN_SEED,
        "hyperdecoder": jax.tree_util.tree_map(np.asarray, hyper),
        "training_autolatent_raw": training_q_raw,
        "direct_predictor_standardized": jax.tree_util.tree_map(np.asarray, predictor),
        "direct_predictor_folded_raw": jax.tree_util.tree_map(
            np.asarray, folded_predictor
        ),
        "predictor_feature_mean": feature_mean,
        "predictor_feature_scale": feature_scale,
        "predictor_fold_identity_max_abs": fold_identity_error,
        "predictor_fold_identity_train_max_abs": fold_identity_train,
        "predictor_fold_identity_selection_max_abs": fold_identity_selection,
        "manifold_optimizer_state": jax.tree_util.tree_map(np.asarray, manifold_opt_state),
        "predictor_optimizer_state": jax.tree_util.tree_map(np.asarray, predictor_opt_state),
        "selection_oracle_optimizer_states": jax.tree_util.tree_map(
            np.asarray, oracle_opt_states
        ),
        "training_affine": arrays["training_affine"],
        "training_features": arrays["training_features"],
        "data_metadata": report["data"], PARENT_GATE_NAME: s0_gate,
        "provenance": report["provenance"], "config": report["config"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(CHECKPOINT)), exist_ok=True)
    with open(CHECKPOINT, "wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **arrays)
    report.update({
        "training": {
            "manifold_history": manifold_history, "manifold_elapsed_s": manifold_elapsed,
            "predictor_history": predictor_history, "predictor_elapsed_s": predictor_elapsed,
            "hyperdecoder_parameters": s.hyperdecoder_parameter_count(
                candidate["R"], candidate["k"]
            ),
            "predictor_parameters": s.predictor_parameter_count(candidate["k"]),
            "autolatent_count": int(training_q_raw.shape[0]),
            "predictor_feature_standardization": {
                "mean": feature_mean.tolist(), "scale": feature_scale.tolist(),
                "empirical_scale": empirical_feature_scale.tolist(),
                "folded_raw_identity_max_abs": fold_identity_error,
                "folded_raw_identity_train_max_abs": fold_identity_train,
                "folded_raw_identity_selection_max_abs": fold_identity_selection,
                "identity_gate": 1e-12,
            },
        },
        "selection_oracle": {
            "starts": oracle_starts, "chosen_start": chosen_start,
            "metrics": oracle_metrics, "gate_pass": oracle_pass,
            "start_selection_metric": "pooled_mean_snapshot_relative_l2_squared",
            "chosen_start_selection_loss": oracle_metrics["pooled"][
                "mean_snapshot_relative_l2_squared"
            ],
        },
        "selection_direct": {
            "metrics": direct_metrics, "degradation_direct_over_oracle": degradation,
            "gate_pass": direct_pass,
        },
        "gates": {
            "representation_oracle_all_N_and_pooled": oracle_pass,
            "direct_all_N_and_pooled": direct_pass,
            "promote_seed": bool(oracle_pass and direct_pass),
            "local_loss_revision_near_miss_condition": bool(
                oracle_pass and not direct_pass and all(
                    (row["trajectory_error_mean"] <= 6e-4
                     and row["trajectory_error_worst"] <= 2e-3
                     and row["all_finite"] and row["exact_binary_boundary"]
                     and degradation[key] <= 3.0)
                    for key, row in list(direct_metrics["meshes"].items())
                    + [("pooled", direct_metrics["pooled"])]
                ),
            ),
            "loss_revision_licensed": False,
            "loss_revision_license_reason": (
                "joint decision requires completed seed-11/29/47 artifacts; "
                "a single-seed cell cannot license revision"
            ),
        },
        "checkpoint": {"path": os.path.basename(CHECKPOINT), "sha256": c.sha256(CHECKPOINT)},
        "npz": {"path": os.path.basename(OUTPUT_NPZ), "sha256": c.sha256(OUTPUT_NPZ)},
    })
    if EXTRA_GATE_FN is not None:
        report["gates"].update(EXTRA_GATE_FN(report))
    report["status"] = "excluded_execution_smoke" if SMOKE else "complete"
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps({
        "status": report["status"], "arm": ARM, "seed": TRAIN_SEED,
        "gates": report["gates"],
    }, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
