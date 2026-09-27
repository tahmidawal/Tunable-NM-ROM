#!/usr/bin/env python
"""Prospectively fixed Phase-11 G2/q32 exact-field training cell.

Scientific execution is cluster-only. ``--smoke`` is synthetic, excluded, and
bounded. It executes the honest conditional decision path: predictor and
selection remain sealed when the synthetic globalized-train gate misses.
"""
from __future__ import annotations

import argparse
import json
import math
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
import b10_phase8_d as p8
import b10_phase9_train as p9
import b10_spline_train as legacy


CONFIG = dict(p9.ARMS["T2"])
TRAIN_MIX = p9.TRAIN_MIX
SELECTION_MIX = p9.SELECTION_MIX
N_ORDER = p9.N_ORDER
BATCH_BY_N = p9.BATCH_BY_N
BATCHES_PER_N = p9.BATCHES_PER_N
IDENTITY_TOL = p9.IDENTITY_TOL
TRUST_ATTEMPTS = 40
SMOKE_TRUST_ATTEMPTS = 1
PHASES = {
    "encoder": {"epochs": 18, "seed": 311011, "segments": ((9, 1e-3, 1e-4), (9, 1e-4, 1e-5))},
    "joint": {"epochs": 54, "seed": 11, "segments": ((27, 1e-3, 1e-5), (27, 1e-5, 1e-6))},
    "predictor": {"epochs": 18, "seed": 100011, "segments": ((18, 5e-4, 5e-6),)},
}
GENERATOR_UPDATES = 705024
PREDICTOR_UPDATES = 176256
MAX_UPDATES = 881280
FINAL_IO_AUDIT_RESERVE_SECONDS = 1800.0
EVALUATION_COUNTS = {
    "pre_gate_epoch_cox_cohorts": 72,
    "terminal_control_cox_cohorts": 1,
    "globalized_train_cox_cohorts": 1,
    "terminal_train_k3_equivalent_cohorts": 2,
    "conditional_predictor_epoch_cox_cohorts": 18,
    "conditional_predictor_terminal_cox_cohorts": 1,
    "conditional_predictor_terminal_k3_equivalent_cohorts": 1,
    "conditional_selection_initial_cox_cohorts": 2,
    "conditional_selection_terminal_cox_cohorts": 2,
    "conditional_selection_terminal_k3_equivalent_cohorts": 2,
}

EXPECTED_EXTRA = {
    "p9_json": "8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d",
    "p9_npz": "d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e",
    "p9_checkpoint": "90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9",
    "p9_audit": "b7bb908addaeb54c81293624c4433c61388c03faf9f514dbe5c8f3ec9d281377",
    "p9_audit_work": "66ef97a0daeb2036e0aa6ccc3f31863e6861702baaa9711311d1839541cd556b",
    "p9_work_checkpoint": "dd7b5fecc07e9021c6264febf5c7187004daaa73a50381921d5190eede639ce0",
    "p9_manifest": "9173714f5a03feef10a9f22f21b4c0c6466caad34e710ccb3131ebb655730edc",
    "p10_json": "f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523",
    "p10_npz": "35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664",
    "p10_work_checkpoint": "5fd879c32f2205d6427b272042c095624d824a8278e1eaf074c5295bcb574d68",
    "p10_audit": "695d5a29829764f3660ef606f6104dbb947761bc52d1f48cdf395969dc8983f4",
    "p10_manifest": "60cd3e3f2d4946c8671f3cb43c3a852fd72a12b86735f3314bf7e78379edb45d",
    "p10_audit_manifest": "44877a42a749b608372e29f13dd3f35c2d6a488aa2c824f563b01b08e506b456",
}

warnings.filterwarnings("error", message=r"(?i).*captured.*large.*constant.*", category=Warning)


def json_normalize(value):
    """Recursively convert NumPy/JAX JSON leaves without changing values."""
    if isinstance(value, dict):
        return {key: json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_normalize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array):
        return json_normalize(np.asarray(value))
    return value


def atomic_json(path, value):
    p9.atomic_json(path, json_normalize(value))


def independent_pre_update_containers():
    permutations, arrays, history = {}, {}, {}
    return permutations, arrays, history


def atomic_pickle(path, value):
    p9.atomic_pickle(path, value)


def validate_chains(args):
    reports, bindings = p9.validate_chains(args, "T1")
    for key, expected in EXPECTED_EXTRA.items():
        path = getattr(args, key)
        if path is None or c.sha256(path) != expected:
            raise SystemExit(f"immutable {key} mismatch")
        bindings[key] = {"basename": os.path.basename(path), "sha256": expected}
    p9_report = p8.load_json(args.p9_json)
    p9_audit = p8.load_json(args.p9_audit)
    p10_report = p8.load_json(args.p10_json)
    p10_audit = p8.load_json(args.p10_audit)
    if not (
        p9_audit.get("status") == "pass"
        and p9_audit.get("negative_aware") is True
        and p9_report.get("decision", {}).get("g2_licensed") is False
        and p10_audit.get("status") == "pass"
        and p10_audit.get("negative_aware") is True
        and p10_audit.get("decision", {}).get("p10_d_valid") is True
        and p10_audit.get("decision", {}).get("fixed_g1_train_representable") is False
        and p10_audit.get("decision", {}).get("architecture_increase_licensed") is False
        and p10_report.get("decision", {}).get("optimizer_updates") == 0
        and p10_report.get("decision", {}).get("selection_evaluated") is False
    ):
        raise SystemExit("accepted P9/P10 evidence chain mismatch")
    bindings["phase11_independent_license"] = {
        "phase10_promotion_used": False,
        "retracted_capacity_used": False,
    }
    return reports, bindings


def load_train(args, smoke):
    legacy.SMOKE = smoke
    if smoke:
        train = p9.p7train.smoke_datasets("train")
        coefficients = np.random.default_rng(92011).normal(0, .05, (4, 3328))
        physical = np.zeros((4, 5), np.float64)
        features = legacy.concatenate(train, "features")
        records = [{"basename": "synthetic-only", "sha256": None,
                    "snapshot_count": 4, "N": 16}]
        return train, coefficients, physical, features, records
    train = legacy.load_mix(TRAIN_MIX, "train")
    p5_report = p8.load_json(args.p5_json)
    coefficients, physical, features, records = p8.load_train_targets(args, p5_report)
    regenerated_affine = legacy.concatenate(train, "affine")
    immutable_affine = np.stack([p9.spline.normalized_state_from_affine(row) for row in physical])
    regenerated_features = legacy.concatenate(train, "features")
    exact_columns = (0, 1, 2, 3, 5, 6)
    viscosity_delta = np.abs(regenerated_features[:, 4] - features[:, 4])
    viscosity_bound = np.abs(np.spacing(np.maximum(np.abs(regenerated_features[:, 4]), np.abs(features[:, 4]))))
    if not (
        np.max(np.abs(regenerated_affine - immutable_affine)) <= p9.AFFINE_ATOL
        and np.array_equal(regenerated_features[:, exact_columns], features[:, exact_columns])
        and np.all(viscosity_delta <= viscosity_bound)
    ):
        raise SystemExit("train affine/feature provenance mismatch")
    return train, coefficients, physical, features, records


def load_selection_after_gate(args, train, physical, features, smoke):
    if smoke:
        return p9.p7train.smoke_datasets("selection"), {"mode": "synthetic"}
    selection = legacy.load_mix(SELECTION_MIX, "selection")
    p7_arrays = {name: value for name, value in np.load(args.p7_npz, allow_pickle=False).items()}
    binding = p8.bind_regeneration(train, selection, physical, features, p7_arrays)
    return selection, binding


def schedule_permutations(phase, epochs, sizes):
    out = {}
    for zero_epoch in range(epochs):
        rng = np.random.Generator(np.random.PCG64(PHASES[phase]["seed"] + zero_epoch))
        for n in N_ORDER:
            out[f"{phase}_e{zero_epoch + 1}_N{n}"] = rng.permutation(sizes[n]).astype(np.int32)
    return out


def piecewise_schedule(segments, steps_per_epoch):
    schedules = []
    boundaries = []
    elapsed = 0
    for epochs, start, stop in segments:
        steps = int(epochs * steps_per_epoch)
        schedules.append(p9.schedule_lr(start, stop, steps))
        elapsed += steps
        boundaries.append(elapsed)
    def value(step):
        result = schedules[-1](step - (boundaries[-2] if len(boundaries) > 1 else 0))
        prior = 0
        for boundary, schedule in reversed(list(zip(boundaries[:-1], schedules[:-1]))):
            result = jnp.where(step < boundary, schedule(step - prior), result)
        return result
    return value


def make_update_kernels(config, smoke=False):
    steps_per_epoch = 1 if smoke else 3 * BATCHES_PER_N
    schedules = {name: piecewise_schedule(row["segments"], steps_per_epoch)
                 for name, row in PHASES.items()}
    opts = tuple(p9.optimizer(schedules[name]) for name in ("encoder", "joint", "predictor"))
    opt_encoder, opt_joint, opt_predictor = opts

    def encoder_loss(params, normalized, truth, affine, coords, mask, mean, scales):
        qraw, q = p9.apply_encoder(params["encoder"], normalized, config)
        states = jnp.concatenate((affine, q), axis=1)
        field, _ = p9.exact_loss(params["generator"], states, truth, coords, mask, mean, scales, config)
        generated = p9.normalized_generated(params["generator"], q, mean, scales, config)
        coeff = .5*jnp.mean((generated[:, :2304]-normalized[:, :2304])**2) + .5*jnp.mean((generated[:, 2304:]-normalized[:, 2304:])**2)
        regularizer = 1e-6*jnp.mean(qraw**2)
        return field + .01*coeff + regularizer, jnp.asarray((field, coeff, regularizer))

    def joint_loss(params, take, normalized, truth, affine, coords, mask, mean, scales):
        raw = params["q_raw"][take]
        q = jnp.tanh(raw)
        states = jnp.concatenate((affine, q), axis=1)
        field, _ = p9.exact_loss(params["generator"], states, truth, coords, mask, mean, scales, config)
        generated = p9.normalized_generated(params["generator"], q, mean, scales, config)
        coeff = .5*jnp.mean((generated[:, :2304]-normalized[:, :2304])**2) + .5*jnp.mean((generated[:, 2304:]-normalized[:, 2304:])**2)
        regularizer = 1e-6*jnp.mean(raw**2)
        return field + .01*coeff + regularizer, jnp.asarray((field, coeff, regularizer))

    def predictor_loss(predictor, generator, standardized, targets, truth, coords, mask, mean, scales):
        states = p9.apply_predictor(predictor, standardized)
        field, _ = p9.exact_loss(generator, states, truth, coords, mask, mean, scales, config)
        state = jnp.mean((states-targets)**2)
        return field + .1*state, jnp.asarray((field, state))

    @jax.jit
    def encoder_update(params, state, normalized, truth, affine, coords, mask, mean, scales):
        (loss, parts), grad = jax.value_and_grad(encoder_loss, has_aux=True)(params, normalized, truth, affine, coords, mask, mean, scales)
        updates, state = opt_encoder.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss, parts

    @jax.jit
    def joint_update(params, state, take, normalized, truth, affine, coords, mask, mean, scales):
        (loss, parts), grad = jax.value_and_grad(joint_loss, has_aux=True)(params, take, normalized, truth, affine, coords, mask, mean, scales)
        updates, state = opt_joint.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss, parts

    @jax.jit
    def predictor_update(predictor, state, generator, standardized, targets, truth, coords, mask, mean, scales):
        (loss, parts), grad = jax.value_and_grad(predictor_loss, has_aux=True)(predictor, generator, standardized, targets, truth, coords, mask, mean, scales)
        updates, state = opt_predictor.update(grad, state, predictor)
        return optax.apply_updates(predictor, updates), state, loss, parts

    return opts, (encoder_update, joint_update, predictor_update)


def _tree_hash(tree):
    leaves = [np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]
    return p9.array_sha(np.concatenate(leaves))


def runtime_preflight(config, views, variables, norm, optimizers, kernels, smoke, job_started):
    if smoke:
        hashes = {name: _tree_hash(tree) for name, tree in variables.items()}
        zero_terms = {
            "pre_gate_updates": 0.0,
            "conditional_predictor_updates": 0.0,
            "pre_gate_train_evaluations": 0.0,
            "conditional_train_evaluations": 0.0,
            "train_globalization": 0.0,
            "conditional_selection_evaluations": 0.0,
            "conditional_selection_globalization": 0.0,
            "audit_regeneration": 0.0,
            "fixed_io_checkpoint_audit": 0.0,
        }
        elapsed = float(time.perf_counter()-job_started)
        allocation = float(os.environ.get("SLURM_TIMELIMIT_SECONDS", 57600))
        return {"raw_seconds": {}, "median_seconds": {},
            "evaluation_counts": dict(EVALUATION_COUNTS),
            "projected_terms_seconds": zero_terms,
            "projected_remaining_seconds": 0.0, "safety_factor": 1.15,
            "required_with_safety_seconds": 0.0, "allocation_seconds": allocation,
            "actual_elapsed_at_decision_seconds": elapsed,
            "actual_remaining_at_decision_seconds": max(0.0, allocation-elapsed),
            "proceed_before_update1": True, "weights_bitwise_unchanged": True,
            "before_hashes": hashes, "after_hashes": hashes,
            "smoke_skipped_duplicate_compile_timing": True}
    repeats = 1 if smoke else 13
    warmups = 0 if smoke else 3
    raw = {}
    medians = {}
    before = {name: _tree_hash(tree) for name, tree in variables.items()}
    states = (
        optimizers[0].init({"generator": variables["generator"], "encoder": variables["encoder"]}),
        optimizers[1].init({"generator": variables["generator"], "q_raw": variables["q_raw"]}),
        optimizers[2].init(variables["predictor"]),
    )
    for n, view in views.items():
        take = np.arange(min(BATCH_BY_N.get(n, 2), len(view["truth"])), dtype=np.int32)
        truth = jnp.asarray(view["truth"][take])
        affine = jnp.asarray(view["affine"][take])
        coords = jnp.asarray(view["coords"])
        mask = jnp.asarray(view["mask"])
        mean = jnp.asarray(norm["mean"])
        scales = jnp.asarray(norm["scales"])
        params = {"generator": variables["generator"], "encoder": variables["encoder"]}
        joint = {"generator": variables["generator"], "q_raw": variables["q_raw"]}
        standardized = (view["features"][take]-norm["feature_mean"])/norm["feature_scale"]
        target = jnp.asarray(variables["target_states"][view["global"][take]])
        invocations = {
            "encoder": lambda: kernels[0](params, states[0], jnp.asarray(view["normalized"][take]), truth, affine, coords, mask, mean, scales),
            "joint": lambda: kernels[1](joint, states[1], jnp.asarray(view["global"][take]), jnp.asarray(view["normalized"][take]), truth, affine, coords, mask, mean, scales),
            "predictor": lambda: kernels[2](variables["predictor"], states[2], variables["generator"], jnp.asarray(standardized), target, truth, coords, mask, mean, scales),
        }
        eval_take = np.arange(min(8, len(view["truth"])), dtype=np.int32)
        evaluation = jax.jit(lambda generator, one_states, one_truth, xy, ma, me, sc:
                             p9.exact_loss(generator, one_states, one_truth, xy, ma, me, sc, config))
        eval_states = jnp.asarray(variables["target_states"][view["global"][eval_take]])
        eval_truth = jnp.asarray(view["truth"][eval_take])
        invocations["evaluation"] = lambda: evaluation(variables["generator"], eval_states, eval_truth, coords, mask, mean, scales)
        trust = p9.make_trust_attempt(config)
        invocations["trust"] = lambda: trust(
            variables["generator"], jnp.asarray(variables["target_states"][view["global"][take], 5:]),
            affine, truth, coords, mask, mean, scales,
            jnp.full((len(take),), p9.DELTA0), jnp.full((len(take),), p9.LAMBDA0),
            jnp.ones((len(take),), dtype=bool), jnp.ones((len(take),)))
        for name, invoke in invocations.items():
            times = []
            for repetition in range(repeats):
                started = time.perf_counter()
                result = invoke()
                jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), result)
                elapsed = time.perf_counter()-started
                if repetition >= warmups:
                    times.append(elapsed)
            raw[f"{name}_N{n}"] = times
            medians[f"{name}_N{n}"] = float(np.median(times))
    train_sizes = {64: 26112, 128: 6528, 256: 3264}
    selection_sizes = {64: 3264, 128: 1632, 256: 816}
    train_eval_unit = sum(math.ceil(train_sizes[n]/8)*medians[f"evaluation_N{n}"] for n in N_ORDER)
    selection_eval_unit = sum(math.ceil(selection_sizes[n]/8)*medians[f"evaluation_N{n}"] for n in N_ORDER)
    terms = {
            "pre_gate_updates": sum(medians[f"{phase}_N{n}"]*BATCHES_PER_N*PHASES[phase]["epochs"]
                                    for phase in ("encoder", "joint") for n in N_ORDER),
            "conditional_predictor_updates": sum(medians[f"predictor_N{n}"]*BATCHES_PER_N*PHASES["predictor"]["epochs"] for n in N_ORDER),
            "pre_gate_train_evaluations": (EVALUATION_COUNTS["pre_gate_epoch_cox_cohorts"]
                + EVALUATION_COUNTS["terminal_control_cox_cohorts"]
                + EVALUATION_COUNTS["globalized_train_cox_cohorts"]
                + EVALUATION_COUNTS["terminal_train_k3_equivalent_cohorts"])*train_eval_unit,
            "conditional_train_evaluations": (EVALUATION_COUNTS["conditional_predictor_epoch_cox_cohorts"]
                + EVALUATION_COUNTS["conditional_predictor_terminal_cox_cohorts"]
                + EVALUATION_COUNTS["conditional_predictor_terminal_k3_equivalent_cohorts"])*train_eval_unit,
            "train_globalization": TRUST_ATTEMPTS*sum(math.ceil(train_sizes[n]/BATCH_BY_N[n])*medians[f"trust_N{n}"] for n in N_ORDER),
            "conditional_selection_evaluations": (EVALUATION_COUNTS["conditional_selection_initial_cox_cohorts"]
                + EVALUATION_COUNTS["conditional_selection_terminal_cox_cohorts"]
                + EVALUATION_COUNTS["conditional_selection_terminal_k3_equivalent_cohorts"])*selection_eval_unit,
            "conditional_selection_globalization": 2*TRUST_ATTEMPTS*sum(math.ceil(selection_sizes[n]/BATCH_BY_N[n])*medians[f"trust_N{n}"] for n in N_ORDER),
            "audit_regeneration": float(time.perf_counter()-job_started),
            "fixed_io_checkpoint_audit": FINAL_IO_AUDIT_RESERVE_SECONDS,
    }
    projected = float(sum(terms.values()))
    after = {name: _tree_hash(tree) for name, tree in variables.items()}
    elapsed = float(time.perf_counter()-job_started)
    allocation = float(os.environ.get("SLURM_TIMELIMIT_SECONDS", 57600))
    remaining = max(0.0, allocation-elapsed)
    return {
        "raw_seconds": raw,
        "median_seconds": medians,
        "evaluation_counts": dict(EVALUATION_COUNTS),
        "projected_terms_seconds": {key: float(value) for key, value in terms.items()},
        "projected_remaining_seconds": projected,
        "safety_factor": 1.15,
        "required_with_safety_seconds": 1.15*projected,
        "allocation_seconds": allocation,
        "actual_elapsed_at_decision_seconds": elapsed,
        "actual_remaining_at_decision_seconds": remaining,
        "proceed_before_update1": bool(smoke or 1.15*projected <= remaining),
        "weights_bitwise_unchanged": before == after,
        "before_hashes": before,
        "after_hashes": after,
    }


def _batches(view, permutation, smoke):
    if smoke:
        return [np.arange(min(2, len(view["truth"])), dtype=np.int32)]
    return permutation.reshape((-1, BATCH_BY_N[view["item"]["N"]]))


def _checkpoint_payload(phase, epoch, update, variables, optimizer_states, evaluation_states):
    return {
        "phase": phase,
        "epoch": int(epoch),
        "global_update": int(update),
        "generator": jax.tree_util.tree_map(np.asarray, variables["generator"]),
        "encoder": jax.tree_util.tree_map(np.asarray, variables["encoder"]),
        "predictor": jax.tree_util.tree_map(np.asarray, variables["predictor"]),
        "q_raw": np.asarray(variables["q_raw"]),
        "globalized_q": np.asarray(variables.get("globalized_q", np.empty((0, CONFIG["q"])) )),
        "final_target_states": np.asarray(variables["target_states"]),
        "folded_predictor": jax.tree_util.tree_map(np.asarray, variables.get("folded_predictor", variables["predictor"])),
        "optimizer_states": jax.tree_util.tree_map(np.asarray, optimizer_states),
        "evaluation_states": np.asarray(evaluation_states),
    }


def train_generator(config, datasets, views, norm, args, smoke, job_started):
    epochs = {name: (1 if smoke else row["epochs"]) for name, row in PHASES.items()}
    sizes = {n: len(view["truth"]) for n, view in views.items()}
    if smoke:
        only_n = next(iter(sizes))
        permutations = {f"{phase}_e1_N{only_n}": np.arange(sizes[only_n], dtype=np.int32)
                        for phase in PHASES}
    else:
        permutations = schedule_permutations("encoder", epochs["encoder"], sizes)
        permutations |= schedule_permutations("joint", epochs["joint"], sizes)
        permutations |= schedule_permutations("predictor", epochs["predictor"], sizes)
    optimizers, kernels = make_update_kernels(config, smoke)
    generator = p9.init_generator(config)
    encoder = p9.init_encoder(config)
    predictor = p9.init_predictor(config)
    q_raw = p9.encode_batches(encoder, norm["normalized"], config)
    target_states = np.concatenate((legacy.concatenate(datasets, "affine"), np.tanh(q_raw)), axis=1)
    variables = {"generator": generator, "encoder": encoder, "predictor": predictor,
                 "q_raw": jnp.asarray(q_raw), "target_states": target_states}
    preflight = runtime_preflight(config, views, variables, norm, optimizers, kernels, smoke, job_started)
    if not (preflight["weights_bitwise_unchanged"] and preflight["proceed_before_update1"]):
        return variables, {}, {}, {}, preflight, False, {}, optimizers
    params = {"generator": generator, "encoder": encoder}
    encoder_state = optimizers[0].init(params)
    history = {}
    arrays = {}
    resolution_order = []
    optimizer_states = {"encoder": None, "joint": None, "predictor": optimizers[2].init(predictor)}
    global_update = 0
    for phase in ("encoder", "joint"):
        if phase == "joint":
            handoff = p9.encode_batches(params["encoder"], norm["normalized"], config)
            arrays["encoder_handoff_q_raw"] = handoff.copy()
            joint = {"generator": params["generator"], "q_raw": jnp.asarray(handoff)}
            joint_state = optimizers[1].init(joint)
        for epoch in range(1, epochs[phase]+1):
            ncycle = tuple(views) if smoke else N_ORDER
            per_n = {n: _batches(view, permutations[f"{phase}_e{epoch}_N{n}"], smoke)
                     for n, view in views.items()}
            batches = 1 if smoke else BATCHES_PER_N
            last = None
            for batch_index in range(batches):
                for n in ncycle:
                    view = views[n]
                    local = np.asarray(per_n[n][batch_index])
                    take = view["global"][local]
                    common = (jnp.asarray(view["truth"][local]), jnp.asarray(view["affine"][local]),
                              jnp.asarray(view["coords"]), jnp.asarray(view["mask"]),
                              jnp.asarray(norm["mean"]), jnp.asarray(norm["scales"]))
                    if phase == "encoder":
                        params, encoder_state, loss, parts = kernels[0](params, encoder_state,
                            jnp.asarray(view["normalized"][local]), *common)
                    else:
                        joint, joint_state, loss, parts = kernels[1](joint, joint_state,
                            jnp.asarray(take), jnp.asarray(view["normalized"][local]), *common)
                    global_update += 1
                    resolution_order.append(n)
                    last = [float(loss), np.asarray(parts).tolist()]
            if phase == "encoder":
                generator_eval = params["generator"]
                q_eval = p9.encode_batches(params["encoder"], norm["normalized"], config)
                optimizer_states["encoder"] = jax.tree_util.tree_map(np.asarray, encoder_state)
            else:
                generator_eval = joint["generator"]
                q_eval = np.asarray(joint["q_raw"])
                optimizer_states["joint"] = jax.tree_util.tree_map(np.asarray, joint_state)
            states_eval = np.concatenate((legacy.concatenate(datasets, "affine"), np.tanh(q_eval)), axis=1)
            metrics = p9.evaluate_full(datasets, generator_eval, states_eval, norm["mean"], norm["scales"], config, arrays, f"{phase}_epoch{epoch}", False)
            history[f"{phase}_epoch{epoch}"] = {"terminal_update": global_update, "last_batch": last, "metrics": metrics}
            variables.update({"generator": generator_eval, "encoder": params["encoder"],
                              "q_raw": jnp.asarray(q_eval), "target_states": states_eval})
            atomic_json(args.progress_json, {"status": "in_progress", "phase": phase,
                        "epoch": epoch, "global_update": global_update,
                        "scientific_metrics_exposed": False})
            atomic_pickle(args.work_checkpoint, _checkpoint_payload(
                phase, epoch, global_update, variables, optimizer_states, states_eval))
    expected = 2 if smoke else GENERATOR_UPDATES
    if global_update != expected:
        raise AssertionError(f"generator update count {global_update} != {expected}")
    variables.update({"generator": joint["generator"], "encoder": params["encoder"],
                      "q_raw": joint["q_raw"], "target_states": states_eval,
                      "optimizer_states": optimizer_states})
    arrays["update_resolution_order"] = np.asarray(resolution_order, np.int16)
    arrays["history_marker"] = np.asarray((global_update,), np.int64)
    return variables, permutations, arrays, history, preflight, True, optimizer_states, optimizers


def run_trust_prefixed(prefix, datasets, generator, starts, norm, config, arrays, args, smoke):
    attempts = SMOKE_TRUST_ATTEMPTS if smoke else TRUST_ATTEMPTS
    nstarts, total, qdim = starts.shape
    trace = {
        "q": np.empty((nstarts, total, attempts+1, qdim)),
        "objective": np.empty((nstarts, total, attempts+1)),
        "delta": np.empty((nstarts, total, attempts+1)),
        "damping": np.empty((nstarts, total, attempts+1)),
        "active": np.zeros((nstarts, total, attempts+1), bool),
        "trial_objective": np.empty((nstarts, total, attempts)),
        "predicted": np.empty((nstarts, total, attempts)),
        "actual": np.empty((nstarts, total, attempts)),
        "rho": np.empty((nstarts, total, attempts)),
        "rho_defined": np.zeros((nstarts, total, attempts), bool),
        "accepted": np.zeros((nstarts, total, attempts), bool),
        "terminated": np.zeros((nstarts, total, attempts), bool),
        "attempted": np.zeros((nstarts, total, attempts), bool),
        "cg_iterations": np.zeros((nstarts, total, attempts), np.int32),
        "cg_relative": np.empty((nstarts, total, attempts)),
        "cg_breakdown": np.zeros((nstarts, total, attempts), bool),
        "finite": np.zeros((nstarts, total, attempts), bool),
        "step": np.empty((nstarts, total, attempts, qdim)),
        "gradient": np.empty((nstarts, total, attempts, qdim)),
        "bound_active_count": np.zeros((nstarts, total, attempts), np.int32),
        "jvp_count": np.zeros((nstarts, total, attempts), np.int32),
        "vjp_count": np.zeros((nstarts, total, attempts), np.int32),
    }
    attempt_fn = p9.make_trust_attempt(config)
    for start_index in range(nstarts):
        q = starts[start_index].copy()
        delta = np.full(total, p9.DELTA0)
        damping = np.full(total, p9.LAMBDA0)
        active = np.ones(total, bool)
        objective = np.empty(total)
        offset = 0
        for item in datasets:
            count = len(item["flat"])
            states = np.concatenate((item["affine"], q[offset:offset+count]), axis=1)
            temporary = {}
            p9.evaluate_full([item], generator, states, norm["mean"], norm["scales"],
                             config, temporary, "initial", False)
            objective[offset:offset+count] = temporary[f"initial_N{item['N']}_numerator"]/temporary[f"initial_N{item['N']}_denominator"]
            offset += count
        trace["q"][start_index, :, 0] = q
        trace["objective"][start_index, :, 0] = objective
        trace["delta"][start_index, :, 0] = delta
        trace["damping"][start_index, :, 0] = damping
        trace["active"][start_index, :, 0] = active
        for attempt in range(attempts):
            attempted = active.copy()
            old_delta = delta.copy()
            old_damping = damping.copy()
            offset = 0
            for item in datasets:
                count = len(item["flat"])
                batch = BATCH_BY_N.get(item["N"], min(2, count))
                for local_start in range(0, count, batch):
                    local_stop = min(local_start+batch, count)
                    sl = slice(offset+local_start, offset+local_stop)
                    local = slice(local_start, local_stop)
                    result = tuple(map(np.asarray, attempt_fn(
                        generator, jnp.asarray(q[sl]), jnp.asarray(item["affine"][local]),
                        jnp.asarray(item["flat"][local]), jnp.asarray(item["coords"]),
                        jnp.asarray(item["mask"]), jnp.asarray(norm["mean"]),
                        jnp.asarray(norm["scales"]), jnp.asarray(delta[sl]),
                        jnp.asarray(damping[sl]), jnp.asarray(active[sl]),
                        jnp.asarray(objective[sl]))))
                    (next_q, next_objective, next_delta, next_damping, next_active,
                     _, trial, predicted, actual, rho, rho_defined, accepted,
                     terminated, iterations, relative, breakdown, finite, step, gradient) = result
                    for name, value in (("trial_objective", trial), ("predicted", predicted),
                        ("actual", actual), ("rho", rho), ("rho_defined", rho_defined),
                        ("accepted", accepted), ("terminated", terminated),
                        ("cg_iterations", iterations), ("cg_relative", relative),
                        ("cg_breakdown", breakdown), ("finite", finite),
                        ("step", step), ("gradient", gradient)):
                        trace[name][start_index, sl, attempt] = value
                    q[sl], objective[sl], delta[sl], damping[sl], active[sl] = (
                        next_q, next_objective, next_delta, next_damping, next_active)
                offset += count
            trace["attempted"][start_index, :, attempt] = attempted
            if not (np.array_equal(old_delta, trace["delta"][start_index, :, attempt])
                    and np.array_equal(old_damping, trace["damping"][start_index, :, attempt])):
                raise SystemExit("trust state transition persistence mismatch")
            work = np.where(attempted, trace["cg_iterations"][start_index, :, attempt]+1, 0)
            trace["jvp_count"][start_index, :, attempt] = work
            trace["vjp_count"][start_index, :, attempt] = work
            trace["q"][start_index, :, attempt+1] = q
            trace["objective"][start_index, :, attempt+1] = objective
            trace["delta"][start_index, :, attempt+1] = delta
            trace["damping"][start_index, :, attempt+1] = damping
            trace["active"][start_index, :, attempt+1] = active
            trace["bound_active_count"][start_index, :, attempt] = np.where(
                attempted, np.sum(np.abs(q) >= 1.0, axis=1), 0)
            atomic_json(args.progress_json, {"status": "in_progress", "phase": prefix,
                "start": start_index, "attempt": attempt+1, "attempt_cap": attempts,
                "scientific_metrics_exposed": False})
    exhaustion = trace["active"][:, :, -1] & (trace["delta"][:, :, -1] <= p9.DELTA_MIN) & (trace["damping"][:, :, -1] >= p9.LAMBDA_MAX)
    for key, value in trace.items():
        arrays[f"{prefix}_{key}"] = value
    arrays[f"{prefix}_unhealthy_exhaustion"] = exhaustion
    return trace


def trust_health(trace, exhaustion):
    attempted = trace["attempted"]
    defined = trace["rho_defined"] & attempted
    undefined = (~trace["rho_defined"]) & attempted
    breakdown_count = int(np.sum(trace["cg_breakdown"] & attempted))
    exhaustion_count = int(np.sum(exhaustion))
    return {
        "finite": bool(np.all(trace["finite"] | ~attempted)),
        "breakdown_count": breakdown_count,
        "no_breakdown": breakdown_count == 0,
        "unhealthy_exhaustion_count": exhaustion_count,
        "no_unhealthy_exhaustion": exhaustion_count == 0,
        "defined_rho_match": bool(np.allclose(trace["rho"][defined],
            trace["actual"][defined]/trace["predicted"][defined], rtol=2e-13, atol=2e-14)),
        "undefined_rho_zero": bool(np.all(trace["rho"][undefined] == 0.0)),
        "undefined_never_accepted": bool(not np.any(trace["accepted"][undefined])),
    }


def train_predictor(config, datasets, views, norm, variables, permutations, arrays,
                    history, optimizer_states, optimizers, args, smoke, start_update):
    _, kernels = make_update_kernels(config, smoke)
    epochs = 1 if smoke else PHASES["predictor"]["epochs"]
    predictor = variables["predictor"]
    state = optimizers[2].init(predictor)
    generator = variables["generator"]
    target_states = variables["target_states"]
    global_update = int(start_update)
    resolution_order = list(np.asarray(arrays["update_resolution_order"], np.int16))
    for epoch in range(1, epochs+1):
        ncycle = tuple(views) if smoke else N_ORDER
        per_n = {n: _batches(view, permutations[f"predictor_e{epoch}_N{n}"], smoke)
                 for n, view in views.items()}
        batches = 1 if smoke else BATCHES_PER_N
        last = None
        for batch_index in range(batches):
            for n in ncycle:
                view = views[n]
                local = np.asarray(per_n[n][batch_index])
                take = view["global"][local]
                standardized = (view["features"][local]-norm["feature_mean"])/norm["feature_scale"]
                predictor, state, loss, parts = kernels[2](predictor, state, generator,
                    jnp.asarray(standardized), jnp.asarray(target_states[take]),
                    jnp.asarray(view["truth"][local]), jnp.asarray(view["coords"]),
                    jnp.asarray(view["mask"]), jnp.asarray(norm["mean"]), jnp.asarray(norm["scales"]))
                global_update += 1
                resolution_order.append(n)
                last = [float(loss), np.asarray(parts).tolist()]
        folded = legacy.fold_predictor_standardization(predictor, norm["feature_mean"], norm["feature_scale"])
        states_eval = p9.predictor_batches(folded, legacy.concatenate(datasets, "features"))
        metrics = p9.evaluate_full(datasets, generator, states_eval, norm["mean"], norm["scales"], config,
                                   arrays, f"predictor_epoch{epoch}", False)
        history[f"predictor_epoch{epoch}"] = {"terminal_update": global_update,
                                               "last_batch": last, "metrics": metrics}
        optimizer_states["predictor"] = jax.tree_util.tree_map(np.asarray, state)
        variables.update({"predictor": predictor, "folded_predictor": folded})
        atomic_json(args.progress_json, {"status": "in_progress", "phase": "predictor",
                    "epoch": epoch, "global_update": global_update,
                    "scientific_metrics_exposed": False})
        atomic_pickle(args.work_checkpoint, _checkpoint_payload(
            "predictor", epoch, global_update, variables, optimizer_states, states_eval))
    expected = 3 if smoke else MAX_UPDATES
    if global_update != expected:
        raise AssertionError(f"terminal update count {global_update} != {expected}")
    arrays["update_resolution_order"] = np.asarray(resolution_order, np.int16)
    arrays["history_marker"] = np.asarray((global_update,), np.int64)
    variables["optimizer_states"] = optimizer_states
    return variables, arrays, history, optimizer_states, global_update


def _gate(metrics, mean_limit, worst_limit, terminal, smoke):
    return p9.gate(metrics, mean_limit, worst_limit, terminal)


def parse_args():
    parser = argparse.ArgumentParser()
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            parser.add_argument(f"--{phase}-{kind}", dest=f"{phase}_{kind}")
    for kind in ("json", "npz", "checkpoint", "audit", "manifest"):
        parser.add_argument(f"--p7-{kind}", dest=f"p7_{kind}")
    for kind in ("json", "npz", "audit", "manifest"):
        parser.add_argument(f"--p8-{kind}", dest=f"p8_{kind}")
    for key in EXPECTED_EXTRA:
        parser.add_argument("--" + key.replace("_", "-"), dest=key)
    parser.add_argument("--target-dir")
    parser.add_argument("--prereg")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--work-checkpoint", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-skip-structural", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    smoke = args.smoke
    if smoke:
        reports = bindings = None
    else:
        required = [getattr(args, f"{phase}_{kind}") for phase in ("p4", "p5", "p6") for kind in ("json", "npz", "audit", "manifest")]
        required += [getattr(args, f"p7_{kind}") for kind in ("json", "npz", "checkpoint", "audit", "manifest")]
        required += [getattr(args, f"p8_{kind}") for kind in ("json", "npz", "audit", "manifest")]
        required += [getattr(args, key) for key in EXPECTED_EXTRA] + [args.target_dir, args.prereg]
        if any(value is None for value in required):
            raise SystemExit("scientific Phase11 requires exact P4-P10 chain")
        reports, bindings = validate_chains(args)
    train, coefficients, physical, immutable_features, target_records = load_train(args, smoke)
    features = legacy.concatenate(train, "features")
    norm = p9.train_normalization(coefficients, features)
    views = p9.per_n_views(train, norm["normalized"])
    if args.smoke_skip_structural and not smoke:
        raise SystemExit("--smoke-skip-structural is synthetic-only")
    structural = (p9.t2_structural_preflight(CONFIG, norm["mean"], norm["scales"], smoke)
                  if not args.smoke_skip_structural else
                  {"scientific": False, "separately_smoked": True, "pass": False})
    if not smoke and not structural["pass"]:
        initial_q = np.zeros((len(coefficients), CONFIG["q"]), np.float64)
        variables = {"generator": p9.init_generator(CONFIG), "encoder": p9.init_encoder(CONFIG),
                     "predictor": p9.init_predictor(CONFIG), "q_raw": initial_q,
                     "target_states": np.concatenate((legacy.concatenate(train, "affine"), initial_q), axis=1),
                     "optimizer_states": {}}
        permutations, arrays, history = independent_pre_update_containers()
        preflight = {"structural_preflight_pass": False, "weights_bitwise_unchanged": True,
                     "proceed_before_update1": False}
        updates_started = False
        optimizer_states = {}
        optimizers = make_update_kernels(CONFIG, smoke)[0]
    else:
        (variables, permutations, arrays, history, preflight, updates_started,
         optimizer_states, optimizers) = train_generator(CONFIG, train, views, norm, args, smoke, started)
    arrays.update({
        "coefficient_mean": norm["mean"], "head_scales": norm["scales"],
        "predictor_feature_mean": norm["feature_mean"], "predictor_feature_scale": norm["feature_scale"],
        "predictor_feature_empirical_scale": norm["feature_empirical"],
        "training_features": features, "normalization_source_indices": norm["source_indices"],
    })
    arrays.update(permutations)
    counts = {"generator": p9.tree_count(variables["generator"]),
              "encoder": p9.tree_count(variables["encoder"]),
              "predictor": p9.tree_count(variables["predictor"])}
    expected_counts = {key: CONFIG[key + "_count"] for key in ("generator", "encoder", "predictor")}
    if counts != expected_counts:
        raise SystemExit("G2 parameter count mismatch")
    decision = {
        "arm": "G2/q32", "updates_started": bool(updates_started),
        "structural_preflight_pass": bool(structural.get("pass", False)),
        "globalized_train_pass": False, "predictor_trained": False,
        "train_direct_pass": False, "selection_evaluated": False,
        "phase11_pass": False, "retracted_capacity_used": False,
        "third_architecture_licensed": False, "scientific_promotion_allowed": False,
        "next_action": "hard stop",
    }
    terminal_train = globalized_train = train_trust_health = train_direct = None
    selection_report = None
    predictor_fold_identity = None
    selection_binding = None
    total_updates = 0
    if updates_started:
        total_updates = 2 if smoke else GENERATOR_UPDATES
        q_terminal = np.tanh(np.asarray(variables["q_raw"]))
        arrays["final_q_raw"] = np.asarray(variables["q_raw"])
        terminal_states = np.concatenate((legacy.concatenate(train, "affine"), q_terminal), axis=1)
        terminal_train = p9.evaluate_full(train, variables["generator"], terminal_states,
            norm["mean"], norm["scales"], CONFIG, arrays, "terminal_train", not smoke)
        trace = run_trust_prefixed("train_trust", train, variables["generator"], q_terminal[None],
                                   norm, CONFIG, arrays, args, smoke)
        globalized_q = np.asarray(trace["q"][0, :, -1])
        arrays["train_trust_terminal_q"] = globalized_q
        variables["globalized_q"] = globalized_q
        variables["target_states"] = np.concatenate((legacy.concatenate(train, "affine"), globalized_q), axis=1)
        globalized_train = p9.evaluate_full(train, variables["generator"], variables["target_states"],
            norm["mean"], norm["scales"], CONFIG, arrays, "globalized_train", not smoke)
        exhaustion = np.asarray(arrays["train_trust_unhealthy_exhaustion"], bool)
        train_trust_health = trust_health(trace, exhaustion)
        trust_gate = p9.trust_health_gate(train_trust_health)
        train_pass = bool(_gate(globalized_train, 2e-4, 7e-4, not smoke, smoke) and trust_gate)
        decision["globalized_train_pass"] = train_pass
        atomic_pickle(args.work_checkpoint, _checkpoint_payload(
            "globalized_train", 40 if not smoke else SMOKE_TRUST_ATTEMPTS, total_updates,
            variables, optimizer_states, variables["target_states"]))
        if train_pass:
            variables, arrays, history, optimizer_states, total_updates = train_predictor(
                CONFIG, train, views, norm, variables, permutations, arrays, history,
                optimizer_states, optimizers, args, smoke, total_updates)
            decision["predictor_trained"] = True
            predictor_fold_identity = legacy.predictor_fold_identity(
                variables["predictor"], variables["folded_predictor"], features,
                norm["feature_mean"], norm["feature_scale"])
            direct_states = p9.predictor_batches(variables["folded_predictor"], features)
            arrays["train_direct_states"] = direct_states
            train_direct = p9.evaluate_full(train, variables["generator"], direct_states,
                norm["mean"], norm["scales"], CONFIG, arrays, "train_direct", not smoke)
            direct_pass = _gate(train_direct, 3e-4, 1e-3, not smoke, smoke)
            decision["train_direct_pass"] = direct_pass
            if direct_pass:
                selection, selection_binding = load_selection_after_gate(
                    args, train, physical, immutable_features, smoke)
                selection_coeff = p9.selection_coefficients(args, selection, smoke)
                normalized_selection = np.concatenate((
                    (selection_coeff[:, :2304]-norm["mean"][:2304])/norm["scales"][0],
                    (selection_coeff[:, 2304:]-norm["mean"][2304:])/norm["scales"][1]), axis=1)
                encoder_raw = p9.encode_batches(variables["encoder"], normalized_selection, CONFIG)
                encoder_q = np.tanh(encoder_raw)
                selection_features = legacy.concatenate(selection, "features")
                direct_selection_states = p9.predictor_batches(variables["folded_predictor"], selection_features)
                predictor_q = direct_selection_states[:, 5:]
                arrays["selection_free_encoder_q_raw"] = encoder_raw
                arrays["selection_direct_states"] = direct_selection_states
                selection_trace = run_trust_prefixed("selection_trust", selection,
                    variables["generator"], np.stack((predictor_q, encoder_q)), norm, CONFIG,
                    arrays, args, smoke)
                chosen = np.argmin(selection_trace["objective"][:, :, -1], axis=0)
                chosen_q = np.where(chosen[:, None] == 0,
                    selection_trace["q"][0, :, -1], selection_trace["q"][1, :, -1])
                arrays["selection_trust_chosen_start"] = chosen.astype(np.int8)
                arrays["selection_trust_chosen_terminal_q"] = chosen_q
                oracle_states = np.concatenate((legacy.concatenate(selection, "affine"), chosen_q), axis=1)
                direct = p9.evaluate_full(selection, variables["generator"], direct_selection_states,
                    norm["mean"], norm["scales"], CONFIG, arrays, "selection_direct", not smoke)
                oracle = p9.evaluate_full(selection, variables["generator"], oracle_states,
                    norm["mean"], norm["scales"], CONFIG, arrays, "selection_oracle", not smoke)
                ratios = {key: (direct["pooled"] if key == "pooled" else direct["meshes"][key])["trajectory_error_mean"]/
                    max((oracle["pooled"] if key == "pooled" else oracle["meshes"][key])["trajectory_error_mean"], 1e-300)
                    for key in [*direct["meshes"], "pooled"]}
                selection_exhaustion = np.asarray(arrays["selection_trust_unhealthy_exhaustion"], bool)
                selection_health = trust_health(selection_trace, selection_exhaustion)
                selection_gate = p9.trust_health_gate(selection_health)
                selection_pass = bool(_gate(direct, 3e-4, 1e-3, not smoke, smoke)
                    and _gate(oracle, 2e-4, 7e-4, not smoke, smoke)
                    and all(value <= 1.5 for value in ratios.values()) and selection_gate)
                selection_report = {"direct": direct, "oracle": oracle,
                    "direct_oracle_mean_ratio": ratios, "trust_health": selection_health,
                    "trust_gate": selection_gate, "pass": selection_pass}
                decision.update({"selection_evaluated": True, "phase11_pass": selection_pass,
                    "scientific_promotion_allowed": selection_pass,
                    "next_action": "separate corrected-rollout proposal" if selection_pass else "hard stop"})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    checkpoint = {
        "status": "excluded_execution_smoke" if smoke else "complete",
        "config": CONFIG,
        "generator": jax.tree_util.tree_map(np.asarray, variables["generator"]),
        "encoder": jax.tree_util.tree_map(np.asarray, variables["encoder"]),
        "predictor": jax.tree_util.tree_map(np.asarray, variables["predictor"]),
        "folded_predictor": jax.tree_util.tree_map(np.asarray, variables.get("folded_predictor", variables["predictor"])),
        "q_raw": np.asarray(variables["q_raw"]),
        "globalized_q": np.asarray(variables.get("globalized_q", np.empty((0, CONFIG["q"])))),
        "encoder_handoff_q_raw": np.asarray(arrays.get("encoder_handoff_q_raw", np.empty((0, CONFIG["q"])))),
        "final_target_states": np.asarray(variables["target_states"]),
        "optimizer_states": jax.tree_util.tree_map(np.asarray, variables.get("optimizer_states", optimizer_states)),
        "normalization": {key: norm[key] for key in ("mean", "scales", "feature_mean", "feature_scale")},
    }
    atomic_pickle(args.checkpoint, checkpoint)
    schedule_epochs = {key: (1 if smoke else value["epochs"]) for key, value in PHASES.items()}
    report = {
        "status": "excluded_execution_smoke" if smoke else "complete",
        "provenance": c.provenance(), "arm": CONFIG, "bindings": bindings,
        "independent_license": {"phase10_promotion_used": False,
            "retracted_capacity_used": False, "free_h1_pass": True,
            "accepted_fixed_g1_globalized_miss": True},
        "parameter_counts": {"reported": counts, "expected": expected_counts},
        "data": {"training": legacy.metadata(train),
            "selection": legacy.metadata(selection) if decision["selection_evaluated"] else None,
            "target_chunks": target_records, "train_snapshot_count": int(len(coefficients)),
            "selection_snapshot_count": int(sum(len(item["flat"]) for item in selection)) if decision["selection_evaluated"] else 0},
        "information_boundary": {"train_only_weights": True,
            "selection_loaded_after_globalized_and_direct_train_pass": bool(not decision["selection_evaluated"] or (decision["globalized_train_pass"] and decision["train_direct_pass"])),
            "selection_target_coefficients_training_use": False,
            "model_validation_touched": False, "confirmation_touched": False,
            "weak_eq_fitting_touched": False, "scaling_touched": False,
            "retracted_capacity_touched": False},
        "selection_affine_feature_binding": selection_binding,
        "normalization": {"definition": "train-only vector mean and centered per-head RMS",
            "elapsed_s": norm["elapsed_s"], "host_bytes": norm["host_bytes"],
            "mean_sha256": p9.array_sha(norm["mean"]), "scales_sha256": p9.array_sha(norm["scales"]),
            "normalized_training_coefficients_sha256": p9.array_sha(norm["normalized"]),
            "training_features_sha256": p9.array_sha(features),
            "feature_mean_sha256": p9.array_sha(norm["feature_mean"]),
            "feature_scale_sha256": p9.array_sha(norm["feature_scale"]),
            "feature_empirical_scale_sha256": p9.array_sha(norm["feature_empirical"])},
        "schedule": {"phase_order": ["encoder", "joint", "predictor_conditional"],
            "epochs": schedule_epochs, "resolution_cycle": [64, 128, 256],
            "batches": {"64": 8, "128": 2, "256": 1},
            "batches_per_N_epoch": 1 if smoke else BATCHES_PER_N,
            "phase_update_counts": {"encoder": 1 if smoke else 176256,
                "joint": 1 if smoke else 528768,
                "predictor": (1 if smoke else PREDICTOR_UPDATES) if decision["predictor_trained"] else 0},
            "actual_total_update_count": int(total_updates), "max_total_update_count": 3 if smoke else MAX_UPDATES,
            "terminal_epochs": {"encoder": 1 if smoke else 18, "joint": 1 if smoke else 54,
                "predictor": (1 if smoke else 18) if decision["predictor_trained"] else None},
            "terminal_only": True, "zero_based_permutation_seed_offsets": True,
            "piecewise_lr_segments": {key: value["segments"] for key, value in PHASES.items()}},
        "preflight": preflight, "history": history,
        "structural_preflight": structural, "terminal_train_control": terminal_train,
        "globalized_train": globalized_train, "train_trust_health": train_trust_health,
        "train_direct": train_direct, "predictor_fold_identity": predictor_fold_identity,
        "selection": selection_report, "decision": decision,
        "npz": {"basename": os.path.basename(args.output_npz), "sha256": c.sha256(args.output_npz)},
        "checkpoint": {"basename": os.path.basename(args.checkpoint), "sha256": c.sha256(args.checkpoint)},
        "work_checkpoint": ({"basename": os.path.basename(args.work_checkpoint), "sha256": c.sha256(args.work_checkpoint)}
                            if os.path.isfile(args.work_checkpoint) else None),
        "elapsed_s": float(time.perf_counter()-started),
    }
    atomic_json(args.output_json, report)
    atomic_json(args.progress_json, {"status": "complete", "scientific_metrics_exposed": False})
    print(json.dumps({"status": report["status"], "decision": decision}, sort_keys=True), flush=True)
    print("ALL-DONE", flush=True)


if __name__ == "__main__":
    main()
