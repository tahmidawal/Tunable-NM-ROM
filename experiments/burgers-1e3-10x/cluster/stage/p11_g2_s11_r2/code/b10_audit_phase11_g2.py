#!/usr/bin/env python
"""Independent negative-aware audit for the immutable Phase-11 G2 cell."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re

import jax
import numpy as np

import b10_audit_phase9_train as a9
import b10_common as c
import b10_phase11_g2 as p11
import b10_phase9_train as p9
import b10_spline_train as legacy


def close(left, right, rtol=2e-13, atol=2e-14):
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol, equal_nan=False))


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def tree_exact(left, right):
    a = jax.tree_util.tree_leaves(left)
    b = jax.tree_util.tree_leaves(right)
    return bool(len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)))


def nested_close(left, right):
    """Audit-grade portable comparison for recomputed JSON metric trees."""
    if isinstance(left, dict) or isinstance(right, dict):
        return bool(isinstance(left, dict) and isinstance(right, dict)
                    and set(left) == set(right)
                    and all(nested_close(left[key], right[key]) for key in left))
    if left is None or right is None or isinstance(left, (bool, str)) or isinstance(right, (bool, str)):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=2e-13, atol=2e-14))
    return left == right


def trace_check(arrays, prefix, chosen):
    key = lambda name: np.asarray(arrays[f"{prefix}_{name}"])
    attempted = key("attempted").astype(bool)
    predicted = key("predicted")
    actual = key("actual")
    rho = key("rho")
    defined = key("rho_defined").astype(bool)
    accepted = key("accepted").astype(bool)
    finite = key("finite").astype(bool)
    breakdown = key("cg_breakdown").astype(bool)
    q = key("q")
    objective = key("objective")
    trial = key("trial_objective")
    step = key("step")
    terminated = key("terminated").astype(bool)
    delta = key("delta")
    damping = key("damping")
    active = key("active").astype(bool)
    iterations = key("cg_iterations")
    jvp = key("jvp_count")
    vjp = key("vjp_count")
    bound = key("bound_active_count")
    positive = attempted & (predicted > 0)
    nonpositive = attempted & (predicted <= 0)
    expected_accept = attempted & finite & ~breakdown & defined & (actual > 0) & (rho >= p9.ACCEPT_RHO)
    relative_objective = (objective[:, :, :-1]-trial)/np.maximum(objective[:, :, :-1], 1e-300)
    relative_step = np.linalg.norm(step, axis=-1)/(1+np.linalg.norm(q[:, :, :-1], axis=-1))
    expected_terminate = expected_accept & ((relative_objective <= 1e-12) | (relative_step <= 1e-12))
    shrink = (~accepted) | (defined & (rho < .25))
    expand = accepted & (rho > .75) & (np.linalg.norm(step, axis=-1) >= .9*delta[:, :, :-1])
    improve = accepted & (rho > .75)
    updated_delta = np.where(shrink, np.maximum(delta[:, :, :-1]/4, p9.DELTA_MIN),
                             np.where(expand, np.minimum(2*delta[:, :, :-1], p9.DELTA_MAX), delta[:, :, :-1]))
    updated_damping = np.where(shrink, np.minimum(10*damping[:, :, :-1], p9.LAMBDA_MAX),
                               np.where(improve, np.maximum(damping[:, :, :-1]/3, p9.LAMBDA_MIN), damping[:, :, :-1]))
    expected_delta = np.where(attempted, updated_delta, delta[:, :, :-1])
    expected_damping = np.where(attempted, updated_damping, damping[:, :, :-1])
    expected_q = np.where(accepted[..., None], q[:, :, :-1]+step, q[:, :, :-1])
    expected_objective = np.where(accepted, trial, objective[:, :, :-1])
    exhaustion = active[:, :, -1] & (delta[:, :, -1] <= p9.DELTA_MIN) & (damping[:, :, -1] >= p9.LAMBDA_MAX)
    checks = {
        "finite_attempt_work": bool(np.all(finite[attempted])
            and all(np.all(np.isfinite(x[attempted])) for x in
                    (predicted, actual, rho, trial, key("cg_relative")))
            and np.all(np.isfinite(step[attempted])) and np.all(np.isfinite(key("gradient")[attempted]))
            and np.all(np.isfinite(q)) and np.all(np.isfinite(objective))
            and np.all(np.isfinite(delta)) and np.all(np.isfinite(damping))),
        "rho_definition": bool(np.array_equal(defined[attempted], predicted[attempted] > 0)),
        "positive_rho": bool(close(rho[positive], actual[positive]/predicted[positive])) if np.any(positive) else True,
        "nonpositive_sentinel": bool(np.all(rho[nonpositive] == 0) and not np.any(accepted[nonpositive])),
        "no_breakdown": bool(not np.any(breakdown[attempted])),
        "attempt_active": bool(np.array_equal(attempted, active[:, :, :-1])),
        "acceptance": bool(np.array_equal(accepted, expected_accept)),
        "termination": bool(np.array_equal(terminated, expected_terminate)),
        "q_transition": close(q[:, :, 1:], expected_q),
        "objective_transition": close(objective[:, :, 1:], expected_objective),
        "delta_transition": close(delta[:, :, 1:], expected_delta),
        "damping_transition": close(damping[:, :, 1:], expected_damping),
        "active_transition": bool(np.array_equal(active[:, :, 1:], active[:, :, :-1] & ~terminated)),
        "work": bool(np.array_equal(jvp, np.where(attempted, iterations+1, 0))
                     and np.array_equal(vjp, np.where(attempted, iterations+1, 0))
                     and np.all(iterations[attempted] >= 0) and np.all(iterations[attempted] <= 32)),
        "inactive_no_work": bool(np.all(jvp[~attempted] == 0) and np.all(vjp[~attempted] == 0)),
        "initial_protocol": bool(np.all(delta[:, :, 0] == p9.DELTA0)
                                 and np.all(damping[:, :, 0] == p9.LAMBDA0)
                                 and np.all(active[:, :, 0])),
        "q_bounds": bool(np.all(np.abs(q) <= 1.0)),
        "bound_count": bool(np.array_equal(bound,
            np.where(attempted, np.sum(np.abs(q[:, :, 1:]) >= 1.0, axis=-1), 0))),
        "exhaustion": bool(np.array_equal(exhaustion, key("unhealthy_exhaustion").astype(bool)) and not np.any(exhaustion)),
    }
    if chosen:
        best = np.argmin(objective[:, :, -1], axis=0)
        terminal = np.where(best[:, None] == 0, q[0, :, -1], q[1, :, -1])
        checks["terminal_choice"] = bool(np.array_equal(best.astype(np.int8), key("chosen_start"))
                                          and close(terminal, key("chosen_terminal_q")))
    else:
        checks["one_start"] = bool(q.shape[0] == 1 and close(q[0, :, -1], np.asarray(arrays["train_trust_terminal_q"])))
    checks["pass"] = bool(all(checks.values()))
    return checks


def schedule_check(report, arrays, smoke):
    if not report["decision"]["updates_started"]:
        return bool(not report["history"] and "update_resolution_order" not in arrays)
    sizes = ({16: 4} if smoke else {64: 26112, 128: 6528, 256: 3264})
    for phase in p11.PHASES:
        epochs = 1 if smoke else p11.PHASES[phase]["epochs"]
        expected = ({f"{phase}_e1_N16": np.arange(4, dtype=np.int32)} if smoke
                    else p11.schedule_permutations(phase, epochs, sizes))
        for name, values in expected.items():
            if name not in arrays or not np.array_equal(np.asarray(arrays[name]), values):
                return False
    predictor = report["decision"]["predictor_trained"]
    expected_updates = 3 if smoke and predictor else 2 if smoke else p11.MAX_UPDATES if predictor else p11.GENERATOR_UPDATES
    expected_order = (np.asarray((16, 16, 16) if predictor else (16, 16), np.int16) if smoke
                      else np.tile(np.asarray((64, 128, 256), np.int16),
                                   (90 if predictor else 72)*p11.BATCHES_PER_N))
    if not np.array_equal(np.asarray(arrays["update_resolution_order"]), expected_order):
        return False
    if not np.array_equal(np.asarray(arrays["history_marker"]), np.asarray((expected_updates,), np.int64)):
        return False
    terminal = 0
    for phase in ("encoder", "joint") + (("predictor",) if predictor else ()):
        epochs = 1 if smoke else p11.PHASES[phase]["epochs"]
        stride = 1 if smoke else 9792
        for epoch in range(1, epochs+1):
            terminal += stride
            if report["history"].get(f"{phase}_epoch{epoch}", {}).get("terminal_update") != terminal:
                return False
    return bool(report["schedule"]["actual_total_update_count"] == expected_updates
                and report["schedule"]["max_total_update_count"] == (3 if smoke else p11.MAX_UPDATES)
                and report["schedule"]["zero_based_permutation_seed_offsets"] is True
                and report["schedule"]["terminal_only"] is True)


def recompute_metric(datasets, checkpoint, states, arrays, prefix, terminal):
    fresh = {}
    report = p9.evaluate_full(datasets, checkpoint["generator"], states,
        checkpoint["normalization"]["mean"], checkpoint["normalization"]["scales"],
        checkpoint["config"], fresh, prefix, terminal)
    exact = True
    for item in datasets:
        for suffix in ("numerator", "denominator", "trajectory", "boundary_count", "identity"):
            left = np.asarray(fresh[f"{prefix}_N{item['N']}_{suffix}"])
            right = np.asarray(arrays[f"{prefix}_N{item['N']}_{suffix}"])
            exact &= (np.array_equal(left, right) if suffix == "boundary_count" else close(left, right))
    return report, bool(exact)


def data_and_field_check(args, report, arrays, checkpoint, smoke):
    legacy.SMOKE = smoke
    if smoke:
        train = p9.p7train.smoke_datasets("train")
        coefficients = np.random.default_rng(92011).normal(0, .05, (4, 3328))
    else:
        train = legacy.load_mix(p11.TRAIN_MIX, "train")
        p5 = load_json(args.p5_json)
        coefficients = []
        next_index = 0
        target_ok = True
        manifest_rows = a9.manifest(args.manifest)
        for row in p5["train_targets"]["chunks"]:
            path = os.path.join(args.target_dir, row["basename"])
            target_ok &= bool(os.path.isfile(path) and c.sha256(path) == row["sha256"]
                              and manifest_rows.get("code/deps/p5/targets/"+row["basename"]) == row["sha256"])
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["coefficients"], np.float64).reshape(-1, 3328)
                indices = np.asarray(data["global_snapshot_index"], np.int64).reshape(-1)
                target_ok &= bool(np.array_equal(indices, np.arange(next_index, next_index+len(values)))
                                  and np.all(data["healthy"]) and np.all(data["boundary"])
                                  and np.max(data["normal"]) <= 1e-8 and np.max(data["pou"]) == 0
                                  and np.max(data["support"]) == 32)
                coefficients.append(values)
                next_index += len(values)
        coefficients = np.concatenate(coefficients)
        target_ok &= next_index == 35904
    features = legacy.concatenate(train, "features")
    norm = p9.train_normalization(coefficients, features)
    affine = legacy.concatenate(train, "affine")
    handoff = p9.encode_batches(checkpoint["encoder"], norm["normalized"], checkpoint["config"])
    state_binding = bool(np.array_equal(handoff, checkpoint["encoder_handoff_q_raw"])
        and np.array_equal(handoff, arrays["encoder_handoff_q_raw"])
        and np.array_equal(checkpoint["q_raw"], arrays["final_q_raw"])
        and np.array_equal(checkpoint["globalized_q"], arrays["train_trust_terminal_q"])
        and np.array_equal(checkpoint["final_target_states"],
                           np.concatenate((affine, checkpoint["globalized_q"]), axis=1))
        and np.array_equal(np.asarray(arrays["train_trust_q"])[0, :, 0],
                           np.tanh(np.asarray(checkpoint["q_raw"]))))
    normalization = bool(np.array_equal(norm["mean"], checkpoint["normalization"]["mean"])
        and np.array_equal(norm["scales"], checkpoint["normalization"]["scales"])
        and np.array_equal(norm["feature_mean"], checkpoint["normalization"]["feature_mean"])
        and np.array_equal(norm["feature_scale"], checkpoint["normalization"]["feature_scale"])
        and np.array_equal(norm["mean"], arrays["coefficient_mean"])
        and np.array_equal(norm["scales"], arrays["head_scales"])
        and np.array_equal(features, arrays["training_features"])
        and p9.array_sha(norm["normalized"]) == report["normalization"]["normalized_training_coefficients_sha256"])
    terminal_states = np.concatenate((affine, np.tanh(np.asarray(checkpoint["q_raw"]))), axis=1)
    terminal, terminal_exact = recompute_metric(train, checkpoint, terminal_states, arrays, "terminal_train", not smoke)
    global_states = np.concatenate((affine, np.asarray(checkpoint["globalized_q"])), axis=1)
    globalized, global_exact = recompute_metric(train, checkpoint, global_states, arrays, "globalized_train", not smoke)
    initial_objective = np.concatenate([
        np.asarray(arrays[f"terminal_train_N{item['N']}_numerator"])
        / np.maximum(np.asarray(arrays[f"terminal_train_N{item['N']}_denominator"]), 1e-300)
        for item in train])
    state_binding &= close(np.asarray(arrays["train_trust_objective"])[0, :, 0], initial_objective)
    metric_match = bool(nested_close(terminal, report["terminal_train_control"])
                        and nested_close(globalized, report["globalized_train"]))
    direct_exact = True
    if report["decision"]["predictor_trained"]:
        folded = legacy.fold_predictor_standardization(checkpoint["predictor"], norm["feature_mean"], norm["feature_scale"])
        direct_states = p9.predictor_batches(folded, features)
        direct, direct_exact = recompute_metric(train, checkpoint, direct_states, arrays, "train_direct", not smoke)
        direct_exact &= bool(nested_close(direct, report["train_direct"])
                             and tree_exact(folded, checkpoint["folded_predictor"])
                             and np.array_equal(direct_states, arrays["train_direct_states"]))
    selection_exact = True
    if report["decision"]["selection_evaluated"]:
        selection = p9.p7train.smoke_datasets("selection") if smoke else legacy.load_mix(p11.SELECTION_MIX, "selection")
        direct_states = np.asarray(arrays["selection_direct_states"])
        chosen_q = np.asarray(arrays["selection_trust_chosen_terminal_q"])
        oracle_states = np.concatenate((legacy.concatenate(selection, "affine"), chosen_q), axis=1)
        direct, direct_ok = recompute_metric(selection, checkpoint, direct_states, arrays, "selection_direct", not smoke)
        oracle, oracle_ok = recompute_metric(selection, checkpoint, oracle_states, arrays, "selection_oracle", not smoke)
        selection_exact = bool(direct_ok and oracle_ok
                               and nested_close(direct, report["selection"]["direct"])
                               and nested_close(oracle, report["selection"]["oracle"]))
    metadata = legacy.metadata(train) == report["data"]["training"]
    result = {"target_files": bool(True if smoke else target_ok), "normalization": normalization,
              "metadata": metadata, "state_handoff": state_binding,
              "terminal_exact": terminal_exact,
              "globalized_exact": global_exact, "reported_metrics": metric_match,
              "train_direct_exact": bool(direct_exact), "selection_exact": selection_exact}
    result["pass"] = bool(all(result.values()))
    return result


def preflight_check(report, smoke):
    preflight = report["preflight"]
    if not report["decision"]["updates_started"]:
        return bool(preflight.get("weights_bitwise_unchanged") and not preflight.get("proceed_before_update1"))
    medians = preflight["median_seconds"]
    if smoke:
        expected_terms = {key: 0.0 for key in preflight["projected_terms_seconds"]}
        return bool(preflight["projected_terms_seconds"] == expected_terms
                    and preflight["projected_remaining_seconds"] == 0
                    and preflight["required_with_safety_seconds"] == 0
                    and preflight["proceed_before_update1"]
                    and preflight["weights_bitwise_unchanged"])
    train_sizes = {64: 26112, 128: 6528, 256: 3264}
    selection_sizes = {64: 3264, 128: 1632, 256: 816}
    train_eval = sum(int(np.ceil(train_sizes[n]/8))*medians[f"evaluation_N{n}"] for n in p11.N_ORDER)
    selection_eval = sum(int(np.ceil(selection_sizes[n]/8))*medians[f"evaluation_N{n}"] for n in p11.N_ORDER)
    counts = p11.EVALUATION_COUNTS
    terms = {
        "pre_gate_updates": sum(medians[f"{phase}_N{n}"]*p11.BATCHES_PER_N*p11.PHASES[phase]["epochs"] for phase in ("encoder", "joint") for n in p11.N_ORDER),
        "conditional_predictor_updates": sum(medians[f"predictor_N{n}"]*p11.BATCHES_PER_N*p11.PHASES["predictor"]["epochs"] for n in p11.N_ORDER),
        "pre_gate_train_evaluations": (counts["pre_gate_epoch_cox_cohorts"]+counts["terminal_control_cox_cohorts"]+counts["globalized_train_cox_cohorts"]+counts["terminal_train_k3_equivalent_cohorts"])*train_eval,
        "conditional_train_evaluations": (counts["conditional_predictor_epoch_cox_cohorts"]+counts["conditional_predictor_terminal_cox_cohorts"]+counts["conditional_predictor_terminal_k3_equivalent_cohorts"])*train_eval,
        "train_globalization": p11.TRUST_ATTEMPTS*sum(int(np.ceil(train_sizes[n]/p11.BATCH_BY_N[n]))*medians[f"trust_N{n}"] for n in p11.N_ORDER),
        "conditional_selection_evaluations": (counts["conditional_selection_initial_cox_cohorts"]+counts["conditional_selection_terminal_cox_cohorts"]+counts["conditional_selection_terminal_k3_equivalent_cohorts"])*selection_eval,
        "conditional_selection_globalization": 2*p11.TRUST_ATTEMPTS*sum(int(np.ceil(selection_sizes[n]/p11.BATCH_BY_N[n]))*medians[f"trust_N{n}"] for n in p11.N_ORDER),
        "audit_regeneration": preflight["projected_terms_seconds"]["audit_regeneration"],
        "fixed_io_checkpoint_audit": p11.FINAL_IO_AUDIT_RESERVE_SECONDS,
    }
    projected = sum(terms.values())
    remaining = max(0.0, preflight["allocation_seconds"]-preflight["actual_elapsed_at_decision_seconds"])
    raw = preflight["raw_seconds"]
    median_check = all(close(medians[key], np.median(raw[key])) for key in raw)
    return bool(preflight["evaluation_counts"] == counts
        and all(close(terms[key], preflight["projected_terms_seconds"][key]) for key in terms)
        and close(projected, preflight["projected_remaining_seconds"])
        and close(remaining, preflight["actual_remaining_at_decision_seconds"])
        and close(1.15*projected, preflight["required_with_safety_seconds"])
        and (1.15*projected <= remaining) == preflight["proceed_before_update1"]
        and preflight["weights_bitwise_unchanged"]
        and preflight["before_hashes"] == preflight["after_hashes"]
        and 0 <= terms["audit_regeneration"] <= preflight["actual_elapsed_at_decision_seconds"]
        and preflight["allocation_seconds"] == 57600 and preflight["safety_factor"] == 1.15
        and set(preflight["raw_seconds"]) == {f"{kind}_N{n}" for kind in ("encoder", "joint", "predictor", "evaluation", "trust") for n in p11.N_ORDER}
        and median_check
        and all(len(values) == 10 and np.all(np.isfinite(values)) and np.all(np.asarray(values) > 0)
                for values in preflight["raw_seconds"].values()))


def structural_check(report, smoke):
    panel = report["structural_preflight"]
    if smoke:
        if panel.get("separately_smoked") is True:
            return True
        identity = panel.get("same_invocation_identity", {})
        return bool(panel.get("scientific") is False and identity.get("finite")
                    and identity.get("boundary") and identity.get("relative_l2", np.inf) <= p11.IDENTITY_TOL
                    and panel.get("compiled_device_bytes", np.inf) <= 20_000_000_000)
    records = panel.get("records", {})
    if set(records) != {"fom", "rom"}:
        return False
    per_case = {method: [float(np.median([row["elapsed_s"] for row in records[method] if row["case_index"] == case])) for case in range(4)] for method in records}
    speed = float(np.median(per_case["fom"])/np.median(per_case["rom"]))
    interval = p9.base.clustered_speedup_ci(per_case["fom"], per_case["rom"], 20266100)
    positions = {method: [sum(row["position"] == position for row in records[method])//4 for position in (0, 1)] for method in records}
    summaries = {method: p9.base.summarize_timing(rows, 4) for method, rows in records.items()}
    fom = bool(all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
                   and row["max_returned_relative_residual"] <= p9.base.FOM_OUTER for row in records["fom"])
               and np.mean([row["trajectory_relative_l2"] for row in records["fom"]]) <= 1e-3
               and np.max([row["trajectory_relative_l2"] for row in records["fom"]]) <= 3e-3)
    rom = bool(all(row["identity_relative_l2"] <= p11.IDENTITY_TOL and row["exact_boundary"]
                   and row["finite"] and row["cold_recovery_finite"] and row["cold_sample_count"] <= 4096
                   and row["cox_weak_evaluations"] == 50
                   and row["k3_coefficient_grid_full_field_evaluations"] == 51
                   and row["weak_jacobian_evaluations"] == 0 and row["trial_evaluations"] == 0
                   and row["failures"] == 0 for row in records["rom"]))
    gate = bool(fom and rom and panel["compiled_device_bytes"] <= 20_000_000_000
                and positions == {"fom": [12, 12], "rom": [12, 12]}
                and panel["burn_count"] > 0 and speed >= 10 and interval[0] >= 8)
    return bool(panel["fom_eligible"] == fom and panel["pass"] == gate
                and panel["position_counts"] == positions and panel["summaries"] == summaries
                and close(panel["paired_median_speedup"], speed)
                and close(panel["clustered_speedup_ci"], interval))


def optimizer_states_finite(checkpoint):
    states = checkpoint.get("optimizer_states", {})
    if set(states) != {"encoder", "joint", "predictor"}:
        return False
    return bool(a9.optimizer_state_check(checkpoint)
                and all(jax.tree_util.tree_leaves(states[name])
                    and all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(states[name]))
                    for name in states))


def work_checkpoint_check(args, report, arrays, checkpoint, smoke):
    row = report.get("work_checkpoint")
    if not report["decision"]["updates_started"]:
        return row is None
    if row is None:
        return False
    path = os.path.join(os.path.dirname(args.source_json), row["basename"])
    if not os.path.isfile(path) or c.sha256(path) != row["sha256"]:
        return False
    with open(path, "rb") as handle:
        work = pickle.load(handle)
    expected_phase = "predictor" if report["decision"]["predictor_trained"] else "globalized_train"
    expected_epoch = (1 if smoke else 18) if expected_phase == "predictor" else (p11.SMOKE_TRUST_ATTEMPTS if smoke else 40)
    expected_update = report["schedule"]["actual_total_update_count"]
    return bool(work.get("phase") == expected_phase and work.get("epoch") == expected_epoch
        and work.get("global_update") == expected_update
        and tree_exact(work.get("generator"), checkpoint["generator"])
        and tree_exact(work.get("encoder"), checkpoint["encoder"])
        and tree_exact(work.get("predictor"), checkpoint["predictor"])
        and tree_exact(work.get("folded_predictor"), checkpoint["folded_predictor"])
        and np.array_equal(np.asarray(work.get("q_raw")), np.asarray(checkpoint["q_raw"]))
        and np.array_equal(np.asarray(work.get("globalized_q")), np.asarray(checkpoint["globalized_q"]))
        and np.array_equal(np.asarray(work.get("final_target_states")), np.asarray(checkpoint["final_target_states"]))
        and np.array_equal(np.asarray(work.get("evaluation_states")),
                           np.asarray(arrays["train_direct_states"] if report["decision"]["predictor_trained"]
                                      else checkpoint["final_target_states"]))
        and tree_exact(work.get("optimizer_states"), checkpoint["optimizer_states"])
        and optimizer_states_finite(checkpoint))


def decision_check(report, smoke):
    decision = report["decision"]
    if not decision["updates_started"]:
        result = {"pre_update_stop": not report["preflight"].get("proceed_before_update1"),
                  "selection_sealed": not decision["selection_evaluated"]}
        result["pass"] = bool(all(result.values()))
        return result
    train = bool(p9.gate(report["globalized_train"], 2e-4, 7e-4, not smoke)
                 and p9.trust_health_gate(report["train_trust_health"]))
    predictor = decision["predictor_trained"]
    direct = bool(predictor and report.get("train_direct") is not None
                  and p9.gate(report["train_direct"], 3e-4, 1e-3, not smoke))
    selection = False
    if decision["selection_evaluated"] and report.get("selection") is not None:
        row = report["selection"]
        selection = bool((p9.gate(row["direct"], 3e-4, 1e-3, not smoke)
                          and p9.gate(row["oracle"], 2e-4, 7e-4, not smoke))
                         and all(value <= 1.5 for value in row["direct_oracle_mean_ratio"].values())
                         and row["trust_gate"])
    checks = {"globalized_train": decision["globalized_train_pass"] == train,
              "predictor_conditional": predictor == train,
              "direct_train": decision["train_direct_pass"] == direct,
              "selection_conditional": decision["selection_evaluated"] == direct,
              "phase11": decision["phase11_pass"] == selection,
              "promotion": decision["scientific_promotion_allowed"] == selection,
              "no_capacity": decision["retracted_capacity_used"] is False,
              "no_third_architecture": decision["third_architecture_licensed"] is False}
    checks["pass"] = bool(all(checks.values()))
    return checks


def negative_self_test():
    valid = {"predicted": np.asarray((1.0, 0.0, -1.0)), "actual": np.asarray((.5, .2, .1)),
             "rho": np.asarray((.5, 0.0, 0.0)), "defined": np.asarray((True, False, False)),
             "accepted": np.asarray((True, False, False)), "finite": np.asarray((True, True, True))}
    def passes(row):
        positive = row["predicted"] > 0
        return bool(np.array_equal(row["defined"], positive)
                    and close(row["rho"][positive], row["actual"][positive]/row["predicted"][positive])
                    and np.all(row["rho"][~positive] == 0) and not np.any(row["accepted"][~positive])
                    and np.all(row["finite"]) and all(np.all(np.isfinite(row[key])) for key in ("predicted", "actual", "rho")))
    corruptions = []
    for key, value in (("rho", np.asarray((.5, 1., 0.))), ("defined", np.asarray((True, True, False))),
                       ("accepted", np.asarray((True, True, False))), ("predicted", np.asarray((1., np.nan, -1.))),
                       ("finite", np.asarray((True, False, True)))):
        row = {name: item.copy() for name, item in valid.items()}
        row[key] = value
        corruptions.append(passes(row))
    result = {"positive_contract": passes(valid), "five_corruptions_rejected": not any(corruptions)}
    result["pass"] = bool(all(result.values()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--p5-json")
    parser.add_argument("--target-dir")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--slurm-out", required=True)
    parser.add_argument("--slurm-err", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = load_json(args.source_json)
    arrays = np.load(args.source_npz, allow_pickle=False)
    with open(args.checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    rows = a9.manifest(args.manifest)
    with open(args.slurm_out, encoding="utf-8") as handle:
        stdout = handle.read()
    with open(args.slurm_err, encoding="utf-8") as handle:
        stderr = handle.read()
    provenance = {
        "commit": report["provenance"].get("commit") == args.expected_commit,
        "job": str(report["provenance"].get("slurm_job_id")) == str(args.expected_job),
        "gpu": report["provenance"].get("jax_backend") == "gpu"
               and (args.smoke or report["provenance"].get("gpu_kind") == "NVIDIA H200"),
        "precision": report["provenance"].get("x64") is True
                     and report["provenance"].get("matmul_precision") == "highest",
        "logs": (args.smoke or "jax_backend=gpu" in stdout) and "ALL-DONE" in stdout
                and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)", stdout+stderr),
    }
    source_binding = bool(args.smoke or all(
        rows.get("code/"+name) == digest
        for name, digest in report["provenance"].get("source_sha256", {}).items()))
    file_binding = bool(args.smoke or (source_binding
        and rows.get("code/b10_phase11_g2.py") is not None
        and rows.get("code/b10_audit_phase11_g2.py") is not None
        and rows.get("code/PHASE-11-PRE-REGISTRATION.md") == c.sha256(args.prereg)
        and report["npz"]["sha256"] == c.sha256(args.source_npz)
        and report["checkpoint"]["sha256"] == c.sha256(args.checkpoint)))
    dependency_binding = True
    if not args.smoke:
        expected = {key: value["sha256"] for key, value in report["bindings"].items()
                    if isinstance(value, dict) and "sha256" in value}
        phase_stems = {"p4": "phase4_d", "p5": "phase5_d", "p6": "phase6_d",
                       "p7": "phase7_train", "p8": "phase8_d"}
        for phase, stem in phase_stems.items():
            kinds = ("json", "npz", "audit", "manifest") + (("checkpoint",) if phase == "p7" else ())
            for kind in kinds:
                basename = ("AUDIT.json" if kind == "audit" else "MANIFEST.sha256" if kind == "manifest"
                            else "checkpoint.pkl" if kind == "checkpoint" else f"{stem}.{kind}")
                dependency_binding &= rows.get(f"code/deps/{phase}/{basename}") == expected[f"{phase}_{kind}"]
        extra_names = {
            "p9_json": "phase9_terminal_recovery.json", "p9_npz": "phase9_terminal_recovery.npz",
            "p9_checkpoint": "checkpoint.pkl", "p9_audit": "AUDIT.json",
            "p9_audit_work": "AUDIT-WORK.npz", "p9_work_checkpoint": "work_checkpoint.pkl",
            "p9_manifest": "MANIFEST.sha256", "p10_json": "phase10_d.json",
            "p10_npz": "phase10_d.npz", "p10_work_checkpoint": "work_checkpoint.pkl",
            "p10_audit": "AUDIT.json", "p10_manifest": "MANIFEST.sha256",
            "p10_audit_manifest": "AUDIT-MANIFEST.sha256",
        }
        for key, basename in extra_names.items():
            group = "p9" if key.startswith("p9_") else "p10"
            dependency_binding &= rows.get(f"code/deps/{group}/{basename}") == expected[key]
        dependency_binding &= report["bindings"]["phase11_independent_license"] == {
            "phase10_promotion_used": False, "retracted_capacity_used": False}
    counts = {"generator": p9.tree_count(checkpoint["generator"]),
              "encoder": p9.tree_count(checkpoint["encoder"]),
              "predictor": p9.tree_count(checkpoint["predictor"])}
    parameter_counts = counts == report["parameter_counts"]["reported"] == report["parameter_counts"]["expected"] == {
        "generator": 165954, "encoder": 164384, "predictor": 2533}
    schedule = schedule_check(report, arrays, args.smoke)
    metrics = True
    for name, row in report.get("history", {}).items():
        metrics &= a9.metric_match(row["metrics"], arrays, name, False)
    for report_key, prefix in (("terminal_train_control", "terminal_train"),
                               ("globalized_train", "globalized_train"),
                               ("train_direct", "train_direct")):
        if report.get(report_key) is not None:
            metrics &= a9.metric_match(report[report_key], arrays, prefix, not args.smoke)
    if report.get("selection") is not None:
        metrics &= a9.metric_match(report["selection"]["direct"], arrays, "selection_direct", not args.smoke)
        metrics &= a9.metric_match(report["selection"]["oracle"], arrays, "selection_oracle", not args.smoke)
    train_trace = ({"pass": True, "present": False} if "train_trust_attempted" not in arrays
                   else trace_check(arrays, "train_trust", False))
    selection_trace = ({"pass": True, "present": False} if "selection_trust_attempted" not in arrays
                       else trace_check(arrays, "selection_trust", True))
    data_fields = ({"pass": True, "skipped": "pre-update stop"} if not report["decision"]["updates_started"]
                   else data_and_field_check(args, report, arrays, checkpoint, args.smoke))
    preflight = preflight_check(report, args.smoke)
    structural = structural_check(report, args.smoke)
    work = work_checkpoint_check(args, report, arrays, checkpoint, args.smoke)
    decision = decision_check(report, args.smoke)
    information = bool(report["information_boundary"] == {
        "train_only_weights": True,
        "selection_loaded_after_globalized_and_direct_train_pass": bool(not report["decision"]["selection_evaluated"] or (report["decision"]["globalized_train_pass"] and report["decision"]["train_direct_pass"])),
        "selection_target_coefficients_training_use": False,
        "model_validation_touched": False, "confirmation_touched": False,
        "weak_eq_fitting_touched": False, "scaling_touched": False,
        "retracted_capacity_touched": False})
    fold = bool(not report["decision"]["predictor_trained"] or
                (np.isfinite(report["predictor_fold_identity"]) and report["predictor_fold_identity"] <= 1e-12))
    negative = negative_self_test()
    health = bool(all(provenance.values()) and file_binding and dependency_binding and parameter_counts
                  and schedule and metrics and train_trace["pass"] and selection_trace["pass"]
                  and data_fields["pass"] and preflight and structural and work and decision["pass"]
                  and information and fold and negative["pass"])
    result = {
        "status": "pass" if health else "fail", "negative_aware": True,
        "source_json_sha256": c.sha256(args.source_json),
        "source_npz_sha256": c.sha256(args.source_npz),
        "checkpoint_sha256": c.sha256(args.checkpoint),
        "manifest_sha256": c.sha256(args.manifest),
        "expected_commit": args.expected_commit, "expected_job": str(args.expected_job),
        "checks": {"provenance": provenance, "file_binding": file_binding,
            "dependency_binding": bool(dependency_binding), "parameter_counts": parameter_counts,
            "schedule": schedule, "metrics": bool(metrics), "train_trust": train_trace,
            "selection_trust": selection_trace, "data_and_full_fields": data_fields,
            "runtime_preflight": preflight, "structural_preflight": structural,
            "work_checkpoint": work, "decision": decision,
            "information_boundary": information, "predictor_fold": fold,
            "negative_self_test": negative},
        "decision": report["decision"],
    }
    p9.atomic_json(args.output, result)
    if not health:
        raise SystemExit("Phase11 independent audit failed")
    print(json.dumps({"status": "pass", "decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
