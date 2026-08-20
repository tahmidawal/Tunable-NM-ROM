"""Direct transported-HG predictor with zero/one bounded weak correction.

This conditional cell compares the direct prediction, one full-grid weak LSPG
correction, and one decoder-output-NNLS EQ correction on identical trajectories.
It never opens model-validation or confirmation data.

Usage: b10_weak_eq.py CHECKPOINT OUTPUT_JSON OUTPUT_NPZ
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

import b10_common as c

CHECKPOINT, OUTPUT_JSON, OUTPUT_NPZ = sys.argv[1:4]
N = int(os.environ.get("N", "64"))
EVAL_SEED = int(os.environ.get("EVAL_SEED", "0"))
EVAL_DRAW_COUNT = int(os.environ.get("EVAL_DRAW_COUNT", "704"))
EVAL_START = int(os.environ.get("EVAL_START", "512"))
EVAL_CASES = int(os.environ.get("EVAL_CASES", "8"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(c.NUM_STEPS)))
M = int(os.environ.get("M", "128"))
EQ_M = int(os.environ.get("EQ_M", str(4 * M)))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "8"))
EQ_POOL_LIMIT = int(os.environ.get("EQ_POOL_LIMIT", "4096"))
EQ_SEED = int(os.environ.get("EQ_SEED", "4321"))
TRUST_RADIUS = float(os.environ.get("TRUST_RADIUS", "0.25"))
LM_DAMPING = float(os.environ.get("LM_DAMPING", "1e-6"))
TIME_REPS = int(os.environ.get("TIME_REPS", "12"))
TIME_WARM = int(os.environ.get("TIME_WARM", "1"))
BURN_SECONDS = float(os.environ.get("BURN_SECONDS", "3"))
SMOKE = os.environ.get("SMOKE", "0") == "1"


def interior_indices(n):
    ii, jj = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1), indexing="ij")
    return (ii * n + jj).reshape(-1)


def stencil_indices(indices, n):
    return np.stack((indices, indices + n, indices - n, indices + 1, indices - 1), axis=1)


def test_modes(n, count):
    dx = 1.0 / (n - 1)
    kk = np.arange(1, n - 1)
    kx_grid, ky_grid = np.meshgrid(kk, kk, indexing="ij")
    eigenvalues = (4.0 / dx**2) * (
        np.sin(np.pi * kx_grid / (2 * (n - 1))) ** 2
        + np.sin(np.pi * ky_grid / (2 * (n - 1))) ** 2
    )
    order = np.argsort(eigenvalues.reshape(-1), kind="stable")[:count]
    kx = kx_grid.reshape(-1)[order]
    ky = ky_grid.reshape(-1)[order]
    xi = kk / (n - 1)
    sx = np.sin(np.pi * np.outer(xi, kx))
    sy = np.sin(np.pi * np.outer(xi, ky))
    phi = (sx[:, None, :] * sy[None, :, :]).reshape(-1, count)
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)
    return phi.astype(np.float64), eigenvalues.reshape(-1)[order].astype(np.float64)


def upwind_adv_field(values, n):
    ni = n - 2
    dx = 1.0 / (n - 1)
    field = jnp.pad(values.reshape(ni, ni), 1)
    center = field[1:-1, 1:-1]
    ux = jnp.where(
        center > 0.0,
        (center - field[:-2, 1:-1]) / dx,
        (field[2:, 1:-1] - center) / dx,
    )
    uy = jnp.where(
        center > 0.0,
        (center - field[1:-1, :-2]) / dx,
        (field[1:-1, 2:] - center) / dx,
    )
    return (center * (ux + uy)).reshape(-1)


def apply_predictor(params, features):
    hidden = features
    for layer in params[:-1]:
        hidden = jax.nn.swish(hidden @ layer["W"] + layer["b"])
    return hidden @ params[-1]["W"] + params[-1]["b"]


def decode_points(state, coords, mask, degree):
    return c.basis_from_warp(
        coords, mask, state[:2], jnp.exp(state[2:4]), degree
    ) @ state[4:]


def exact_initial_state(recovered, degree):
    state = np.zeros(4 + len(c.hermite_pairs(degree)), np.float64)
    state[:2] = recovered[:2]
    state[2:4] = np.log(recovered[2])
    state[4] = recovered[3]
    return state


def deterministic_candidate_indices(n, limit):
    interior = interior_indices(n)
    if interior.size <= limit:
        return interior, "all interior target-grid stencil centers"
    side = int(np.floor(np.sqrt(limit)))
    axis = np.unique(np.rint(np.linspace(1, n - 2, side)).astype(int))
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    return np.sort((ii * n + jj).reshape(-1)), f"deterministic {axis.size}x{axis.size} target-grid lattice"


def nnls_capped(matrix, target, max_support, tol=1e-10, inner_max=200):
    """Audited Lawson-Hanson active-set NNLS with an ECSW support cap."""
    n = matrix.shape[1]
    weights = np.zeros(n, np.float64)
    passive = np.zeros(n, bool)
    residual = target.copy()
    outer = 0
    while outer < 5 * max_support + 10:
        gradient = matrix.T @ residual
        candidates = np.where(~passive)[0]
        if candidates.size == 0 or passive.sum() >= max_support:
            break
        add = candidates[np.argmax(gradient[candidates])]
        if gradient[add] <= tol * (np.linalg.norm(target) + 1e-300):
            break
        passive[add] = True
        outer += 1
        for _ in range(inner_max):
            selected = np.where(passive)[0]
            solution, *_ = np.linalg.lstsq(matrix[:, selected], target, rcond=None)
            if np.all(solution > 0.0):
                weights[:] = 0.0
                weights[selected] = solution
                break
            nonpositive = solution <= 0.0
            alpha = np.min(
                weights[selected][nonpositive]
                / (weights[selected][nonpositive] - solution[nonpositive] + 1e-300)
            )
            weights[selected] += alpha * (solution - weights[selected])
            passive[selected[weights[selected] <= 1e-14]] = False
            weights[~passive] = 0.0
        residual = target - matrix @ weights
    return weights, float(np.linalg.norm(residual)), outer


def predictor_states(checkpoint, parameters, n):
    features = c.trajectory_features(parameters, n).reshape(-1, 7)
    normalized = (features - checkpoint["feature_mean"]) / checkpoint["feature_scale"]
    predict = jax.jit(lambda value: apply_predictor(checkpoint["params"], value))
    states = np.asarray(predict(jnp.asarray(normalized, jnp.float64)))
    return states * checkpoint["state_scale"] + checkpoint["state_mean"]


def eq_training_states(checkpoint, degree):
    cx, cy, width, amplitude, nu, normalized = c.bf.sample_params(
        seed=0, m=EVAL_DRAW_COUNT
    )
    count = min(512, EVAL_DRAW_COUNT)
    parameters = {
        "cx": np.asarray(cx[:count]), "cy": np.asarray(cy[:count]),
        "width": np.asarray(width[:count]), "amplitude": np.asarray(amplitude[:count]),
        "nu": np.asarray(nu[:count]), "normalized": np.asarray(normalized[:count]),
    }
    states = predictor_states(checkpoint, parameters, N).reshape(count, c.NUM_STEPS + 1, -1)
    # Deterministic seeded subsample of decoder outputs; no residual snapshots.
    rng = np.random.default_rng(EQ_SEED)
    choose = np.sort(rng.choice(states.shape[0] * states.shape[1], EQ_SNAPS, replace=False))
    selected = states.reshape(-1, states.shape[-1])[choose]
    if selected.shape[1] != 4 + len(c.hermite_pairs(degree)):
        raise SystemExit("EQ state dimension does not match decoder architecture")
    return selected, choose


def fit_eq_weights(checkpoint, degree):
    started = time.perf_counter()
    phi, _ = test_modes(N, M)
    interior = interior_indices(N)
    candidates, pool_rule = deterministic_candidate_indices(N, EQ_POOL_LIMIT)
    if candidates.size < EQ_M:
        raise SystemExit("EQ candidate pool is smaller than m")
    positions = np.searchsorted(interior, candidates)
    if not np.all(interior[positions] == candidates):
        raise SystemExit("EQ candidate is not an interior target-grid node")
    coords = jnp.asarray(c.grid_coords(N), jnp.float64)
    mask = jnp.asarray(c.binary_boundary_mask(N), jnp.float64)
    coords_interior = coords[jnp.asarray(interior)]
    states, state_indices = eq_training_states(checkpoint, degree)
    decode_interior = jax.jit(lambda state: decode_points(
        state, coords_interior, jnp.ones(interior.size, jnp.float64), degree
    ))
    matrices = []
    targets = []
    for state in states:
        values = np.asarray(decode_interior(jnp.asarray(state, jnp.float64)))
        advection = np.asarray(upwind_adv_field(jnp.asarray(values), N))
        for field in (values, advection):
            targets.append(phi.T @ field)
            matrices.append(phi[positions].T * field[positions][None, :])
    matrix = np.concatenate(matrices, axis=0)
    target = np.concatenate(targets)
    row_scale = np.linalg.norm(matrix, axis=1) + 1e-300
    scaled_matrix = matrix / row_scale[:, None]
    scaled_target = target / row_scale
    weights, capped_norm, outer = nnls_capped(
        scaled_matrix, scaled_target, max_support=EQ_M
    )
    support = np.nonzero(weights > 0.0)[0]
    if support.size >= EQ_M:
        keep = support[np.argsort(-weights[support])[:EQ_M]]
    else:
        remaining = np.setdiff1d(np.arange(candidates.size), support)
        score = np.mean(np.abs(scaled_matrix), axis=0)
        padding = remaining[np.argsort(-score[remaining])[:EQ_M - support.size]]
        keep = np.concatenate((support, padding))
    final_weights, _, final_outer = nnls_capped(
        scaled_matrix[:, keep], scaled_target, max_support=keep.size
    )
    positive_floor = 1e-8 * max(float(np.max(final_weights)), 1e-300)
    final_weights = np.where(final_weights > 0.0, final_weights, positive_floor)
    row_residual = scaled_matrix[:, keep] @ final_weights - scaled_target
    row_relative = np.abs(row_residual) / (np.abs(scaled_target) + 1e-300)
    info = {
        "fit_source": "direct-predictor decoder-output snapshots (u and FOM-exact upwind N(u))",
        "snapshot_flat_indices": state_indices.tolist(),
        "snapshot_count": int(states.shape[0]),
        "candidate_pool_rule": pool_rule,
        "candidate_count": int(candidates.size),
        "M": M, "m": int(keep.size), "support_before_padding": int(support.size),
        "padded": int(keep.size - support.size), "nnls_outer_capped": int(outer),
        "nnls_outer_refit": int(final_outer), "capped_residual_norm": capped_norm,
        "relative_fit_l2": float(np.linalg.norm(row_residual) / np.linalg.norm(scaled_target)),
        "row_relative_median": float(np.median(row_relative)),
        "row_relative_p95": float(np.quantile(row_relative, 0.95)),
        "row_relative_worst": float(np.max(row_relative)),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return {"idx": candidates[keep], "w": final_weights, "info": info}


def make_weak_corrector(degree, checkpoint, collocation):
    phi, eigenvalues = test_modes(N, M)
    interior = interior_indices(N)
    indices = np.asarray(collocation["idx"], np.int64)
    positions = np.searchsorted(interior, indices)
    weights = np.asarray(collocation["w"], np.float64)
    phi_weighted = jnp.asarray(phi[positions] * weights[:, None], jnp.float64)
    stencil = stencil_indices(indices, N)
    coords = jnp.asarray(c.grid_coords(N), jnp.float64)
    mask = jnp.asarray(c.binary_boundary_mask(N), jnp.float64)
    stencil_coords = coords[jnp.asarray(stencil.reshape(-1))]
    stencil_mask = mask[jnp.asarray(stencil.reshape(-1))]
    center_coords = coords[jnp.asarray(indices)]
    center_mask = mask[jnp.asarray(indices)]
    state_mean = jnp.asarray(checkpoint["state_mean"], jnp.float64)
    state_scale = jnp.asarray(checkpoint["state_scale"], jnp.float64)
    eigenvalues = jnp.asarray(eigenvalues, jnp.float64)

    def centers(state):
        return decode_points(state, center_coords, center_mask, degree)

    def residual_normalized(normalized_state, previous_state, viscosity):
        state = normalized_state * state_scale + state_mean
        stencil_values = decode_points(
            state, stencil_coords, stencil_mask, degree
        ).reshape(indices.size, 5)
        center, xp, xm, yp, ym = (
            stencil_values[:, 0], stencil_values[:, 1], stencil_values[:, 2],
            stencil_values[:, 3], stencil_values[:, 4]
        )
        dx = 1.0 / (N - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        previous = centers(previous_state)
        projected_u = phi_weighted.T @ center
        weight_modes = (1.0 + c.DT * viscosity * eigenvalues) ** -1.0
        return weight_modes * (
            phi_weighted.T @ (center - previous)
            + c.DT * (phi_weighted.T @ advection + viscosity * eigenvalues * projected_u)
        )

    def one_update(predicted_state, previous_state, viscosity):
        normalized = (predicted_state - state_mean) / state_scale
        weak_residual = residual_normalized(normalized, previous_state, viscosity)
        weak_jacobian = jax.jacfwd(residual_normalized)(
            normalized, previous_state, viscosity
        )
        hessian = weak_jacobian.T @ weak_jacobian
        gradient = weak_jacobian.T @ weak_residual
        diagonal = jnp.diag(jnp.diag(hessian)) + jnp.eye(normalized.size, dtype=jnp.float64) * 1e-12
        raw_step = jnp.linalg.solve(hessian + LM_DAMPING * diagonal, -gradient)
        raw_step = jnp.where(jnp.all(jnp.isfinite(raw_step)), raw_step, jnp.zeros_like(raw_step))
        raw_norm = jnp.linalg.norm(raw_step)
        bounded_step = raw_step * jnp.minimum(1.0, TRUST_RADIUS / jnp.maximum(raw_norm, 1e-300))
        alphas = jnp.asarray((1.0, 0.5, 0.25, 0.0), jnp.float64)
        candidates = normalized[None] + alphas[:, None] * bounded_step[None]
        candidate_norms = jax.vmap(lambda value: jnp.linalg.norm(
            residual_normalized(value, previous_state, viscosity)
        ))(candidates)
        candidate_norms = jnp.where(jnp.isfinite(candidate_norms), candidate_norms, jnp.inf)
        best = jnp.argmin(candidate_norms)
        corrected = candidates[best] * state_scale + state_mean
        return (
            corrected, jnp.linalg.norm(weak_residual), candidate_norms[best],
            alphas[best], raw_norm, jnp.linalg.norm(alphas[best] * bounded_step),
            jnp.int32(1),  # exactly one weak Jacobian evaluation
            jnp.int32(4),  # four bounded residual candidates, including no-op
        )

    return jax.jit(one_update)


def summarize_records(records):
    cases = sorted(set(item["case_index"] for item in records))
    case_medians = [float(np.median([
        item["elapsed_s"] for item in records if item["case_index"] == case
    ])) for case in cases]
    canonical = [next(item for item in records if item["case_index"] == case)
                 for case in cases]
    errors = np.asarray([item["trajectory_relative_l2"] for item in canonical])
    elapsed = np.asarray([item["elapsed_s"] for item in records])
    median = float(np.median(elapsed))
    return {
        "trajectory_error_mean": float(np.mean(errors)),
        "trajectory_error_median": float(np.median(errors)),
        "trajectory_error_worst": float(np.max(errors)),
        "trajectory_error_all": errors.tolist(),
        "zero_failures": bool(all(item["finite"] and item["steps_completed"] == EVAL_STEPS
                                  for item in records)),
        "median_elapsed_s": median,
        "elapsed_all_s": elapsed.tolist(),
        "per_case_median_elapsed_s": case_medians,
        "outliers_gt_1p5_median": int(np.sum(elapsed > 1.5 * median)),
        "mean_weak_jacobians": float(np.mean([
            item["weak_jacobians"] for item in canonical
        ])),
        "mean_accepted_corrections": float(np.mean([
            item["accepted_corrections"] for item in canonical
        ])),
    }


def main():
    c.require_gpu_highest()
    if not SMOKE and EVAL_STEPS != c.NUM_STEPS:
        raise SystemExit("scientific cells must execute all 50 steps")
    if TIME_REPS % 6:
        raise SystemExit("TIME_REPS must be divisible by six for balanced method order")
    load_started = time.perf_counter()
    with open(CHECKPOINT, "rb") as handle:
        checkpoint = pickle.load(handle)
    checkpoint_load_s = time.perf_counter() - load_started
    degree = int(checkpoint["config"]["degree"])
    latent_dimension = 4 + len(c.hermite_pairs(degree))
    if degree not in (4, 5) or checkpoint["state_mean"].size != latent_dimension:
        raise SystemExit("checkpoint is not a preregistered HG4/HG5 predictor")
    if M < max(64, 4 * latent_dimension):
        raise SystemExit("M must satisfy M >= max(64,4k)")
    if EQ_M != 4 * M:
        raise SystemExit("m must equal 4M")
    if M > (N - 2) ** 2:
        raise SystemExit("mesh has fewer interior modes than M")

    indices = np.arange(EVAL_START, EVAL_START + EVAL_CASES)
    truth, parameters, reference_health = c.generate_population(
        N, EVAL_SEED, EVAL_DRAW_COUNT, indices, chunk=max(1, min(8, EVAL_CASES))
    )
    if reference_health["independent_max_relative_residual"] > 1e-8:
        raise SystemExit("reference residual gate failed")
    truth = truth[:, :EVAL_STEPS + 1]

    eq = fit_eq_weights(checkpoint, degree)
    full = {"idx": interior_indices(N), "w": np.ones((N - 2) ** 2, np.float64)}
    full_correct = make_weak_corrector(degree, checkpoint, full)
    eq_correct = make_weak_corrector(degree, checkpoint, eq)
    coords = jnp.asarray(c.grid_coords(N), jnp.float64)
    mask = jnp.asarray(c.binary_boundary_mask(N), jnp.float64)
    decode_full = jax.jit(lambda state: decode_points(state, coords, mask, degree))
    predict = jax.jit(lambda features: apply_predictor(checkpoint["params"], features))

    def run(case, method):
        started = time.perf_counter()
        recovered, sample_indices = c.recover_blob_parameters_fixed_sample(truth[case, 0], N)
        one = {
            "cx": recovered[0:1], "cy": recovered[1:2],
            "width": recovered[2:3], "amplitude": recovered[3:4],
            "nu": parameters["nu"][case:case + 1],
        }
        features = c.trajectory_features(one, N)[0, :EVAL_STEPS + 1]
        normalized_features = (
            features - checkpoint["feature_mean"]
        ) / checkpoint["feature_scale"]
        states = np.asarray(predict(jnp.asarray(normalized_features, jnp.float64)))
        states = states * checkpoint["state_scale"] + checkpoint["state_mean"]
        states[0] = exact_initial_state(recovered, degree)
        weak_jacobians = 0
        weak_residual_evaluations = 0
        accepted = 0
        residual_before, residual_after, step_norms, step_alphas = [], [], [], []
        if method != "direct_zero":
            correct = full_correct if method == "full_one" else eq_correct
            corrected = [states[0]]
            for step in range(1, EVAL_STEPS + 1):
                output = correct(
                    jnp.asarray(states[step]), jnp.asarray(corrected[-1]),
                    jnp.asarray(parameters["nu"][case], jnp.float64)
                )
                values = [np.asarray(value) for value in output]
                corrected.append(values[0])
                before, after, alpha, _, step_norm, n_jac, n_res = values[1:]
                residual_before.append(float(before)); residual_after.append(float(after))
                step_alphas.append(float(alpha)); step_norms.append(float(step_norm))
                weak_jacobians += int(n_jac); weak_residual_evaluations += int(n_res)
                accepted += int(float(alpha) > 0.0 and float(after) < float(before))
            states = np.stack(corrected)
        fields = [np.asarray(truth[case, 0])]
        for state in states[1:]:
            fields.append(np.asarray(decode_full(jnp.asarray(state))))
        fields = np.stack(fields)
        elapsed = time.perf_counter() - started
        error = float(np.linalg.norm(fields - truth[case]) / np.linalg.norm(truth[case]))
        record = {
            "case_index": int(case), "source_draw_index": int(indices[case]),
            "method": method, "elapsed_s": float(elapsed),
            "trajectory_relative_l2": error,
            "finite": bool(np.all(np.isfinite(fields))),
            "steps_completed": EVAL_STEPS,
            "cold_sample_count": int(sample_indices.size),
            "recovered_parameters": recovered.tolist(),
            "weak_jacobians": weak_jacobians,
            "weak_residual_evaluations": weak_residual_evaluations,
            "accepted_corrections": accepted,
            "residual_before": residual_before, "residual_after": residual_after,
            "correction_step_norm": step_norms, "correction_alpha": step_alphas,
        }
        return fields, record

    methods = ("direct_zero", "full_one", "eq_one")
    first_use = {}
    for method in methods:
        _, item = run(0, method)
        first_use[method] = item
    for _ in range(TIME_WARM):
        for case in range(EVAL_CASES):
            for method in methods:
                run(case, method)
    burn_count = c.gpu_burn(BURN_SECONDS)

    records = {method: [] for method in methods}
    timing_orders = []
    canonical_fields = {}
    balanced_orders = (
        ("direct_zero", "full_one", "eq_one"),
        ("direct_zero", "eq_one", "full_one"),
        ("full_one", "direct_zero", "eq_one"),
        ("full_one", "eq_one", "direct_zero"),
        ("eq_one", "direct_zero", "full_one"),
        ("eq_one", "full_one", "direct_zero"),
    )
    for repetition in range(TIME_REPS):
        order = list(balanced_orders[repetition % len(balanced_orders)])
        timing_orders.append(order)
        case_order = list(range(EVAL_CASES))
        case_order = case_order[repetition % EVAL_CASES:] + case_order[:repetition % EVAL_CASES]
        for case in case_order:
            for method in order:
                fields, item = run(case, method)
                item["repetition"] = repetition
                records[method].append(item)
                canonical_fields.setdefault((method, case), fields)

    summaries = {method: summarize_records(items) for method, items in records.items()}
    eq_degradation = (
        summaries["eq_one"]["trajectory_error_mean"]
        / max(summaries["full_one"]["trajectory_error_mean"], 1e-300)
    )
    arrays = {"truth": truth}
    for method in methods:
        arrays[f"{method}_fields"] = np.stack([
            canonical_fields[(method, case)] for case in range(EVAL_CASES)
        ])
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **arrays)

    report = {
        "stage": "direct predictor and zero/one bounded weak/EQ correction",
        "status": "complete" if not SMOKE else "excluded_execution_smoke",
        "config": {
            "N": N, "dt": c.DT, "num_steps": EVAL_STEPS,
            "eval_seed": EVAL_SEED, "eval_draw_count": EVAL_DRAW_COUNT,
            "eval_indices": indices.tolist(), "degree": degree,
            "latent_dimension": latent_dimension, "M": M, "m": EQ_M,
            "M_gate": max(64, 4 * latent_dimension),
            "trust_radius_normalized_state": TRUST_RADIUS,
            "lm_damping": LM_DAMPING,
            "maximum_weak_updates_per_step": 1,
            "weak_objective": "discrete sine test modes with FOM-exact upwind advection",
            "cold_start": "Gaussian fit on fixed at-most-64x64 target-grid sample",
            "model_validation_touched": False, "confirmation_touched": False,
            "f64": True, "smoke": SMOKE,
        },
        "provenance": c.provenance(),
        "checkpoint": {"path": os.path.basename(CHECKPOINT), "sha256": c.sha256(CHECKPOINT)},
        "reference_health": reference_health,
        "viscosity_all": parameters["nu"].tolist(),
        "eq": eq["info"],
        "setup": {
            "checkpoint_load_s": float(checkpoint_load_s),
            "eq_nnls_s": eq["info"]["elapsed_s"],
            "eq_nnls_amortized_per_eval_case_s": eq["info"]["elapsed_s"] / EVAL_CASES,
            "first_use_compile_and_online_invocation": first_use,
            "burn_count": burn_count,
        },
        "timing_orders": timing_orders,
        "records": records,
        "methods": summaries,
        "eq_over_full_mean_error": float(eq_degradation),
        "gates": {
            "full_mean_le_7e-4": bool(summaries["full_one"]["trajectory_error_mean"] <= 7e-4),
            "eq_mean_le_1e-3": bool(summaries["eq_one"]["trajectory_error_mean"] <= 1e-3),
            "eq_over_full_le_1p05": bool(eq_degradation <= 1.05),
            "zero_failures": bool(all(summary["zero_failures"] for summary in summaries.values())),
        },
        "npz": {"path": os.path.basename(OUTPUT_NPZ), "sha256": c.sha256(OUTPUT_NPZ)},
    }
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps({
        "status": report["status"], "methods": report["methods"],
        "eq_over_full_mean_error": report["eq_over_full_mean_error"],
        "gates": report["gates"],
    }, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
