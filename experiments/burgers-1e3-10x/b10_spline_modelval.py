"""Joint all-seed Phase-2 spline model-validation, full-weak, and EQ cell.

Scientific execution requires exactly three independently audited, passing trainer
artifacts for seeds 11/29/47.  SMOKE=1 ignores artifact paths, uses synthetic fields,
and is permanently excluded; it cannot generate any model-validation draw.

Usage: b10_spline_modelval.py OUTPUT.json OUTPUT.npz TRAIN11.json TRAIN29.json TRAIN47.json
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_spline as s

HERE = os.path.dirname(os.path.abspath(__file__))
HYBRID = os.path.abspath(os.path.join(HERE, "..", "burgers-hybrid-1024"))
if os.path.isfile(os.path.join(HYBRID, "bh_common.py")):
    sys.path.insert(0, HYBRID)
import bh_common as bc  # noqa: E402

if len(sys.argv) != 6:
    raise SystemExit(__doc__)
OUTPUT_JSON, OUTPUT_NPZ = sys.argv[1:3]
TRAIN_JSONS = tuple(sys.argv[3:6])
SMOKE = os.environ.get("SMOKE", "0") == "1"
TIME_REPS = int(os.environ.get("TIME_REPS", "1" if SMOKE else "6"))
TIME_WARM = int(os.environ.get("TIME_WARM", "0" if SMOKE else "1"))
BURN_SECONDS = float(os.environ.get("BURN_SECONDS", "0.05" if SMOKE else "3"))
TRUST_RADIUS = 0.25
LM_DAMPING = 1e-6
RHO_STOP = 1e-3
REFERENCE_OUTER = 1e-12
REFERENCE_INNER = 1e-7
AUDIT_OUTER = 3e-13
AUDIT_INNER = 3e-8
EQ_SEED = 20260824
EQ_SNAPSHOTS = 256
SEEDS = (11, 29, 47)
MODELVAL_MIX = ((64, 576, 64), (128, 576, 32), (256, 576, 16))


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load_training_artifact(path, expected_seed, shared):
    with open(path) as handle:
        report = json.load(handle)
    directory = os.path.dirname(os.path.abspath(path))
    npz_path = os.path.join(directory, report.get("npz", {}).get("path", ""))
    checkpoint_path = os.path.join(
        directory, report.get("checkpoint", {}).get("path", "")
    )
    audit_path = os.path.join(directory, "AUDIT.json")
    require(report.get("status") == "complete", f"trainer seed {expected_seed} incomplete")
    require(os.path.isfile(npz_path) and c.sha256(npz_path) == report["npz"]["sha256"],
            f"trainer seed {expected_seed} NPZ checksum")
    require(os.path.isfile(checkpoint_path)
            and c.sha256(checkpoint_path) == report["checkpoint"]["sha256"],
            f"trainer seed {expected_seed} checkpoint checksum")
    require(os.path.isfile(audit_path), f"trainer seed {expected_seed} audit missing")
    with open(audit_path) as handle:
        audit = json.load(handle)
    require(
        audit.get("status") == "pass"
        and audit.get("source_json_sha256") == c.sha256(path)
        and audit.get("source_npz_sha256") == c.sha256(npz_path)
        and audit.get("source_checkpoint_sha256") == c.sha256(checkpoint_path)
        and audit.get("seed") == expected_seed,
        f"trainer seed {expected_seed} independent audit mismatch",
    )
    config = report["config"]
    require(config["training_seed"] == expected_seed, "trainer seed identity")
    require(config["model_validation_touched"] is False
            and config["confirmation_touched"] is False, "trainer touched locked data")
    require(report["selection_oracle"]["gate_pass"] is True
            and report["selection_direct"]["gate_pass"] is True
            and report["gates"]["promote_seed"] is True,
            f"trainer seed {expected_seed} did not pass all selection gates")
    if shared:
        require(config["candidate"] == shared["candidate"], "trainer candidate mismatch")
        require(audit["s0_json_sha256"] == shared["s0_json_sha256"], "trainer S0 mismatch")
    with open(checkpoint_path, "rb") as handle:
        checkpoint = pickle.load(handle)
    require(checkpoint["status"] == "complete"
            and checkpoint["training_seed"] == expected_seed, "checkpoint identity")
    record = {
        "json_path": os.path.abspath(path), "json_sha256": c.sha256(path),
        "npz_path": npz_path, "npz_sha256": c.sha256(npz_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": c.sha256(checkpoint_path),
        "audit_path": audit_path, "audit_sha256": c.sha256(audit_path),
        "seed": expected_seed, "candidate": config["candidate"],
        "s0_json_sha256": audit["s0_json_sha256"],
    }
    return report, checkpoint, record


def fake_checkpoint(candidate, seed):
    one, two = jax.random.split(jax.random.PRNGKey(seed))
    features = np.random.default_rng(seed).normal(0.0, 0.25, size=(12, 7))
    return {
        "status": "excluded_execution_smoke",
        "candidate": candidate, "training_seed": seed,
        "hyperdecoder": s.init_mlp(
            one, (candidate["k"] - 5, 32, 32, candidate["R"] ** 2)
        ),
        "direct_predictor_folded_raw": s.init_mlp(
            two, (7, 32, 32, candidate["k"])
        ),
        "training_features": features.astype(np.float64),
    }


def load_all_trainers():
    if SMOKE:
        candidate = dict(s.CANDIDATES[0])
        return ({seed: fake_checkpoint(candidate, seed) for seed in SEEDS}, {
            "status": "excluded synthetic smoke; no artifact or locked draw opened",
            "candidate": candidate, "artifacts": [],
        })
    checkpoints, records = {}, []
    shared = None
    for path, seed in zip(TRAIN_JSONS, SEEDS):
        report, checkpoint, record = load_training_artifact(path, seed, shared)
        if shared is None:
            shared = record
        checkpoints[seed] = checkpoint
        records.append(record)
    require(records[0]["candidate"] == records[1]["candidate"] == records[2]["candidate"],
            "all seeds must use one finalist")
    require([item["seed"] for item in records] == list(SEEDS), "trainer seed set")
    return checkpoints, {
        "status": "three independently audited passing trainers",
        "candidate": records[0]["candidate"], "artifacts": records,
        "deployable_seed_policy": 11,
    }


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
    kx, ky = kx_grid.reshape(-1)[order], ky_grid.reshape(-1)[order]
    xi = kk / (n - 1)
    sx, sy = np.sin(np.pi * np.outer(xi, kx)), np.sin(np.pi * np.outer(xi, ky))
    phi = (sx[:, None, :] * sy[None, :, :]).reshape(-1, count)
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)
    return phi.astype(np.float64), eigenvalues.reshape(-1)[order].astype(np.float64)


def deterministic_candidate_indices(n, limit=4096):
    interior = interior_indices(n)
    if interior.size <= limit:
        return interior, "all interior target-grid stencil centers"
    side = int(np.floor(np.sqrt(limit)))
    axis = np.unique(np.rint(np.linspace(1, n - 2, side)).astype(np.int64))
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    return np.sort((ii * n + jj).reshape(-1)), (
        f"lexicographically sorted deterministic {axis.size}x{axis.size} target-grid lattice"
    )


def upwind_adv_field(values, n):
    ni, dx = n - 2, 1.0 / (n - 1)
    field = jnp.pad(values.reshape(ni, ni), 1)
    center = field[1:-1, 1:-1]
    ux = jnp.where(center > 0.0, (center - field[:-2, 1:-1]) / dx,
                   (field[2:, 1:-1] - center) / dx)
    uy = jnp.where(center > 0.0, (center - field[1:-1, :-2]) / dx,
                   (field[1:-1, 2:] - center) / dx)
    return (center * (ux + uy)).reshape(-1)


def nnls_capped(matrix, target, max_support, tol=1e-10, inner_max=200):
    weights = np.zeros(matrix.shape[1], np.float64)
    passive = np.zeros(matrix.shape[1], bool)
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


def decode_points(checkpoint, state, coords, mask):
    candidate = checkpoint["candidate"]
    coefficients = s.apply_mlp(checkpoint["hyperdecoder"], state[5:])
    return s.decode_one_jax(state, coefficients, coords, mask, candidate["R"])


def predictor_states(checkpoint, raw_features):
    return jnp.tanh(s.apply_mlp(
        checkpoint["direct_predictor_folded_raw"], raw_features
    ))


def fom_grade(output, truth):
    fields, newton, linear, breakdowns, flags, residuals = [np.asarray(x) for x in output]
    return {
        "trajectory_relative_l2": float(np.linalg.norm(fields - truth[1:])
                                        / np.linalg.norm(truth[1:])),
        "max_returned_relative_residual": float(np.max(residuals)),
        "newton_total": int(np.sum(newton)), "linear_total": int(np.sum(linear)),
        "breakdowns": int(np.sum(breakdowns)), "flags_nonzero": int(np.sum(flags != 0)),
        "finite": bool(np.all(np.isfinite(fields)) and np.all(np.isfinite(residuals))),
    }


def tight_reference(n, start, count):
    cx, cy, width, amplitude, nu, normalized = c.bf.sample_params(seed=0, m=704)
    selected = np.arange(start, start + count)
    u0 = np.stack([c.bf.blob_ic(n, cx[i], cy[i], width[i], amplitude[i])
                   for i in selected]).astype(np.float64)
    dummy = jnp.zeros((c.NUM_STEPS, n * n), jnp.float64)
    reference, _ = bc.make_chain(
        n, REFERENCE_OUTER, lin_tol=REFERENCE_INNER, preconditioner="helmholtz"
    )
    audit, _ = bc.make_chain(
        n, AUDIT_OUTER, lin_tol=AUDIT_INNER, preconditioner="helmholtz"
    )
    truth, ref_records, audit_records, differences = [], [], [], []
    for local, draw in enumerate(selected):
        output = reference(jnp.asarray(u0[local]), nu[draw], dummy, jnp.int32(5))
        jax.block_until_ready(output)
        trajectory = np.concatenate((u0[local:local + 1], np.asarray(output[0])), axis=0)
        audit_output = audit(jnp.asarray(u0[local]), nu[draw], dummy, jnp.int32(5))
        jax.block_until_ready(audit_output)
        ref_records.append(fom_grade(output, trajectory))
        audit_records.append(fom_grade(audit_output, trajectory))
        differences.append(audit_records[-1]["trajectory_relative_l2"])
        truth.append(trajectory)
    ref_ok = all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
                 and row["max_returned_relative_residual"] <= REFERENCE_OUTER
                 for row in ref_records)
    audit_ok = all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
                   and row["max_returned_relative_residual"] <= AUDIT_OUTER
                   for row in audit_records)
    require(ref_ok and audit_ok and max(differences) <= 1e-4,
            f"tight reference health failed at N={n}")
    parameters = {
        "cx": np.asarray(cx[selected]), "cy": np.asarray(cy[selected]),
        "width": np.asarray(width[selected]), "amplitude": np.asarray(amplitude[selected]),
        "nu": np.asarray(nu[selected]), "normalized": np.asarray(normalized[selected]),
    }
    health = {
        "N": n, "seed": 0, "draw_count": 704, "indices": selected.tolist(),
        "reference_outer": REFERENCE_OUTER, "reference_inner": REFERENCE_INNER,
        "audit_outer": AUDIT_OUTER, "audit_inner": AUDIT_INNER,
        "reference_records": ref_records, "audit_records": audit_records,
        "cross_chain_trajectory_relative_l2_all": differences,
        "cross_chain_worst": float(max(differences)),
        "all_finite_zero_flags_breakdowns": True,
    }
    return np.stack(truth), parameters, health


def synthetic_reference():
    n, count, steps = 24, 2, 2
    params = {
        "cx": np.asarray((0.40, 0.57)), "cy": np.asarray((0.55, 0.43)),
        "width": np.asarray((0.11, 0.13)), "amplitude": np.asarray((1.1, 1.3)),
        "nu": np.asarray((0.03, 0.02)),
    }
    truth = np.empty((count, steps + 1, n * n), np.float64)
    for case in range(count):
        for step in range(steps + 1):
            tau = step / steps
            truth[case, step] = c.bf.blob_ic(
                n, params["cx"][case] + .01 * tau, params["cy"][case] - .01 * tau,
                params["width"][case] * (1 + .05 * tau),
                params["amplitude"][case] * (1 - .05 * tau),
            )
    params["normalized"] = np.zeros((count, 5), np.float64)
    health = {"status": "excluded synthetic smoke", "N": n, "indices": []}
    return {(n, 0, count): (truth, params, health)}, steps


def generate_modelval_data():
    if SMOKE:
        return synthetic_reference()
    data = {}
    for n, start, count in MODELVAL_MIX:
        data[(n, start, count)] = tight_reference(n, start, count)
    return data, c.NUM_STEPS


def decode_trajectory(checkpoint, raw_features, n):
    coords = jnp.asarray(c.grid_coords(n), jnp.float64)
    mask = jnp.asarray(c.binary_boundary_mask(n), jnp.float64)

    @jax.jit
    def evaluate(hyper, predictor, features, coordinate_arg, mask_arg):
        states = jnp.tanh(s.apply_mlp(predictor, features))
        coefficients = jax.vmap(lambda state: s.apply_mlp(hyper, state[5:]))(states)
        fields = jax.lax.map(
            lambda pair: s.decode_one_jax(
                pair[0], pair[1], coordinate_arg, mask_arg, checkpoint["candidate"]["R"]
            ), (states, coefficients)
        )
        return fields, states

    return evaluate(
        checkpoint["hyperdecoder"], checkpoint["direct_predictor_folded_raw"],
        jnp.asarray(raw_features, jnp.float64), coords, mask,
    )


def reconstruction_metrics(checkpoint, data, steps):
    result, pooled = {"meshes": {}}, []
    for (n, start, count), (truth, parameters, _) in data.items():
        features = c.trajectory_features(parameters, n)[:, :steps + 1]
        fields, _ = decode_trajectory(checkpoint, features.reshape(-1, 7), n)
        prediction = np.asarray(fields).reshape(count, steps + 1, n * n)
        errors = np.linalg.norm((prediction - truth).reshape(count, -1), axis=1) \
            / np.maximum(np.linalg.norm(truth.reshape(count, -1), axis=1), 1e-300)
        pooled.extend(errors.tolist())
        result["meshes"][str(n)] = {
            "trajectory_error_mean": float(np.mean(errors)),
            "trajectory_error_worst": float(np.max(errors)),
            "trajectory_error_all": errors.tolist(),
            "all_finite": bool(np.all(np.isfinite(prediction))),
            "exact_binary_boundary": bool(np.all(
                prediction[:, :, c.binary_boundary_mask(n) == 0] == 0.0
            )),
        }
    pooled = np.asarray(pooled, np.float64)
    result["pooled"] = {
        "trajectory_error_mean": float(np.mean(pooled)),
        "trajectory_error_worst": float(np.max(pooled)),
        "trajectory_error_all": pooled.tolist(), "all_finite": bool(np.all(np.isfinite(pooled))),
    }
    result["gate_pass"] = bool(
        all(row["trajectory_error_mean"] <= 3e-4
            and row["trajectory_error_worst"] <= 1e-3
            and row["all_finite"] and row["exact_binary_boundary"]
            for row in result["meshes"].values())
        and result["pooled"]["trajectory_error_mean"] <= 3e-4
        and result["pooled"]["trajectory_error_worst"] <= 1e-3
    )
    return result


def fit_eq(checkpoint, n, m_modes, m_points):
    started = time.perf_counter()
    phi, _ = test_modes(n, m_modes)
    interior = interior_indices(n)
    candidates, rule = deterministic_candidate_indices(n)
    require(candidates.size >= m_points, "EQ candidate pool smaller than m")
    positions = np.searchsorted(interior, candidates)
    features = np.asarray(checkpoint["training_features"][-64 * 51:], np.float64)
    snapshot_count = min(EQ_SNAPSHOTS, features.shape[0])
    rng = np.random.default_rng(EQ_SEED)
    selected = np.sort(rng.choice(features.shape[0], snapshot_count, replace=False))
    states = np.asarray(predictor_states(
        checkpoint, jnp.asarray(features[selected], jnp.float64)
    ))
    coords = jnp.asarray(c.grid_coords(n)[interior], jnp.float64)
    mask = jnp.ones(interior.size, jnp.float64)

    @jax.jit
    def decode(hyper, state, coordinate_arg, mask_arg):
        coefficients = s.apply_mlp(hyper, state[5:])
        return s.decode_one_jax(
            state, coefficients, coordinate_arg, mask_arg, checkpoint["candidate"]["R"]
        )

    matrices, targets = [], []
    for state in states:
        values = np.asarray(decode(
            checkpoint["hyperdecoder"], jnp.asarray(state), coords, mask
        ))
        advection = np.asarray(upwind_adv_field(jnp.asarray(values), n))
        for field in (values, advection):
            targets.append(phi.T @ field)
            matrices.append(phi[positions].T * field[positions][None, :])
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    row_scale = np.linalg.norm(matrix, axis=1) + 1e-300
    matrix, target = matrix / row_scale[:, None], target / row_scale
    weights, residual, outer = nnls_capped(matrix, target, m_points)
    support = np.flatnonzero(weights > 0)
    if support.size < m_points:
        remaining = np.setdiff1d(np.arange(candidates.size), support)
        score = np.mean(np.abs(matrix), axis=0)
        support = np.concatenate((support, remaining[np.argsort(-score[remaining])[
            :m_points - support.size]]))
    elif support.size > m_points:
        support = support[np.argsort(-weights[support])[:m_points]]
    final_weights, final_residual, final_outer = nnls_capped(
        matrix[:, support], target, support.size
    )
    floor = 1e-8 * max(float(np.max(final_weights)), 1e-300)
    final_weights = np.where(final_weights > 0, final_weights, floor)
    row_residual = matrix[:, support] @ final_weights - target
    info = {
        "fit_source": "seed-specific spline decoder-output u and exact FOM-upwind images",
        "training_mesh": n, "training_seed": 0, "training_indices": [0, 63],
        "snapshot_flat_indices": selected.tolist(), "snapshot_selection_seed": EQ_SEED,
        "snapshot_count": int(snapshot_count), "candidate_pool_rule": rule,
        "candidate_count": int(candidates.size), "M": m_modes, "m": int(support.size),
        "nnls_outer": int(outer), "nnls_refit_outer": int(final_outer),
        "initial_residual_norm": residual, "refit_residual_norm": final_residual,
        "relative_fit_l2": float(
            np.linalg.norm(row_residual) / max(np.linalg.norm(target), 1e-300)
        ),
        "row_relative_median": float(np.median(
            np.abs(row_residual) / (np.abs(target) + 1e-300)
        )),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return {"indices": candidates[support], "weights": final_weights, "info": info}


def weak_geometry(n, m_modes, collocation):
    phi, eigenvalues = test_modes(n, m_modes)
    interior = interior_indices(n)
    indices = np.asarray(collocation["indices"], np.int64)
    positions = np.searchsorted(interior, indices)
    require(np.all(interior[positions] == indices), "weak points not target-grid interior")
    stencil = stencil_indices(indices, n)
    coords = c.grid_coords(n)
    mask = c.binary_boundary_mask(n)
    weights = np.asarray(collocation["weights"], np.float64)
    return {
        "indices": indices, "stencil_coords": coords[stencil.reshape(-1)],
        "stencil_mask": mask[stencil.reshape(-1)], "center_coords": coords[indices],
        "center_mask": mask[indices], "phi_weighted": phi[positions] * weights[:, None],
        "eigenvalues": eigenvalues,
    }


def make_corrector(checkpoint, n, geometry):
    r = checkpoint["candidate"]["R"]

    def residual(state, hyper, previous_values, viscosity, stencil_coords,
                 stencil_mask, phi_weighted, eigenvalues):
        coefficients = s.apply_mlp(hyper, state[5:])
        values = s.decode_one_jax(
            state, coefficients, stencil_coords, stencil_mask, r
        ).reshape(-1, 5)
        center, xp, xm, yp, ym = [values[:, column] for column in range(5)]
        dx = 1.0 / (n - 1)
        ux = jnp.where(center > 0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected = phi_weighted.T @ center
        preconditioner = (1.0 + c.DT * viscosity * eigenvalues) ** -1
        value = preconditioner * (
            phi_weighted.T @ (center - previous_values)
            + c.DT * (phi_weighted.T @ advection + viscosity * eigenvalues * projected)
        )
        rho = jnp.linalg.norm(value) / jnp.maximum(
            jnp.linalg.norm(phi_weighted.T @ previous_values), 1e-12
        )
        return value, rho

    @jax.jit
    def correct(state, hyper, previous_values, viscosity, stencil_coords,
                stencil_mask, phi_weighted, eigenvalues):
        before, rho_before = residual(
            state, hyper, previous_values, viscosity, stencil_coords,
            stencil_mask, phi_weighted, eigenvalues,
        )

        def attempt(_):
            jacobian = jax.jacfwd(lambda value: residual(
                value, hyper, previous_values, viscosity, stencil_coords,
                stencil_mask, phi_weighted, eigenvalues,
            )[0])(state)
            hessian, gradient = jacobian.T @ jacobian, jacobian.T @ before
            diagonal = jnp.diag(jnp.diag(hessian)) + 1e-12 * jnp.eye(state.size)
            raw = jnp.linalg.solve(hessian + LM_DAMPING * diagonal, -gradient)
            raw = jnp.where(jnp.all(jnp.isfinite(raw)), raw, jnp.zeros_like(raw))
            bounded = raw * jnp.minimum(1.0, TRUST_RADIUS / jnp.maximum(
                jnp.linalg.norm(raw), 1e-300
            ))
            factors = jnp.asarray((1.0, .5, .25, 0.0), jnp.float64)
            trials = jnp.clip(state[None] + factors[:, None] * bounded,
                              -1 + 1e-8, 1 - 1e-8)
            trial = jax.vmap(lambda value: residual(
                value, hyper, previous_values, viscosity, stencil_coords,
                stencil_mask, phi_weighted, eigenvalues,
            ))(trials)
            norms = jnp.where(jnp.isfinite(trial[1]), trial[1], jnp.inf)
            best = jnp.argmin(norms)
            accepted = norms[best] < rho_before
            corrected = jnp.where(accepted, trials[best], state)
            after = jnp.where(accepted, trial[0][best], before)
            rho_after = jnp.where(accepted, norms[best], rho_before)
            return (corrected, before, after, rho_before, rho_after, jacobian,
                    jnp.int32(1), jnp.int32(5), accepted,
                    jnp.where(accepted, factors[best], 0.0),
                    jnp.linalg.norm(jnp.where(accepted, factors[best], 0.0) * bounded),
                    jnp.where(accepted, jnp.int32(1), jnp.int32(2)))

        def stopped(_):
            jacobian = jnp.zeros((before.size, state.size), jnp.float64)
            return (state, before, before, rho_before, rho_before, jacobian,
                    jnp.int32(0), jnp.int32(1), jnp.bool_(False), jnp.float64(0),
                    jnp.float64(0), jnp.int32(0))

        return jax.lax.cond(jnp.isfinite(rho_before) & (rho_before <= RHO_STOP),
                            stopped, attempt, operand=None)

    @jax.jit
    def centers(state, hyper, coords, mask):
        coefficients = s.apply_mlp(hyper, state[5:])
        return s.decode_one_jax(state, coefficients, coords, mask, r)

    args = tuple(jnp.asarray(geometry[key], jnp.float64) for key in (
        "stencil_coords", "stencil_mask", "phi_weighted", "eigenvalues"
    ))
    center_args = (
        jnp.asarray(geometry["center_coords"], jnp.float64),
        jnp.asarray(geometry["center_mask"], jnp.float64),
    )
    return correct, centers, args, center_args


def run_online(checkpoint, truth, parameters, case, n, steps, method, correctors):
    started = time.perf_counter()
    recovered, sample_indices = c.recover_blob_parameters_fixed_sample(truth[case, 0], n)
    one = {"cx": recovered[0:1], "cy": recovered[1:2], "width": recovered[2:3],
           "amplitude": recovered[3:4], "nu": parameters["nu"][case:case + 1]}
    features = c.trajectory_features(one, n)[0, :steps + 1]
    predicted = np.asarray(predictor_states(checkpoint, jnp.asarray(features)))
    states = [predicted[0]]
    work = {key: [] for key in (
        "residual_before", "residual_after", "rho_before", "rho_after", "jacobian",
        "attempted", "residual_evaluations", "accepted", "factor", "step_norm",
        "stopping_code",
    )}
    if method == "direct":
        states = list(predicted)
    else:
        correct, centers, args, center_args, indices = correctors[method]
        previous_values = np.asarray(truth[case, 0, indices])
        for step in range(1, steps + 1):
            output = correct(
                jnp.asarray(predicted[step]), checkpoint["hyperdecoder"],
                jnp.asarray(previous_values), jnp.asarray(parameters["nu"][case]), *args,
            )
            values = [np.asarray(value) for value in output]
            states.append(values[0])
            for key, value in zip(work, values[1:]):
                work[key].append(value)
            previous_values = np.asarray(centers(
                jnp.asarray(states[-1]), checkpoint["hyperdecoder"], *center_args
            ))
    states = np.stack(states)
    coords, mask = jnp.asarray(c.grid_coords(n)), jnp.asarray(c.binary_boundary_mask(n))

    @jax.jit
    def decode_all(hyper, state_arg, coordinate_arg, mask_arg):
        coefficients = jax.vmap(lambda state: s.apply_mlp(hyper, state[5:]))(state_arg)
        return jax.lax.map(lambda pair: s.decode_one_jax(
            pair[0], pair[1], coordinate_arg, mask_arg, checkpoint["candidate"]["R"]
        ), (state_arg, coefficients))

    fields = np.asarray(decode_all(
        checkpoint["hyperdecoder"], jnp.asarray(states), coords, mask
    ))
    jax.block_until_ready(fields)
    elapsed = time.perf_counter() - started
    error = float(np.linalg.norm(fields - truth[case]) / np.linalg.norm(truth[case]))
    record = {
        "case_index": case, "method": method, "elapsed_s": float(elapsed),
        "trajectory_relative_l2": error, "finite": bool(np.all(np.isfinite(fields))),
        "steps_completed": steps, "cold_sample_count": int(sample_indices.size),
        "recovered_parameters": recovered.tolist(),
    }
    if method != "direct":
        for key in ("rho_before", "rho_after", "attempted", "residual_evaluations",
                    "accepted", "factor", "step_norm", "stopping_code"):
            value = np.asarray(work[key])
            record[f"{key}_all"] = value.tolist()
        record["jacobian_evaluations"] = int(np.sum(work["attempted"]))
        record["weak_residual_evaluations"] = int(np.sum(work["residual_evaluations"]))
        record["accepted_corrections"] = int(np.sum(work["accepted"]))
    return fields, states, work, record


def summarize(records, case_count):
    canonical = [next(row for row in records if row["case_index"] == case
                      and row["repetition"] == 0) for case in range(case_count)]
    errors = np.asarray([row["trajectory_relative_l2"] for row in canonical])
    elapsed = np.asarray([row["elapsed_s"] for row in records])
    case_medians, outliers = [], []
    for case in range(case_count):
        values = np.asarray([row["elapsed_s"] for row in records
                             if row["case_index"] == case])
        median = float(np.median(values)); case_medians.append(median)
        outliers.append(int(np.sum(values > 1.5 * median)))
    rng = np.random.default_rng(20260826 + case_count + len(records))
    resampled = errors[rng.integers(0, errors.size, size=(10_000, errors.size))]
    error_ci = np.quantile(np.mean(resampled, axis=1), (0.025, 0.975))
    case_median_values = np.asarray(case_medians, np.float64)
    timing_resampled = case_median_values[
        rng.integers(0, case_count, size=(10_000, case_count))
    ]
    timing_ci = np.quantile(np.median(timing_resampled, axis=1), (0.025, 0.975))
    return {
        "trajectory_error_mean": float(np.mean(errors)),
        "trajectory_error_median": float(np.median(errors)),
        "trajectory_error_worst": float(np.max(errors)),
        "trajectory_error_all": errors.tolist(),
        "trajectory_error_mean_clustered_95ci": error_ci.tolist(),
        "zero_failures": bool(all(row["finite"] and row["steps_completed"] == canonical[0][
            "steps_completed"] for row in records)),
        "median_elapsed_s": float(np.median(elapsed)), "elapsed_all_s": elapsed.tolist(),
        "per_case_median_elapsed_s": case_medians,
        "median_elapsed_clustered_95ci_s": timing_ci.tolist(),
        "outliers_gt_1p5_within_trajectory_all": outliers,
        "outliers_gt_1p5_within_trajectory_total": int(sum(outliers)),
    }


def main():
    c.require_gpu_highest()
    require(TIME_REPS == 1 if SMOKE else TIME_REPS == 6,
            "scientific model-validation requires six balanced timing repetitions")
    checkpoints, artifact_gate = load_all_trainers()
    candidate = artifact_gate["candidate"]
    require(candidate in s.CANDIDATES, "candidate outside preregistered bracket")
    m_modes, m_points = candidate["M"], candidate["m"]
    require(m_modes >= max(64, 4 * candidate["k"]) and m_points == 4 * m_modes,
            "M/m gate failed")
    data, steps = generate_modelval_data()
    execution_seeds = SEEDS[:1] if SMOKE else SEEDS
    report = {
        "stage": "Phase-2 joint all-seed model-validation/full-weak/EQ",
        "status": "excluded_execution_smoke" if SMOKE else "running",
        "config": {
            "candidate": candidate, "seeds": list(SEEDS), "N_full_eq": 24 if SMOKE else 256,
            "execution_seeds": list(execution_seeds),
            "model_validation_mix": [] if SMOKE else [list(row) for row in MODELVAL_MIX],
            "synthetic_smoke_only": SMOKE, "model_validation_touched": not SMOKE,
            "confirmation_touched": False, "num_steps": steps, "M": m_modes,
            "m": m_points, "rho_stop": RHO_STOP, "trust_radius": TRUST_RADIUS,
            "lm_damping": LM_DAMPING, "time_repetitions": TIME_REPS, "f64": True,
            "time_grid": [float(index * c.DT) for index in range(steps + 1)],
            "viscosity_family": "parameterized 2D viscous Burgers inherited family",
        },
        "provenance": c.provenance(), "trainer_gate": artifact_gate,
        "reference_health": {
            str(key[0]): value[2] for key, value in data.items()
        }, "seeds": {},
    }
    c.save_json(OUTPUT_JSON, report)
    n_full = 24 if SMOKE else 256
    full_key = next(key for key in data if key[0] == n_full)
    truth, parameters, _ = data[full_key]
    arrays = {}
    for seed in execution_seeds:
        checkpoint = checkpoints[seed]
        reconstruction = reconstruction_metrics(checkpoint, data, steps)
        eq = fit_eq(checkpoint, n_full, m_modes, m_points)
        arrays[f"seed{seed}_eq_indices"] = eq["indices"]
        arrays[f"seed{seed}_eq_weights"] = eq["weights"]
        full_collocation = {
            "indices": interior_indices(n_full),
            "weights": np.ones((n_full - 2) ** 2, np.float64),
        }
        geometries = {
            "full": weak_geometry(n_full, m_modes, full_collocation),
            "eq": weak_geometry(n_full, m_modes, eq),
        }
        correctors = {}
        for name, geometry in geometries.items():
            correct, centers, args, center_args = make_corrector(
                checkpoint, n_full, geometry
            )
            correctors[name] = (correct, centers, args, center_args, geometry["indices"])
        methods = ("direct", "full", "eq")
        first_use = {}
        for method in methods:
            _, _, _, first_use[method] = run_online(
                checkpoint, truth, parameters, 0, n_full, steps, method, correctors
            )
        for _ in range(TIME_WARM):
            for case in range(truth.shape[0]):
                for method in methods:
                    run_online(checkpoint, truth, parameters, case, n_full, steps,
                               method, correctors)
        burn_count = c.gpu_burn(BURN_SECONDS)
        orders = (
            ("direct", "full", "eq"), ("direct", "eq", "full"),
            ("full", "direct", "eq"), ("full", "eq", "direct"),
            ("eq", "direct", "full"), ("eq", "full", "direct"),
        )[:TIME_REPS]
        records = {method: [] for method in methods}
        for repetition, order in enumerate(orders):
            cases = list(range(truth.shape[0]))
            cases = cases[repetition % len(cases):] + cases[:repetition % len(cases)]
            for case in cases:
                for method in order:
                    fields, states, work, record = run_online(
                        checkpoint, truth, parameters, case, n_full, steps,
                        method, correctors,
                    )
                    record["repetition"] = repetition
                    records[method].append(record)
                    if repetition == 0:
                        arrays[f"seed{seed}_{method}_case{case}_states"] = states
                        if method != "direct":
                            for name, values in work.items():
                                arrays[f"seed{seed}_{method}_case{case}_{name}"] = np.asarray(values)
        summaries = {method: summarize(rows, truth.shape[0])
                     for method, rows in records.items()}
        degradation = summaries["eq"]["trajectory_error_mean"] / max(
            summaries["full"]["trajectory_error_mean"], 1e-300
        )
        gates = {
            "reconstruction_mean_le_3e-4_worst_le_1e-3": reconstruction["gate_pass"],
            "full_mean_le_7e-4": summaries["full"]["trajectory_error_mean"] <= 7e-4,
            "eq_mean_le_1e-3": summaries["eq"]["trajectory_error_mean"] <= 1e-3,
            "eq_over_full_le_1p05": degradation <= 1.05,
            "zero_failures": all(row["zero_failures"] for row in summaries.values()),
        }
        report["seeds"][str(seed)] = {
            "reconstruction": reconstruction, "eq_fit": eq["info"],
            "first_use": first_use, "burn_count": burn_count,
            "timing_orders": [list(row) for row in orders], "records": records,
            "methods": summaries, "eq_over_full_mean_error": float(degradation),
            "gates": gates, "all_gates_pass": bool(all(gates.values())),
            "viscosity_all": parameters["nu"].tolist(),
        }
        c.save_json(OUTPUT_JSON, report)
    all_seed_pass = all(report["seeds"][str(seed)]["all_gates_pass"]
                        for seed in execution_seeds)
    base_eq_zero = all(report["seeds"][str(seed)]["methods"]["eq"]["zero_failures"]
                       for seed in execution_seeds)
    full_all = all(report["seeds"][str(seed)]["gates"]["full_mean_le_7e-4"]
                   for seed in execution_seeds)
    conditional_band = all(
        report["seeds"][str(seed)]["methods"]["eq"]["trajectory_error_mean"] <= 1.2e-3
        and report["seeds"][str(seed)]["eq_over_full_mean_error"] <= 1.20
        for seed in execution_seeds
    )
    base_miss = any(
        not report["seeds"][str(seed)]["gates"]["eq_mean_le_1e-3"]
        or not report["seeds"][str(seed)]["gates"]["eq_over_full_le_1p05"]
        for seed in execution_seeds
    )
    report["joint_gates"] = {
        "all_three_seeds_pass": all_seed_pass,
        "next_M_cell_licensed": bool(full_all and base_eq_zero and base_miss
                                     and conditional_band),
        "hard_stop": bool(not all_seed_pass and not (
            full_all and base_eq_zero and base_miss and conditional_band
        )),
    }
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **arrays)
    report["npz"] = {"path": os.path.basename(OUTPUT_NPZ), "sha256": c.sha256(OUTPUT_NPZ)}
    report["status"] = "excluded_execution_smoke" if SMOKE else "complete"
    c.save_json(OUTPUT_JSON, report)
    c.log(json.dumps({"status": report["status"], "joint_gates": report["joint_gates"]},
                     indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
