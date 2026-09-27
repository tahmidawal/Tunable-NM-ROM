#!/usr/bin/env python
"""Phase-8 cluster-only exact-full-grid representation/inference diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)
from jax import lax
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase3 as p3
import b10_phase4 as p4
import b10_phase7 as p7
import b10_phase7_train as p7train
import b10_spline as spline
import b10_spline_train as legacy


TRAIN_MIX = ((64, 0, 512), (128, 0, 128), (256, 0, 64))
SELECTION_MIX = ((64, 512, 64), (128, 512, 32), (256, 512, 16))
STARTS = ("predictor_q", "free_target_encoder_q")
BATCH = 8
MAX_ATTEMPTS = 40
CG_MAX = 19
CG_TOL = 1e-12
DELTA0, DELTA_MIN, DELTA_MAX = 0.25, 2.0 ** -20, 1.0
LAMBDA0, LAMBDA_MIN, LAMBDA_MAX = 1e-6, 1e-12, 1e12
ACCEPT_RHO = 1e-4
IDENTITY_TOL = 2e-14
AFFINE_ATOL = 2e-15
_DECODERS = {}
_GENERATOR_EVALUATE = jax.jit(p7.apply_generator)

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

warnings.filterwarnings("error", message=r"(?i).*captured.*large.*constant.*", category=Warning)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_pickle(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.partial"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def exact(path, key):
    digest = c.sha256(path)
    if digest != EXPECTED[key]:
        raise SystemExit(f"immutable {key} hash mismatch")
    return {"basename": os.path.basename(path), "sha256": digest}


def validate_chains(args):
    bindings = {}
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            key = f"{phase}_{kind}"
            bindings[key] = exact(getattr(args, key), key)
    for kind in ("json", "npz", "checkpoint", "audit", "manifest"):
        key = f"p7_{kind}"
        bindings[key] = exact(getattr(args, key), key)
    for phase in ("p4", "p5", "p6", "p7"):
        audit = load_json(getattr(args, f"{phase}_audit"))
        if audit.get("status") != "pass" or audit.get("negative_aware") is not True:
            raise SystemExit(f"immutable {phase} audit is not passing")
    p4r, p5r, p6r, p7r = (load_json(getattr(args, f"{phase}_json"))
                            for phase in ("p4", "p5", "p6", "p7"))
    if not (
        p4r["decision"]["selected_spatial_arm"] == "H1"
        and p5r["train_targets"]["integrity_pass"] is True
        and p5r["train_targets"]["snapshot_count"] == 35_904
        and p6r["decision"]["repair_licensed"] is True
        and p7r["decision"] == {
            "g1_seed11_pass": False,
            "seeds29_47_proposal_licensed": False,
            "phase7_hard_stop": True,
            "training_authorized": False,
            "next_action": "hard stop",
            "scientific_promotion_allowed": True,
        }
        and p7r["checkpoint"]["sha256"] == EXPECTED["p7_checkpoint"]
        and p7r["npz"]["sha256"] == EXPECTED["p7_npz"]
    ):
        raise SystemExit("immutable P4/P5/P6/P7 decision chain mismatch")
    bindings["phase8_preregistration"] = {
        "basename": os.path.basename(args.prereg), "sha256": c.sha256(args.prereg)
    }
    return p4r, p5r, p6r, p7r, bindings


def normalized_coefficients(coefficients, mean, scales):
    return np.concatenate(((coefficients[:, :2304] - mean[:2304]) / scales[0],
                           (coefficients[:, 2304:] - mean[2304:]) / scales[1]), axis=1)


def load_train_targets(args, p5_report):
    coefficients, physical, features, records = [], [], [], []
    for row in p5_report["train_targets"]["chunks"]:
        path = os.path.join(args.target_dir, row["basename"])
        if c.sha256(path) != row["sha256"]:
            raise SystemExit(f"immutable target mismatch: {row['basename']}")
        with np.load(path, allow_pickle=False) as data:
            coefficients.append(np.asarray(data["coefficients"], np.float64).reshape(-1, 3328))
            physical.append(np.asarray(data["affine"], np.float64).reshape(-1, 5))
            features.append(np.asarray(data["features"], np.float64).reshape(-1, 7))
            if not (np.all(data["healthy"]) and np.all(data["boundary"])
                    and np.max(data["normal"]) <= 1e-8 and np.max(data["pou"]) == 0.0
                    and np.max(data["support"]) == 32):
                raise SystemExit(f"unhealthy target: {row['basename']}")
        records.append({k: row[k] for k in ("basename", "sha256", "snapshot_count", "N")})
    values = np.concatenate(coefficients)
    return values, np.concatenate(physical), np.concatenate(features), records


def bind_regeneration(train, selection, train_physical, train_features, p7_arrays):
    train_affine_regen = legacy.concatenate(train, "affine")
    train_affine_immutable = np.stack([
        spline.normalized_state_from_affine(row) for row in train_physical
    ])
    selection_affine = legacy.concatenate(selection, "affine")
    selection_features = legacy.concatenate(selection, "features")
    train_features_regen = legacy.concatenate(train, "features")
    exact_columns = (0, 1, 2, 3, 5, 6)

    def feature_check(left, right):
        viscosity_delta = np.abs(left[:, 4] - right[:, 4])
        tolerance = np.abs(np.spacing(np.maximum(np.abs(left[:, 4]), np.abs(right[:, 4]))))
        return (bool(np.array_equal(left[:, exact_columns], right[:, exact_columns])),
                float(np.max(viscosity_delta)), bool(np.all(viscosity_delta <= tolerance)))

    train_exact, train_vmax, train_vpass = feature_check(train_features_regen, train_features)
    select_exact, select_vmax, select_vpass = feature_check(
        selection_features, p7_arrays["selection_features"]
    )
    deltas = {
        "train_regenerated_vs_physical_mapping": float(np.max(np.abs(
            train_affine_regen - train_affine_immutable))),
        "train_r3_vs_physical_mapping": float(np.max(np.abs(
            p7_arrays["training_affine"] - train_affine_immutable))),
        "selection_regenerated_vs_r3": float(np.max(np.abs(
            selection_affine - p7_arrays["selection_affine"]))),
    }
    if max(deltas.values()) > AFFINE_ATOL or not (
        train_exact and train_vpass and select_exact and select_vpass
        and np.array_equal(p7_arrays["training_features"], train_features)
    ):
        raise SystemExit("affine/feature provenance mismatch")
    return {
        "absolute_tolerance": AFFINE_ATOL, "relative_tolerance": 0.0,
        **deltas, "train_feature_exact_columns_bitwise": train_exact,
        "train_viscosity_feature_max_abs": train_vmax,
        "train_viscosity_feature_within_ulp": train_vpass,
        "selection_feature_exact_columns_bitwise": select_exact,
        "selection_viscosity_feature_max_abs": select_vmax,
        "selection_viscosity_feature_within_ulp": select_vpass,
    }


def decode_metrics(datasets, states, coefficients, arrays, prefix, batch=BATCH):
    report, pooled_trajectory, identities = {"meshes": {}}, [], []
    global_offset = 0
    snapshot_total, snapshot_count = 0.0, 0
    for item in datasets:
        count = item["flat"].shape[0]
        one_states = np.asarray(states[global_offset:global_offset + count], np.float64)
        one_coeff = np.asarray(coefficients[global_offset:global_offset + count], np.float64)
        global_offset += count
        npoints = item["coords"].shape[0]
        padded = ((npoints + 127) // 128) * 128
        coords = np.pad(item["coords"], ((0, padded - npoints), (0, 0)))
        mask = np.pad(item["mask"], (0, padded - npoints))
        coarse = jnp.asarray(p3.span_polynomial_table_np(48), jnp.float64)
        fine = jnp.asarray(p3.span_polynomial_table_np(32), jnp.float64)
        cache_key = (batch, padded)
        if cache_key not in _DECODERS:
            k3 = p4.make_pallas_hierarchical_decoder(batch, padded, p7.H1)

            @jax.jit
            def evaluate(state_arg, coefficient_arg, coords_arg, mask_arg,
                         coarse_arg, fine_arg):
                actual = k3(state_arg, coefficient_arg, coords_arg, mask_arg,
                            coarse_arg, fine_arg)
                control = jax.vmap(lambda s, a: p4.decode_one_cox_jax(
                    s, a, coords_arg, mask_arg, p7.H1))(state_arg, coefficient_arg)
                return actual, control
            _DECODERS[cache_key] = evaluate
        evaluate = _DECODERS[cache_key]

        prediction = np.empty_like(item["flat"])
        identity = np.empty(count, np.float64)
        for start in range(0, count, batch):
            take = min(batch, count - start)
            sb, cb = one_states[start:start + take], one_coeff[start:start + take]
            if take < batch:
                sb = np.concatenate((sb, np.repeat(sb[-1:], batch - take, axis=0)))
                cb = np.concatenate((cb, np.repeat(cb[-1:], batch - take, axis=0)))
            actual, control = evaluate(jnp.asarray(sb), jnp.asarray(cb),
                                       jnp.asarray(coords), jnp.asarray(mask), coarse, fine)
            actual, control = np.asarray(actual)[:, :npoints], np.asarray(control)[:, :npoints]
            # Cox is the sole exact-full-grid scientific objective. K3 remains
            # only as the separately reported route-identity control.
            prediction[start:start + take] = control[:take]
            identity[start:start + take] = np.linalg.norm(
                actual[:take] - control[:take], axis=1
            ) / np.maximum(np.linalg.norm(control[:take], axis=1), 1e-300)
        difference = prediction.reshape(item["fields"].shape) - item["fields"]
        snapshot_num = np.sum(np.square(difference), axis=2)
        snapshot_den = np.sum(np.square(item["fields"]), axis=2)
        trajectory = np.sqrt(np.sum(snapshot_num, axis=1) /
                             np.maximum(np.sum(snapshot_den, axis=1), 1e-300))
        snapshot = np.sqrt(snapshot_num / np.maximum(snapshot_den, 1e-300))
        finite = bool(np.all(np.isfinite(prediction)))
        boundary_violations = int(np.count_nonzero(
            prediction[:, item["mask"] == 0.0] != 0.0
        ))
        boundary = boundary_violations == 0
        key = f"{prefix}_N{item['N']}"
        arrays[f"{key}_snapshot_numerator_sq"] = snapshot_num
        arrays[f"{key}_truth_norm_sq"] = snapshot_den
        arrays[f"{key}_identity"] = identity
        arrays[f"{key}_boundary_violation_count"] = np.asarray(
            boundary_violations, np.int64
        )
        pooled_trajectory.extend(trajectory.tolist()); identities.extend(identity.tolist())
        snapshot_total += float(np.sum(snapshot_num / np.maximum(snapshot_den, 1e-300)))
        snapshot_count += int(snapshot.size)
        report["meshes"][str(item["N"])] = {
            "trajectory_error_mean": float(np.mean(trajectory)),
            "trajectory_error_worst": float(np.max(trajectory)),
            "trajectory_error_all": trajectory.tolist(),
            "snapshot_error_mean": float(np.mean(snapshot)),
            "snapshot_error_worst": float(np.max(snapshot)),
            "k3_cox_identity_worst": float(np.max(identity)),
            "all_finite": finite, "exact_binary_boundary": boundary,
            "boundary_violation_count": boundary_violations,
        }
    pooled = np.asarray(pooled_trajectory)
    report["pooled"] = {
        "trajectory_error_mean": float(np.mean(pooled)),
        "trajectory_error_worst": float(np.max(pooled)),
        "trajectory_error_all": pooled.tolist(),
        "mean_snapshot_relative_l2_squared": snapshot_total / snapshot_count,
        "k3_cox_identity_worst": float(np.max(identities)),
        "all_finite": bool(all(v["all_finite"] for v in report["meshes"].values())),
        "exact_binary_boundary": bool(all(v["exact_binary_boundary"] for v in report["meshes"].values())),
        "boundary_violation_count": int(sum(
            v["boundary_violation_count"] for v in report["meshes"].values()
        )),
    }
    return report


def generator_coefficients(generator, q, mean, scales, batch=128):
    output = np.empty((len(q), 3328), np.float64)
    for start in range(0, len(q), batch):
        output[start:start + batch] = np.asarray(_GENERATOR_EVALUATE(
            generator, jnp.asarray(q[start:start + batch]),
            jnp.asarray(mean), jnp.asarray(scales)))
    return output


def make_trust_attempt():
    def residual_one(generator, q, affine, truth, coords, mask, mean, scales):
        state = jnp.concatenate((affine, q))
        coefficient = p7.apply_generator(generator, q[None], mean, scales)[0]
        prediction = p4.decode_one_cox_jax(state, coefficient, coords, mask, p7.H1)
        return (prediction - truth) / jnp.maximum(jnp.linalg.norm(truth), 1e-150)

    def one(generator, q, affine, truth, coords, mask, mean, scales, delta,
            damping, active, recorded_objective):
        def perform(_):
            function = lambda value: residual_one(
                generator, value, affine, truth, coords, mask, mean, scales)
            residual, pullback = jax.vjp(function, q)
            gradient = pullback(residual)[0]

            def matvec(value):
                jvalue = jax.jvp(function, (q,), (value,))[1]
                return pullback(jvalue)[0] + damping * value

            right = -gradient
            rr0 = jnp.vdot(right, right)

            def body(carry, _):
                x, r, direction, rr, running, count, breakdown = carry

                def cg_step(values):
                    x, r, direction, rr, _, count, breakdown = values
                    product = matvec(direction)
                    denominator = jnp.vdot(direction, product)
                    valid = jnp.isfinite(denominator) & (denominator > 0.0)
                    alpha = jnp.where(valid, rr / denominator, 0.0)
                    x_new = x + alpha * direction
                    r_new = r - alpha * product
                    rr_new = jnp.vdot(r_new, r_new)
                    converged = (jnp.sqrt(rr_new)
                                 <= CG_TOL * jnp.maximum(jnp.sqrt(rr0), 1e-300))
                    beta = jnp.where(valid & (rr > 0.0), rr_new / rr, 0.0)
                    direction_new = r_new + beta * direction
                    return (jnp.where(valid, x_new, x), jnp.where(valid, r_new, r),
                            jnp.where(valid, direction_new, direction),
                            jnp.where(valid, rr_new, rr), valid & ~converged,
                            count + valid.astype(jnp.int32), breakdown | ~valid)
                return lax.cond(running, cg_step, lambda values: values, carry), None

            initial = (jnp.zeros_like(q), right, right, rr0, rr0 > 0.0,
                       jnp.int32(0), jnp.bool_(False))
            (step, _, _, rr, _, iterations, breakdown), _ = lax.scan(
                body, initial, xs=None, length=CG_MAX)
            step = step * jnp.minimum(
                1.0, delta / jnp.maximum(jnp.linalg.norm(step), 1e-300))
            trial = jnp.clip(q + step, -1.0, 1.0)
            actual_step = trial - q
            jstep = jax.jvp(function, (q,), (actual_step,))[1]
            predicted = (-jnp.vdot(gradient, actual_step)
                         - 0.5 * jnp.vdot(jstep, jstep))
            trial_residual = function(trial)
            objective = jnp.vdot(residual, residual)
            trial_objective = jnp.vdot(trial_residual, trial_residual)
            actual = 0.5 * (objective - trial_objective)
            rho = jnp.where(predicted > 0.0, actual / predicted, -jnp.inf)
            finite = (jnp.all(jnp.isfinite(trial_residual))
                      & jnp.isfinite(predicted) & jnp.isfinite(actual)
                      & jnp.isfinite(rho))
            accepted = (~breakdown & finite & (predicted > 0.0)
                        & (actual > 0.0) & (rho >= ACCEPT_RHO))
            next_q = jnp.where(accepted, trial, q)
            next_objective = jnp.where(accepted, trial_objective, objective)
            terminate = accepted & (((objective - trial_objective)
                         / jnp.maximum(objective, 1e-300) <= 1e-12)
                        | (jnp.linalg.norm(actual_step)
                           / (1.0 + jnp.linalg.norm(q)) <= 1e-12))
            shrink = ~accepted | (rho < 0.25)
            expand = (accepted & (rho > 0.75)
                      & (jnp.linalg.norm(actual_step) >= 0.9 * delta))
            improve = accepted & (rho > 0.75)
            next_delta = jnp.where(
                shrink, jnp.maximum(delta / 4.0, DELTA_MIN),
                jnp.where(expand, jnp.minimum(2.0 * delta, DELTA_MAX), delta))
            next_damping = jnp.where(
                shrink, jnp.minimum(10.0 * damping, LAMBDA_MAX),
                jnp.where(improve, jnp.maximum(damping / 3.0, LAMBDA_MIN), damping))
            cg_relative = jnp.sqrt(rr) / jnp.maximum(jnp.sqrt(rr0), 1e-300)
            return (next_q, next_objective, next_delta, next_damping, ~terminate,
                    objective, trial_objective, gradient, actual_step, predicted,
                    actual, rho, accepted, terminate, iterations, cg_relative,
                    breakdown, finite)

        def inactive(_):
            zero = jnp.asarray(0.0, jnp.float64)
            return (q, recorded_objective, delta, damping, jnp.bool_(False),
                    recorded_objective, recorded_objective, jnp.zeros_like(q),
                    jnp.zeros_like(q), zero, zero, zero, jnp.bool_(False),
                    jnp.bool_(False), jnp.int32(0), zero, jnp.bool_(False),
                    jnp.bool_(True))

        return lax.cond(active, perform, inactive, operand=None)

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, None, None, None, None,
                                          0, 0, 0, 0)))


def run_trust(datasets, generator, starts, initial_objectives, mean, scales,
              arrays, attempts, progress_path, checkpoint_path, wall_started):
    total = starts.shape[1]
    shape = (2, total)
    trace = {
        "q": np.empty((2, total, attempts + 1, 19), np.float64),
        "objective": np.empty((2, total, attempts + 1), np.float64),
        "delta": np.empty((2, total, attempts), np.float64),
        "damping": np.empty((2, total, attempts), np.float64),
        "gradient_norm": np.empty((2, total, attempts), np.float64),
        "step": np.empty((2, total, attempts, 19), np.float64),
        "trial_objective": np.empty((2, total, attempts), np.float64),
        "predicted": np.empty((2, total, attempts), np.float64),
        "actual": np.empty((2, total, attempts), np.float64),
        "rho": np.empty((2, total, attempts), np.float64),
        "accepted": np.zeros((2, total, attempts), bool),
        "terminated": np.zeros((2, total, attempts), bool),
        "attempted": np.zeros((2, total, attempts), bool),
        "cg_iterations": np.zeros((2, total, attempts), np.int32),
        "cg_relative_residual": np.empty((2, total, attempts), np.float64),
        "cg_breakdown": np.zeros((2, total, attempts), bool),
        "finite": np.zeros((2, total, attempts), bool),
        "jvp_count": np.zeros((2, total, attempts), np.int32),
        "vjp_count": np.zeros((2, total, attempts), np.int32),
        "bound_active_count": np.zeros((2, total, attempts), np.int8),
    }
    attempt_fn = make_trust_attempt()
    for start_index in range(2):
        q = starts[start_index].copy()
        delta = np.full(total, DELTA0); damping = np.full(total, LAMBDA0)
        active = np.ones(total, bool); objective = initial_objectives[start_index].copy()
        trace["q"][start_index, :, 0] = q
        trace["objective"][start_index, :, 0] = objective
        for attempt in range(attempts):
            attempted_before = active.copy()
            for item in datasets:
                lo, hi = item["global_start"], item["global_stop"]
                for batch_start in range(lo, hi, BATCH):
                    sl = slice(batch_start, batch_start+BATCH); local = batch_start - lo
                    result = attempt_fn(generator, jnp.asarray(q[sl]),
                        jnp.asarray(item["affine"][local:local+BATCH]),
                        jnp.asarray(item["flat"][local:local+BATCH]),
                        jnp.asarray(item["coords"]), jnp.asarray(item["mask"]),
                        jnp.asarray(mean), jnp.asarray(scales), jnp.asarray(delta[sl]),
                        jnp.asarray(damping[sl]), jnp.asarray(active[sl]),
                        jnp.asarray(objective[sl]))
                    (next_q, next_obj, next_delta, next_damping, next_active,
                     old_obj, trial_obj, gradient, step, pred, actual, rho,
                     accepted, terminated, cg_iter, cg_rel, breakdown, finite) = map(np.asarray, result)
                    if not np.allclose(old_obj, objective[sl], rtol=2e-13, atol=2e-14):
                        raise SystemExit("Cox trust objective/control trace mismatch")
                    trace["delta"][start_index, sl, attempt] = delta[sl]
                    trace["damping"][start_index, sl, attempt] = damping[sl]
                    trace["gradient_norm"][start_index, sl, attempt] = np.linalg.norm(gradient, axis=1)
                    trace["step"][start_index, sl, attempt] = step
                    trace["trial_objective"][start_index, sl, attempt] = trial_obj
                    trace["predicted"][start_index, sl, attempt] = pred
                    trace["actual"][start_index, sl, attempt] = actual
                    trace["rho"][start_index, sl, attempt] = rho
                    trace["accepted"][start_index, sl, attempt] = accepted
                    trace["terminated"][start_index, sl, attempt] = terminated
                    trace["cg_iterations"][start_index, sl, attempt] = cg_iter
                    trace["cg_relative_residual"][start_index, sl, attempt] = cg_rel
                    trace["cg_breakdown"][start_index, sl, attempt] = breakdown
                    trace["finite"][start_index, sl, attempt] = finite
                    work_count = np.where(attempted_before[sl], cg_iter + 1, 0)
                    trace["jvp_count"][start_index, sl, attempt] = work_count
                    trace["vjp_count"][start_index, sl, attempt] = work_count
                    trace["bound_active_count"][start_index, sl, attempt] = np.sum(
                        np.abs(next_q) >= 1.0, axis=1
                    )
                    q[sl], objective[sl], delta[sl], damping[sl], active[sl] = (
                        next_q, next_obj, next_delta, next_damping, next_active)
            trace["attempted"][start_index, :, attempt] = attempted_before
            trace["q"][start_index, :, attempt + 1] = q
            trace["objective"][start_index, :, attempt + 1] = objective
            atomic_json(progress_path, {
                "status": "in_progress", "stage": "trust",
                "start_index": start_index, "start_name": STARTS[start_index],
                "attempt_completed": attempt + 1, "attempt_cap": attempts,
                "active_count": int(np.sum(active)),
                "accepted_count": int(np.sum(trace["accepted"][start_index,:,:attempt+1])),
                "cg_breakdown_count": int(np.sum(
                    trace["cg_breakdown"][start_index,:,:attempt+1]
                    & trace["attempted"][start_index,:,:attempt+1])),
                "elapsed_s": float(time.perf_counter() - wall_started),
                "scientific_metrics_exposed": False,
            })
            atomic_pickle(checkpoint_path, {
                "status": "in_progress", "start_index": start_index,
                "attempt_completed": attempt + 1, "q": q,
                "objective": objective, "delta": delta, "damping": damping,
                "active": active,
            })
        arrays[f"trust_start{start_index}_terminal_delta"] = delta
        arrays[f"trust_start{start_index}_terminal_damping"] = damping
        arrays[f"trust_start{start_index}_terminal_active"] = active
    for name, value in trace.items():
        arrays[f"trust_{name}"] = value
    arrays["trust_starts_q"] = starts
    arrays["trust_terminal_q"] = trace["q"][:, :, -1]
    atomic_pickle(checkpoint_path, {
        "status": "complete", "attempt_cap": attempts,
        "terminal_q": arrays["trust_terminal_q"],
        "terminal_objective": trace["objective"][:,:,-1],
    })
    return trace


def trust_checkpoint_summary(datasets, objective, attempted, accepted):
    result = {}
    final_attempt = objective.shape[2] - 1
    checkpoints = sorted(set((0, final_attempt) + tuple(
        value for value in (10, 20, 30, 40) if value <= final_attempt
    )))
    for start_index, start_name in enumerate(STARTS):
        rows = []
        for checkpoint in checkpoints:
            meshes, pooled = {}, []
            for item in datasets:
                lo, hi = item["global_start"], item["global_stop"]
                values = objective[start_index, lo:hi, checkpoint]
                pooled.extend(values.tolist())
                meshes[str(item["N"])] = {
                    "mean_snapshot_relative_l2_squared": float(np.mean(values)),
                    "worst_snapshot_relative_l2": float(np.sqrt(np.max(values))),
                }
            pooled = np.asarray(pooled)
            attempt_slice = slice(0, checkpoint)
            rows.append({
                "attempt": checkpoint, "meshes": meshes,
                "pooled_mean_snapshot_relative_l2_squared": float(np.mean(pooled)),
                "pooled_worst_snapshot_relative_l2": float(np.sqrt(np.max(pooled))),
                "attempted_total_through_checkpoint": int(np.sum(
                    attempted[start_index, :, attempt_slice])),
                "accepted_total_through_checkpoint": int(np.sum(
                    accepted[start_index, :, attempt_slice])),
            })
        result[start_name] = rows
    return result


def exact_objectives_from_control(datasets, arrays, prefix):
    return np.concatenate([
        (arrays[f"{prefix}_N{item['N']}_snapshot_numerator_sq"] /
         np.maximum(arrays[f"{prefix}_N{item['N']}_truth_norm_sq"], 1e-300)).reshape(-1)
        for item in datasets
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            parser.add_argument(f"--{phase}-{kind}", dest=f"{phase}_{kind}")
    for kind in ("json", "npz", "checkpoint", "audit", "manifest"):
        parser.add_argument(f"--p7-{kind}", dest=f"p7_{kind}")
    parser.add_argument("--target-dir")
    parser.add_argument("--prereg")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--work-checkpoint", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    c.require_gpu_highest()
    started = time.perf_counter()
    legacy.SMOKE = args.smoke
    attempts = 1 if args.smoke else MAX_ATTEMPTS
    if args.smoke:
        p4r = p5r = p6r = p7r = bindings = None
        train = p7train.smoke_datasets("train")
        selection = p7train.smoke_datasets("selection")
        generator, encoder = p7.init_generator(), p7.init_encoder()
        predictor = spline.init_mlp(jax.random.PRNGKey(12), (7, 32, 32, 24))
        mean, scales = np.zeros(3328), np.ones(2)
        train_coeff = np.zeros((4, 3328)); selection_coeff = np.zeros((4, 3328))
        train_physical = np.stack([spline.affine_state_from_field(row, train[0]["coords"])
                                   for row in train[0]["flat"]])
        train_features = legacy.concatenate(train, "features")
        p7_arrays = {
            "training_affine": legacy.concatenate(train, "affine"),
            "selection_affine": legacy.concatenate(selection, "affine"),
            "training_features": train_features,
            "selection_features": legacy.concatenate(selection, "features"),
            "encoder_handoff_q_raw": np.zeros((4, 19)),
            "training_q_raw": np.zeros((4, 19)),
            "training_states": np.concatenate((legacy.concatenate(train, "affine"), np.zeros((4,19))), axis=1),
            "selection_oracle_states": np.concatenate((legacy.concatenate(selection, "affine"), np.zeros((4,19))), axis=1),
            "selection_predictor_states": np.zeros((4,24)),
        }
        target_records = [{"basename": "synthetic-only", "sha256": None,
                           "snapshot_count": 4, "N": 16}]
    else:
        required = [getattr(args, f"{phase}_{kind}") for phase in ("p4","p5","p6")
                    for kind in ("json","npz","audit","manifest")]
        required += [getattr(args, f"p7_{kind}") for kind in
                     ("json","npz","checkpoint","audit","manifest")]
        required += [args.target_dir, args.prereg]
        if any(value is None for value in required):
            raise SystemExit("scientific P8-D requires all immutable chains")
        p4r, p5r, p6r, p7r, bindings = validate_chains(args)
        train = legacy.load_mix(TRAIN_MIX, "train")
        selection = legacy.load_mix(SELECTION_MIX, "selection")
        train_coeff, train_physical, train_features, target_records = load_train_targets(args, p5r)
        with np.load(args.p5_npz, allow_pickle=False) as data:
            mean = np.asarray(data["coefficient_mean"], np.float64)
            scales = np.asarray(data["head_scales"], np.float64)
        with np.load(args.p4_npz, allow_pickle=False) as data:
            selection_coeff = np.concatenate([
                np.asarray(data[f"H1_N{item['N']}_coefficients"], np.float64).reshape(-1,3328)
                for item in selection])
            selection_p4_physical = np.concatenate([
                np.asarray(data[f"H1_N{item['N']}_affine"], np.float64).reshape(-1,5)
                for item in selection])
        p4_normalized = np.stack([spline.normalized_state_from_affine(row)
                                  for row in selection_p4_physical])
        if np.max(np.abs(p4_normalized - legacy.concatenate(selection, "affine"))) > AFFINE_ATOL:
            raise SystemExit("P4 exposed affine/regenerated selection mismatch")
        with np.load(args.p7_npz, allow_pickle=False) as data:
            p7_arrays = {name: np.asarray(data[name]) for name in data.files}
        with open(args.p7_checkpoint, "rb") as handle:
            checkpoint = pickle.load(handle)
        generator = checkpoint["generator"]
        encoder = checkpoint["encoder"]
        predictor = checkpoint["direct_predictor_folded_raw"]

    binding = bind_regeneration(train, selection, train_physical, train_features, p7_arrays)
    arrays = {
        "training_affine": legacy.concatenate(train, "affine"),
        "training_features": legacy.concatenate(train, "features"),
        "selection_affine": legacy.concatenate(selection, "affine"),
        "selection_features": legacy.concatenate(selection, "features"),
        "coefficient_mean": mean, "head_scales": scales,
        "training_encoder_handoff_q_raw": p7_arrays["encoder_handoff_q_raw"],
        "training_final_q_raw": p7_arrays["training_q_raw"],
        "training_final_states": p7_arrays["training_states"],
        "selection_locked_p7_oracle_states": p7_arrays["selection_oracle_states"],
    }
    controls = {}
    completed_controls = []

    def control_progress(name):
        completed_controls.append(name)
        atomic_json(args.progress_json, {
            "status": "in_progress", "stage": "controls",
            "completed_control_count": len(completed_controls),
            "completed_control_names": completed_controls,
            "elapsed_s": float(time.perf_counter() - started),
            "scientific_metrics_exposed": False,
        })

    atomic_json(args.progress_json, {
        "status": "in_progress", "stage": "data_regenerated",
        "elapsed_s": float(time.perf_counter() - started),
        "scientific_metrics_exposed": False,
    })
    train_free_states = np.concatenate((arrays["training_affine"], np.zeros((len(train_coeff),19))), axis=1)
    controls["train_free_h1"] = decode_metrics(train, train_free_states, train_coeff, arrays, "train_free_h1")
    control_progress("train_free_h1")
    handoff_q = np.tanh(p7_arrays["encoder_handoff_q_raw"])
    handoff_states = np.concatenate((arrays["training_affine"], handoff_q), axis=1)
    controls["train_encoder_handoff"] = decode_metrics(
        train, handoff_states, generator_coefficients(generator, handoff_q, mean, scales),
        arrays, "train_encoder_handoff")
    control_progress("train_encoder_handoff")
    final_train_q = np.tanh(p7_arrays["training_q_raw"])
    controls["train_final_autolatent"] = decode_metrics(
        train, p7_arrays["training_states"],
        generator_coefficients(generator, final_train_q, mean, scales),
        arrays, "train_final_autolatent")
    control_progress("train_final_autolatent")
    selection_free_states = np.concatenate((arrays["selection_affine"], np.zeros((len(selection_coeff),19))), axis=1)
    controls["selection_free_h1"] = decode_metrics(
        selection, selection_free_states, selection_coeff, arrays, "selection_free_h1")
    control_progress("selection_free_h1")
    normalized_selection = normalized_coefficients(selection_coeff, mean, scales)
    encoder_q_raw = p7train.encode_all(encoder, normalized_selection)
    encoder_q = np.tanh(encoder_q_raw)
    encoder_states = np.concatenate((arrays["selection_affine"], encoder_q), axis=1)
    controls["selection_free_target_encoder"] = decode_metrics(
        selection, encoder_states, generator_coefficients(generator, encoder_q, mean, scales),
        arrays, "selection_free_target_encoder")
    control_progress("selection_free_target_encoder")
    predictor_states = np.asarray(jnp.tanh(spline.apply_mlp(
        predictor, jnp.asarray(arrays["selection_features"]))))
    predictor_q = predictor_states[:, 5:]
    predictor_exact_affine_states = np.concatenate((arrays["selection_affine"], predictor_q), axis=1)
    controls["selection_predictor_q_exact_affine"] = decode_metrics(
        selection, predictor_exact_affine_states,
        generator_coefficients(generator, predictor_q, mean, scales), arrays,
        "selection_predictor_q_exact_affine")
    control_progress("selection_predictor_q_exact_affine")
    controls["selection_deployed_predictor"] = decode_metrics(
        selection, predictor_states,
        generator_coefficients(generator, predictor_q, mean, scales), arrays,
        "selection_deployed_predictor")
    control_progress("selection_deployed_predictor")
    locked_states = p7_arrays["selection_oracle_states"]
    controls["selection_locked_p7_oracle"] = decode_metrics(
        selection, locked_states,
        generator_coefficients(generator, locked_states[:,5:], mean, scales), arrays,
        "selection_locked_p7_oracle")
    control_progress("selection_locked_p7_oracle")

    atomic_json(args.progress_json, {
        "status": "in_progress", "stage": "controls_complete",
        "elapsed_s": float(time.perf_counter() - started),
        "scientific_metrics_exposed": False,
    })

    starts = np.stack((predictor_q, encoder_q))
    initial_objectives = np.stack((
        exact_objectives_from_control(selection, arrays,
                                      "selection_predictor_q_exact_affine"),
        exact_objectives_from_control(selection, arrays,
                                      "selection_free_target_encoder"),
    ))
    trace = run_trust(selection, generator, starts, initial_objectives,
                      mean, scales, arrays, attempts, args.progress_json,
                      args.work_checkpoint, started)
    trust_metrics = {}
    for index, name in enumerate(STARTS):
        q = trace["q"][index, :, -1]
        states = np.concatenate((arrays["selection_affine"], q), axis=1)
        trust_metrics[name] = decode_metrics(
            selection, states, generator_coefficients(generator, q, mean, scales),
            arrays, f"trust_{name}")
    chosen = np.argmin(trace["objective"][:, :, -1], axis=0)
    terminal_q = np.where(chosen[:,None] == 0, trace["q"][0,:,-1], trace["q"][1,:,-1])
    arrays["trust_chosen_start"] = chosen.astype(np.int8)
    arrays["trust_chosen_q"] = terminal_q
    chosen_states = np.concatenate((arrays["selection_affine"], terminal_q), axis=1)
    chosen_metrics = decode_metrics(selection, chosen_states,
        generator_coefficients(generator, terminal_q, mean, scales), arrays, "trust_two_start")
    controls["selection_trust_starts"] = trust_metrics
    controls["selection_trust_two_start"] = chosen_metrics

    trust_checkpoints = trust_checkpoint_summary(
        selection, trace["objective"], trace["attempted"], trace["accepted"]
    )
    exhausted = []
    for index in range(2):
        terminal_active = arrays[f"trust_start{index}_terminal_active"]
        terminal_delta = arrays[f"trust_start{index}_terminal_delta"]
        terminal_damping = arrays[f"trust_start{index}_terminal_damping"]
        never_accepted = ~np.any(trace["accepted"][index], axis=1)
        exhausted.append(terminal_active & never_accepted
                         & (terminal_delta <= DELTA_MIN)
                         & (terminal_damping >= LAMBDA_MAX))
    exhausted = np.stack(exhausted)
    arrays["trust_unhealthy_exhaustion"] = exhausted

    trace_health = {
        "all_attempt_values_finite": bool(np.all(trace["finite"] | ~trace["attempted"])),
        "any_cg_breakdown": bool(np.any(trace["cg_breakdown"] & trace["attempted"])),
        "attempted_total": int(np.sum(trace["attempted"])),
        "accepted_total": int(np.sum(trace["accepted"])),
        "terminated_total": int(np.sum(trace["terminated"])),
        "jvp_total": int(np.sum(trace["jvp_count"])),
        "vjp_total": int(np.sum(trace["vjp_count"])),
        "unhealthy_exhaustion_count": int(np.sum(exhausted)),
        "max_attempts": attempts,
    }
    identity_pass = bool(all(
        row["pooled"]["k3_cox_identity_worst"] <= IDENTITY_TOL
        for name, row in controls.items() if name != "selection_trust_starts"
    ) and all(row["pooled"]["k3_cox_identity_worst"] <= IDENTITY_TOL
              for row in trust_metrics.values()))
    health_pass = bool(trace_health["all_attempt_values_finite"]
                       and not trace_health["any_cg_breakdown"]
                       and trace_health["unhealthy_exhaustion_count"] == 0
                       and identity_pass)
    decision = {
        "p8_d_valid": bool(not args.smoke and health_pass),
        "t1_implementation_authorized": False,
        "t1_submission_authorized": False,
        "t2_authorized": False,
        "next_action": "root audit of P8-D" if not args.smoke else "excluded smoke only",
        "scientific_promotion_allowed": False,
    }
    config = {
        "objective": "exact discrete full-grid FOM relative-L2-squared",
        "scientific_full_grid_route": "Cox",
        "K3_role": "identity_control_only",
        "candidate": p7.G1, "latent_dimension": 19,
        "train_mix": TRAIN_MIX, "selection_mix": SELECTION_MIX,
        "starts": list(STARTS), "direct_q_optimization": True,
        "trust": {"max_attempts": attempts, "cg_max": CG_MAX, "cg_tolerance": CG_TOL,
                  "delta0": DELTA0, "delta_min": DELTA_MIN, "delta_max": DELTA_MAX,
                  "lambda0": LAMBDA0, "lambda_min": LAMBDA_MIN,
                  "lambda_max": LAMBDA_MAX, "accept_rho": ACCEPT_RHO},
        "free_target_encoder_deployable": False, "direct_predictor_deployable": True,
        "model_validation_touched": False, "confirmation_touched": False,
        "training_touched": False, "weak_eq_touched": False, "scaling_touched": False,
        "terminated_local_diagnostic_reused": False, "smoke": args.smoke,
        "f64": True, "matmul_precision": "highest",
    }
    arrays["selection_encoder_q_raw"] = encoder_q_raw
    arrays["selection_encoder_q"] = encoder_q
    arrays["selection_predictor_states"] = predictor_states
    arrays["selection_predictor_q"] = predictor_q
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    atomic_json(args.progress_json, {
        "status": "complete", "stage": "complete", "attempt_cap": attempts,
        "elapsed_s": float(time.perf_counter() - started),
        "scientific_metrics_exposed": False,
    })
    report = {
        "status": "excluded_execution_smoke" if args.smoke else "complete",
        "provenance": c.provenance(), "config": config, "bindings": bindings,
        "data": {"training": legacy.metadata(train), "selection": legacy.metadata(selection),
                 "target_chunks": target_records, "affine_feature_binding": binding},
        "controls": controls, "trust_checkpoints": trust_checkpoints,
        "trust_health": trace_health,
        "gates": {"identity": identity_pass, "health": health_pass},
        "decision": decision, "npz": {"basename": os.path.basename(args.output_npz),
                                        "sha256": c.sha256(args.output_npz)},
        "progress": {"basename": os.path.basename(args.progress_json),
                     "sha256": c.sha256(args.progress_json)},
        "work_checkpoint": {"basename": os.path.basename(args.work_checkpoint),
                            "sha256": c.sha256(args.work_checkpoint)},
        "elapsed_s": float(time.perf_counter() - started),
    }
    c.save_json(args.output_json, report)
    c.log({"status": report["status"], "trust_health": trace_health,
           "decision": decision}, "\nALL-DONE")


if __name__ == "__main__":
    main()
