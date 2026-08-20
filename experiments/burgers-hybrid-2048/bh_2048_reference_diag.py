"""Bounded diagnostic for the failed Burgers N=2048 offline reference.

This script never runs or times a warm-start arm.  It first reproduces the
public-JAX fixed-25 reference on every frozen trajectory and persists the final
nonlinear residual at every time step.  It then instruments only the
deterministically worst trajectory's worst step and compares three
independently frozen whole-trajectory counting routes.  The output diagnoses a
reference generator; it cannot promote a replacement or become a performance
result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc


OUT = sys.argv[1]
N = 2048
TEST_SEED = 20260827
DRAW_COUNT = 16
TRAJECTORY_INDICES = (0, 1, 2, 3)
FIXED_NEWTON_ITERS = 25
REFERENCE_GATE = 1e-11
PUBLIC_LINEAR_TOL = 1e-10
PUBLIC_LINEAR_MAXITER = 2000
EXPECTED_JAX_VERSION = "0.10.2"
INSTRUMENTED_UPDATE_MATCH_GATE = 1e-12
TIGHT_ROUTE_AGREEMENT_GATE = 1e-10
COUNTING_ROUTES = (
    {
        "name": "unpreconditioned_control",
        "outer_tol": 1e-12,
        "linear_tol": 1e-10,
        "preconditioner": "none",
    },
    {
        "name": "helmholtz_candidate",
        "outer_tol": 1e-12,
        "linear_tol": 1e-8,
        "preconditioner": "helmholtz",
    },
    {
        "name": "helmholtz_strict_ladder",
        "outer_tol": 1e-13,
        "linear_tol": 1e-10,
        "preconditioner": "helmholtz",
    },
)


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def device_memory():
    stats = jax.devices()[0].memory_stats() or {}
    converted = {}
    for key, value in stats.items():
        if isinstance(value, (int, np.integer)):
            converted[key] = int(value)
        elif isinstance(value, (float, np.floating)) and np.isfinite(value):
            converted[key] = float(value)
    peak = converted.get("peak_bytes_in_use")
    limit = converted.get("bytes_limit")
    converted["peak_fraction_of_limit"] = (
        None if peak is None or limit in (None, 0) else float(peak / limit)
    )
    return converted


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def float_list(values):
    return [finite_float(value) for value in np.asarray(values).reshape(-1)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def worst_step_index(values):
    """Zero-based worst step, with the first nonfinite step ranked worst."""
    values = np.asarray(values)
    nonfinite = np.flatnonzero(~np.isfinite(values))
    return int(nonfinite[0]) if len(nonfinite) else int(np.argmax(values))


def instrumented_jax_bicgstab(operator, rhs):
    """Line-for-line numerical clone of JAX 0.10.2 ``_bicgstab_solve``.

    The public JAX API deliberately returns ``info=None``.  This clone exposes
    its otherwise-hidden final iteration/breakdown code and recursive residual;
    every correction also records its difference from the public solver output.
    Negative codes -10 and -11 retain their JAX meanings (rho breakdown and
    alpha/omega breakdown).
    """
    x0 = jnp.zeros_like(rhs)
    rhs_norm_sq = jnp.vdot(rhs, rhs).real
    tolerance_sq = PUBLIC_LINEAR_TOL**2 * rhs_norm_sq

    def cond(state):
        _, residual, _, _, _, _, _, _, iteration = state
        residual_sq = jnp.vdot(residual, residual).real
        return (
            (residual_sq > tolerance_sq)
            & (iteration < PUBLIC_LINEAR_MAXITER)
            & (iteration >= 0)
        )

    def body(state):
        x, residual, shadow, alpha, omega, rho, direction, image, iteration = state
        rho_new = jnp.vdot(shadow, residual)
        beta = rho_new / rho * alpha / omega
        direction_new = residual + beta * (direction - omega * image)
        image_new = operator(direction_new)
        alpha_new = rho_new / jnp.vdot(shadow, image_new)
        intermediate = residual - alpha_new * image_new
        exit_early = jnp.vdot(intermediate, intermediate).real < tolerance_sq
        image_intermediate = operator(intermediate)
        omega_new = (
            jnp.vdot(image_intermediate, intermediate)
            / jnp.vdot(image_intermediate, image_intermediate)
        )
        full_x = x + alpha_new * direction_new + omega_new * intermediate
        early_x = x + alpha_new * direction_new
        x_new = jnp.where(exit_early, early_x, full_x)
        residual_new = jnp.where(
            exit_early,
            intermediate,
            intermediate - omega_new * image_intermediate,
        )
        iteration_new = jnp.where(
            (omega_new == 0) | (alpha_new == 0), -11, iteration + 1
        )
        iteration_new = jnp.where(rho_new == 0, -10, iteration_new)
        return (
            x_new,
            residual_new,
            shadow,
            alpha_new,
            omega_new,
            rho_new,
            direction_new,
            image_new,
            iteration_new,
        )

    residual0 = rhs - operator(x0)
    one = jnp.asarray(1.0, rhs.dtype)
    initial = (
        x0,
        residual0,
        residual0,
        one,
        one,
        one,
        residual0,
        residual0,
        jnp.int32(0),
    )
    x, residual, _, _, _, _, _, _, iteration = jax.lax.while_loop(
        cond, body, initial
    )
    recursive_relative_residual = (
        jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), 1e-300)
    )
    return x, iteration, recursive_relative_residual


def make_detailed_fixed_step(residual):
    """Expose every hidden linear outcome while reproducing the fixed route."""
    def step(u_prev, nu):
        scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def body(u, _):
            nonlinear = residual(u, u_prev, nu)
            rhs = -nonlinear
            operator = lambda vector: jax.jvp(
                lambda state: residual(state, u_prev, nu),
                (u,),
                (vector,),
            )[1]
            public_update, public_info = jax.scipy.sparse.linalg.bicgstab(
                operator,
                rhs,
                tol=PUBLIC_LINEAR_TOL,
                maxiter=PUBLIC_LINEAR_MAXITER,
            )
            # JAX 0.10.2 documents and returns this as a static None placeholder.
            assert public_info is None
            cloned_update, exit_code, recursive_relative = instrumented_jax_bicgstab(
                operator, rhs
            )
            rhs_norm = jnp.linalg.norm(rhs)
            public_true_relative = (
                jnp.linalg.norm(operator(public_update) - rhs)
                / jnp.maximum(rhs_norm, 1e-300)
            )
            clone_true_relative = (
                jnp.linalg.norm(operator(cloned_update) - rhs)
                / jnp.maximum(rhs_norm, 1e-300)
            )
            update_difference = (
                jnp.linalg.norm(public_update - cloned_update)
                / jnp.maximum(jnp.linalg.norm(public_update), 1e-300)
            )
            pre_relative = jnp.linalg.norm(nonlinear) / scale
            active = jnp.linalg.norm(nonlinear) > 1e-12 * scale
            finite_update = jnp.all(jnp.isfinite(public_update))
            accepted = active & finite_update
            u_new = u + jnp.where(accepted, public_update, 0.0)
            post_relative = jnp.linalg.norm(residual(u_new, u_prev, nu)) / scale
            telemetry = (
                pre_relative,
                post_relative,
                rhs_norm,
                public_true_relative,
                clone_true_relative,
                recursive_relative,
                update_difference,
                jnp.linalg.norm(public_update),
                finite_update,
                active,
                accepted,
                exit_code,
            )
            return u_new, telemetry

        return jax.lax.scan(body, u_prev, None, length=FIXED_NEWTON_ITERS)

    return jax.jit(step)


def trajectory_parameters():
    cx, cy, width, amplitude, nu, _ = bc.bf.sample_params(
        seed=TEST_SEED, m=DRAW_COUNT
    )
    parameters = {}
    for index in TRAJECTORY_INDICES:
        parameters[index] = {
            "cx": float(cx[index]),
            "cy": float(cy[index]),
            "width": float(width[index]),
            "amplitude": float(amplitude[index]),
            "nu": float(nu[index]),
        }
    return parameters


def initial_condition(index, parameters):
    item = parameters[index]
    return bc.bf.blob_ic(
        N,
        item["cx"],
        item["cy"],
        item["width"],
        item["amplitude"],
    )


def run_fixed_phase(report, parameters):
    rollout, _ = bc.bf.make_rollout(N)
    phase = report["fixed25_all_trajectories"]
    for index in TRAJECTORY_INDICES:
        u0 = initial_condition(index, parameters)
        _, residuals = rollout(
            jnp.asarray(u0)[None], jnp.asarray([parameters[index]["nu"]])
        )
        residuals = np.asarray(residuals)[:, 0]
        finite = np.isfinite(residuals)
        maximum = float(np.max(residuals)) if finite.all() else math.inf
        worst_step = worst_step_index(residuals)
        phase.append({
            "trajectory_index": index,
            "parameters": parameters[index],
            "per_step_final_nonlinear_relative_residual": float_list(residuals),
            "all_finite": bool(finite.all()),
            "max_final_nonlinear_relative_residual": finite_float(maximum),
            "worst_step": worst_step + 1,
            "passes_reference_gate": bool(finite.all() and maximum <= REFERENCE_GATE),
        })
        save(report)


def select_detailed_trajectory(report):
    def key(item):
        value = item["max_final_nonlinear_relative_residual"]
        return (
            math.inf if value is None else value,
            -item["trajectory_index"],
        )

    return max(report["fixed25_all_trajectories"], key=key)["trajectory_index"]


def run_detailed_phase(report, parameters, selected_index):
    rollout, residual = bc.bf.make_rollout(N)
    diagnostic_step = make_detailed_fixed_step(residual)
    u0 = jnp.asarray(initial_condition(selected_index, parameters))
    nu = parameters[selected_index]["nu"]
    phase_record = next(
        item
        for item in report["fixed25_all_trajectories"]
        if item["trajectory_index"] == selected_index
    )
    selected_step = int(phase_record["worst_step"])
    snapshots, _ = rollout(u0[None], jnp.asarray([nu]))
    u_prev = snapshots[selected_step - 1, 0]
    expected_u = snapshots[selected_step, 0]
    u, telemetry = diagnostic_step(u_prev, nu)
    telemetry = [np.asarray(value) for value in telemetry]
    (
        pre_relative,
        post_relative,
        rhs_norm,
        public_true_relative,
        clone_true_relative,
        recursive_relative,
        update_difference,
        update_norm,
        finite_update,
        active,
        accepted,
        exit_code,
    ) = telemetry
    details = report["fixed25_detailed_worst_trajectory"]
    details["trajectory_index"] = selected_index
    details["selected_worst_step"] = selected_step
    details["step"] = {
        "pre_nonlinear_relative_residual": float_list(pre_relative),
        "post_nonlinear_relative_residual": float_list(post_relative),
        "rhs_norm": float_list(rhs_norm),
        "public_true_linear_relative_residual": float_list(public_true_relative),
        "instrumented_true_linear_relative_residual": float_list(
            clone_true_relative
        ),
        "instrumented_recursive_linear_relative_residual": float_list(
            recursive_relative
        ),
        "public_vs_instrumented_update_relative_difference": float_list(
            update_difference
        ),
        "public_update_norm": float_list(update_norm),
        "public_update_finite": [bool(value) for value in finite_update],
        "nonlinear_correction_active": [bool(value) for value in active],
        "public_update_accepted": [bool(value) for value in accepted],
        "instrumented_jax_exit_code": [int(value) for value in exit_code],
        "final_nonlinear_relative_residual": finite_float(post_relative[-1]),
    }
    expected_residual = phase_record[
        "per_step_final_nonlinear_relative_residual"
    ][selected_step - 1]
    reproduced_residual = finite_float(post_relative[-1])
    residual_mismatch = (
        None
        if expected_residual is None or reproduced_residual is None
        else abs(expected_residual - reproduced_residual)
    )
    expected_u_np = np.asarray(expected_u)
    u_np = np.asarray(u)
    field_difference = float(
        np.linalg.norm(u_np - expected_u_np)
        / np.maximum(np.linalg.norm(expected_u_np), 1e-300)
    )
    active_true_residuals = [
        float(true_residual)
        for is_active, true_residual in zip(active, public_true_relative)
        if is_active and np.isfinite(true_residual)
    ]
    details["summary"] = {
        "phase1_reproduction_abs_residual_difference": residual_mismatch,
        "phase1_reproduction_field_relative_difference": field_difference,
        "max_active_public_true_linear_relative_residual": (
            max(active_true_residuals) if active_true_residuals else None
        ),
        "active_instrumented_negative_exit_codes": int(
            np.sum(active & (exit_code < 0))
        ),
        "active_nonfinite_public_updates": int(np.sum(active & ~finite_update)),
        "max_active_public_vs_instrumented_update_relative_difference": (
            max(
                float(value)
                for is_active, value in zip(active, update_difference)
                if is_active and np.isfinite(value)
            )
            if np.any(active & np.isfinite(update_difference))
            else None
        ),
        "public_info_values": "None for every solve by the JAX 0.10.2 API contract",
    }
    if (
        residual_mismatch is None
        or residual_mismatch > 1e-12
        or not np.isfinite(field_difference)
        or field_difference > 1e-12
    ):
        raise SystemExit(
            "detailed fixed route failed to reproduce phase 1: "
            f"residual={residual_mismatch}, field={field_difference}"
        )
    save(report)


def run_counting_routes(report, parameters, selected_index):
    u0 = jnp.asarray(initial_condition(selected_index, parameters))
    nu = parameters[selected_index]["nu"]
    dummy = jnp.zeros((bc.T, N * N), jnp.float64)
    fields = {}
    terminal_fields = {}
    for route in COUNTING_ROUTES:
        chain, _ = bc.make_chain(
            N,
            route["outer_tol"],
            lin_tol=route["linear_tol"],
            preconditioner=route["preconditioner"],
        )
        outputs = chain(u0, nu, dummy, jnp.int32(0))
        outputs[0].block_until_ready()
        U, newton, linear, breakdowns, flags, residuals = [
            np.asarray(value) for value in outputs
        ]
        fields[route["name"]] = U
        terminal_fields[route["name"]] = U[-1]
        record = {
            **route,
            "trajectory_index": selected_index,
            "per_step_newton_iterations": [int(value) for value in newton],
            "per_step_linear_iterations": [int(value) for value in linear],
            "per_step_breakdowns_or_linear_max": [int(value) for value in breakdowns],
            "per_step_flags": [int(value) for value in flags],
            "per_step_final_nonlinear_relative_residual": float_list(residuals),
            "max_final_nonlinear_relative_residual": finite_float(np.max(residuals)),
            "all_fields_finite": bool(np.isfinite(U).all()),
            "breakdowns_or_linear_max_total": int(np.sum(breakdowns)),
            "flags_nonzero": int(np.sum(flags != 0)),
            "newton_total": int(np.sum(newton)),
            "linear_total": int(np.sum(linear)),
            "per_step_field_l2_norm": float_list(np.linalg.norm(U, axis=1)),
        }
        report["counting_routes"].append(record)
        save(report)

    route_names = [route["name"] for route in COUNTING_ROUTES]
    report["pairwise_field_agreement"] = {}
    for left_index, left_name in enumerate(route_names):
        for right_name in route_names[left_index + 1:]:
            left = fields[left_name]
            right = fields[right_name]
            step_differences = np.linalg.norm(left - right, axis=1) / np.maximum(
                np.linalg.norm(right, axis=1), 1e-300
            )
            pair_name = f"{left_name}_vs_{right_name}"
            report["pairwise_field_agreement"][pair_name] = {
                "per_step_relative_difference": float_list(step_differences),
                "max_step_relative_difference": finite_float(
                    np.max(step_differences)
                ),
                "trajectory_relative_difference": finite_float(
                    np.linalg.norm(left - right)
                    / np.maximum(np.linalg.norm(right), 1e-300)
                ),
            }
    fields_path = os.path.splitext(OUT)[0] + "-terminal-fields.npz"
    np.savez(fields_path, **terminal_fields)
    report["persisted_terminal_fields"] = {
        "path": os.path.basename(fields_path),
        "sha256": sha256(fields_path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in terminal_fields.items()
        },
    }
    save(report)


def main():
    if (
        jax.__version__ != EXPECTED_JAX_VERSION
        or bc.bf.NEWTON_ITERS != FIXED_NEWTON_ITERS
        or bc.bf.LIN_TOL != PUBLIC_LINEAR_TOL
        or bc.bf.LIN_MAXITER != PUBLIC_LINEAR_MAXITER
        or bc.MAX_NEWTON != 25
    ):
        raise SystemExit("reference diagnostic environment drifted")
    provenance = bc.provenance()
    if (
        provenance["jax_backend"] != "gpu"
        or provenance["jax_version"] != "0.10.2"
        or not provenance["x64"]
        or provenance["matmul_precision"] != "highest"
        or "H200" not in provenance["gpu_kind"]
    ):
        raise SystemExit(f"invalid execution environment: {provenance}")

    report = {
        "config": {
            "purpose": "diagnose failed N=2048 offline reference only",
            "scientific_timing_or_method_rows": False,
            "candidate_can_self_promote": False,
            "N": N,
            "test_seed": TEST_SEED,
            "canonical_draw_count": DRAW_COUNT,
            "trajectory_indices": list(TRAJECTORY_INDICES),
            "fixed_public_jax_newton_iterations": FIXED_NEWTON_ITERS,
            "public_jax_linear_tol": PUBLIC_LINEAR_TOL,
            "public_jax_linear_maxiter": PUBLIC_LINEAR_MAXITER,
            "instrumented_jax_version": EXPECTED_JAX_VERSION,
            "public_jax_info_contract": "None in JAX 0.10.2",
            "reference_gate": REFERENCE_GATE,
            "instrumented_update_match_gate": INSTRUMENTED_UPDATE_MATCH_GATE,
            "tight_route_agreement_gate": TIGHT_ROUTE_AGREEMENT_GATE,
            "detailed_selection_rule": (
                "largest nonfinite-or-finite fixed25 per-trajectory maximum; "
                "ties choose lowest trajectory index"
            ),
            "detailed_scope": "selected worst trajectory and worst step only",
            "counting_routes": list(COUNTING_ROUTES),
            "counting_scope": "selected worst trajectory only",
            "f64": True,
        },
        "provenance": provenance,
        "device_memory_at_start": device_memory(),
        "fixed25_all_trajectories": [],
        "fixed25_detailed_worst_trajectory": {},
        "counting_routes": [],
        "pairwise_field_agreement": {},
        "persisted_terminal_fields": {},
        "device_memory_at_end": {},
        "complete": False,
    }
    save(report)
    parameters = trajectory_parameters()
    run_fixed_phase(report, parameters)
    selected_index = select_detailed_trajectory(report)
    report["config"]["selected_detailed_trajectory"] = selected_index
    save(report)
    run_detailed_phase(report, parameters, selected_index)
    run_counting_routes(report, parameters, selected_index)
    report["device_memory_at_end"] = device_memory()
    report["complete"] = True
    save(report)
    bc.log("REFERENCE-DIAGNOSTIC-DONE")


if __name__ == "__main__":
    main()
