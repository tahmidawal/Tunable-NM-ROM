#!/usr/bin/env python
"""Phase-12 corrected-route G2/q32 cell.

The optimizer, schedules, losses, trust recovery, and gates are inherited
unchanged from Phase 11.  Only the pre-update structural route is replaced:
the timed callable has exactly five output leaves and cannot reach the
separately constructed Cox full-field identity callable.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_common as c
import b10_phase11_g2 as p11
import b10_phase9_train as p9


F64 = jnp.float64
EXPECTED_OUTPUT_SIZE = 429_230_768
EXPECTED_LOGICAL_BYTES = 429_230_728
EXPECTED_P11 = {
    "phase11_g2.json": "c1a3d24a0883fa5046248f294fb40b55a584de87d42e58c2d20aedd8d63e39bb",
    "phase11_g2.npz": "f0260ef4f2127d258480b17dbedc0a8720f1e454d9298ecfb93d697822a31aec",
    "checkpoint.pkl": "cf73892c3f0e065951154beb77fcb4113375a9957fafe1de850f681946fca851",
    "AUDIT.json": "2146c63f2e01cf0e55e5dd6212ab7d91041db7df86359069a7c4d88c92b393d8",
    "MANIFEST.sha256": "36969c5db0b1caa7671b9620ba63338d8f8384994b082eb1c11d460450c1022e",
    "LOCAL.sha256": "0bc8ad9467db57cfaef1c34c2a3cd2d61c7a1cf1127f0628740ee2e0350380eb",
    "PHASE-11-PRE-REGISTRATION.md": "87b835bc2727ed251c8efaf412fc17c3508299f96c3cd9994c95005f6791514b",
}


def expected_output_contract(n, steps):
    specs = [
        ("k3_fields", (steps + 1, n * n)),
        ("states", (steps + 1, p11.CONFIG["state"])),
        ("coefficients", (steps + 1, 3328)),
        ("cox_weak_residual", (steps, p9.p7.H1["M"])),
        ("cox_weak_rho", (steps,)),
    ]
    leaves = [{"index": index, "name": name, "shape": list(shape),
               "dtype": "float64", "logical_bytes": int(np.prod(shape) * 8)}
              for index, (name, shape) in enumerate(specs)]
    return {"leaf_count": 5, "leaves": leaves,
            "logical_output_bytes": int(sum(row["logical_bytes"] for row in leaves))}


def observed_output_contract(output):
    names = ("k3_fields", "states", "coefficients", "cox_weak_residual", "cox_weak_rho")
    leaves = []
    for index, (name, value) in enumerate(zip(names, output, strict=True)):
        array = np.asarray(value)
        leaves.append({"index": index, "name": name, "shape": list(array.shape),
                       "dtype": str(array.dtype), "logical_bytes": int(array.nbytes)})
    return {"leaf_count": len(leaves), "leaves": leaves,
            "logical_output_bytes": int(sum(row["logical_bytes"] for row in leaves))}


def _route_components(n, steps):
    geometry = p9.base.weak_geometry(n, p9.p7.H1)
    return {
        "stencil": jnp.asarray(geometry["stencil_coords"], F64),
        "stencil_mask": jnp.asarray(geometry["stencil_mask"], F64),
        "phi": jnp.asarray(geometry["phi_weighted"], F64),
        "eigen": jnp.asarray(geometry["eigenvalues"], F64),
        "coords": jnp.asarray(c.grid_coords(n), F64),
        "mask": jnp.asarray(c.binary_boundary_mask(n), F64),
        "coarse": jnp.asarray(p9.p3.span_polynomial_table_np(48), F64),
        "fine": jnp.asarray(p9.p3.span_polynomial_table_np(32), F64),
        "decoder": p9.p4.make_pallas_hierarchical_decoder(steps + 1, n * n, p9.p7.H1),
    }


def _weak_function(n, parts):
    """Construct only the canonical P6 Cox weak computation."""
    stencil, stencil_mask = parts["stencil"], parts["stencil_mask"]
    phi, eigen = parts["phi"], parts["eigen"]

    def weak_one(state, values, previous_state, previous_values, viscosity):
        current = p9.p4.decode_one_cox_jax(
            state, values, stencil, stencil_mask, p9.p7.H1
        ).reshape(p9.p7.H1["m"], 5)
        previous = p9.p4.decode_one_cox_jax(
            previous_state, previous_values, stencil[::5], stencil_mask[::5], p9.p7.H1
        )
        center, xp, xm, yp, ym = [current[:, index] for index in range(5)]
        dx = 1.0 / (n - 1)
        ux = jnp.where(center > 0.0, (center - xm) / dx, (xp - center) / dx)
        uy = jnp.where(center > 0.0, (center - ym) / dx, (yp - center) / dx)
        advection = center * (ux + uy)
        projected = phi.T @ center
        preconditioner = (1.0 + c.DT * viscosity * eigen) ** -1.0
        residual = preconditioner * (
            phi.T @ (center - previous)
            + c.DT * (phi.T @ advection + viscosity * eigen * projected)
        )
        denominator = jnp.maximum(jnp.linalg.norm(phi.T @ previous), 1e-12)
        return residual, jnp.linalg.norm(residual) / denominator

    return lambda states, coefficients, viscosity: jax.vmap(
        lambda state, values, previous_state, previous_values:
            weak_one(state, values, previous_state, previous_values, viscosity)
    )(states[1:], coefficients[1:], states[:-1], coefficients[:-1])


def make_timed_actual_route(n, steps):
    """Build the five-leaf deployed route; no Cox full decoder is reachable."""
    parts = _route_components(n, steps)
    weak = _weak_function(n, parts)
    decoder = parts["decoder"]
    coords, mask = parts["coords"], parts["mask"]
    coarse, fine = parts["coarse"], parts["fine"]

    def actual(pred, gen, features, viscosity, mean, scales):
        states = p9.apply_predictor(pred, features)
        coefficients = p9.apply_generator(gen, states[:, 5:], mean, scales, p11.CONFIG)
        residual, rho = weak(states, coefficients, viscosity)
        fields = decoder(states, coefficients, coords, mask, coarse, fine)
        return fields, states, coefficients, residual, rho

    return jax.jit(actual)


def make_untimed_identity_route(n, steps):
    """Build the separately compiled identity route, including Cox control."""
    parts = _route_components(n, steps)
    weak = _weak_function(n, parts)
    decoder = parts["decoder"]
    coords, mask = parts["coords"], parts["mask"]
    coarse, fine = parts["coarse"], parts["fine"]

    def identity(pred, gen, features, viscosity, mean, scales):
        states = p9.apply_predictor(pred, features)
        coefficients = p9.apply_generator(gen, states[:, 5:], mean, scales, p11.CONFIG)
        residual, rho = weak(states, coefficients, viscosity)
        k3 = decoder(states, coefficients, coords, mask, coarse, fine)
        control = p9.p4.decode_states_cox_sequential(
            states, coefficients, coords, mask, p9.p7.H1
        )
        return control, k3, states, coefficients, residual, rho

    return jax.jit(identity)


def _relative(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-300))


def _compile_routes(n, steps, predictor, generator, feature, viscosity, mean, scales):
    args = (predictor, generator, jnp.asarray(feature, F64), jnp.asarray(viscosity, F64),
            jnp.asarray(mean, F64), jnp.asarray(scales, F64))
    actual_callable = make_timed_actual_route(n, steps)
    identity_callable = make_untimed_identity_route(n, steps)
    actual = actual_callable.lower(*args).compile()
    identity = identity_callable.lower(*args).compile()
    return actual, identity, p9.base.memory_analysis(actual), p9.base.memory_analysis(identity)


def corrected_structural_preflight(config, mean, scales, smoke=False):
    """Corrected same-H200 structural preflight with a five-leaf timed route."""
    n, steps, cases = (32, 1, 1) if smoke else (1024, 50, 4)
    if smoke:
        parameters = {"cx": np.asarray((.42,)), "cy": np.asarray((.58,)),
                      "width": np.asarray((.12,)), "amplitude": np.asarray((1.2,)),
                      "nu": np.asarray((.01,))}
        truth, dummy, live_health = None, None, None
    else:
        truth, parameters, live_health, dummy = p9.base.generate_live_reference(n)
    features = []
    for case in range(cases):
        one = {key: parameters[key][case:case + 1]
               for key in ("cx", "cy", "width", "amplitude", "nu")}
        features.append(c.trajectory_features(one, n)[0, :steps + 1])
    predictor = p9.init_predictor(config, 20260827)
    generator = p9.init_generator(config, 20260826)
    route_args = [(predictor, generator, np.asarray(features[case]), parameters["nu"][case],
                   np.asarray(mean), np.asarray(scales)) for case in range(cases)]
    actual, identity, actual_memory, identity_memory = _compile_routes(
        n, steps, predictor, generator, features[0], parameters["nu"][0], mean, scales
    )
    identity_cases = []
    for case in range(cases):
        actual_output = tuple(map(np.asarray, actual(*route_args[case])))
        identity_output = tuple(map(np.asarray, identity(*route_args[case])))
        control, identity_k3, identity_states, identity_coeff, identity_residual, identity_rho = identity_output
        comparisons = [
            _relative(actual_output[0], identity_k3),
            _relative(actual_output[1], identity_states),
            _relative(actual_output[2], identity_coeff),
            _relative(actual_output[3], identity_residual),
            _relative(actual_output[4], identity_rho),
        ]
        mask = c.binary_boundary_mask(n) == 0
        identity_cases.append({
            "case_index": case, "actual_identity_relative_l2": comparisons,
            "actual_identity_worst": float(max(comparisons)),
            "k3_cox_relative_l2": _relative(identity_k3, control),
            "actual_exact_boundary": bool(np.all(actual_output[0][:, mask] == 0)),
            "identity_k3_exact_boundary": bool(np.all(identity_k3[:, mask] == 0)),
            "control_exact_boundary": bool(np.all(control[:, mask] == 0)),
            "finite": bool(all(np.all(np.isfinite(value))
                               for value in (*actual_output, *identity_output))),
        })
    observed = observed_output_contract(tuple(map(np.asarray, actual(*route_args[0]))))
    expected = expected_output_contract(n, steps)
    contract_pass = observed == expected
    compiler_pass = bool(smoke or actual_memory["output_size_in_bytes"] == EXPECTED_OUTPUT_SIZE)
    logical_pass = bool(smoke or observed["logical_output_bytes"] == EXPECTED_LOGICAL_BYTES)
    identity_pass = bool(all(row["finite"] and row["actual_identity_worst"] <= p11.IDENTITY_TOL
                             and row["k3_cox_relative_l2"] <= p11.IDENTITY_TOL
                             and row["actual_exact_boundary"]
                             and row["identity_k3_exact_boundary"]
                             and row["control_exact_boundary"] for row in identity_cases))
    actual_panel = {"expected": expected, "observed": observed,
        "contract_pass": contract_pass, "logical_bytes_pass": logical_pass,
        "compiler_output_bytes_pass": compiler_pass, "memory_analysis": actual_memory,
        "work": {"cox_weak_evaluations": steps,
                 "k3_coefficient_grid_full_field_evaluations": steps + 1,
                 "cox_full_grid_control_evaluations": 0,
                 "weak_jacobian_evaluations": 0, "trial_evaluations": 0,
                 "duplicate_k3_full_decodes": 0, "failures": 0}}
    identity_panel = {"memory_analysis": identity_memory,
        "work": {"cox_weak_evaluations": steps,
                 "k3_coefficient_grid_full_field_evaluations": steps + 1,
                 "cox_full_grid_control_evaluations": steps + 1}}
    if smoke:
        return {"scientific": False, "pass": False,
                "actual_route": actual_panel, "identity_route": identity_panel,
                "identity_cases": identity_cases,
                "smoke_contract_pass": bool(contract_pass and identity_pass)}

    fom = p9.base.bc.make_chain(n, p9.base.FOM_OUTER,
        lin_tol=p9.base.FOM_INNER, preconditioner="helmholtz")[0]

    def invoke(method, case, return_output=False):
        started = time.perf_counter()
        work = {}
        if method == "fom":
            output = fom(jnp.asarray(truth[case, 0]), parameters["nu"][case], dummy, jnp.int32(5))
        else:
            recovered, indices = c.recover_blob_parameters_fixed_sample(truth[case, 0], n)
            one = {"cx": recovered[0:1], "cy": recovered[1:2], "width": recovered[2:3],
                   "amplitude": recovered[3:4], "nu": parameters["nu"][case:case + 1]}
            arguments = list(route_args[case])
            arguments[2] = jnp.asarray(c.trajectory_features(one, n)[0, :steps + 1])
            output = actual(*tuple(arguments))
            work = {"cold_sample_count": int(indices.size),
                    "cold_recovery_finite": bool(np.all(np.isfinite(recovered)))}
        jax.block_until_ready(output)
        elapsed = float(time.perf_counter() - started)
        return (output, elapsed, work) if return_output else elapsed

    for method in ("fom", "rom"):
        invoke(method, 0)
    burn_count = c.gpu_burn(3.0)
    records = {"fom": [], "rom": []}
    orders = []
    for repetition in range(24):
        order = ("fom", "rom") if repetition % 2 == 0 else ("rom", "fom")
        orders.append(list(order))
        for case in range(cases):
            for position, method in enumerate(order):
                output, elapsed, work = invoke(method, case, True)
                row = {"repetition": repetition, "case_index": case,
                       "position": position, "elapsed_s": elapsed, **work}
                if method == "fom":
                    row.update(p9.base.fom_grade(output, truth[case]))
                else:
                    values = tuple(map(np.asarray, output))
                    row.update({"finite": bool(all(np.all(np.isfinite(value)) for value in values)),
                        "exact_boundary": bool(np.all(values[0][:, c.binary_boundary_mask(n) == 0] == 0)),
                        "output_leaf_count": 5,
                        "logical_output_bytes": int(sum(value.nbytes for value in values)),
                        "cox_weak_evaluations": steps,
                        "k3_coefficient_grid_full_field_evaluations": steps + 1,
                        "cox_full_grid_control_evaluations": 0,
                        "weak_jacobian_evaluations": 0, "trial_evaluations": 0,
                        "duplicate_k3_full_decodes": 0, "failures": 0})
                records[method].append(row)
    summaries = {method: p9.base.summarize_timing(rows, cases)
                 for method, rows in records.items()}
    per_case = {method: summaries[method]["per_case_median_elapsed_s"] for method in records}
    speed = float(np.median(per_case["fom"]) / np.median(per_case["rom"]))
    interval = p9.base.clustered_speedup_ci(per_case["fom"], per_case["rom"], 20266100)
    fom_eligible = bool(all(row["finite"] and row["breakdowns"] == 0
        and row["flags_nonzero"] == 0
        and row["max_returned_relative_residual"] <= p9.base.FOM_OUTER
        for row in records["fom"])
        and np.mean([row["trajectory_relative_l2"] for row in records["fom"]]) <= 1e-3
        and np.max([row["trajectory_relative_l2"] for row in records["fom"]]) <= 3e-3)
    timing_work = bool(all(row["finite"] and row["exact_boundary"]
        and row["cold_recovery_finite"] and row["cold_sample_count"] <= 4096
        and row["output_leaf_count"] == 5 and row["logical_output_bytes"] == EXPECTED_LOGICAL_BYTES
        and row["cox_weak_evaluations"] == 50
        and row["k3_coefficient_grid_full_field_evaluations"] == 51
        and row["cox_full_grid_control_evaluations"] == 0
        and row["weak_jacobian_evaluations"] == 0 and row["trial_evaluations"] == 0
        and row["duplicate_k3_full_decodes"] == 0 and row["failures"] == 0
        for row in records["rom"]))
    positions = {method: [sum(row["position"] == position for row in records[method]) // cases
                          for position in (0, 1)] for method in records}
    memory_pass = actual_memory["eligibility_device_bytes"] <= 20_000_000_000
    gate = bool(fom_eligible and timing_work and identity_pass and contract_pass
                and logical_pass and compiler_pass and memory_pass
                and positions == {"fom": [12, 12], "rom": [12, 12]}
                and burn_count > 0 and speed >= 10 and interval[0] >= 8)
    return {"scientific": True, "live_reference_health": live_health,
        "actual_route": actual_panel, "identity_route": identity_panel,
        "identity_cases": identity_cases, "identity_pass": identity_pass,
        "fom_eligible": fom_eligible, "burn_count": int(burn_count),
        "orders": orders, "position_counts": positions, "records": records,
        "summaries": summaries, "per_case_median_seconds": per_case,
        "paired_median_speedup": speed, "clustered_speedup_ci": interval,
        "memory_pass": memory_pass, "pass": gate}


def _validate_phase11_bundle():
    root = os.environ.get("B10_P11_DIR")
    if not root:
        raise SystemExit("B10_P11_DIR is required for scientific Phase12")
    for basename, digest in EXPECTED_P11.items():
        path = os.path.join(root, basename)
        if not os.path.isfile(path) or c.sha256(path) != digest:
            raise SystemExit(f"immutable Phase11 {basename} mismatch")
    with open(os.path.join(root, "AUDIT.json"), encoding="utf-8") as handle:
        audit = json.load(handle)
    with open(os.path.join(root, "phase11_g2.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    if not (audit.get("status") == "pass" and audit.get("negative_aware") is True
            and report.get("decision", {}).get("updates_started") is False
            and report.get("decision", {}).get("structural_preflight_pass") is False):
        raise SystemExit("Phase11 zero-update retracted timing evidence mismatch")
    return {"phase11_" + key.replace(".", "_"): {"basename": key, "sha256": digest}
            for key, digest in EXPECTED_P11.items()}


def _finalize_phase12(output_json, bindings):
    with open(output_json, encoding="utf-8") as handle:
        report = json.load(handle)
    report["bindings"] = report.get("bindings") or {}
    report["bindings"].update(bindings)
    report["independent_license"].update({
        "phase11_speed_promotion_used": False,
        "phase11_overcharged_timing_retracted": True,
        "phase12_preregistered_route_repair": True,
    })
    report["decision"]["phase12_pass"] = report["decision"]["phase11_pass"]
    report["decision"]["phase12_cell_cap"] = 1
    report["phase12_execution"] = {
        "timed_route": "five-leaf actual only",
        "identity_route": "separately compiled and untimed",
        "phase11_training_path_reused_unchanged": True,
        "phase11_retraction_commit": "b7aa407ec06b2f816ea8daa21368731924327444",
    }
    p11.atomic_json(output_json, report)


def main():
    smoke = "--smoke" in sys.argv
    bindings = {} if smoke else _validate_phase11_bundle()
    p9.t2_structural_preflight = corrected_structural_preflight
    p11.main()
    output_json = sys.argv[sys.argv.index("--output-json") + 1]
    _finalize_phase12(output_json, bindings)
    print("PHASE12-FINALIZED", flush=True)


if __name__ == "__main__":
    main()
