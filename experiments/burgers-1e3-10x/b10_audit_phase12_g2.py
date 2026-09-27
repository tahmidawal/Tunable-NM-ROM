#!/usr/bin/env python
"""Independent negative-aware audit for the Phase-12 corrected G2 route."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import b10_audit_phase9_train as a9
import b10_audit_phase11_g2 as a11
import b10_common as c
import b10_phase12_g2 as p12
import b10_phase11_g2 as p11
import b10_phase9_train as p9


def close(left, right, rtol=2e-13, atol=2e-14):
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=rtol,
                            atol=atol, equal_nan=False))


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _independent_routes(n, steps):
    """Independently express the declared actual and identity call graphs."""
    geometry = p9.base.weak_geometry(n, p9.p7.H1)
    stencil = jnp.asarray(geometry["stencil_coords"], jnp.float64)
    stencil_mask = jnp.asarray(geometry["stencil_mask"], jnp.float64)
    phi = jnp.asarray(geometry["phi_weighted"], jnp.float64)
    eigen = jnp.asarray(geometry["eigenvalues"], jnp.float64)
    coords = jnp.asarray(c.grid_coords(n), jnp.float64)
    mask = jnp.asarray(c.binary_boundary_mask(n), jnp.float64)
    coarse = jnp.asarray(p9.p3.span_polynomial_table_np(48), jnp.float64)
    fine = jnp.asarray(p9.p3.span_polynomial_table_np(32), jnp.float64)
    decoder = p9.p4.make_pallas_hierarchical_decoder(steps + 1, n * n, p9.p7.H1)

    def shared(pred, gen, features, viscosity, mean, scales):
        states = p9.apply_predictor(pred, features)
        coefficients = p9.apply_generator(gen, states[:, 5:], mean, scales, p11.CONFIG)
        def one(state, values, previous_state, previous_values):
            current = p9.p4.decode_one_cox_jax(
                state, values, stencil, stencil_mask, p9.p7.H1
            ).reshape(p9.p7.H1["m"], 5)
            previous = p9.p4.decode_one_cox_jax(
                previous_state, previous_values, stencil[::5], stencil_mask[::5], p9.p7.H1
            )
            center, xp, xm, yp, ym = [current[:, index] for index in range(5)]
            dx = 1.0 / (n - 1)
            advection = center * (
                jnp.where(center > 0, (center - xm) / dx, (xp - center) / dx)
                + jnp.where(center > 0, (center - ym) / dx, (yp - center) / dx)
            )
            projected = phi.T @ center
            residual = (1 + c.DT * viscosity * eigen) ** -1 * (
                phi.T @ (center - previous)
                + c.DT * (phi.T @ advection + viscosity * eigen * projected)
            )
            rho = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(phi.T @ previous), 1e-12)
            return residual, rho
        residual, rho = jax.vmap(one)(
            states[1:], coefficients[1:], states[:-1], coefficients[:-1])
        k3 = decoder(states, coefficients, coords, mask, coarse, fine)
        return states, coefficients, residual, rho, k3

    def actual(pred, gen, features, viscosity, mean, scales):
        states, coefficients, residual, rho, k3 = shared(
            pred, gen, features, viscosity, mean, scales)
        return k3, states, coefficients, residual, rho

    def identity(pred, gen, features, viscosity, mean, scales):
        states, coefficients, residual, rho, k3 = shared(
            pred, gen, features, viscosity, mean, scales)
        control = p9.p4.decode_states_cox_sequential(
            states, coefficients, coords, mask, p9.p7.H1)
        return control, k3, states, coefficients, residual, rho

    return jax.jit(actual), jax.jit(identity)


def _parameters(cases):
    cx, cy, width, amplitude, nu, _ = c.bf.sample_params(
        seed=p9.base.FOM_SEED, m=p9.base.FOM_DRAW_COUNT)
    return {"cx": np.asarray(cx[:cases]), "cy": np.asarray(cy[:cases]),
            "width": np.asarray(width[:cases]), "amplitude": np.asarray(amplitude[:cases]),
            "nu": np.asarray(nu[:cases])}


def rebuild_structural(mean, scales, smoke):
    n, steps, cases = (32, 1, 1) if smoke else (1024, 50, 4)
    parameters = ({"cx": np.asarray((.42,)), "cy": np.asarray((.58,)),
                   "width": np.asarray((.12,)), "amplitude": np.asarray((1.2,)),
                   "nu": np.asarray((.01,))} if smoke else _parameters(cases))
    features = []
    for case in range(cases):
        one = {key: parameters[key][case:case + 1]
               for key in ("cx", "cy", "width", "amplitude", "nu")}
        features.append(c.trajectory_features(one, n)[0, :steps + 1])
    predictor = p9.init_predictor(p11.CONFIG, 20260827)
    generator = p9.init_generator(p11.CONFIG, 20260826)
    actual_fn, identity_fn = _independent_routes(n, steps)
    args0 = (predictor, generator, jnp.asarray(features[0]), jnp.asarray(parameters["nu"][0]),
             jnp.asarray(mean), jnp.asarray(scales))
    actual = actual_fn.lower(*args0).compile()
    identity = identity_fn.lower(*args0).compile()
    cases_out = []
    for case in range(cases):
        args = (predictor, generator, jnp.asarray(features[case]), jnp.asarray(parameters["nu"][case]),
                jnp.asarray(mean), jnp.asarray(scales))
        aout = tuple(map(np.asarray, actual(*args)))
        iout = tuple(map(np.asarray, identity(*args)))
        control, k3, states, coefficients, residual, rho = iout
        comparisons = [p12._relative(aout[0], k3), p12._relative(aout[1], states),
                       p12._relative(aout[2], coefficients), p12._relative(aout[3], residual),
                       p12._relative(aout[4], rho)]
        boundary = c.binary_boundary_mask(n) == 0
        cases_out.append({"case_index": case, "actual_identity_relative_l2": comparisons,
            "actual_identity_worst": float(max(comparisons)),
            "k3_cox_relative_l2": p12._relative(k3, control),
            "actual_exact_boundary": bool(np.all(aout[0][:, boundary] == 0)),
            "identity_k3_exact_boundary": bool(np.all(k3[:, boundary] == 0)),
            "control_exact_boundary": bool(np.all(control[:, boundary] == 0)),
            "finite": bool(all(np.all(np.isfinite(value)) for value in (*aout, *iout)))})
    return {"expected": p12.expected_output_contract(n, steps),
            "observed": p12.observed_output_contract(tuple(map(np.asarray, actual(*args0)))),
            "actual_memory": p9.base.memory_analysis(actual),
            "identity_memory": p9.base.memory_analysis(identity),
            "identity_cases": cases_out}


def structural_check(report, smoke, independent=None):
    panel = report["structural_preflight"]
    if smoke and panel.get("separately_smoked") is True:
        return True
    n, steps, cases = (32, 1, 1) if smoke else (1024, 50, 4)
    expected = p12.expected_output_contract(n, steps)
    actual = panel.get("actual_route", {})
    identity = panel.get("identity_route", {})
    memory_keys = {"argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                   "alias_size_in_bytes", "host_argument_size_in_bytes",
                   "host_output_size_in_bytes", "host_temp_size_in_bytes",
                   "eligibility_device_bytes"}
    work = {"cox_weak_evaluations": steps,
            "k3_coefficient_grid_full_field_evaluations": steps + 1,
            "cox_full_grid_control_evaluations": 0, "weak_jacobian_evaluations": 0,
            "trial_evaluations": 0, "duplicate_k3_full_decodes": 0, "failures": 0}
    identity_work = {"cox_weak_evaluations": steps,
                     "k3_coefficient_grid_full_field_evaluations": steps + 1,
                     "cox_full_grid_control_evaluations": steps + 1}
    contract = bool(actual.get("expected") == expected and actual.get("observed") == expected
        and actual.get("contract_pass") is True and actual.get("logical_bytes_pass") is True
        and actual.get("compiler_output_bytes_pass") is True
        and set(actual.get("memory_analysis", {})) == memory_keys
        and set(identity.get("memory_analysis", {})) == memory_keys
        and actual.get("memory_analysis", {}).get("output_size_in_bytes")
            == expected["logical_output_bytes"] + 5 * 8
        and actual.get("work") == work and identity.get("work") == identity_work)
    if not smoke:
        contract &= bool(actual["memory_analysis"]["output_size_in_bytes"] == p12.EXPECTED_OUTPUT_SIZE
            and actual["observed"]["logical_output_bytes"] == p12.EXPECTED_LOGICAL_BYTES
            and actual["memory_analysis"]["eligibility_device_bytes"] <= 20_000_000_000)
    identities = panel.get("identity_cases", [])
    identity_ok = bool(len(identities) == cases and all(
        row["case_index"] == index and row["finite"]
        and len(row["actual_identity_relative_l2"]) == 5
        and close(max(row["actual_identity_relative_l2"]), row["actual_identity_worst"])
        and row["actual_identity_worst"] <= p11.IDENTITY_TOL
        and row["k3_cox_relative_l2"] <= p11.IDENTITY_TOL
        and row["actual_exact_boundary"] and row["identity_k3_exact_boundary"]
        and row["control_exact_boundary"] for index, row in enumerate(identities)))
    if independent is not None:
        contract &= bool(independent["expected"] == expected
            and independent["observed"] == expected
            and independent["actual_memory"] == actual["memory_analysis"]
            and independent["identity_memory"] == identity["memory_analysis"])
        identity_ok &= bool(a11.nested_close(independent["identity_cases"], identities))
    if smoke:
        return bool(panel.get("scientific") is False and panel.get("pass") is False
                    and panel.get("smoke_contract_pass") is True and contract and identity_ok)
    records = panel.get("records", {})
    if set(records) != {"fom", "rom"} or any(len(records[key]) != 96 for key in records):
        return False
    summaries = {method: p9.base.summarize_timing(rows, cases)
                 for method, rows in records.items()}
    per_case = {method: summaries[method]["per_case_median_elapsed_s"] for method in records}
    speed = float(np.median(per_case["fom"]) / np.median(per_case["rom"]))
    interval = p9.base.clustered_speedup_ci(per_case["fom"], per_case["rom"], 20266100)
    positions = {method: [sum(row["position"] == position for row in records[method]) // cases
                          for position in (0, 1)] for method in records}
    order = [["fom", "rom"] if rep % 2 == 0 else ["rom", "fom"] for rep in range(24)]
    timing_indices = bool(panel.get("orders") == order and all(
        row["repetition"] == rep and row["case_index"] == case
        for method in records for rep in range(24) for case in range(cases)
        for row in [records[method][rep * cases + case]]))
    fom = bool(all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
        and row["max_returned_relative_residual"] <= p9.base.FOM_OUTER for row in records["fom"])
        and np.mean([row["trajectory_relative_l2"] for row in records["fom"]]) <= 1e-3
        and np.max([row["trajectory_relative_l2"] for row in records["fom"]]) <= 3e-3)
    rom = bool(all(row["finite"] and row["exact_boundary"] and row["cold_recovery_finite"]
        and row["cold_sample_count"] <= 4096 and row["output_leaf_count"] == 5
        and row["logical_output_bytes"] == p12.EXPECTED_LOGICAL_BYTES
        and row["cox_weak_evaluations"] == 50
        and row["k3_coefficient_grid_full_field_evaluations"] == 51
        and row["cox_full_grid_control_evaluations"] == 0
        and row["weak_jacobian_evaluations"] == 0 and row["trial_evaluations"] == 0
        and row["duplicate_k3_full_decodes"] == 0 and row["failures"] == 0
        for row in records["rom"]))
    gate = bool(contract and identity_ok and fom and rom and panel.get("memory_pass")
        and positions == {"fom": [12, 12], "rom": [12, 12]}
        and timing_indices and panel.get("burn_count", 0) > 0 and speed >= 10 and interval[0] >= 8)
    return bool(panel.get("scientific") is True and panel.get("identity_pass") == identity_ok
        and panel.get("fom_eligible") == fom and panel.get("pass") == gate
        and panel.get("position_counts") == positions
        and a11.nested_close(panel.get("summaries"), summaries)
        and a11.nested_close(panel.get("per_case_median_seconds"), per_case)
        and close(panel.get("paired_median_speedup"), speed)
        and close(panel.get("clustered_speedup_ci"), interval))


def structural_corruption_self_test():
    expected = p12.expected_output_contract(1024, 50)
    smoke_expected = p12.expected_output_contract(32, 1)
    memory = {name: 0 for name in ("argument_size_in_bytes", "output_size_in_bytes",
        "temp_size_in_bytes", "alias_size_in_bytes", "host_argument_size_in_bytes",
        "host_output_size_in_bytes", "host_temp_size_in_bytes", "eligibility_device_bytes")}
    memory["output_size_in_bytes"] = smoke_expected["logical_output_bytes"] + 5 * 8
    panel = {"scientific": False, "pass": False, "smoke_contract_pass": True,
        "actual_route": {"expected": smoke_expected,
            "observed": smoke_expected, "contract_pass": True,
            "logical_bytes_pass": True, "compiler_output_bytes_pass": True,
            "memory_analysis": dict(memory),
            "work": {"cox_weak_evaluations": 1,
                "k3_coefficient_grid_full_field_evaluations": 2,
                "cox_full_grid_control_evaluations": 0, "weak_jacobian_evaluations": 0,
                "trial_evaluations": 0, "duplicate_k3_full_decodes": 0, "failures": 0}},
        "identity_route": {"memory_analysis": dict(memory),
            "work": {"cox_weak_evaluations": 1,
                "k3_coefficient_grid_full_field_evaluations": 2,
                "cox_full_grid_control_evaluations": 2}},
        "identity_cases": [{"case_index": 0, "actual_identity_relative_l2": [0.] * 5,
            "actual_identity_worst": 0., "k3_cox_relative_l2": 0.,
            "actual_exact_boundary": True, "identity_k3_exact_boundary": True,
            "control_exact_boundary": True, "finite": True}]}
    report = {"structural_preflight": panel}
    positive = structural_check(report, True)
    corruptions = {}
    for mutation in ("sixth_leaf", "hidden_count", "removed", "reordered", "dtype",
                     "shape", "logical", "compiler", "identity", "boundary"):
        bad = copy.deepcopy(report)
        actual = bad["structural_preflight"]["actual_route"]
        if mutation == "sixth_leaf":
            extra = {"index": 5, "name": "cox_control", "shape": [51, 1048576],
                     "dtype": "float64", "logical_bytes": 427819008}
            actual["observed"]["leaves"].append(extra)
            actual["observed"]["leaf_count"] = 6
            actual["observed"]["logical_output_bytes"] += 427819008
            actual["memory_analysis"]["output_size_in_bytes"] += 427819008
        elif mutation == "hidden_count": actual["work"]["cox_full_grid_control_evaluations"] = 51
        elif mutation == "removed": actual["observed"]["leaves"].pop()
        elif mutation == "reordered": actual["observed"]["leaves"][0:2] = reversed(actual["observed"]["leaves"][0:2])
        elif mutation == "dtype": actual["observed"]["leaves"][0]["dtype"] = "float32"
        elif mutation == "shape": actual["observed"]["leaves"][0]["shape"][0] += 1
        elif mutation == "logical": actual["observed"]["logical_output_bytes"] += 8
        elif mutation == "compiler": actual["memory_analysis"]["output_size_in_bytes"] += 8
        elif mutation == "identity": bad["structural_preflight"]["identity_cases"][0]["actual_identity_worst"] = 1e-6
        elif mutation == "boundary": bad["structural_preflight"]["identity_cases"][0]["control_exact_boundary"] = False
        corruptions[mutation] = structural_check(bad, True)
    return {"positive_five_leaf_contract": positive,
            "hidden_control_extra_logical_bytes": expected["leaves"][0]["logical_bytes"],
            "corruption_accepted": corruptions,
            "ten_corruptions_rejected": not any(corruptions.values()),
            "pass": bool(positive and not any(corruptions.values()))}


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
    stdout = open(args.slurm_out, encoding="utf-8").read()
    stderr = open(args.slurm_err, encoding="utf-8").read()
    provenance = {"commit": report["provenance"].get("commit") == args.expected_commit,
        "job": str(report["provenance"].get("slurm_job_id")) == str(args.expected_job),
        "gpu": report["provenance"].get("jax_backend") == "gpu"
               and (args.smoke or report["provenance"].get("gpu_kind") == "NVIDIA H200"),
        "precision": report["provenance"].get("x64") is True
                     and report["provenance"].get("matmul_precision") == "highest",
        "logs": (args.smoke or "jax_backend=gpu" in stdout) and "ALL-DONE" in stdout
                and "PHASE12-FINALIZED" in stdout
                and not re.search(r"(?i)(captured.*large.*constant|oom|out of memory|traceback|disk.*full)", stdout + stderr)}
    source_binding = bool(args.smoke or all(rows.get("code/" + name) == digest
        for name, digest in report["provenance"].get("source_sha256", {}).items()))
    file_binding = bool(args.smoke or (source_binding
        and rows.get("code/b10_phase12_g2.py") is not None
        and rows.get("code/b10_audit_phase12_g2.py") is not None
        and rows.get("code/PHASE-12-PRE-REGISTRATION.md") == c.sha256(args.prereg)
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
        extra_names = {"p9_json": "phase9_terminal_recovery.json", "p9_npz": "phase9_terminal_recovery.npz",
            "p9_checkpoint": "checkpoint.pkl", "p9_audit": "AUDIT.json", "p9_audit_work": "AUDIT-WORK.npz",
            "p9_work_checkpoint": "work_checkpoint.pkl", "p9_manifest": "MANIFEST.sha256",
            "p10_json": "phase10_d.json", "p10_npz": "phase10_d.npz",
            "p10_work_checkpoint": "work_checkpoint.pkl", "p10_audit": "AUDIT.json",
            "p10_manifest": "MANIFEST.sha256", "p10_audit_manifest": "AUDIT-MANIFEST.sha256"}
        for key, basename in extra_names.items():
            group = "p9" if key.startswith("p9_") else "p10"
            dependency_binding &= rows.get(f"code/deps/{group}/{basename}") == expected[key]
        for basename, digest in p12.EXPECTED_P11.items():
            key = "phase11_" + basename.replace(".", "_")
            dependency_binding &= rows.get(f"code/deps/p11/{basename}") == expected[key] == digest
    parameter_counts = {"generator": p9.tree_count(checkpoint["generator"]),
        "encoder": p9.tree_count(checkpoint["encoder"]),
        "predictor": p9.tree_count(checkpoint["predictor"])} == {
            "generator": 165954, "encoder": 164384, "predictor": 2533}
    schedule = a11.schedule_check(report, arrays, args.smoke)
    metrics = True
    for name, row in report.get("history", {}).items():
        metrics &= a9.metric_match(row["metrics"], arrays, name, False)
    for report_key, prefix in (("terminal_train_control", "terminal_train"),
                               ("globalized_train", "globalized_train"),
                               ("train_direct", "train_direct")):
        if report.get(report_key) is not None:
            metrics &= a9.metric_match(report[report_key], arrays, prefix, not args.smoke)
    train_trace = ({"pass": True, "present": False} if "train_trust_attempted" not in arrays
                   else a11.trace_check(arrays, "train_trust", False))
    selection_trace = ({"pass": True, "present": False} if "selection_trust_attempted" not in arrays
                       else a11.trace_check(arrays, "selection_trust", True))
    data_fields = ({"pass": True, "skipped": "pre-update stop"}
                   if not report["decision"]["updates_started"]
                   else a11.data_and_field_check(args, report, arrays, checkpoint, args.smoke))
    independent = None
    if not (args.smoke and report["structural_preflight"].get("separately_smoked") is True):
        independent = rebuild_structural(arrays["coefficient_mean"], arrays["head_scales"], args.smoke)
    structural = structural_check(report, args.smoke, independent)
    preflight = a11.preflight_check(report, args.smoke)
    work = a11.work_checkpoint_check(args, report, arrays, checkpoint, args.smoke)
    decision = a11.decision_check(report, args.smoke)
    phase12_decision = bool(report["decision"].get("phase12_pass") == report["decision"]["phase11_pass"]
        and report["decision"].get("phase12_cell_cap") == 1
        and report["independent_license"].get("phase11_speed_promotion_used") is False
        and report["independent_license"].get("phase11_overcharged_timing_retracted") is True)
    information = bool(report["information_boundary"] == {
        "train_only_weights": True,
        "selection_loaded_after_globalized_and_direct_train_pass": bool(not report["decision"]["selection_evaluated"] or (report["decision"]["globalized_train_pass"] and report["decision"]["train_direct_pass"])),
        "selection_target_coefficients_training_use": False, "model_validation_touched": False,
        "confirmation_touched": False, "weak_eq_fitting_touched": False,
        "scaling_touched": False, "retracted_capacity_touched": False})
    fold = bool(not report["decision"]["predictor_trained"] or
                (np.isfinite(report["predictor_fold_identity"]) and report["predictor_fold_identity"] <= 1e-12))
    negative = {"trust": a11.negative_self_test(), "structural": structural_corruption_self_test()}
    negative["pass"] = bool(negative["trust"]["pass"] and negative["structural"]["pass"])
    health = bool(all(provenance.values()) and file_binding and dependency_binding and parameter_counts
        and schedule and metrics and train_trace["pass"] and selection_trace["pass"]
        and data_fields["pass"] and preflight and structural and work and decision["pass"]
        and phase12_decision and information and fold and negative["pass"])
    result = {"status": "pass" if health else "fail", "negative_aware": True,
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
            "work_checkpoint": work, "decision": decision, "phase12_decision": phase12_decision,
            "information_boundary": information, "predictor_fold": fold,
            "negative_self_test": negative}, "decision": report["decision"]}
    p11.atomic_json(args.output, result)
    if not health:
        raise SystemExit("Phase12 independent audit failed")
    print(json.dumps({"status": "pass", "decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
