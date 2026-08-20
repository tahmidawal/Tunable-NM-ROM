#!/usr/bin/env python
"""Independent negative-aware audit for the Burgers Phase-5 diagnostic."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os

import numpy as np

import b10_common as c
import b10_phase5 as p5
import b10_phase5_d as d
import b10_s0_spline as base
import b10_spline as s


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def close(observed, expected, tolerance=2e-12):
    return math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=1e-15)


def parse_manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            rows[relative.removeprefix("./")] = digest
    return rows


def compare_summary(observed, recomputed, name):
    scalar_keys = (
        "median_elapsed_s", "outliers_gt_1p5_within_trajectory_total",
    )
    for key in scalar_keys:
        if key not in observed or key not in recomputed:
            raise SystemExit(f"{name} missing timing summary {key}")
        if isinstance(observed[key], int):
            okay = observed[key] == recomputed[key]
        else:
            okay = close(observed[key], recomputed[key])
        if not okay:
            raise SystemExit(f"{name} timing summary mismatch: {key}")
    for key in (
        "elapsed_all_s", "per_case_median_elapsed_s",
        "outliers_gt_1p5_within_trajectory_all",
    ):
        if not np.allclose(observed[key], recomputed[key], rtol=2e-12, atol=1e-15):
            raise SystemExit(f"{name} timing array mismatch: {key}")


def audit_targets(report, main, json_path, smoke):
    target = report["train_targets"]
    expected = {24: (0, 2, 1)} if smoke else {
        n: (start, count, 51) for n, (start, count, _) in d.TRAIN.items()
    }
    chunks = target["chunks"]
    total = 0
    sums = np.zeros(3328, np.float64)
    seen = {n: [] for n in expected}
    paths = []
    next_global_snapshot = 0
    sampled = None
    if not smoke:
        values = c.bf.sample_params(seed=d.TARGET_SEED, m=d.TARGET_DRAW_COUNT)
        sampled = dict(zip(
            ("cx", "cy", "width", "amplitude", "nu", "normalized"),
            (np.asarray(value, np.float64) for value in values),
        ))
    for row in chunks:
        n = int(row["N"])
        if n not in expected:
            raise SystemExit("unexpected target mesh")
        path = os.path.join(os.path.dirname(json_path), "targets", row["basename"])
        if c.sha256(path) != row["sha256"]:
            raise SystemExit(f"target chunk checksum mismatch: {row['basename']}")
        paths.append(path)
        with np.load(path, allow_pickle=False) as arrays:
            cases, times = len(row["indices"]), expected[n][2]
            shape = (cases, times)
            if arrays["coefficients"].shape != shape + (3328,):
                raise SystemExit("target coefficient shape mismatch")
            if arrays["affine"].shape != shape + (5,) or arrays["features"].shape != shape + (7,):
                raise SystemExit("target state/feature shape mismatch")
            indices = np.asarray(row["indices"], np.int64)
            expected_index = np.broadcast_to(indices[:, None], shape)
            if not np.array_equal(arrays["source_draw_index"], expected_index):
                raise SystemExit("target source-draw metadata mismatch")
            expected_global = np.arange(
                next_global_snapshot, next_global_snapshot + cases * times,
                dtype=np.int64,
            ).reshape(shape)
            if not np.array_equal(arrays["global_snapshot_index"], expected_global):
                raise SystemExit("target global-snapshot metadata mismatch")
            if (row["global_snapshot_start"] != next_global_snapshot
                    or row["global_snapshot_stop"] != next_global_snapshot + cases * times):
                raise SystemExit("target global-snapshot range mismatch")
            next_global_snapshot += cases * times
            if not np.array_equal(arrays["time_index"], np.broadcast_to(np.arange(times), shape)):
                raise SystemExit("target time metadata mismatch")
            if not np.array_equal(arrays["N"], np.full(shape, n)):
                raise SystemExit("target mesh metadata mismatch")
            parameters = {name: np.asarray(arrays[f"parameter_{name}"])
                          for name in ("cx", "cy", "width", "amplitude", "nu")}
            if not smoke:
                for name, parameter in parameters.items():
                    if not np.array_equal(parameter, sampled[name][indices]):
                        raise SystemExit(f"target regenerated parameter mismatch: {name}")
                if not np.array_equal(
                    arrays["normalized_parameters"], sampled["normalized"][indices]
                ):
                    raise SystemExit("target regenerated normalized-parameter mismatch")
            features = c.trajectory_features(parameters, n)[:, :times]
            if not np.array_equal(arrays["features"], features):
                raise SystemExit("target feature recomputation mismatch")
            coefficients = np.asarray(arrays["coefficients"])
            normal = np.asarray(arrays["normal"])
            health = (np.all(np.isfinite(coefficients), axis=-1)
                      & np.asarray(arrays["coefficient_finite"], bool)
                      & np.asarray(arrays["prediction_finite"], bool)
                      & np.asarray(arrays["rhs_finite"], bool)
                      & np.isfinite(normal) & (normal <= s.ORACLE_NORMAL_TOL)
                      & np.asarray(arrays["boundary"], bool)
                      & (np.asarray(arrays["pou"]) == 0.0)
                      & (np.asarray(arrays["support"]) == 32))
            if not np.array_equal(health, np.asarray(arrays["healthy"], bool)):
                raise SystemExit("target independently reconstructed health mismatch")
            checks = (
                row["snapshot_count"] == int(coefficients.shape[0] * coefficients.shape[1]),
                row["healthy_count"] == int(np.sum(health)),
                close(row["normal_worst"], np.max(normal)),
                row["boundary_all"] == bool(np.all(arrays["boundary"])),
                close(row["pou_worst"], np.max(arrays["pou"])),
                row["support_min"] == int(np.min(arrays["support"])),
                row["support_max"] == int(np.max(arrays["support"])),
                row["rhs_finite_all"] == bool(np.all(arrays["rhs_finite"])),
                row["prediction_finite_all"] == bool(np.all(arrays["prediction_finite"])),
                row["coefficient_finite_all"] == bool(np.all(arrays["coefficient_finite"])),
            )
            if not all(checks):
                raise SystemExit("target chunk summary mismatch")
            flattened = coefficients.reshape(-1, 3328)
            sums += np.sum(flattened, axis=0)
            total += flattened.shape[0]
            seen[n].extend(indices.tolist())
        if not smoke:
            health_record = row["reference_health"]
            if not (np.isfinite(health_record["reported_max_relative_residual"])
                    and np.isfinite(health_record["independent_max_relative_residual"])
                    and health_record["reported_max_relative_residual"] <= 1e-8
                    and health_record["independent_max_relative_residual"] <= 1e-8):
                raise SystemExit("target reference health failure")
            if (health_record["seed"] != d.TARGET_SEED
                    or health_record["draw_count"] != d.TARGET_DRAW_COUNT
                    or health_record["indices"] != row["indices"]):
                raise SystemExit("target reference metadata mismatch")
    for n, (start, count, _) in expected.items():
        if seen[n] != list(range(start, start + count)):
            raise SystemExit(f"N{n} target cohort mismatch")
    expected_total = 2 if smoke else d.TARGET_COUNT
    if (total != expected_total or target["snapshot_count"] != total
            or next_global_snapshot != expected_total):
        raise SystemExit("target total mismatch")
    mean = sums / total
    sumsq = np.zeros(2, np.float64)
    source_min = np.full(2, np.inf)
    source_max = np.full(2, -np.inf)
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            values = np.asarray(arrays["coefficients"]).reshape(-1, 3328)
        for head, part in enumerate((values[:, :2304], values[:, 2304:])):
            center = mean[:2304] if head == 0 else mean[2304:]
            sumsq[head] += np.sum((part - center) ** 2)
            source_min[head] = min(source_min[head], float(np.min(part)))
            source_max[head] = max(source_max[head], float(np.max(part)))
    rms = np.sqrt(sumsq / (total * np.asarray((2304, 1024))))
    scales = np.maximum(rms, 1e-12)
    normalization = target["normalization"]
    for stored, expected_value, label in (
        (normalization["coefficient_mean"], mean, "mean"),
        (normalization["head_rms"], rms, "rms"),
        (normalization["head_scales"], scales, "scales"),
        (normalization["source_min"], source_min, "source_min"),
        (normalization["source_max"], source_max, "source_max"),
        (main["coefficient_mean"], mean, "main mean"),
        (main["head_rms"], rms, "main rms"),
        (main["head_scales"], scales, "main scales"),
    ):
        if not np.allclose(stored, expected_value, rtol=2e-13, atol=1e-15):
            raise SystemExit(f"target normalization mismatch: {label}")
    integrity = bool(total == expected_total and np.all(np.isfinite(mean))
                     and np.all(np.isfinite(rms)) and np.all(rms > 0.0)
                     and all(row["healthy_count"] == row["snapshot_count"]
                             and row["normal_worst"] <= s.ORACLE_NORMAL_TOL
                             and row["boundary_all"] and row["pou_worst"] == 0.0
                             and row["support_min"] == 32 and row["support_max"] == 32
                             and row["rhs_finite_all"] and row["prediction_finite_all"]
                             and row["coefficient_finite_all"] for row in chunks))
    if target["integrity_pass"] != integrity:
        raise SystemExit("target integrity decision mismatch")
    return integrity, mean, scales


def expected_timing_orders():
    methods = ["fom", "G1_mandatory", "G1_maximum_one",
               "G2_mandatory", "G2_maximum_one"]
    orders = []
    for repetition in range(d.TIME_REPS):
        offset = repetition % len(methods)
        order = methods[offset:] + methods[:offset]
        if (repetition // len(methods)) % 2:
            order = list(reversed(order))
        orders.append(order)
    return methods, orders


def independent_work_pass(row, suffix):
    expected_shape = [c.NUM_STEPS + 1, 1024 * 1024]
    common = bool(
        row.get("finite") and row.get("zero_failures")
        and row.get("output_shape") == expected_shape
        and row.get("weak_objective_evaluations") == c.NUM_STEPS
        and len(row.get("weak_residual_norm_all", ())) == c.NUM_STEPS
        and np.all(np.isfinite(row.get("weak_residual_norm_all", ())))
    )
    if suffix == "mandatory":
        return bool(
            common and row.get("weak_jacobian_evaluations") == 0
            and row.get("trial_residual_evaluations") == 0
            and row.get("coefficient_grid_evaluations") == 51
            and len(row.get("rho_all", ())) == 50
            and np.all(np.isfinite(row.get("rho_all", ())))
        )
    names = ("rho_before_all", "rho_after_all", "jacobian_frobenius_all",
             "trial_factor_all", "bounded_step_norm_all")
    factors = np.asarray(row.get("trial_factor_all", ()), np.float64)
    return bool(
        common and row.get("weak_jacobian_evaluations") == 50
        and row.get("trial_residual_evaluations") == 200
        and row.get("coefficient_grid_evaluations") == 401
        and all(len(row.get(name, ())) == 50 for name in names + ("accepted_all",))
        and all(np.all(np.isfinite(row.get(name, ()))) for name in names)
        and np.all(np.isin(factors, np.asarray((1.0, 0.5, 0.25, 0.0))))
    )


def audit_science_panel_shape(panel, main):
    methods, orders = expected_timing_orders()
    if panel.get("timing_orders") != orders:
        raise SystemExit("exact cyclic/reversed timing order mismatch")
    position = {method: [sum(order[pos] == method for order in orders)
                         for pos in range(len(methods))] for method in methods}
    if (panel.get("position_counts") != position
            or panel.get("exact_position_balance") is not True
            or not all(value == 4 for counts in position.values() for value in counts)):
        raise SystemExit("exact timing position balance mismatch")
    expected_coverage = {(case, repetition) for case in range(4)
                         for repetition in range(d.TIME_REPS)}
    work = {}
    for method in methods:
        rows = panel["records"][method]
        coverage = {(int(row["case_index"]), int(row["repetition"])) for row in rows}
        if len(rows) != 80 or coverage != expected_coverage or len(coverage) != len(rows):
            raise SystemExit(f"{method} timing coverage mismatch")
        recomputed = base.summarize_timing(rows, 4)
        compare_summary(panel["summaries"][method], recomputed, method)
        observed = np.asarray(main[f"timing_{method}"])
        expected_array = np.asarray([
            [row["case_index"], row["repetition"], row["elapsed_s"]] for row in rows
        ], np.float64)
        if not np.array_equal(observed, expected_array):
            raise SystemExit(f"{method} NPZ timing mismatch")
        if method == "fom":
            continue
        arm, suffix = method.split("_", 1)
        canonical = panel["canonical_work"][method]
        passed = bool(
            len(canonical) == 4
            and sorted(int(row["case_index"]) for row in canonical) == list(range(4))
            and all(row["method"] == method and independent_work_pass(row, suffix)
                    for row in canonical)
        )
        work[method] = passed
    return work


def audit_cost(report, main, mean, scales, smoke):
    panel = report["cost_panel"]
    science_work = None if smoke else audit_science_panel_shape(panel, main)
    arm_pass = {}
    for index, candidate in enumerate(p5.CANDIDATES):
        arm = candidate["arm"]
        parameters = p5.init_generator(candidate, p5.COST_SEED, nonzero_bias=True)
        if p5._parameter_count(parameters) != candidate["parameter_count"]:
            raise SystemExit(f"{arm} parameter count mismatch")
        observed = panel["geometry"][arm]
        recomputed = p5.output_geometry(candidate, parameters, mean, scales)
        if (observed["rank_at_tolerance"] != recomputed["rank_at_tolerance"]
                or not np.allclose(observed["singular_value_ratios"], recomputed["singular_value_ratios"], rtol=2e-12, atol=1e-15)
                or not np.allclose(observed["curvature_all"], recomputed["curvature_all"], rtol=2e-12, atol=1e-15)
                or observed["pass"] != recomputed["pass"]):
            raise SystemExit(f"{arm} output geometry mismatch")
        identity = bool(panel["setup"][arm]["basis_identity"]["pass"]
                        and panel["setup"][arm]["pallas_basis_identity"]["pass"]
                        and all(row["max_relative_l2"] <= d.IDENTITY_TOL
                                and row["exact_boundary"] and row["pass"]
                                for row in panel["identity"][arm]))
        noncollapse = bool(observed["pass"] and all(
            row["finite"] and row["q_std"] >= 1e-3
            and row["coefficient_std"] >= 1e-3 for row in panel["variance"][arm]))
        if smoke:
            gate = panel["gates"][arm]
            if not (gate["identity_pass"] == identity
                    and gate["noncollapse_pass"] == noncollapse
                    and gate["scientific_promotion_allowed"] is False
                    and all(row["finite"] for suffix in ("mandatory", "maximum_one")
                            for row in panel["canonical_work"][f"{arm}_{suffix}"])):
                raise SystemExit(f"{arm} smoke cost gate mismatch")
            arm_pass[arm] = False
            continue
        gate = panel["gates"][arm]
        for suffix in ("mandatory", "maximum_one"):
            method = f"{arm}_{suffix}"
            work = science_work[method]
            summary = base.summarize_timing(panel["records"][method], 4)
            speed = panel["summaries"]["fom"]["median_elapsed_s"] / summary["median_elapsed_s"]
            ci = base.clustered_speedup_ci(
                panel["summaries"]["fom"]["per_case_median_elapsed_s"],
                summary["per_case_median_elapsed_s"],
                20265100 + 20 * index + (suffix == "maximum_one"),
            )
            memory = panel["setup"][arm]["kernels"][suffix]["memory_analysis"]["eligibility_device_bytes"]
            passed = bool(panel["fom_accuracy"]["eligible"] and identity and noncollapse
                          and work and memory <= 20_000_000_000
                          and speed >= 10.0 and ci[0] >= 8.0)
            stored = gate[suffix]
            if not (close(stored["paired_median_speedup"], speed)
                    and np.allclose(stored["clustered_speedup_ci"], ci, rtol=2e-12, atol=1e-15)
                    and stored["canonical_work_pass"] == work
                    and stored["compiled_device_bytes"] == memory
                    and stored["memory_pass"] == (memory <= 20_000_000_000)
                    and stored["pass"] == passed):
                raise SystemExit(f"{method} gate mismatch")
        license_value = gate["mandatory"]["pass"]
        classification = ("correction-capable" if gate["maximum_one"]["pass"] else
                          "conditional-zero-or-occasional-attempt" if license_value else "cost-fail")
        if gate["training_cost_license"] != license_value or gate["correction_classification"] != classification:
            raise SystemExit(f"{arm} cost classification mismatch")
        arm_pass[arm] = license_value
    if not smoke:
        first = [row for row in panel["records"]["fom"] if row["repetition"] == 0]
        mean_error = float(np.mean([row["trajectory_relative_l2"] for row in first]))
        worst_error = float(np.max([row["trajectory_relative_l2"] for row in first]))
        healthy = bool(all(row["finite"] and row["breakdowns"] == 0
                           and row["flags_nonzero"] == 0
                           and row["max_returned_relative_residual"] <= base.FOM_OUTER
                           for row in panel["records"]["fom"]))
        eligible = bool(healthy and mean_error <= 1e-3 and worst_error <= 3e-3)
        accuracy = panel["fom_accuracy"]
        if not (close(accuracy["mean"], mean_error) and close(accuracy["worst"], worst_error)
                and accuracy["healthy"] == healthy and accuracy["eligible"] == eligible):
            raise SystemExit("FOM accuracy eligibility mismatch")
        reference = panel["reference_health"]
        for chain, tolerance in (("reference", base.REFERENCE_OUTER), ("audit", base.AUDIT_OUTER)):
            rows = reference[f"{chain}_records"]
            if not all(row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
                       and row["max_returned_relative_residual"] <= tolerance for row in rows):
                raise SystemExit(f"{chain} reference health mismatch")
        if reference["cross_chain_worst"] > 1e-4:
            raise SystemExit("tight/tighter reference difference failed")
    return arm_pass


def audit_bindings(report, args, manifest):
    p4 = report["bindings"]["P4"]
    for label, path in (("json", args.p4_json), ("npz", args.p4_npz),
                        ("audit", args.p4_audit), ("manifest", args.p4_manifest)):
        if c.sha256(path) != p4[f"{label}_sha256"]:
            raise SystemExit(f"P4 {label} binding mismatch")
    p4_report, p4_audit = load(args.p4_json), load(args.p4_audit)
    if (p4_audit["status"] != "pass" or p4_audit["decision"] != p4_report["decision"]
            or p4_audit["source_json_sha256"] != c.sha256(args.p4_json)
            or p4_audit["source_npz_sha256"] != c.sha256(args.p4_npz)
            or p4_report["npz"]["sha256"] != c.sha256(args.p4_npz)):
        raise SystemExit("P4 independent decision mismatch")
    if not (
        p4["commit"] == p4_report["provenance"]["commit"]
        and str(p4["job_id"]) == str(p4_report["provenance"]["slurm_job_id"])
        and p4["manifest_sha256"] == c.sha256(args.p4_manifest)
        and p4_report["provenance"]["jax_backend"] == "gpu"
        and p4_report["provenance"]["x64"] is True
        and p4_report["provenance"]["matmul_precision"] == "highest"
    ):
        raise SystemExit("P4 provenance/backend binding mismatch")
    phase5 = report["bindings"]["Phase5"]
    for label, path in (("json", args.rank_json), ("script", args.rank_script),
                        ("checkpoint", args.rank_checkpoint), ("preregistration", args.prereg)):
        if c.sha256(path) != phase5[f"{label}_sha256"]:
            raise SystemExit(f"Phase5 {label} binding mismatch")
    root = parse_manifest(args.manifest)
    prior = parse_manifest(args.p4_manifest)
    if root.get("code/deps/p4/MANIFEST.sha256") != c.sha256(args.p4_manifest):
        raise SystemExit("root manifest does not bind nested P4 manifest")
    for staged, digest in (
        ("code/deps/p4/phase4_d.json", c.sha256(args.p4_json)),
        ("code/deps/p4/phase4_d.npz", c.sha256(args.p4_npz)),
        ("code/deps/p4/AUDIT.json", c.sha256(args.p4_audit)),
        ("code/deps/phase5/phase5_rank_diagnostic.json", c.sha256(args.rank_json)),
        ("code/deps/phase5/PHASE-5-RANK-CHECKPOINT.md", c.sha256(args.rank_checkpoint)),
        ("code/deps/phase5/PHASE-5-PRE-REGISTRATION.md", c.sha256(args.prereg)),
        ("code/b10_phase5_rank_diagnostic.py", c.sha256(args.rank_script)),
    ):
        if root.get(staged) != digest:
            raise SystemExit(f"root manifest artifact mismatch: {staged}")
    rank = load(args.rank_json)
    expected_immutable = {"phase4_d.json": c.sha256(args.p4_json),
                          "phase4_d.npz": c.sha256(args.p4_npz),
                          "AUDIT.json": c.sha256(args.p4_audit),
                          "MANIFEST.sha256": c.sha256(args.p4_manifest)}
    if (rank.get("status") != "pass"
            or rank.get("diagnostic_source_sha256") != c.sha256(args.rank_script)
            or rank.get("phase5_bracket_adaptation_allowed") is not False
            or rank.get("immutable_inputs") != expected_immutable):
        raise SystemExit("rank diagnostic chain mismatch")
    dependencies = report["bindings"]["runtime_dependencies"]
    for key, label in (("code/bh_common.py", "bh_common_sha256"),
                       ("code/deps/burgers2d-coord-rom/burgers2d_film.py", "burgers2d_film_sha256")):
        if root.get(key) != prior.get(key) or root.get(key) != dependencies[label]:
            raise SystemExit("runtime dependency chain mismatch")
    for source, digest in report["provenance"]["source_sha256"].items():
        if root.get(f"code/{source}") != digest:
            raise SystemExit(f"staged source mismatch: {source}")


def audit_config(report):
    config = report["config"]
    expected_train = {
        str(n): {"start": start, "count": count, "times": 51,
                 "chunk_cases": chunk}
        for n, (start, count, chunk) in d.TRAIN.items()
    }
    checks = (
        config["train"] == expected_train,
        config["target_seed"] == d.TARGET_SEED,
        config["target_draw_count"] == d.TARGET_DRAW_COUNT,
        config["target_workers"] == 8,
        config["target_fit_count"] == d.TARGET_COUNT,
        config["candidates"] == list(p5.CANDIDATES),
        config["H1"] == p5.H1,
        config["time_repetitions"] == d.TIME_REPS,
        config["time_warmups"] == d.TIME_WARM,
        close(config["burn_seconds"], d.BURN_SECONDS),
        close(config["identity_tolerance"], d.IDENTITY_TOL),
        close(config["normal_tolerance"], s.ORACLE_NORMAL_TOL),
        config["model_validation_touched"] is False,
        config["confirmation_touched"] is False,
        config["smoke"] is False,
        config["f64"] is True,
        config["matmul_precision"] == "highest",
        report["provenance"]["jax_backend"] == "gpu",
        report["provenance"]["x64"] is True,
        report["provenance"]["matmul_precision"] == "highest",
    )
    if not all(checks):
        raise SystemExit("P5-D locked config/backend mismatch")


def science_shape_self_test():
    methods, orders = expected_timing_orders()
    records, summaries, main = {}, {}, {}
    for method_index, method in enumerate(methods):
        rows = []
        for repetition in range(20):
            for case in range(4):
                rows.append({"case_index": case, "repetition": repetition,
                             "elapsed_s": 1.0 + 0.01 * method_index
                             + 0.001 * case + 1e-5 * repetition})
        records[method] = rows
        summaries[method] = base.summarize_timing(rows, 4)
        main[f"timing_{method}"] = np.asarray([
            [row["case_index"], row["repetition"], row["elapsed_s"]] for row in rows
        ], np.float64)
    canonical = {}
    for arm in ("G1", "G2"):
        mandatory, maximum = [], []
        for case in range(4):
            common = {"case_index": case, "finite": True, "zero_failures": True,
                      "output_shape": [51, 1024 * 1024],
                      "weak_objective_evaluations": 50,
                      "weak_residual_norm_all": [1.0] * 50}
            mandatory.append(dict(common, method=f"{arm}_mandatory",
                                  weak_jacobian_evaluations=0,
                                  trial_residual_evaluations=0,
                                  coefficient_grid_evaluations=51,
                                  rho_all=[1.0] * 50))
            maximum.append(dict(common, method=f"{arm}_maximum_one",
                                weak_jacobian_evaluations=50,
                                trial_residual_evaluations=200,
                                coefficient_grid_evaluations=401,
                                rho_before_all=[1.0] * 50,
                                rho_after_all=[0.5] * 50,
                                jacobian_frobenius_all=[2.0] * 50,
                                trial_factor_all=[0.5] * 50,
                                bounded_step_norm_all=[0.1] * 50,
                                accepted_all=[True] * 50))
        canonical[f"{arm}_mandatory"] = mandatory
        canonical[f"{arm}_maximum_one"] = maximum
    position = {method: [sum(order[pos] == method for order in orders)
                         for pos in range(5)] for method in methods}
    panel = {"timing_orders": orders, "position_counts": position,
             "exact_position_balance": True, "records": records,
             "summaries": summaries, "canonical_work": canonical}
    passed = audit_science_panel_shape(panel, main)
    if not all(passed.values()):
        raise SystemExit("science-shape positive fixture failed")
    mutations = []
    wrong_summary = copy.deepcopy(panel)
    wrong_summary["summaries"]["fom"]["outliers_gt_1p5_within_trajectory_total"] += 1
    mutations.append(wrong_summary)
    wrong_order = copy.deepcopy(panel)
    wrong_order["timing_orders"][0] = list(reversed(wrong_order["timing_orders"][0]))
    mutations.append(wrong_order)
    wrong_work = copy.deepcopy(panel)
    wrong_work["canonical_work"]["G1_mandatory"][0]["coefficient_grid_evaluations"] = 50
    mutations.append(wrong_work)
    for mutated in mutations:
        try:
            result = audit_science_panel_shape(mutated, main)
        except SystemExit:
            continue
        if not all(result.values()):
            continue
        raise SystemExit("science-shape negative fixture was not rejected")
    print("phase5_d_science_shape_fixture=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("npz")
    parser.add_argument("audit")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-job")
    parser.add_argument("--manifest")
    parser.add_argument("--p4-json")
    parser.add_argument("--p4-npz")
    parser.add_argument("--p4-audit")
    parser.add_argument("--p4-manifest")
    parser.add_argument("--rank-json")
    parser.add_argument("--rank-script")
    parser.add_argument("--rank-checkpoint")
    parser.add_argument("--prereg")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--science-shape-self-test", action="store_true")
    args = parser.parse_args()
    if args.science_shape_self_test:
        science_shape_self_test()
        return
    report = load(args.json)
    expected_status = "excluded_execution_smoke_pass" if args.smoke else "complete"
    if report.get("status") != expected_status:
        raise SystemExit("P5-D report status mismatch")
    if report["npz"]["basename"] != os.path.basename(args.npz) or report["npz"]["sha256"] != c.sha256(args.npz):
        raise SystemExit("P5-D NPZ binding mismatch")
    if not args.smoke:
        if report["provenance"]["commit"] != args.expected_commit or str(report["provenance"]["slurm_job_id"]) != str(args.expected_job):
            raise SystemExit("P5-D expected provenance mismatch")
        audit_bindings(report, args, args.manifest)
        audit_config(report)
    with np.load(args.npz, allow_pickle=False) as main_arrays:
        integrity, mean, scales = audit_targets(report, main_arrays, args.json, args.smoke)
        arm_cost = audit_cost(report, main_arrays, mean, scales, args.smoke)
    licenses = {arm: bool(integrity and passed) for arm, passed in arm_cost.items()}
    next_arm = next((candidate["arm"] for candidate in p5.CANDIDATES
                     if licenses[candidate["arm"]]), None)
    decision = {"target_integrity_pass": integrity, "arm_training_licenses": licenses,
                "next_seed11_arm": next_arm,
                "phase5_hard_stop": bool(not args.smoke and next_arm is None),
                "scientific_promotion_allowed": not args.smoke}
    if report["decision"] != decision:
        raise SystemExit("P5-D decision mismatch")
    result = {"status": "pass", "source_json": os.path.basename(args.json),
              "source_json_sha256": c.sha256(args.json),
              "source_npz_sha256": c.sha256(args.npz),
              "negative_aware": True, "decision": decision,
              "model_validation_touched": False, "confirmation_touched": False}
    c.save_json(args.audit, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
