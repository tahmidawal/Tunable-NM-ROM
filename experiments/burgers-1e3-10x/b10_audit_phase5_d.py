#!/usr/bin/env python
"""Independent negative-aware audit for the Burgers Phase-5 diagnostic."""
from __future__ import annotations

import argparse
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
    scalar_keys = ("median_elapsed_s", "outlier_count")
    for key in scalar_keys:
        if key not in observed or key not in recomputed:
            raise SystemExit(f"{name} missing timing summary {key}")
        if isinstance(observed[key], int):
            okay = observed[key] == recomputed[key]
        else:
            okay = close(observed[key], recomputed[key])
        if not okay:
            raise SystemExit(f"{name} timing summary mismatch: {key}")
    for key in ("per_case_median_elapsed_s", "per_case_outlier_count"):
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
            if not np.array_equal(arrays["global_index"], expected_index):
                raise SystemExit("target global-index metadata mismatch")
            if not np.array_equal(arrays["time_index"], np.broadcast_to(np.arange(times), shape)):
                raise SystemExit("target time metadata mismatch")
            if not np.array_equal(arrays["N"], np.full(shape, n)):
                raise SystemExit("target mesh metadata mismatch")
            parameters = {name: np.asarray(arrays[f"parameter_{name}"])
                          for name in ("cx", "cy", "width", "amplitude", "nu")}
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
    for n, (start, count, _) in expected.items():
        if seen[n] != list(range(start, start + count)):
            raise SystemExit(f"N{n} target cohort mismatch")
    expected_total = 2 if smoke else d.TARGET_COUNT
    if total != expected_total or target["snapshot_count"] != total:
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


def audit_cost(report, main, mean, scales, smoke):
    panel = report["cost_panel"]
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
            work = bool(len(panel["canonical_work"][method]) == 4 and all(
                row["finite"] and row["zero_failures"]
                and row["weak_objective_evaluations"] == 50
                and row["weak_jacobian_evaluations"] == (0 if suffix == "mandatory" else 50)
                and row["trial_residual_evaluations"] == (0 if suffix == "mandatory" else 200)
                for row in panel["canonical_work"][method]))
            summary = base.summarize_timing(panel["records"][method], 4)
            compare_summary(panel["summaries"][method], summary, method)
            npz_rows = np.asarray(main[f"timing_{method}"])
            json_rows = np.asarray([[row["case_index"], row["repetition"], row["elapsed_s"]]
                                    for row in panel["records"][method]])
            if not np.array_equal(npz_rows, json_rows):
                raise SystemExit(f"{method} NPZ timing mismatch")
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
        if len(panel["timing_orders"]) != 20 or any(len(order) != 5 for order in panel["timing_orders"]):
            raise SystemExit("timing order shape mismatch")
        position = {method: [sum(order[pos] == method for order in panel["timing_orders"])
                             for pos in range(5)] for method in panel["records"]}
        if position != panel["position_counts"] or not all(
                value == 4 for counts in position.values() for value in counts):
            raise SystemExit("timing position balance mismatch")
        for method, rows in panel["records"].items():
            if len(rows) != 80:
                raise SystemExit(f"{method} timing count mismatch")
            if method == "fom":
                compare_summary(panel["summaries"][method], base.summarize_timing(rows, 4), method)
                if not np.array_equal(np.asarray(main[f"timing_{method}"]),
                                      np.asarray([[row["case_index"], row["repetition"], row["elapsed_s"]] for row in rows])):
                    raise SystemExit("FOM NPZ timing mismatch")
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
    args = parser.parse_args()
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
