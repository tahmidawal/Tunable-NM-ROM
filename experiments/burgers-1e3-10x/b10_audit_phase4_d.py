#!/usr/bin/env python
"""Independent, negative-aware audit for the Burgers Phase-4 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess

import numpy as np

import b10_common as c
import b10_phase4 as p4
import b10_phase4_d as d
import b10_s0_spline as base
import b10_spline as s


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def close(observed, expected, tolerance=2e-13):
    return math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=1e-15)


def parse_manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            rows[relative.removeprefix("./")] = digest
    return rows


def verify_chain(report, name, paths):
    bound = report["bindings"][name]
    labels = ("json", "npz", "audit")
    for label, path in zip(labels, paths):
        if c.sha256(path) != bound[f"{label}_sha256"]:
            raise SystemExit(f"{name} {label} immutable binding failed")
    source, audit = load(paths[0]), load(paths[2])
    if (
        audit["status"] != "pass"
        or d.normalized_audit_decision(audit, name) != source["decision"]
    ):
        raise SystemExit(f"{name} independent chain status failed")
    if audit["source_json_sha256"] != c.sha256(paths[0]):
        raise SystemExit(f"{name} audit JSON binding failed")
    if audit["source_npz_sha256"] != c.sha256(paths[1]):
        raise SystemExit(f"{name} audit NPZ binding failed")
    if source["npz"]["sha256"] != c.sha256(paths[1]):
        raise SystemExit(f"{name} source NPZ binding failed")
    return source


def audit_free(report, arrays, smoke, s0_npz_path=None, p3_report=None):
    candidates = p4.CANDIDATES[:1] if smoke else p4.CANDIDATES
    meshes = {24: (0, 1)} if smoke else d.SELECTION
    recomputed = {}
    old = None if smoke else np.load(s0_npz_path, allow_pickle=False)
    p3_values = {} if smoke else {
        (int(row["N"]), int(row["draw_index"]), int(row["time_index"])):
            float(row["field_relative_l2"])
        for row in p3_report["solver_diagnostic"]["records"]["S2"]
    }
    for candidate in candidates:
        arm = candidate["arm"]
        basis = report["free_oracle"][arm]["basis_identity"]
        basis_pass = bool(
            basis["window_max_abs"] <= d.IDENTITY_TOL
            and basis["max_tensor_support"] == 32
            and basis["exact_taper_endpoints"]
            and all(
                row["support_indices_exact"]
                and row["weight_max_abs"] <= d.IDENTITY_TOL
                and row["max_nonzero_support"] <= 4
                for row in basis["routes"].values()
            )
        )
        if basis["pass"] != basis_pass:
            raise SystemExit(f"{arm} polynomial basis identity mismatch")
        pooled = []
        mesh_passes = []
        for n, (_, count) in meshes.items():
            key = f"{arm}_N{n}"
            expected_shape = (count, 1 if smoke else 51)
            snapshot = np.sqrt(
                arrays[f"{key}_error_numerator_sq"]
                / np.maximum(arrays[f"{key}_truth_norm_sq"], 1e-300)
            )
            trajectory = np.sqrt(
                np.sum(arrays[f"{key}_error_numerator_sq"], axis=1)
                / np.maximum(np.sum(arrays[f"{key}_truth_norm_sq"], axis=1), 1e-300)
            )
            if snapshot.shape != expected_shape or trajectory.shape != (count,):
                raise SystemExit(f"{key} metric shape mismatch")
            if not np.allclose(snapshot, arrays[f"{key}_snapshot_error"], rtol=2e-13, atol=1e-15):
                raise SystemExit(f"{key} snapshot metric mismatch")
            if not np.allclose(trajectory, arrays[f"{key}_trajectory_error"], rtol=2e-13, atol=1e-15):
                raise SystemExit(f"{key} trajectory metric mismatch")
            normal = arrays[f"{key}_normal"]
            boundary = arrays[f"{key}_boundary"].astype(bool)
            pou = arrays[f"{key}_pou"]
            support = arrays[f"{key}_support"]
            coefficients = arrays[f"{key}_coefficients"]
            independent_health = (
                np.all(np.isfinite(coefficients), axis=-1)
                & np.isfinite(normal) & (normal <= s.ORACLE_NORMAL_TOL)
                & boundary & (pou <= 1e-15) & (support == 32)
            )
            if not np.array_equal(independent_health, arrays[f"{key}_healthy"].astype(bool)):
                raise SystemExit(f"{key} independent fit-health mismatch")
            row = report["free_oracle"][arm]["meshes"][str(n)]
            if not smoke:
                truth_health = row["truth_health"]
                if not (
                    np.isfinite(truth_health["reported_max_relative_residual"])
                    and np.isfinite(truth_health["independent_max_relative_residual"])
                    and truth_health["reported_max_relative_residual"] <= 1e-8
                    and truth_health["independent_max_relative_residual"] <= 1e-8
                ):
                    raise SystemExit(f"{key} selection truth health failed")
            checks = (
                close(row["trajectory_mean"], np.mean(trajectory)),
                close(row["trajectory_worst"], np.max(trajectory)),
                close(row["snapshot_mean"], np.mean(snapshot)),
                close(row["snapshot_worst"], np.max(snapshot)),
                row["fit_count"] == int(independent_health.size),
                row["healthy_count"] == int(np.sum(independent_health)),
                row["zero_unhealthy"] == bool(np.all(independent_health)),
                row["exact_boundary"] == bool(np.all(boundary)),
                close(row["normal_worst"], np.max(normal)),
                row["max_support"] == int(np.max(support)),
            )
            if not all(checks):
                raise SystemExit(f"{key} reported oracle summary mismatch")
            if smoke:
                s0_pass = True
            else:
                immutable_old = np.asarray(old[f"C_N{n}_snapshot_error"])
                if not np.array_equal(
                    immutable_old, arrays[f"{key}_s0_snapshot_error"]
                ):
                    raise SystemExit(f"{key} copied S0 control mismatch")
                s0_pass = bool(np.all(
                    snapshot <= immutable_old + d.NO_REGRESSION
                ))
            if row["s0_no_regression"] != s0_pass:
                raise SystemExit(f"{key} S0 no-regression mismatch")
            comparisons = []
            for item in row["p3_fixed_subset"]:
                index = (
                    n, int(item["draw_index"]), int(item["time_index"])
                )
                immutable_value = p3_values[index]
                observed = float(
                    snapshot[int(item["draw_index"]) - meshes[n][0], int(item["time_index"])]
                )
                no_regression = bool(
                    observed <= immutable_value + d.NO_REGRESSION
                )
                if not close(item["p4_error"], observed) or not close(item["p3_s2_error"], immutable_value) or item["no_regression"] != no_regression:
                    raise SystemExit(f"{key} immutable P3 comparison mismatch")
                comparisons.append(no_regression)
            p3_pass = bool(all(comparisons))
            if row["p3_fixed_subset_no_regression"] != p3_pass:
                raise SystemExit(f"{key} P3 no-regression mismatch")
            pooled.extend(trajectory.tolist())
            mesh_passes.append(row)
        pooled = np.asarray(pooled)
        summary = report["free_oracle"][arm]["summary"]
        accuracy = bool(smoke or (
            np.mean(pooled) <= 2e-4 and np.max(pooled) <= 7e-4
            and all(row["trajectory_mean"] <= 2e-4 and row["trajectory_worst"] <= 7e-4 for row in mesh_passes)
        ))
        zero_unhealthy = bool(all(row["zero_unhealthy"] for row in mesh_passes))
        exact_structure = bool(
            basis_pass and all(
                row["exact_boundary"] and row["pou_worst"] <= 1e-15
                and row["max_support"] == 32 for row in mesh_passes
            )
        )
        s0_all = bool(all(row["s0_no_regression"] for row in mesh_passes))
        p3_all = bool(all(
            row["p3_fixed_subset_no_regression"] for row in mesh_passes
        ))
        if not (
            close(summary["trajectory_mean"], np.mean(pooled))
            and close(summary["trajectory_worst"], np.max(pooled))
            and summary["all_mesh_and_pooled_accuracy_pass"] == accuracy
            and summary["zero_unhealthy"] == zero_unhealthy
            and summary["exact_boundary_pou_support"] == exact_structure
            and summary["s0_no_regression"] == s0_all
            and summary["p3_fixed_subset_no_regression"] == p3_all
        ):
            raise SystemExit(f"{arm} pooled oracle summary mismatch")
        if smoke:
            target_pass = True
        else:
            target = float(arrays[f"{arm}_N128_trajectory_error"][530 - 512])
            p3_target = float(p3_report["solver_diagnostic"]["summaries"]["S2"]["target_N128_draw530_trajectory_relative_l2"])
            target_pass = bool(
                target <= 7e-4 and target <= p3_target + d.NO_REGRESSION
            )
            if not close(summary["N128_draw530_trajectory"], target) or summary["N128_draw530_pass"] != target_pass:
                raise SystemExit(f"{arm} N128 draw530 gate mismatch")
        passed = bool(
            accuracy and zero_unhealthy and exact_structure
            and s0_all and p3_all
            and target_pass
        )
        if summary["pass"] != passed:
            raise SystemExit(f"{arm} free-oracle gate mismatch")
        recomputed[arm] = passed
    return recomputed


def audit_cost(report, smoke):
    panel = report["cost_panel"]
    candidates = p4.CANDIDATES[:1] if smoke else p4.CANDIDATES
    cost_pass = {}
    if not smoke:
        health = panel["reference_health"]
        reference_ok = all(
            row["finite"] and row["breakdowns"] == 0
            and row["flags_nonzero"] == 0
            and row["max_returned_relative_residual"] <= base.REFERENCE_OUTER
            for row in health["reference_records"]
        )
        audit_ok = all(
            row["finite"] and row["breakdowns"] == 0
            and row["flags_nonzero"] == 0
            and row["max_returned_relative_residual"] <= base.AUDIT_OUTER
            for row in health["audit_records"]
        )
        if not reference_ok or not audit_ok or health["cross_chain_worst"] > 1e-4:
            raise SystemExit("tight/tighter reference health mismatch")
        methods = ["fom"] + [
            f"{candidate['arm']}_{suffix}" for candidate in candidates
            for suffix in ("mandatory", "maximum_one")
        ]
        orders = panel["timing_orders"]
        expected_orders = []
        for repetition in range(d.TIME_REPS):
            offset = repetition % len(methods)
            order = methods[offset:] + methods[:offset]
            if (repetition // len(methods)) % 2:
                order = list(reversed(order))
            expected_orders.append(order)
        if orders != expected_orders:
            raise SystemExit("timing order schedule mismatch")
        position_counts = {
            method: [
                int(sum(order[position] == method for order in orders))
                for position in range(len(methods))
            ] for method in methods
        }
        if panel["position_counts"] != position_counts or not panel["exact_position_balance"]:
            raise SystemExit("five-method exact position balance mismatch")
        if any(value != 4 for counts in position_counts.values() for value in counts):
            raise SystemExit("expected four observations per clock position")
        expected_pairs = {
            (case, repetition) for repetition in range(d.TIME_REPS)
            for case in range(d.FOM_CASES)
        }
        for method in methods:
            rows = panel["records"][method]
            observed_pairs = {
                (int(row["case_index"]), int(row["repetition"]))
                for row in rows
            }
            if len(rows) != 80 or observed_pairs != expected_pairs:
                raise SystemExit(f"{method} timing repetition coverage mismatch")
            summary = base.summarize_timing(rows, d.FOM_CASES)
            for key in (
                "median_elapsed_s", "elapsed_all_s",
                "per_case_median_elapsed_s",
                "outliers_gt_1p5_within_trajectory_all",
                "outliers_gt_1p5_within_trajectory_total",
            ):
                if panel["summaries"][method][key] != summary[key]:
                    raise SystemExit(
                        f"{method} timing/outlier summary mismatch"
                    )
    for candidate in candidates:
        arm = candidate["arm"]
        pallas = panel["setup"][arm]["pallas_basis_identity"]
        pallas_pass = bool(
            pallas["available"] and pallas["executed"]
            and all(
                row["support_indices_exact"]
                and row["local_support"] == 4
                and row["weight_max_abs"] <= d.IDENTITY_TOL
                and row["pass"]
                for row in pallas["routes"].values()
            )
        )
        if pallas["pass"] != pallas_pass or not pallas_pass:
            raise SystemExit(f"{arm} K3-specific basis identity failed")
        identity_pass = bool(all(
            row["max_relative_l2"] <= d.IDENTITY_TOL
            and row["exact_boundary"] and row["pass"]
            for row in panel["identity"][arm]
        ))
        if not identity_pass:
            raise SystemExit(f"{arm} K3/Cox field+weak identity failed")
        for suffix in ("mandatory", "maximum_one"):
            method = f"{arm}_{suffix}"
            rows = panel["canonical_work"][method]
            expected_steps = 1 if smoke else 50
            expected_jacobian = 0 if suffix == "mandatory" else expected_steps
            expected_trials = 0 if suffix == "mandatory" else 4 * expected_steps
            expected_coefficients = expected_steps + 1 if suffix == "mandatory" else 8 * expected_steps + 1
            work_pass = bool(
                len(rows) == (1 if smoke else 4)
                and all(
                    row["finite"] and row["zero_failures"]
                    and row["weak_objective_evaluations"] == expected_steps
                    and row["weak_jacobian_evaluations"] == expected_jacobian
                    and row["trial_residual_evaluations"] == expected_trials
                    and row["coefficient_grid_evaluations"] == expected_coefficients
                    and np.all(np.isfinite(row["weak_residual_norm_all"]))
                    for row in rows
                )
            )
            if not work_pass:
                raise SystemExit(f"{method} canonical work mismatch")
        if smoke:
            cost_pass[arm] = False
            continue
        gates = panel["gates"][arm]
        fom_rows = [row for row in panel["records"]["fom"] if row["repetition"] == 0]
        fom_mean = float(np.mean([row["trajectory_relative_l2"] for row in fom_rows]))
        fom_worst = float(np.max([row["trajectory_relative_l2"] for row in fom_rows]))
        fom_healthy = bool(all(
            row["finite"] and row["breakdowns"] == 0
            and row["flags_nonzero"] == 0
            and row["max_returned_relative_residual"] <= base.FOM_OUTER
            for row in panel["records"]["fom"]
        ))
        fom_eligible = bool(fom_healthy and fom_mean <= 1e-3 and fom_worst <= 3e-3)
        if panel["fom_accuracy"] != {
            "mean": fom_mean, "worst": fom_worst,
            "healthy": fom_healthy, "eligible": fom_eligible,
        }:
            raise SystemExit("live FOM eligibility mismatch")
        identity = all(row["pass"] for row in panel["identity"][arm])
        if gates["fom_eligible"] != fom_eligible or gates["identity_pass"] != identity:
            raise SystemExit(f"{arm} shared cost-gate mismatch")
        for suffix in ("mandatory", "maximum_one"):
            method = f"{arm}_{suffix}"
            summary = base.summarize_timing(panel["records"][method], 4)
            fom_summary = base.summarize_timing(panel["records"]["fom"], 4)
            for key in (
                "median_elapsed_s", "elapsed_all_s",
                "per_case_median_elapsed_s",
                "outliers_gt_1p5_within_trajectory_all",
                "outliers_gt_1p5_within_trajectory_total",
            ):
                if panel["summaries"][method][key] != summary[key]:
                    raise SystemExit(f"{method} timing/outlier summary mismatch")
            speed = fom_summary["median_elapsed_s"] / summary["median_elapsed_s"]
            seed = 20264100 + (0 if arm == "H1" else 20) + (0 if suffix == "mandatory" else 1)
            interval = base.clustered_speedup_ci(
                fom_summary["per_case_median_elapsed_s"],
                summary["per_case_median_elapsed_s"], seed,
            )
            memory = panel["setup"][arm]["kernels"][suffix]["memory_analysis"]["eligibility_device_bytes"] <= 20_000_000_000
            passed = bool(fom_eligible and identity and memory and speed >= 10 and interval[0] >= 8)
            if not close(gates[suffix]["paired_median_speedup"], speed) or not np.allclose(gates[suffix]["clustered_speedup_ci"], interval):
                raise SystemExit(f"{method} timing metric mismatch")
            if gates[suffix]["pass"] != passed:
                raise SystemExit(f"{method} cost gate mismatch")
            if not gates[suffix]["canonical_work_pass"] or gates[suffix]["memory_pass"] != memory:
                raise SystemExit(f"{method} reported work/memory gate mismatch")
        training_license = bool(gates["mandatory"]["pass"])
        classification = (
            "correction-capable" if gates["maximum_one"]["pass"]
            else "conditional-zero-or-occasional-attempt"
            if training_license else "cost-fail"
        )
        if gates["training_cost_license"] != training_license or gates["correction_classification"] != classification:
            raise SystemExit(f"{arm} training-cost classification mismatch")
        cost_pass[arm] = bool(gates["training_cost_license"])
    return cost_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("npz")
    parser.add_argument("audit")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-job")
    parser.add_argument("--manifest")
    parser.add_argument("--s0-json")
    parser.add_argument("--s0-npz")
    parser.add_argument("--s0-audit")
    parser.add_argument("--p3-json")
    parser.add_argument("--p3-npz")
    parser.add_argument("--p3-audit")
    parser.add_argument("--p3-manifest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = load(args.report)
    smoke = args.smoke
    expected_status = "excluded_execution_smoke_pass" if smoke else "complete"
    if report["status"] != expected_status:
        raise SystemExit("P4-D status mismatch")
    if report["npz"]["basename"] != os.path.basename(args.npz):
        raise SystemExit("P4-D NPZ path is not relocatable")
    if report["npz"]["sha256"] != c.sha256(args.npz):
        raise SystemExit("P4-D NPZ checksum mismatch")
    if report["config"]["model_validation_touched"] or report["config"]["confirmation_touched"]:
        raise SystemExit("forbidden cohort-touch flag")
    chain_reports = {}
    if not smoke:
        required = (args.expected_commit, args.expected_job, args.manifest, args.s0_json, args.s0_npz, args.s0_audit, args.p3_json, args.p3_npz, args.p3_audit, args.p3_manifest)
        if any(value is None for value in required):
            raise SystemExit("scientific audit requires provenance and both chains")
        if report["provenance"]["commit"] != args.expected_commit or str(report["provenance"]["slurm_job_id"]) != str(args.expected_job):
            raise SystemExit("scientific commit/job mismatch")
        if report["provenance"]["jax_backend"] != "gpu" or not report["provenance"]["x64"] or report["provenance"]["matmul_precision"] != "highest":
            raise SystemExit("scientific backend/precision mismatch")
        manifest = parse_manifest(args.manifest)
        sources = ("b10_common.py", "b10_spline.py", "b10_s0_spline.py", "b10_phase3.py", "b10_phase4.py", "b10_phase4_d.py", "b10_audit_phase4_d.py")
        for source in sources:
            if manifest.get(f"code/{source}") != report["provenance"]["source_sha256"].get(source):
                raise SystemExit(f"manifest/source mismatch for {source}")
        chain_reports["S0"] = verify_chain(report, "S0", (args.s0_json, args.s0_npz, args.s0_audit))
        chain_reports["P3"] = verify_chain(report, "P3", (args.p3_json, args.p3_npz, args.p3_audit))
        for prefix, paths in (
            ("code/deps/s0", (args.s0_json, args.s0_npz, args.s0_audit)),
            ("code/deps/p3", (args.p3_json, args.p3_npz, args.p3_audit)),
        ):
            for path in paths:
                key = f"{prefix}/{os.path.basename(path)}"
                if manifest.get(key) != c.sha256(path):
                    raise SystemExit(f"manifest immutable dependency mismatch: {key}")
        prior_manifest = parse_manifest(args.p3_manifest)
        if c.sha256(args.p3_manifest) != report["bindings"]["P3"]["staged_manifest_sha256"]:
            raise SystemExit("P3 staged-manifest report binding mismatch")
        if manifest.get("code/deps/p3/MANIFEST.sha256") != c.sha256(args.p3_manifest):
            raise SystemExit("P4 manifest does not bind the P3 staged manifest")
        film_key = "code/deps/burgers2d-coord-rom/burgers2d_film.py"
        bh_key = "code/bh_common.py"
        dependencies = report["bindings"]["runtime_dependencies"]
        if not (
            prior_manifest.get(film_key) == manifest.get(film_key)
            == dependencies["burgers2d_film_sha256"]
            and prior_manifest.get(bh_key) == manifest.get(bh_key)
            == dependencies["bh_common_sha256"]
            and dependencies["source_manifest"] == "P3"
        ):
            raise SystemExit("runtime dependency provenance mismatch")
        worktree = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        bh_bytes = subprocess.check_output((
            "git", "-C", worktree, "show",
            f"{args.expected_commit}:experiments/burgers-hybrid-1024/bh_common.py",
        ))
        if hashlib.sha256(bh_bytes).hexdigest() != dependencies["bh_common_sha256"]:
            raise SystemExit("bh_common is not the expected-commit content")
    with np.load(args.npz, allow_pickle=False) as arrays:
        free = audit_free(
            report, arrays, smoke,
            None if smoke else args.s0_npz,
            None if smoke else chain_reports["P3"],
        )
    cost = audit_cost(report, smoke)
    selected = None
    if not smoke:
        passing = [candidate["arm"] for candidate in p4.CANDIDATES if free[candidate["arm"]] and cost[candidate["arm"]]]
        selected = passing[0] if passing else None
    expected_decision = {
        "selected_spatial_arm": selected,
        "training_seed11_k24_licensed": selected is not None,
        "correction_classification": None if selected is None else report["cost_panel"]["gates"][selected]["correction_classification"],
        "phase4_hard_stop": bool(not smoke and selected is None),
        "scientific_promotion_allowed": not smoke,
    }
    if report["decision"] != expected_decision:
        raise SystemExit("negative-aware P4-D decision mismatch")
    result = {
        "status": "pass", "source_json": os.path.basename(args.report),
        "source_json_sha256": c.sha256(args.report),
        "source_npz": os.path.basename(args.npz),
        "source_npz_sha256": c.sha256(args.npz),
        "expected_commit": args.expected_commit,
        "expected_job": args.expected_job,
        "manifest_sha256": None if smoke else c.sha256(args.manifest),
        "negative_aware": True, "smoke": smoke,
        "fom_timing_summary_recomputed": bool(not smoke),
        "decision": expected_decision,
    }
    c.save_json(args.audit, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
