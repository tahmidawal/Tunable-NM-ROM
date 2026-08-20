"""Independent, deterministic audit of a completed scientific Phase-2 S0 artifact."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np


if len(sys.argv) != 6:
    raise SystemExit(
        "usage: b10_audit_s0.py S0.json S0.npz AUDIT.json EXPECTED_COMMIT JOB_ID"
    )
JSON_PATH, NPZ_PATH, AUDIT_PATH, EXPECTED_COMMIT, JOB_ID = sys.argv[1:]
ARMS = ("A", "B", "C")
MESHES = (64, 128, 256)
CANDIDATES = {
    "A": {"arm": "A", "R": 24, "k": 12, "M": 64, "m": 256},
    "B": {"arm": "B", "R": 32, "k": 16, "M": 64, "m": 256},
    "C": {"arm": "C", "R": 48, "k": 24, "M": 96, "m": 384},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"S0 AUDIT FAILED: {message}")


def same(actual, recorded, name):
    require(
        np.isclose(float(actual), float(recorded), rtol=1e-12, atol=1e-15),
        f"{name}: recomputed {actual!r} != recorded {recorded!r}",
    )


def timing_summary(records):
    elapsed = np.asarray([row["elapsed_s"] for row in records], np.float64)
    require(elapsed.size == 40 and np.all(np.isfinite(elapsed)) and np.all(elapsed > 0),
            "timing repetition array must have 4 cases x 10 positive finite values")
    case_medians = []
    outliers = []
    for case in range(4):
        values = np.asarray(
            [row["elapsed_s"] for row in records if row["case_index"] == case],
            np.float64,
        )
        require(values.size == 10, f"case {case} does not have ten timing repetitions")
        median = float(np.median(values))
        case_medians.append(median)
        outliers.append(int(np.sum(values > 1.5 * median)))
    return {
        "median_elapsed_s": float(np.median(elapsed)),
        "per_case_median_elapsed_s": case_medians,
        "outliers": outliers,
    }


def clustered_ci(fom, method, seed):
    ratios = np.asarray(fom, np.float64) / np.asarray(method, np.float64)
    rng = np.random.default_rng(int(seed))
    sample = ratios[rng.integers(0, ratios.size, size=(10_000, ratios.size))]
    return np.quantile(np.median(sample, axis=1), (0.025, 0.975))


def healthy_fom_record(row, tolerance):
    return bool(
        row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
        and np.isfinite(row["max_returned_relative_residual"])
        and row["max_returned_relative_residual"] <= tolerance
    )


with open(JSON_PATH) as handle:
    report = json.load(handle)
require(JOB_ID.isdigit(), "job id is not numeric")
provenance = report.get("provenance") or {}
require(report.get("status") == "complete", "scientific report status is not complete")
require(provenance.get("commit") == EXPECTED_COMMIT, "commit mismatch")
require(provenance.get("slurm_job_id") == JOB_ID, "Slurm job id mismatch")
require(provenance.get("jax_backend") == "gpu", "backend is not GPU")
require(provenance.get("x64") is True, "JAX x64 is not true")
require(provenance.get("matmul_precision") == "highest", "matmul precision is not highest")
require(report["config"].get("smoke") is False, "artifact is a smoke")
require(report["config"].get("model_validation_touched") is False,
        "model validation was touched")
require(report["config"].get("confirmation_touched") is False,
        "confirmation was touched")
require(report["config"].get("candidate_bracket") is not None, "candidate bracket missing")
require(report.get("npz", {}).get("sha256") == sha256(NPZ_PATH),
        "NPZ internal checksum mismatch")

selection_health = report.get("selection_reference_health") or {}
require(set(selection_health) == {str(n) for n in MESHES},
        "selection reference health mesh set mismatch")
for n in MESHES:
    row = selection_health[str(n)]
    require(
        np.isfinite(row["reported_max_relative_residual"])
        and np.isfinite(row["independent_max_relative_residual"])
        and row["reported_max_relative_residual"] <= 1e-8
        and row["independent_max_relative_residual"] <= 1e-8,
        f"N={n} selection reference health failed",
    )

audit = {
    "status": "running", "source_json": os.path.abspath(JSON_PATH),
    "source_json_sha256": sha256(JSON_PATH), "source_npz_sha256": sha256(NPZ_PATH),
    "expected_commit": EXPECTED_COMMIT, "job_id": JOB_ID,
    "provenance_health": "pass", "free_oracle": {}, "cost_panel": {},
}
with np.load(NPZ_PATH, allow_pickle=False) as arrays:
    for name in arrays.files:
        value = arrays[name]
        if np.issubdtype(value.dtype, np.number):
            require(np.all(np.isfinite(value)), f"nonfinite values in NPZ array {name}")
    for arm in ARMS:
        require(report["free_oracle"][arm]["candidate"] == CANDIDATES[arm],
                f"{arm} candidate changed")
        pooled = []
        all_healthy = True
        for n in MESHES:
            key = f"{arm}_N{n}"
            trajectory = np.asarray(arrays[f"{key}_trajectory_error"], np.float64)
            snapshot = np.asarray(arrays[f"{key}_snapshot_error"], np.float64)
            fit_healthy = np.asarray(arrays[f"{key}_fit_healthy"], bool)
            fit_normal = np.asarray(arrays[f"{key}_relative_normal"], np.float64)
            iterations = np.asarray(arrays[f"{key}_fit_iterations"], np.int64)
            row = report["free_oracle"][arm]["meshes"][str(n)]
            require(trajectory.size == len(row["indices"]), f"{key} trajectory count")
            require(fit_healthy.size == trajectory.size * 51, f"{key} fit count")
            require(np.all(fit_healthy) == row["zero_unhealthy_fits"],
                    f"{key} fit health mismatch")
            require(np.max(fit_normal) <= 1e-8, f"{key} normal residual gate")
            require(np.all((iterations >= 0) & (iterations <= 500)),
                    f"{key} LSMR iteration range")
            same(np.mean(trajectory), row["trajectory_error_mean"], f"{key} mean")
            same(np.max(trajectory), row["trajectory_error_worst"], f"{key} worst")
            same(np.mean(snapshot), row["snapshot_error_mean"], f"{key} snapshot mean")
            same(np.max(snapshot), row["snapshot_error_worst"], f"{key} snapshot worst")
            require(row["exact_binary_boundary"] is True, f"{key} exact boundary")
            require(row["max_partition_sum_error"] <= 2e-15,
                    f"{key} partition of unity")
            pooled.extend(trajectory.tolist())
            all_healthy = all_healthy and bool(np.all(fit_healthy))
        pooled = np.asarray(pooled, np.float64)
        summary = report["free_oracle"][arm]["summary"]
        same(np.mean(pooled), summary["trajectory_error_mean"], f"{arm} pooled mean")
        same(np.max(pooled), summary["trajectory_error_worst"], f"{arm} pooled worst")
        accuracy_pass = bool(
            np.mean(pooled) <= 2e-4 and np.max(pooled) <= 7e-4
            and all(
                row["trajectory_error_mean"] <= 2e-4
                and row["trajectory_error_worst"] <= 7e-4
                for row in report["free_oracle"][arm]["meshes"].values()
            )
        )
        require(summary["every_mesh_and_pooled_accuracy_pass"] == accuracy_pass,
                f"{arm} accuracy gate mismatch")
        require(summary["zero_unhealthy_fits"] == all_healthy,
                f"{arm} pooled health mismatch")
        audit["free_oracle"][arm] = {
            "trajectory_error_mean": float(np.mean(pooled)),
            "trajectory_error_worst": float(np.max(pooled)),
            "all_mesh_and_pooled_accuracy_pass": accuracy_pass,
            "zero_unhealthy_fits": all_healthy,
        }

panel = report.get("cost_panel") or {}
reference = panel.get("reference_health") or {}
require(reference.get("cross_chain_worst", np.inf) <= 1e-4,
        "live reference/tighter-chain difference exceeds 1e-4")
require(reference.get("all_finite_zero_flags_breakdowns") is True,
        "live reference aggregate health flag failed")
for row in reference.get("reference_records", []):
    require(healthy_fom_record(row, reference["reference_outer"]),
            "live reference record unhealthy")
for row in reference.get("audit_records", []):
    require(healthy_fom_record(row, reference["audit_outer"]),
            "tighter audit record unhealthy")
require(len(reference.get("reference_records", [])) == 4, "reference case count")
require(len(reference.get("audit_records", [])) == 4, "audit case count")

methods = ["fom"] + [
    f"{arm}_{suffix}" for arm in ARMS
    for suffix in ("direct", "mandatory", "maximum_one")
]
require(set(panel["records"]) == set(methods), "timing method set mismatch")
orders = panel["timing_orders"]
require(len(orders) == 10 and all(len(order) == 10 and set(order) == set(methods)
                                  for order in orders), "invalid timing orders")
position_counts = {
    method: [sum(order[position] == method for order in orders)
             for position in range(10)] for method in methods
}
require(all(counts == [1] * 10 for counts in position_counts.values()),
        "ten-method cyclic position balance failed")
require(panel["exact_ten_method_position_balance"] is True,
        "recorded position-balance flag false")
require(panel["timing_position_counts"] == position_counts,
        "recorded position counts mismatch")

recomputed = {}
for method in methods:
    one = timing_summary(panel["records"][method])
    stored = panel["summaries"][method]
    same(one["median_elapsed_s"], stored["median_elapsed_s"], f"{method} median")
    require(np.allclose(one["per_case_median_elapsed_s"],
                        stored["per_case_median_elapsed_s"], rtol=1e-12, atol=1e-15),
            f"{method} case medians")
    require(one["outliers"] == stored["outliers_gt_1p5_within_trajectory_all"],
            f"{method} within-trajectory outliers")
    require(sum(one["outliers"]) == stored["outliers_gt_1p5_within_trajectory_total"],
            f"{method} total outliers")
    recomputed[method] = one

fom_records = panel["records"]["fom"]
fom_healthy = all(healthy_fom_record(row, 3e-3) for row in fom_records)
accuracy_rows = [row for row in fom_records if row["repetition"] == 0]
require(len(accuracy_rows) == 4, "FOM same-invocation accuracy case count")
fom_mean = float(np.mean([row["trajectory_relative_l2"] for row in accuracy_rows]))
fom_worst = float(np.max([row["trajectory_relative_l2"] for row in accuracy_rows]))
fom_eligible = bool(fom_healthy and fom_mean <= 1e-3 and fom_worst <= 3e-3)
recorded_fom = panel["fom_same_invocation_accuracy"]
same(fom_mean, recorded_fom["trajectory_error_mean"], "FOM accuracy mean")
same(fom_worst, recorded_fom["trajectory_error_worst"], "FOM accuracy worst")
require(recorded_fom["healthy"] == fom_healthy, "FOM health mismatch")
require(recorded_fom["accuracy_eligible"] == fom_eligible, "FOM eligibility mismatch")

for arm in ARMS:
    method = f"{arm}_mandatory"
    speedup = recomputed["fom"]["median_elapsed_s"] / recomputed[method]["median_elapsed_s"]
    ci = clustered_ci(
        recomputed["fom"]["per_case_median_elapsed_s"],
        recomputed[method]["per_case_median_elapsed_s"], 20260822 + CANDIDATES[arm]["R"],
    )
    memory_ok = all(
        row["eligibility_device_bytes"] <= 20_000_000_000
        for row in panel["setup"][arm]["memory_analysis"].values()
    )
    gate = panel["gates"][arm]
    same(speedup, gate["paired_median_speedup_mandatory_lower_bound"], f"{arm} speedup")
    require(np.allclose(ci, gate["clustered_speedup_ci"], rtol=1e-12, atol=1e-15),
            f"{arm} clustered CI")
    expected_pass = bool(speedup >= 10 and ci[0] >= 8 and memory_ok and fom_eligible)
    require(gate["pass"] == expected_pass, f"{arm} cost gate mismatch")
    for suffix in ("mandatory", "maximum_one"):
        work = panel["canonical_untimed_work"][f"{arm}_{suffix}"]
        require(len(work) == 4 and all(row["finite"] for row in work),
                f"{arm}_{suffix} canonical work health")
        require(all(row["weak_objective_evaluations"] == 50 for row in work),
                f"{arm}_{suffix} weak evaluation count")
        if suffix == "mandatory":
            require(all(row["weak_jacobian_evaluations"] == 0
                        and row["trial_residual_evaluations"] == 0 for row in work),
                    f"{arm} mandatory work accounting")
        else:
            require(all(row["weak_jacobian_evaluations"] == 50
                        and row["trial_residual_evaluations"] == 200 for row in work),
                    f"{arm} maximum-one work accounting")
    recomputed_promote = bool(
        audit["free_oracle"][arm]["all_mesh_and_pooled_accuracy_pass"]
        and audit["free_oracle"][arm]["zero_unhealthy_fits"]
        and report["free_oracle"][arm]["summary"]["exact_binary_boundary"]
        and expected_pass
    )
    require(report["free_oracle"][arm]["promote"] == recomputed_promote,
            f"{arm} final promotion mismatch")
    audit["cost_panel"][arm] = {
        "mandatory_median_elapsed_s": recomputed[method]["median_elapsed_s"],
        "paired_fom_median_elapsed_s": recomputed["fom"]["median_elapsed_s"],
        "paired_median_speedup": float(speedup),
        "clustered_speedup_ci": ci.tolist(), "memory_ok": memory_ok,
        "paired_fom_accuracy_eligible": fom_eligible,
        "cost_pass": expected_pass, "promote": recomputed_promote,
    }

promoted = [arm for arm in ARMS if audit["cost_panel"][arm]["promote"]]
decision = report.get("decision") or {}
require(decision.get("promoted_arms") == promoted, "promoted arm list mismatch")
require(decision.get("selected_smallest_promoted") == (promoted[0] if promoted else None),
        "selected smallest promoted mismatch")
require(decision.get("phase2_hard_stop") == (not bool(promoted)),
        "Phase-2 hard-stop decision mismatch")
audit["cost_panel"]["paired_fom"] = {
    "trajectory_error_mean": fom_mean, "trajectory_error_worst": fom_worst,
    "healthy": fom_healthy, "accuracy_eligible": fom_eligible,
    "median_elapsed_s": recomputed["fom"]["median_elapsed_s"],
}
audit["decision"] = {
    "promoted_arms": promoted,
    "selected_smallest_promoted": promoted[0] if promoted else None,
    "phase2_hard_stop": not bool(promoted),
}
audit["status"] = "pass"
os.makedirs(os.path.dirname(os.path.abspath(AUDIT_PATH)), exist_ok=True)
with open(AUDIT_PATH, "w") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps({"status": "pass", "decision": audit["decision"]}, indent=2))
