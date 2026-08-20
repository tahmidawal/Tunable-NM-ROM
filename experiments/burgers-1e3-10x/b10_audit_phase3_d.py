"""Independent deterministic audit of a completed Phase-3 P3-D artifact."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np


if sys.argv[1:] == ["--self-test-negative"]:
    # Integrity PASS is intentionally orthogonal to scientific promotion.  This
    # fixed fixture guards against repeating the S0 checker bug that rejected an
    # honestly persisted negative gate.
    fixture = {
        "selected_solver": None, "selected_kernel": None,
        "run_P3_F": False, "phase3_hard_stop": True,
        "K3_absent_classified": True,
    }
    assert fixture["phase3_hard_stop"] == (
        not bool(fixture["selected_solver"] and fixture["selected_kernel"])
    )
    assert fixture["K3_absent_classified"]
    print(json.dumps({
        "status": "pass", "negative_scientific_decision_accepted": True,
        "fixture": fixture,
    }, indent=1))
    raise SystemExit(0)


if len(sys.argv) != 7:
    raise SystemExit(
        "usage: b10_audit_phase3_d.py P3.json P3.npz AUDIT.json "
        "EXPECTED_COMMIT JOB_ID MANIFEST.sha256"
    )
JSON_PATH, NPZ_PATH, AUDIT_PATH, EXPECTED_COMMIT, JOB_ID, MANIFEST_PATH = sys.argv[1:]
SOLVER_NAMES = ("S1", "S2")
KERNEL_ORDER = ("K0", "K1", "K2", "K3")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"P3-D AUDIT FAILED: {message}")


def same(actual, recorded, name):
    require(
        np.isclose(float(actual), float(recorded), rtol=1e-12, atol=1e-15),
        f"{name}: {actual!r} != {recorded!r}",
    )


def git_blob_sha(commit, relative):
    value = subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{commit}:{relative}"]
    )
    return hashlib.sha256(value).hexdigest()


def manifest_entries(path):
    entries = {}
    with open(path) as handle:
        for line in handle:
            digest, relative = line.rstrip().split("  ", 1)
            entries[relative.removeprefix("./")] = digest
    return entries


def timing_summary(records):
    require(len(records) == 80, "timing array is not 4 cases x 20 repetitions")
    elapsed = np.asarray([row["elapsed_s"] for row in records], np.float64)
    require(np.all(np.isfinite(elapsed)) and np.all(elapsed > 0), "bad timing value")
    case_medians, outliers = [], []
    for case in range(4):
        values = np.asarray([
            row["elapsed_s"] for row in records if row["case_index"] == case
        ], np.float64)
        require(values.size == 20, f"case {case} timing count")
        median = float(np.median(values))
        case_medians.append(median)
        outliers.append(int(np.sum(values > 1.5 * median)))
    return {
        "median_elapsed_s": float(np.median(elapsed)),
        "per_case_median_elapsed_s": case_medians,
        "outliers_gt_1p5_within_trajectory_all": outliers,
        "outliers_gt_1p5_within_trajectory_total": int(sum(outliers)),
    }


def clustered_ci(fom, method, seed):
    ratios = np.asarray(fom, np.float64) / np.asarray(method, np.float64)
    rng = np.random.default_rng(int(seed))
    sample = ratios[rng.integers(0, ratios.size, size=(10_000, ratios.size))]
    return np.quantile(np.median(sample, axis=1), (0.025, 0.975))


with open(JSON_PATH) as handle:
    report = json.load(handle)
require(JOB_ID.isdigit(), "job id is not numeric")
provenance = report.get("provenance") or {}
require(report.get("status") == "complete", "report is not complete")
require(provenance.get("commit") == EXPECTED_COMMIT, "commit mismatch")
require(provenance.get("slurm_job_id") == JOB_ID, "job id mismatch")
require(provenance.get("jax_backend") == "gpu", "backend is not GPU")
require(provenance.get("x64") is True, "x64 false")
require(provenance.get("matmul_precision") == "highest", "precision mismatch")
config = report.get("config") or {}
require(config.get("smoke") is False, "artifact is a smoke")
require(config.get("model_validation_touched") is False, "model validation touched")
require(config.get("confirmation_touched") is False, "confirmation touched")
require(config.get("candidate") == {"arm": "C", "R": 48, "k": 24, "M": 96, "m": 384},
        "candidate changed")
require(config.get("solver_subsets") == {
    "64": [512, 533, 554, 575], "128": [512, 523, 530, 543],
    "256": [512, 517, 522, 527],
}, "top-level solver subsets")
require(config.get("solver_times") == [0, 1, 10, 20, 30, 40, 50],
        "top-level solver times")
require(config.get("kernel_routes") == list(KERNEL_ORDER), "kernel routes")
require(config.get("time_repetitions") == 20, "timing repetitions")
require(config.get("identity_relative_l2_tolerance") == 2e-14,
        "identity tolerance")
require(report.get("npz", {}).get("sha256") == sha256(NPZ_PATH),
        "NPZ checksum mismatch")

binding = report.get("s0_binding") or {}
for name in ("json", "npz", "audit"):
    require(len(binding.get(f"{name}_sha256", "")) == 64, f"S0 {name} hash missing")
require(binding.get("audit_decision", {}).get("phase2_hard_stop") is True,
        "S0 hard-stop binding absent")
manifest = manifest_entries(MANIFEST_PATH)
source_hashes = provenance.get("source_sha256") or {}
required_sources = {
    "b10_common.py", "b10_spline.py", "b10_s0_spline.py",
    "b10_phase3.py", "b10_phase3_d.py",
}
require(set(source_hashes) == required_sources, "staged source set mismatch")
for name, expected in source_hashes.items():
    relative = f"experiments/burgers-1e3-10x/{name}"
    require(manifest.get(f"code/{name}") == expected, f"manifest source hash {name}")
    require(git_blob_sha(EXPECTED_COMMIT, relative) == expected,
            f"commit source hash {name}")
require(manifest.get("code/deps/s0/s0.json") == binding["json_sha256"],
        "manifest S0 JSON binding")
require(manifest.get("code/deps/s0/s0.npz") == binding["npz_sha256"],
        "manifest S0 NPZ binding")
require(manifest.get("code/deps/s0/AUDIT.json") == binding["audit_sha256"],
        "manifest S0 audit binding")

with np.load(NPZ_PATH, allow_pickle=False) as arrays:
    require(set(arrays.files) == {"S1_coefficients", "S2_coefficients"},
            "unexpected solver NPZ keys")
    for name in SOLVER_NAMES:
        values = np.asarray(arrays[f"{name}_coefficients"], np.float64)
        require(values.shape == (128, 48 * 48), f"{name} coefficient shape")
        require(np.all(np.isfinite(values)), f"{name} coefficient finiteness")

solver = report["solver_diagnostic"]
require(solver["config"] == {
    "subsets": {
        "64": [512, 533, 554, 575],
        "128": [512, 523, 530, 543],
        "256": [512, 517, 522, 527],
    },
    "times": [0, 1, 10, 20, 30, 40, 50],
    "target_full_trajectory": {"N": 128, "draw_index": 530},
    "normal_tolerance": 1e-8,
    "no_regression_absolute": 1e-5,
}, "solver diagnostic config mismatch")
solver_pass = {}
solver_worst = {}
for name in SOLVER_NAMES:
    records = solver["records"][name]
    require(len(records) == 128, f"{name} fit count")
    target = sorted(
        [row for row in records if row["N"] == 128 and row["draw_index"] == 530],
        key=lambda row: row["time_index"],
    )
    require([row["time_index"] for row in target] == list(range(51)),
            f"{name} targeted trajectory coverage")
    for row in records:
        same(
            np.sqrt(row["error_numerator_sq"] / max(row["truth_norm_sq"], 1e-300)),
            row["field_relative_l2"], f"{name} field error",
        )
        require(row["no_regression"] == (
            row["field_relative_l2"] <= row["s0_field_relative_l2"] + 1e-5
        ), f"{name} no-regression bit")
    trajectory = np.sqrt(
        sum(row["error_numerator_sq"] for row in target)
        / max(sum(row["truth_norm_sq"] for row in target), 1e-300)
    )
    summary = solver["summaries"][name]
    same(trajectory, summary["target_N128_draw530_trajectory_relative_l2"],
         f"{name} target trajectory")
    all_health = all(
        row["healthy"] and row["relative_normal_residual"] <= 1e-8
        and row["boundary_exact"] for row in records
    )
    no_regression = all(row["no_regression"] for row in records)
    target_pass = bool(
        trajectory <= 7e-4
        and trajectory <= summary["target_N128_draw530_s0_trajectory_relative_l2"] + 1e-5
    )
    solver_worst[name] = max(row["relative_normal_residual"] for row in records)
    solver_pass[name] = bool(all_health and no_regression and target_pass)
    require(summary["all_fit_health_pass"] == all_health, f"{name} health summary")
    require(summary["no_snapshot_regression_pass"] == no_regression,
            f"{name} regression summary")
    require(summary["target_N128_draw530_pass"] == target_pass, f"{name} target gate")
    require(summary["pass"] == solver_pass[name], f"{name} final solver gate")
passing_solvers = [name for name in SOLVER_NAMES if solver_pass[name]]
selected_solver = None if not passing_solvers else min(
    passing_solvers, key=lambda name: (solver_worst[name], 0 if name == "S1" else 1)
)
require(solver["selected_solver"] == selected_solver, "selected solver mismatch")

kernel = report["kernel_diagnostic"]
basis_identity = kernel["basis_identity"]
require(
    basis_identity == {
        "seed": 20260820, "probe_count": 4234,
        "unique_knots_nextafter_random": True,
        "support_indices_exact": True, "local_support": 16,
        "weight_max_abs": basis_identity["weight_max_abs"],
        "weight_max_scaled_eps": basis_identity["weight_max_scaled_eps"],
        "pass": True,
    }
    and np.isfinite(basis_identity["weight_max_abs"])
    and basis_identity["weight_max_abs"] <= 2e-14,
    "basis knot/nextafter/random identity",
)
pallas_basis = kernel["pallas_basis_identity"]
if "K3" in kernel["setup"]:
    require(
        pallas_basis.get("available") is True
        and pallas_basis.get("executed") is True
        and pallas_basis.get("support_indices_exact") is True
        and pallas_basis.get("local_support") == 4
        and pallas_basis.get("probe_count") == 4234
        and np.isfinite(pallas_basis.get("weight_max_abs", np.inf))
        and pallas_basis["weight_max_abs"] <= 2e-14
        and pallas_basis.get("pass") is True,
        "K3-specific knot/nextafter/random basis identity",
    )
else:
    require(
        "K3" in kernel.get("compile_failures", {})
        and pallas_basis.get("pass") is False
        and isinstance(kernel["compile_failures"]["K3"], str)
        and len(kernel["compile_failures"]["K3"]) > 0,
        "absent K3 lacks a classified probe/compile failure",
    )
methods = ["fom"] + [name for name in KERNEL_ORDER if name in kernel["setup"]]
require(methods[:4] == ["fom", "K0", "K1", "K2"], "mandatory methods missing/order")
require(set(kernel["compile_failures"]).issubset({"K3"}), "non-K3 compile failure")
require(len(methods) in (4, 5), "unexpected timed method count")
orders = kernel["timing_orders"]
require(len(orders) == 20 and all(set(order) == set(methods) for order in orders),
        "timing orders malformed")
position_counts = {
    method: [sum(order[position] == method for order in orders) for position in range(len(methods))]
    for method in methods
}
expected_position = 20 // len(methods)
require(20 % len(methods) == 0 and all(
    value == expected_position for counts in position_counts.values() for value in counts
), "timing position balance")
require(position_counts == kernel["position_counts"], "position counts mismatch")
require(kernel["exact_position_balance"] is True, "balance flag false")
summaries = {}
for method in methods:
    summaries[method] = timing_summary(kernel["records"][method])
    recorded = kernel["summaries"][method]
    same(summaries[method]["median_elapsed_s"], recorded["median_elapsed_s"],
         f"{method} median")
    require(summaries[method]["outliers_gt_1p5_within_trajectory_all"]
            == recorded["outliers_gt_1p5_within_trajectory_all"],
            f"{method} outliers")

fom_rows = kernel["records"]["fom"]
fom_accuracy_rows = [row for row in fom_rows if row["repetition"] == 0]
fom_mean = np.mean([row["trajectory_relative_l2"] for row in fom_accuracy_rows])
fom_worst = np.max([row["trajectory_relative_l2"] for row in fom_accuracy_rows])
fom_healthy = all(
    row["finite"] and row["breakdowns"] == 0 and row["flags_nonzero"] == 0
    and row["max_returned_relative_residual"] <= 3e-3 for row in fom_rows
)
fom_eligible = bool(fom_healthy and fom_mean <= 1e-3 and fom_worst <= 3e-3)
same(fom_mean, kernel["fom_accuracy"]["mean"], "FOM mean")
same(fom_worst, kernel["fom_accuracy"]["worst"], "FOM worst")
require(kernel["fom_accuracy"]["healthy"] == fom_healthy, "FOM health")
require(kernel["fom_accuracy"]["eligible"] == fom_eligible, "FOM eligibility")

kernel_pass, kernel_medians = {}, {}
for index, route in enumerate(name for name in KERNEL_ORDER if name in kernel["setup"]):
    work = kernel["canonical_work"][route]
    work_pass = bool(len(work) == 4 and all(
        row["finite"] and row["weak_objective_evaluations"] == 50
        and row["weak_jacobian_evaluations"] == 0
        and row["trial_residual_evaluations"] == 0 for row in work
    ))
    require(work_pass, f"{route} work accounting")
    identity_pass = route != "K0" and all(
        row["pass"] and row["max_relative_l2"] <= 2e-14
        and row["exact_boundary"] for row in kernel["identity"].get(route, [])
    ) and len(kernel["identity"].get(route, [])) == 4 and (
        route != "K3" or pallas_basis.get("pass") is True
    )
    memory_ok = kernel["setup"][route]["memory_analysis"]["eligibility_device_bytes"] <= 20_000_000_000
    speedup = summaries["fom"]["median_elapsed_s"] / summaries[route]["median_elapsed_s"]
    ci = clustered_ci(
        summaries["fom"]["per_case_median_elapsed_s"],
        summaries[route]["per_case_median_elapsed_s"], 20263000 + index,
    )
    expected_pass = bool(
        identity_pass and work_pass and memory_ok and fom_eligible
        and speedup >= 10.0 and ci[0] >= 8.0
    )
    gate = kernel["gates"][route]
    same(speedup, gate["paired_median_speedup"], f"{route} speedup")
    require(np.allclose(ci, gate["clustered_speedup_ci"], rtol=1e-12, atol=1e-15),
            f"{route} clustered CI")
    require(gate["pass"] == expected_pass, f"{route} gate")
    require(gate["canonical_work_pass"] == work_pass, f"{route} work gate")
    kernel_pass[route] = expected_pass
    kernel_medians[route] = summaries[route]["median_elapsed_s"]
passing_kernels = [name for name in ("K1", "K2", "K3") if kernel_pass.get(name)]
selected_kernel = None if not passing_kernels else min(
    passing_kernels, key=lambda name: (kernel_medians[name], KERNEL_ORDER.index(name))
)
require(kernel["selected_kernel"] == selected_kernel, "selected kernel mismatch")

decision = report["decision"]
run_p3_f = bool(selected_solver and selected_kernel)
require(decision["selected_solver"] == selected_solver, "decision solver")
require(decision["selected_kernel"] == selected_kernel, "decision kernel")
require(decision["run_P3_F"] == run_p3_f, "P3-F decision")
require(decision["phase3_hard_stop"] == (not run_p3_f), "hard-stop decision")

audit = {
    "status": "pass", "source_json_basename": os.path.basename(JSON_PATH),
    "source_json_sha256": sha256(JSON_PATH),
    "source_npz_sha256": sha256(NPZ_PATH),
    "manifest_file_sha256": sha256(MANIFEST_PATH),
    "expected_commit": EXPECTED_COMMIT, "job_id": JOB_ID,
    "selected_solver": selected_solver, "selected_kernel": selected_kernel,
    "run_P3_F": run_p3_f, "phase3_hard_stop": not run_p3_f,
    "fom_eligible": fom_eligible,
}
with open(AUDIT_PATH, "w") as handle:
    json.dump(audit, handle, indent=1, allow_nan=False, sort_keys=True)
print(json.dumps(audit, indent=1))
