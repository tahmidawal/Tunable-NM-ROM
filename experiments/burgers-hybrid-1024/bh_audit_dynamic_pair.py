"""Validate the disjoint dynamic AB/BA result and freeze its mesh policy."""
from __future__ import annotations

import hashlib
import json
import math
import sys

SOURCE, SELECTION, OUT = sys.argv[1:4]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


with open(SOURCE) as handle:
    report = json.load(handle)
with open(SELECTION) as handle:
    selection = json.load(handle)
if not report.get("complete"):
    raise SystemExit("dynamic AB/BA report is incomplete")

config = report["config"]
expected = {
    "classification": "classical FOM warm-start control; not learned and not NM-ROM",
    "N": 256,
    "coarse_n": 256,
    "relaxation": 1.0,
    "fom_tau": 1e-6,
    "lin_tol": 1e-2,
    "preconditioner": "exact target-grid Helmholtz",
    "development_seed": 20260818,
    "development_indices": [8, 9, 10, 11],
    "draw_count": 16,
    "selection_indices_reused": False,
    "pair_blocks": 6,
    "repetitions_per_trajectory": 12,
    "burn_before_every_AB_and_BA_block": True,
    "burn_seconds": 3.0,
    "accuracy_work_residual_from_every_timed_invocation": True,
    "new_final_seed_touched": False,
    "f64": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"dynamic pair {key} drifted: {config.get(key)!r} != {value!r}")
provenance = report["provenance"]
for key, value in {"jax_backend": "gpu", "x64": True, "matmul_precision": "highest"}.items():
    if provenance.get(key) != value:
        raise SystemExit(f"dynamic pair provenance {key} drifted")
if not provenance.get("gpu_kind") or not provenance.get("slurm_job_id"):
    raise SystemExit("dynamic pair is missing GPU/Slurm provenance")
if config["selection_sha256"] != sha256(SELECTION):
    raise SystemExit("dynamic pair did not use the retained selection artifact")
selected = selection.get("selected")
if (
    selection.get("status") != "promoted"
    or selected is None
    or selected.get("arm") != "coarse256_a1"
    or selected.get("coarse_n") != 256
    or selected.get("relaxation") != 1.0
):
    raise SystemExit("selection artifact does not promote q=N=256, alpha=1")

expected_pairs = {
    (trajectory, repetition)
    for trajectory in range(8, 12)
    for repetition in range(48)
    if repetition // 8 < 6 and (repetition % 8) // 2 == trajectory - 8
}
# ``repetition`` advances AB then BA across the four trajectories in every
# block: each trajectory therefore owns exactly two consecutive ids per block.
for arm in ("cubic", "dynamic"):
    records = report["records"].get(arm, [])
    if len(records) != 48:
        raise SystemExit(f"dynamic pair {arm} has {len(records)} rather than 48 records")
    record_pairs = {(item["trajectory_index"], item["repetition"]) for item in records}
    if record_pairs != expected_pairs:
        raise SystemExit(f"dynamic pair record grid drifted for {arm}")
    if any(
        not math.isfinite(item["elapsed_s"])
        or item["elapsed_s"] <= 0
        or item["breakdowns"]
        or item["flags_nonzero"]
        or item["max_returned_residual"] > 1e-6
        for item in records
    ):
        raise SystemExit(f"dynamic pair has invalid timed invocation for {arm}")
    by_trajectory = {
        trajectory: [item for item in records if item["trajectory_index"] == trajectory]
        for trajectory in range(8, 12)
    }
    if any(
        len(values) != 12
        or sum(item["order"] == "cubic/dynamic" for item in values) != 6
        or sum(item["order"] == "dynamic/cubic" for item in values) != 6
        for values in by_trajectory.values()
    ):
        raise SystemExit(f"dynamic pair AB/BA balance drifted for {arm}")

summary = report["summary"]
savings = summary["saving_vs_cubic_case_medians_ms"]
interval = summary["saving_vs_cubic_trajectory_cluster_95ci_ms"]
gates = {
    "all_trajectory_median_savings_positive": min(savings) > 0.0,
    "trajectory_cluster_interval_positive": interval[0] > 0.0,
    "solver_health": summary["max_returned_residual"] <= 1e-6
    and summary["max_trajectory_rel_l2"] < 1e-5,
    "wall_speedup_positive": summary["speedup_cubic_over_dynamic"] > 1.0,
}
status = "promoted" if all(gates.values()) else "hard_stop"
audit = {
    "status": status,
    "source": SOURCE,
    "source_sha256": sha256(SOURCE),
    "selection": SELECTION,
    "selection_sha256": sha256(SELECTION),
    "classification": "classical FOM warm start; not learned and not NM-ROM",
    "gates": gates,
    "selected_policy": {
        "relaxation": 1.0,
        "resolution_policy": "coarse_n=N at every mesh (full-grid fraction 1.0)",
        "basis": "only q=N passed at N=256; q=64 and q=128 failed",
    } if status == "promoted" else None,
    "timing": {
        "cubic_median_ms": summary["cubic_median_ms"],
        "dynamic_median_ms": summary["dynamic_median_ms"],
        "speedup_cubic_over_dynamic": summary["speedup_cubic_over_dynamic"],
        "saving_vs_cubic_case_medians_ms": savings,
        "saving_trajectory_cluster_95ci_ms": interval,
    },
    "finish_work": {
        "interpretation": (
            "Newton/BiCGStab are finish-only counters; wall time also charges "
            "the dynamic corrections listed separately"
        ),
        "cubic_newton_median": summary["cubic_newton_total_median"],
        "dynamic_newton_median": summary["dynamic_newton_total_median"],
        "cubic_bicgstab_median": summary["cubic_linear_total_median"],
        "dynamic_bicgstab_median": summary["dynamic_linear_total_median"],
    },
    "dynamic_charged_correction_work_per_trajectory": {
        "full_grid_residual_evaluations": 50,
        "exact_helmholtz_inverses": 50,
    },
    "outliers_retained": summary["tukey_1p5iqr_outliers_retained"],
}
with open(OUT, "w") as handle:
    json.dump(audit, handle, indent=1, allow_nan=False)
print(json.dumps(audit, indent=1))
