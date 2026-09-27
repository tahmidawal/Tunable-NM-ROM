"""Apply the pre-registered dynamic-correction promotion gates."""
from __future__ import annotations

import hashlib
import json
import math
import sys

SOURCE, OUT = sys.argv[1:3]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


with open(SOURCE) as handle:
    report = json.load(handle)
if not report.get("complete"):
    raise SystemExit("dynamic selection gate is incomplete")
config = report["config"]
provenance = report["provenance"]
expected_config = {
    "classification": "classical reduced correction control; not a learned NM-ROM",
    "ns": [256],
    "coarse_ns": [64, 128, 256],
    "relaxations": [1.0],
    "fom_tau": 1e-6,
    "lin_tol": 1e-2,
    "preconditioner": "exact target-grid Helmholtz FOM; one exact coarse Helmholtz correction",
    "history": "live cubic",
    "development_seed": 20260818,
    "development_indices": [0, 1, 2, 3],
    "draw_count": 16,
    "confirmation_touched": False,
    "time_reps": 7,
    "burn_seconds": 3.0,
    "f64": True,
}
for key, expected in expected_config.items():
    if config.get(key) != expected:
        raise SystemExit(
            f"dynamic selection {key} drifted: {config.get(key)!r} != {expected!r}"
        )
expected_provenance = {
    "jax_backend": "gpu",
    "x64": True,
    "matmul_precision": "highest",
}
for key, expected in expected_provenance.items():
    if provenance.get(key) != expected:
        raise SystemExit(
            f"dynamic selection provenance {key} drifted: "
            f"{provenance.get(key)!r} != {expected!r}"
        )
if not provenance.get("gpu_kind") or not provenance.get("slurm_job_id"):
    raise SystemExit("dynamic selection is missing GPU/Slurm provenance")
rows = {row["arm"]: row for row in report["rows"]}
expected_arms = {"cubic", "coarse64_a1", "coarse128_a1", "coarse256_a1"}
if set(rows) != expected_arms or len(report["rows"]) != len(expected_arms):
    raise SystemExit(
        f"dynamic selection arm grid drifted: {sorted(rows)} != {sorted(expected_arms)}"
    )
for arm, row in rows.items():
    expected_coarse = None if arm == "cubic" else int(arm.split("_")[0][6:])
    expected_relaxation = None if arm == "cubic" else 1.0
    if (
        row.get("N") != 256
        or row.get("coarse_n") != expected_coarse
        or row.get("relaxation") != expected_relaxation
    ):
        raise SystemExit(f"dynamic selection row metadata drifted for {arm}")
    records = row.get("timed_records", [])
    if len(records) != 4 * 7:
        raise SystemExit(f"dynamic selection {arm} has {len(records)} rather than 28 records")
    record_grid = {
        (item.get("trajectory_index"), item.get("repetition")) for item in records
    }
    expected_grid = {(case, repetition) for case in range(4) for repetition in range(7)}
    if record_grid != expected_grid or len(row.get("case_timing_repetitions_s", [])) != 4:
        raise SystemExit(f"dynamic selection timed record grid drifted for {arm}")
    if any(len(values) != 7 for values in row["case_timing_repetitions_s"]):
        raise SystemExit(f"dynamic selection timing repetitions drifted for {arm}")
    if any(not math.isfinite(item["elapsed_s"]) or item["elapsed_s"] <= 0 for item in records):
        raise SystemExit(f"dynamic selection has invalid timing for {arm}")
cubic = rows["cubic"]
cubic_cases = cubic["case_medians_s"]
decisions = []
for arm, row in rows.items():
    if arm == "cubic":
        continue
    case_ratios = [
        candidate / baseline
        for candidate, baseline in zip(row["case_medians_s"], cubic_cases)
    ]
    gates = {
        "newton_reduction_at_least_20pct": row["newton_total_median"]
        <= 0.8 * cubic["newton_total_median"],
        "bicgstab_reduction_at_least_20pct": row["linear_total_median"]
        <= 0.8 * cubic["linear_total_median"],
        "positive_median_total_saving": row["saving_vs_cubic_median_ms"] > 0.0,
        "no_case_more_than_10pct_slower": max(case_ratios) <= 1.1,
        "solver_health": row["max_returned_residual"] <= report["config"]["fom_tau"]
        and row["max_trajectory_rel_l2"] < 1e-5,
    }
    decisions.append(
        {
            "arm": arm,
            "coarse_n": row["coarse_n"],
            "relaxation": row["relaxation"],
            "timing_median_ms": row["timing_median_ms"],
            "saving_vs_cubic_median_ms": row["saving_vs_cubic_median_ms"],
            "case_time_ratios_vs_cubic": case_ratios,
            "gates": gates,
            "eligible": all(gates.values()),
        }
    )
eligible = [item for item in decisions if item["eligible"]]
selected = min(eligible, key=lambda item: item["timing_median_ms"]) if eligible else None
choice = {
    "status": "promoted" if selected else "hard_stop",
    "source": SOURCE,
    "source_sha256": sha256(SOURCE),
    "aggregation": {
        "time": "median repetitions within trajectory, then median trajectories",
        "work": "deterministic per trajectory, then median trajectories",
    },
    "decisions": decisions,
    "selected": selected,
}
with open(OUT, "w") as handle:
    json.dump(choice, handle, indent=1, allow_nan=False)
print(json.dumps(choice, indent=1))
