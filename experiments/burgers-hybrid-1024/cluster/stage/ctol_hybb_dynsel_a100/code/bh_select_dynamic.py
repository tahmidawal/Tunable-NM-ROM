"""Apply the pre-registered dynamic-correction promotion gates."""
from __future__ import annotations

import hashlib
import json
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
if report["config"]["ns"] != [256] or report["config"]["fom_tau"] != 1e-6:
    raise SystemExit("dynamic selection configuration does not match preregistration")
rows = {row["arm"]: row for row in report["rows"]}
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
