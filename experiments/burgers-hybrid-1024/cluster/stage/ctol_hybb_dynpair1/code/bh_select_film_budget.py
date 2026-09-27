"""Lock the cheapest bounded FiLM arm that produces a usable reduced update."""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


INPUT, OUTPUT = sys.argv[1:3]
J1 = "film_nmrom_extrapolation_j1"
J2 = "film_nmrom_extrapolation_j2"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def paired(rows, left, right, field):
    maps = {}
    for arm in (left, right):
        maps[arm] = {
            (record["trajectory_index"], record["repetition"]): record
            for record in rows[arm]["timed_records"]
        }
    if maps[left].keys() != maps[right].keys():
        raise SystemExit("unmatched paired records")
    return np.asarray([
        maps[left][key][field] - maps[right][key][field]
        for key in sorted(maps[left])
    ], np.float64)


def bootstrap_ci(values):
    rng = np.random.default_rng(20260819)
    sample = values[rng.integers(0, values.size, size=(10000, values.size))]
    return np.quantile(np.median(sample, axis=1), [0.025, 0.975])


with open(INPUT) as handle:
    gate = json.load(handle)
if not gate.get("complete"):
    raise SystemExit("incomplete gate")
rows = {row["arm"]: row for row in gate["rows"]}
if J1 not in rows or J2 not in rows:
    raise SystemExit("missing bounded FiLM arm")

j2_minus_j1 = -paired(rows, J1, J2, "elapsed_s")
j1_records = rows[J1]["timed_records"]
j2_records = rows[J2]["timed_records"]
j1_ever_accepted = any(record["film_guard_accepted_count"] for record in j1_records)
j2_ever_accepted = any(record["film_guard_accepted_count"] for record in j2_records)
if j1_ever_accepted or not j2_ever_accepted:
    raise SystemExit("expected zero-accept J1 and nonzero-accept J2 gate")

ci = bootstrap_ci(j2_minus_j1)
report = {
    "input": INPUT,
    "input_sha256": digest(INPUT),
    "selection_population": "held-out canonical draw-16 indices 4:8",
    "selection_rule": (
        "minimum max-step-Jacobian budget that produces and passes at least one "
        "actual reduced update through the exact live-cubic FOM-residual guard"
    ),
    "selected_max_step_jacobians": 2,
    "selected_history_mode": "extrapolation",
    "j1_classification": "zero-update safety-floor diagnostic; excluded from final",
    "j1_guard_accepted_fraction_median": rows[J1][
        "film_guard_accepted_fraction_median"
    ],
    "j2_guard_accepted_fraction_median": rows[J2][
        "film_guard_accepted_fraction_median"
    ],
    "j2_guard_accepted_count_median": rows[J2]["film_guard_accepted_count_median"],
    "j1_timing_median_ms": rows[J1]["timing_median_ms"],
    "j2_timing_median_ms": rows[J2]["timing_median_ms"],
    "j2_extra_time_vs_j1_median_ms": float(np.median(j2_minus_j1) * 1e3),
    "j2_extra_time_vs_j1_median_95ci_ms": [float(value * 1e3) for value in ci],
    "j1_reduced_jacobians_median": rows[J1]["reduced_jacobians_total_median"],
    "j2_reduced_jacobians_median": rows[J2]["reduced_jacobians_total_median"],
    "j1_linear_total_median": rows[J1]["linear_total_median"],
    "j2_linear_total_median": rows[J2]["linear_total_median"],
    "max_returned_outer_residual": max(
        rows[J1]["max_timed_outer_residual"],
        rows[J2]["max_timed_outer_residual"],
    ),
    "complete": True,
}
with open(OUTPUT, "w") as handle:
    json.dump(report, handle, indent=1, allow_nan=False)
