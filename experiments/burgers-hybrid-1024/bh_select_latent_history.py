"""Derive the locked FiLM latent-history choice from a paired gate JSON."""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


INPUT, OUTPUT = sys.argv[1:3]
PREVIOUS = "film_nmrom_previous"
EXTRAPOLATION = "film_nmrom_extrapolation"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_median_ci(values, seed=20260819):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(10000, values.size))]
    return [float(value) for value in np.quantile(
        np.median(samples, axis=1), [0.025, 0.975]
    )]


with open(INPUT) as handle:
    gate = json.load(handle)
if not gate.get("complete"):
    raise SystemExit("gate is incomplete")
rows = {row["arm"]: row for row in gate["rows"]}
if PREVIOUS not in rows or EXTRAPOLATION not in rows:
    raise SystemExit("gate does not contain both latent-history variants")

records = {}
for arm in (PREVIOUS, EXTRAPOLATION):
    for record in rows[arm]["timed_records"]:
        key = (record["trajectory_index"], record["repetition"])
        records[(arm, key)] = record
keys = sorted(key for arm, key in records if arm == PREVIOUS)
if keys != sorted(key for arm, key in records if arm == EXTRAPOLATION):
    raise SystemExit("latent variants do not have matched timing records")

previous = [records[(PREVIOUS, key)] for key in keys]
extrapolation = [records[(EXTRAPOLATION, key)] for key in keys]
paired_saving = np.asarray([
    left["elapsed_s"] - right["elapsed_s"]
    for left, right in zip(previous, extrapolation)
])
if not np.all(np.isfinite(paired_saving)):
    raise SystemExit("non-finite timing delta")

health_fields = ("finite", "breakdowns", "flags_nonzero")
healthy = all(
    record["finite"] and not record["breakdowns"] and not record["flags_nonzero"]
    for record in previous + extrapolation
)
same_fom_work = all(
    left["newton_total"] == right["newton_total"]
    and left["linear_total"] == right["linear_total"]
    for left, right in zip(previous, extrapolation)
)
if not healthy:
    raise SystemExit(f"unhealthy gate records: {health_fields}")

ci = bootstrap_median_ci(paired_saving)
selected = (
    "extrapolation"
    if float(np.median(paired_saving)) > 0.0 and ci[0] > 0.0
    else "previous"
)
report = {
    "input": INPUT,
    "input_sha256": sha256(INPUT),
    "selection_population": "held-out canonical draw-16 indices 4:8",
    "selection_metric": "paired end-to-end NM-ROM construction plus locked FOM time",
    "selected_mode": selected,
    "selected_latent_extrapolation_scale": 1.0 if selected == "extrapolation" else 0.0,
    "n_pairs": int(paired_saving.size),
    "all_pairs_extrapolation_faster": bool(np.all(paired_saving > 0.0)),
    "paired_extrapolation_saving_vs_previous_s": paired_saving.tolist(),
    "paired_extrapolation_saving_vs_previous_median_ms": float(
        np.median(paired_saving) * 1e3
    ),
    "paired_extrapolation_saving_vs_previous_median_95ci_ms": [
        float(value * 1e3) for value in ci
    ],
    "same_fom_newton_and_bicg_work_every_pair": same_fom_work,
    "previous_reduced_jacobians_median": float(np.median([
        record["reduced_jacobians_total"] for record in previous
    ])),
    "extrapolation_reduced_jacobians_median": float(np.median([
        record["reduced_jacobians_total"] for record in extrapolation
    ])),
    "previous_reduced_attempts_median": float(np.median([
        record["reduced_attempts_total"] for record in previous
    ])),
    "extrapolation_reduced_attempts_median": float(np.median([
        record["reduced_attempts_total"] for record in extrapolation
    ])),
    "max_returned_outer_residual": float(max(
        record["max_final_rel_residual"] for record in previous + extrapolation
    )),
    "gate_fom_tau": rows[PREVIOUS]["fom_tau"],
    "complete": True,
}
with open(OUTPUT, "w") as handle:
    json.dump(report, handle, indent=1, allow_nan=False)
