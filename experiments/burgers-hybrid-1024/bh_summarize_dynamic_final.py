"""Validate and render the fresh dynamic-warm-start confirmation."""
from __future__ import annotations

import hashlib
import json
import math
import sys

SOURCE, SELECTION, PAIR_AUDIT, SUMMARY_OUT, TABLE_OUT = sys.argv[1:6]
EXPECTED_NS = [32, 64, 128, 256, 512, 1024]
EXPECTED_CONDITIONS = [(1e-6, 1e-2), (1e-8, 1e-4), (1e-10, 1e-5)]
EXPECTED_EXECUTION_COMMIT = "9cdeca5aabc9f911f3d6057e371813e11d3e6c89"
EXPECTED_FINAL_SOURCE_SHA256 = "cbc067f08a5aba834008599fabd0aa979c4bab736de8c523d6e7f9281cff4b18"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


with open(SOURCE) as handle:
    report = json.load(handle)
if not report.get("complete"):
    raise SystemExit("dynamic final report is incomplete")
config = report["config"]
expected_config = {
    "ns": EXPECTED_NS,
    "conditions": [
        {"fom_tau": tau, "linear_tol": linear_tol}
        for tau, linear_tol in EXPECTED_CONDITIONS
    ],
    "resolution_policy": "coarse_n=N at every mesh (full-grid fraction 1.0)",
    "relaxation": 1.0,
    "preconditioner": "exact target-grid Helmholtz for both FOM finishes",
    "dynamic_correction_steps_per_trajectory": 50,
    "dynamic_extra_full_grid_residual_evaluations": 50,
    "dynamic_extra_exact_helmholtz_inverses": 50,
    "test_seed": 20260825,
    "canonical_draw_count": 16,
    "selected_test_indices": [0, 1, 2, 3],
    "pair_blocks": 6,
    "repetitions_per_arm_per_trajectory": 12,
    "burn_before_every_AB_and_BA_block": True,
    "burn_seconds": 3.0,
    "accuracy_work_residual_from_every_timed_invocation": True,
    "reference_residual_gate": 1e-11,
    "fresh_seed_touched": True,
    "smoke": False,
    "f64": True,
}
for key, expected in expected_config.items():
    if config.get(key) != expected:
        raise SystemExit(f"dynamic final {key} drifted: {config.get(key)!r} != {expected!r}")
if config["selection_sha256"] != sha256(SELECTION):
    raise SystemExit("dynamic final selection artifact hash mismatch")
if config["pair_audit_sha256"] != sha256(PAIR_AUDIT):
    raise SystemExit("dynamic final pair-audit artifact hash mismatch")
provenance = report["provenance"]
for key, expected in {"jax_backend": "gpu", "x64": True, "matmul_precision": "highest"}.items():
    if provenance.get(key) != expected:
        raise SystemExit(f"dynamic final provenance {key} drifted")
if not provenance.get("gpu_kind") or not provenance.get("slurm_job_id"):
    raise SystemExit("dynamic final is missing GPU/Slurm provenance")
if provenance.get("commit") != EXPECTED_EXECUTION_COMMIT:
    raise SystemExit("dynamic final execution commit drifted")
if provenance.get("source_sha256", {}).get("bh_dynamic_final.py") != EXPECTED_FINAL_SOURCE_SHA256:
    raise SystemExit("dynamic final staged source hash drifted")
if "bh_summarize_dynamic_final.py" in provenance.get("source_sha256", {}):
    raise SystemExit("post-stage validator unexpectedly affected the execution cell")
if set(report["reference_health"]) != {str(n) for n in EXPECTED_NS}:
    raise SystemExit("dynamic final reference-health mesh grid drifted")
if any(
    value["max_reference_newton_residual"] > 1e-11
    for value in report["reference_health"].values()
):
    raise SystemExit("dynamic final reference generation failed its residual gate")

rows = {(row["N"], row["fom_tau"]): row for row in report["rows"]}
expected_grid = {(n, tau) for n in EXPECTED_NS for tau, _ in EXPECTED_CONDITIONS}
if set(rows) != expected_grid or len(report["rows"]) != len(expected_grid):
    raise SystemExit("dynamic final condition grid is incomplete or duplicated")

rendered = []
for n in EXPECTED_NS:
    for tau, linear_tol in EXPECTED_CONDITIONS:
        row = rows[(n, tau)]
        if (
            not row.get("complete")
            or row.get("linear_tol") != linear_tol
            or row.get("coarse_n") != n
            or row.get("relaxation") != 1.0
        ):
            raise SystemExit(f"dynamic final row metadata drifted for N={n}, tau={tau}")
        for arm in ("cubic", "dynamic"):
            records = row["records"].get(arm, [])
            if len(records) != 48:
                raise SystemExit(f"N={n}, tau={tau}, {arm}: expected 48 records")
            expected_records = {
                (trajectory, sample)
                for trajectory in range(4)
                for sample in range(12)
            }
            record_grid = {
                (record["trajectory_index"], record["sample_index"])
                for record in records
            }
            if record_grid != expected_records:
                raise SystemExit(f"N={n}, tau={tau}, {arm}: record grid drifted")
            for trajectory in range(4):
                case = [
                    record for record in records
                    if record["trajectory_index"] == trajectory
                ]
                if (
                    len(case) != 12
                    or sum(record["order"] == "cubic/dynamic" for record in case) != 6
                    or sum(record["order"] == "dynamic/cubic" for record in case) != 6
                ):
                    raise SystemExit(f"N={n}, tau={tau}, {arm}: AB/BA imbalance")
            expected_extra = 50 if arm == "dynamic" else 0
            if any(
                not math.isfinite(record["elapsed_s"])
                or record["elapsed_s"] <= 0
                or record["breakdowns"]
                or record["flags_nonzero"]
                or record["max_returned_residual"] > tau
                or record["extra_full_grid_residual_evaluations"] != expected_extra
                or record["extra_exact_helmholtz_inverses"] != expected_extra
                for record in records
            ):
                raise SystemExit(f"N={n}, tau={tau}, {arm}: timed invocation failed")
        if len(row["burn_records"]) != 48:
            raise SystemExit(f"N={n}, tau={tau}: burn grid drifted")
        summary = row["summary"]
        if (
            summary["dynamic_extra_full_grid_residual_evaluations"] != 50
            or summary["dynamic_extra_exact_helmholtz_inverses"] != 50
            or summary["max_returned_residual"] > tau
            or summary["max_trajectory_rel_l2"] >= 1e-4
        ):
            raise SystemExit(f"N={n}, tau={tau}: summary health failed")
        equivalence = row["equivalence"]
        if any(
            item["breakdowns"] or item["flags_nonzero"]
            for item in equivalence["per_trajectory"]
        ) or equivalence["linear_solver"]["ours_flag"]:
            raise SystemExit(f"N={n}, tau={tau}: solver equivalence failed")
        ci = summary["paired_saving_trajectory_cluster_95ci_ms"]
        rendered.append({
            "N": n,
            "fom_tau": tau,
            "linear_tol": linear_tol,
            "cubic_median_ms": summary["cubic_median_ms"],
            "dynamic_median_ms": summary["dynamic_median_ms"],
            "speedup_cubic_over_dynamic": summary["speedup_cubic_over_dynamic"],
            "paired_saving_median_ms": summary["paired_saving_vs_cubic_median_ms"],
            "paired_saving_trajectory_cluster_95ci_ms": ci,
            "all_case_savings_positive": min(
                summary["paired_saving_vs_cubic_case_medians_ms"]
            ) > 0,
            "supported_speedup": ci[0] > 0,
            "finish_newton_median": summary["finish_newton_total_median"],
            "finish_bicgstab_median": summary["finish_bicgstab_total_median"],
            "dynamic_extra_full_grid_residual_evaluations": 50,
            "dynamic_extra_exact_helmholtz_inverses": 50,
            "max_returned_residual": summary["max_returned_residual"],
            "max_trajectory_rel_l2": summary["max_trajectory_rel_l2"],
            "outliers_retained": summary["tukey_1p5iqr_outliers_retained"],
        })

summary_out = {
    "status": "complete",
    "source": SOURCE,
    "source_sha256": sha256(SOURCE),
    "classification": "classical FOM warm start; not learned and not NM-ROM",
    "timing_scope": config["timing_scope"],
    "rows": rendered,
    "supported_speedup_cells": sum(row["supported_speedup"] for row in rendered),
    "positive_median_cells": sum(
        row["paired_saving_median_ms"] > 0 for row in rendered
    ),
    "all_case_positive_cells": sum(row["all_case_savings_positive"] for row in rendered),
    "tukey_1p5iqr_outliers_retained_total": int(sum(
        arm["total"]
        for row in rendered
        for arm in row["outliers_retained"].values()
    )),
    "total_cells": len(rendered),
}
with open(SUMMARY_OUT, "w") as handle:
    json.dump(summary_out, handle, indent=1, allow_nan=False)

lines = [
    "| N | FOM tol | cubic ms | dynamic ms | speedup | paired saving ms [95% cluster CI] | finish Newton c/d | finish BiCG c/d | outliers c/d | supported |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
]
for row in rendered:
    ci = row["paired_saving_trajectory_cluster_95ci_ms"]
    newton = row["finish_newton_median"]
    bicg = row["finish_bicgstab_median"]
    lines.append(
        f"| {row['N']} | {row['fom_tau']:.0e} | {row['cubic_median_ms']:.3f} | "
        f"{row['dynamic_median_ms']:.3f} | {row['speedup_cubic_over_dynamic']:.3f}x | "
        f"{row['paired_saving_median_ms']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
        f"{newton['cubic']:.1f}/{newton['dynamic']:.1f} | "
        f"{bicg['cubic']:.1f}/{bicg['dynamic']:.1f} | "
        f"{row['outliers_retained']['cubic']['total']}/"
        f"{row['outliers_retained']['dynamic']['total']} | "
        f"{'yes' if row['supported_speedup'] else 'no'} |"
    )
with open(TABLE_OUT, "w") as handle:
    handle.write("\n".join(lines) + "\n")
print(json.dumps({
    "supported_speedup_cells": summary_out["supported_speedup_cells"],
    "positive_median_cells": summary_out["positive_median_cells"],
    "all_case_positive_cells": summary_out["all_case_positive_cells"],
    "tukey_1p5iqr_outliers_retained_total": summary_out[
        "tukey_1p5iqr_outliers_retained_total"
    ],
    "total_cells": summary_out["total_cells"],
}, indent=1))
