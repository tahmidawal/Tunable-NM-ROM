"""Recompute and render the final Burgers N=2048 evidence from run JSONs."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(HERE, "runs", "final3")
FINAL_JSON = os.path.join(FINAL_DIR, "out", "final.json")
FINAL_AUDIT = os.path.join(FINAL_DIR, "out", "final-audit.json")
LEARNED_AUDIT = os.path.join(HERE, "runs", "learned1", "LEARNED-AUDIT.json")
INDEPENDENT_OUT = os.path.join(FINAL_DIR, "INDEPENDENT-AUDIT.json")
CLASSIFICATION_OUT = os.path.join(FINAL_DIR, "CLASSIFICATION.json")
MARKDOWN_OUT = os.path.join(HERE, "FINAL-RESULT.generated.md")


def load(path):
    with open(path) as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def bootstrap_interval(values, seed, draws=20000):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [
        float(value)
        for value in np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
    ]


def tukey_total(records, trajectory_indices):
    total = 0
    for trajectory_index in trajectory_indices:
        values = np.asarray([
            record["elapsed_s"]
            for record in records
            if record["trajectory_index"] == trajectory_index
        ])
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        total += int(np.sum(
            (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        ))
    return total


def close(left, right, atol=1e-12):
    return bool(np.allclose(left, right, rtol=0.0, atol=atol))


def recompute(final, audit):
    indices = final["config"]["selected_test_indices"]
    require(indices == [0, 1, 2, 3], "unexpected final cohort")
    require(final["complete"] and audit["status"] == "complete", "incomplete final")
    require(sha256(FINAL_JSON) == audit["source_sha256"], "final hash drift")
    rows = []
    for condition_index, row in enumerate(final["rows"]):
        by_arm = {}
        expected_record_grid = {
            (index, sample) for index in indices for sample in range(12)
        }
        for arm in ("cubic", "dynamic"):
            records = row["records"][arm]
            expected_extra = 50 if arm == "dynamic" else 0
            require(len(records) == 48, f"{arm} record-count drift")
            require({
                (record["trajectory_index"], record["sample_index"])
                for record in records
            } == expected_record_grid, f"{arm} record-grid drift")
            require(all(
                record["breakdowns"] == 0
                and record["flags_nonzero"] == 0
                and record["max_returned_residual"] <= row["fom_tau"]
                and np.isfinite(record["trajectory_rel_l2"])
                and record["extra_full_grid_residual_evaluations"] == expected_extra
                and record["extra_exact_helmholtz_inverses"] == expected_extra
                and record["block"] == record["sample_index"] // 2
                and record["order"] == (
                    "cubic/dynamic"
                    if record["sample_index"] % 2 == 0
                    else "dynamic/cubic"
                )
                for record in records
            ), f"{arm} health failure")
            by_arm[arm] = [
                float(np.median([
                    record["elapsed_s"]
                    for record in records
                    if record["trajectory_index"] == index
                ]))
                for index in indices
            ]
        require(len(row["burn_records"]) == 48, "burn-count drift")
        require({
            (record["trajectory_index"], record["sample_index"])
            for record in row["burn_records"]
        } == expected_record_grid,
                "burn-grid drift")
        require(all(
            record["block"] == record["sample_index"] // 2
            and record["order"] == (
                ["cubic", "dynamic"]
                if record["sample_index"] % 2 == 0
                else ["dynamic", "cubic"]
            )
            and record["iterations"] > 0
            for record in row["burn_records"]
        ), "burn block/order drift")
        savings = []
        for index in indices:
            cubic = {
                record["sample_index"]: record["elapsed_s"]
                for record in row["records"]["cubic"]
                if record["trajectory_index"] == index
            }
            dynamic = {
                record["sample_index"]: record["elapsed_s"]
                for record in row["records"]["dynamic"]
                if record["trajectory_index"] == index
            }
            require(cubic.keys() == dynamic.keys() and len(cubic) == 12,
                    "paired grid drift")
            savings.append(float(np.median([
                1e3 * (cubic[sample] - dynamic[sample])
                for sample in sorted(cubic)
            ])))
        recomputed = {
            "fom_tau": row["fom_tau"],
            "linear_tol": row["linear_tol"],
            "cubic_median_ms": float(np.median(by_arm["cubic"]) * 1e3),
            "dynamic_median_ms": float(np.median(by_arm["dynamic"]) * 1e3),
            "speedup_cubic_over_dynamic": float(
                np.median(by_arm["cubic"]) / np.median(by_arm["dynamic"])
            ),
            "paired_saving_case_medians_ms": savings,
            "paired_saving_median_ms": float(np.median(savings)),
            "paired_saving_trajectory_cluster_95ci_ms": bootstrap_interval(
                savings,
                final["config"]["test_seed"] + row["N"] + 1009 * condition_index,
            ),
            "supported_speedup": False,
            "finish_newton_median": {
                arm: float(np.median([
                    record["finish_newton_total"] for record in row["records"][arm]
                ]))
                for arm in ("cubic", "dynamic")
            },
            "finish_bicgstab_median": {
                arm: float(np.median([
                    record["finish_bicgstab_total"] for record in row["records"][arm]
                ]))
                for arm in ("cubic", "dynamic")
            },
            "max_returned_residual": max(
                record["max_returned_residual"]
                for arm in ("cubic", "dynamic")
                for record in row["records"][arm]
            ),
            "max_trajectory_rel_l2": max(
                record["trajectory_rel_l2"]
                for arm in ("cubic", "dynamic")
                for record in row["records"][arm]
            ),
            "outliers_retained": {
                arm: tukey_total(row["records"][arm], indices)
                for arm in ("cubic", "dynamic")
            },
        }
        recomputed["supported_speedup"] = (
            recomputed["paired_saving_trajectory_cluster_95ci_ms"][0] > 0
        )
        published = audit["rows"][condition_index]
        for key in (
            "cubic_median_ms",
            "dynamic_median_ms",
            "speedup_cubic_over_dynamic",
            "paired_saving_case_medians_ms",
            "paired_saving_median_ms",
            "paired_saving_trajectory_cluster_95ci_ms",
            "max_returned_residual",
            "max_trajectory_rel_l2",
        ):
            require(close(recomputed[key], published[key]), f"{key} recompute drift")
        for key in ("finish_newton_median", "finish_bicgstab_median"):
            require(recomputed[key] == published[key], f"{key} recompute drift")
        require(recomputed["supported_speedup"] == published["supported_speedup"],
                "support classification drift")
        for arm in ("cubic", "dynamic"):
            require(
                recomputed["outliers_retained"][arm]
                == published["outliers_retained"][arm]["total"],
                "outlier recompute drift",
            )
        equivalence = row["equivalence"]
        require(
            equivalence["reference_kind"]
            == "prospective_dual_exact_helmholtz_candidate"
            and equivalence["max_step_rel_difference"] <= 10 * row["fom_tau"]
            and equivalence["max_trajectory_rel_difference"] <= 10 * row["fom_tau"]
            and all(
                item["breakdowns"] == 0
                and item["flags_nonzero"] == 0
                and item["all_fields_finite"]
                and item["counting_max_rel_newton_residual"] <= row["fom_tau"]
                for item in equivalence["per_trajectory"]
            ),
            "equivalence health drift",
        )
        rows.append(recomputed)

    reference = final["reference_health"]["2048"]
    raw_residuals = [
        value
        for trajectory in reference["trajectories"]
        for route in trajectory["routes"]
        for value in route["per_step_actual_outer_relative_residual"]
    ]
    raw_step_agreement = [
        value
        for trajectory in reference["trajectories"]
        for value in trajectory["candidate_vs_inner_strict"][
            "per_step_relative_field_difference"
        ]
    ]
    reference_recompute = {
        "max_actual_outer_relative_residual": max(raw_residuals),
        "max_step_relative_field_difference": max(raw_step_agreement),
        "max_trajectory_relative_field_difference": max(
            trajectory["candidate_vs_inner_strict"][
                "trajectory_relative_field_difference"
            ]
            for trajectory in reference["trajectories"]
        ),
        "route_flags_nonzero": sum(
            route["flags_nonzero"]
            for trajectory in reference["trajectories"]
            for route in trajectory["routes"]
        ),
        "route_breakdowns_or_linear_max_total": sum(
            route["breakdowns_or_linear_max_total"]
            for trajectory in reference["trajectories"]
            for route in trajectory["routes"]
        ),
        "all_pass": reference["all_pass"],
    }
    for key in (
        "max_actual_outer_relative_residual",
        "max_step_relative_field_difference",
        "max_trajectory_relative_field_difference",
    ):
        require(close(reference_recompute[key], audit["reference_health"][key]),
                f"reference {key} drift")
    require(
        reference_recompute["all_pass"]
        and reference_recompute["route_flags_nonzero"] == 0
        and reference_recompute["route_breakdowns_or_linear_max_total"] == 0,
        "reference health failed",
    )
    return {
        "status": "pass",
        "classification": "independent local recomputation from raw records",
        "source_sha256": sha256(FINAL_JSON),
        "generated_audit_sha256": sha256(FINAL_AUDIT),
        "timing_record_count": sum(
            len(records)
            for row in final["rows"]
            for records in row["records"].values()
        ),
        "burn_record_count": sum(len(row["burn_records"]) for row in final["rows"]),
        "reference": reference_recompute,
        "rows": rows,
        "supported_speedup_cells": sum(row["supported_speedup"] for row in rows),
    }


def render(final, audit, learned, independent):
    lines = [
        "# Burgers-2D hybrid extension at N=2048",
        "",
        "These are final, independently audited N=2048 results. The primary candidate is a "
        "classical residual-plus-exact-Helmholtz warm start, not a learned method or NM-ROM; "
        "its speedup is supported at tolerance 1e-6 only.",
        "",
        "## Authoritative classical panel",
        "",
        "| FOM tolerance | Linear tolerance | Cubic (ms) | Classical warm start (ms) | "
        "Speedup | Paired saving (ms) | Trajectory-cluster 95% CI (ms) | Supported |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in audit["rows"]:
        ci = row["paired_saving_trajectory_cluster_95ci_ms"]
        lines.append(
            f"| {row['fom_tau']:.0e} | {row['linear_tol']:.0e} | "
            f"{row['cubic_median_ms']:.3f} | {row['dynamic_median_ms']:.3f} | "
            f"{row['speedup_cubic_over_dynamic']:.6f}x | "
            f"{row['paired_saving_median_ms']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{'yes' if row['supported_speedup'] else 'no'} |"
        )
    reference = audit["reference_health"]
    lines.extend([
        "",
        "Only tolerance 1e-6 is a supported speedup. The 1e-8 and 1e-10 point "
        "paired-saving medians are positive while their aggregate speed ratios are below one; "
        "these noncommuting reducers disagree, and both clustered intervals cross zero, so no "
        "speedup is claimed. All timing outliers were retained.",
        "",
        "The candidate is charged for 50 full-grid exact-upwind residual evaluations and 50 "
        "exact Helmholtz inverses per trajectory. Timing is warmed compiled online latency; "
        "reference generation, compilation, loading, and first-query latency are excluded.",
        "",
        "## Reference gate",
        "",
        "| Trajectories | Maximum actual residual | Maximum step disagreement | "
        "Maximum trajectory disagreement | Flags/breakdowns | Pass |",
        "|---:|---:|---:|---:|---:|:---:|",
        f"| {reference['trajectory_count']} | "
        f"{reference['max_actual_outer_relative_residual']:.6e} | "
        f"{reference['max_step_relative_field_difference']:.6e} | "
        f"{reference['max_trajectory_relative_field_difference']:.6e} | 0 / 0 | "
        f"{'yes' if reference['all_pass'] else 'no'} |",
        "",
        "This reference uses two prospectively gated exact-Helmholtz routes on untouched seed "
        f"{final['config']['test_seed']}. The fixed public-JAX truth path is excluded after "
        "the development diagnostic reproduced its nonfinite-update freeze.",
        "",
        "## Genuine weak FiLM NM-ROM sensitivity",
        "",
        "The learned sensitivity used a separate preregistered job and seed, so its absolute "
        "times are compared only within that job, never against the primary job above.",
        "",
        "| FOM tolerance | Comparison | Control (ms) | FiLM NM-ROM (ms) | "
        "Control / FiLM | Paired saving (ms) | 95% CI (ms) | Supported | Guard accepts |",
        "|---:|:---|---:|---:|---:|---:|---:|:---:|---:|",
    ])
    for row in learned["rows"]:
        for label, key in (("cubic", "cubic_vs_film"), ("classical", "dynamic_vs_film")):
            item = row[key]
            ci = item["paired_saving_trajectory_cluster_95ci_ms"]
            lines.append(
                f"| {row['fom_tau']:.0e} | {label} vs FiLM | "
                f"{item['control_median_ms']:.3f} | {item['film_nmrom_median_ms']:.3f} | "
                f"{item['speedup_control_over_film_nmrom']:.6f}x | "
                f"{item['paired_saving_control_minus_film_median_ms']:.3f} | "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{'yes' if item['supported_film_speedup'] else 'no'} | "
                f"{item['film_guard_accepted_count_median']:.1f} / 50 |"
            )
    lines.extend([
        "",
        f"The FiLM arm is supported in {learned['film_supported_vs_cubic_cells']} of "
        f"{learned['total_cells_per_comparison']} cells against cubic and "
        f"{learned['film_supported_vs_dynamic_cells']} of "
        f"{learned['total_cells_per_comparison']} against the classical warm start. It performs "
        "100 reduced Jacobians per trajectory but its exact-residual guard accepts a median 4 "
        "of 50 steps. Conversely, cubic is supported faster than FiLM at all three tolerances; "
        "the classical arm is supported faster at 1e-6 and 1e-10, while 1e-8 is inconclusive.",
        "",
        "## Audit and exclusions",
        "",
        f"The primary bundle contains {independent['timing_record_count']} raw timing records "
        f"and {independent['burn_record_count']} immediate burn records. The generated audit "
        "and an independent recomputation of paired medians, bootstrap intervals, outliers, "
        "reference histories, work, accuracy, and record grids both pass.",
        "",
        "Excluded from claims: fixed-eight and fixed-25 public-JAX reference attempts (zero "
        "timing rows), the first unbatched diagnostic replay, the negative repaired diagnostic, "
        "and the accidental seed-20260829 N=32 implementation smoke. The authoritative primary "
        "uses seed 20260830 and job 2680178.",
    ])
    return "\n".join(lines) + "\n"


def main():
    final = load(FINAL_JSON)
    audit = load(FINAL_AUDIT)
    learned = load(LEARNED_AUDIT)
    require(learned["status"] == "complete", "learned audit incomplete")
    learned_source = os.path.join(HERE, "runs", "learned1", "out", "learned.json")
    require(sha256(learned_source) == learned["source_sha256"], "learned hash drift")
    independent = recompute(final, audit)
    classification = {
        "status": "final_audited",
        "classification": "authoritative fresh-seed classical FOM warm-start comparison; not learned and not NM-ROM",
        "job_id": final["provenance"]["slurm_job_id"],
        "execution_commit": audit["execution_commit"],
        "gpu_kind": final["provenance"]["gpu_kind"],
        "test_seed": final["config"]["test_seed"],
        "result_sha256": sha256(FINAL_JSON),
        "generated_audit_sha256": sha256(FINAL_AUDIT),
        "independent_recompute_status": independent["status"],
        "reference_health": audit["reference_health"],
        "rows": audit["rows"],
        "supported_speedup_cells": audit["supported_speedup_cells"],
        "total_cells": audit["total_cells"],
        "learned_supported_vs_cubic_cells": learned["film_supported_vs_cubic_cells"],
        "learned_supported_vs_classical_cells": learned["film_supported_vs_dynamic_cells"],
        "remote_directory_deleted_after_independent_audit": True,
    }
    with open(INDEPENDENT_OUT, "w") as handle:
        json.dump(independent, handle, indent=1, allow_nan=False)
    with open(CLASSIFICATION_OUT, "w") as handle:
        json.dump(classification, handle, indent=1, allow_nan=False)
    with open(MARKDOWN_OUT, "w") as handle:
        handle.write(render(final, audit, learned, independent))


if __name__ == "__main__":
    main()
