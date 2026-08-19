#!/usr/bin/env python3
"""Generate the audited Burgers confirm2 summary directly from its run JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_NS = [32, 64, 128, 256, 512, 1024]
EXPECTED_TAUS = [1e-6, 1e-8, 1e-10]
EXPECTED_ARMS = ["linear", "cubic", "film_nmrom"]


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def fmt_ci(interval: list[float]) -> str:
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def fmt_work(value: float) -> str:
    return f"{value:.1f}" if not float(value).is_integer() else f"{int(value)}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(data: dict) -> dict[tuple[int, float, str], dict]:
    if data.get("complete") is not True:
        raise ValueError("confirmation is not marked complete")
    config = data["config"]
    expected_config = {
        "test_seed": 20260819,
        "canonical_draw_count": 16,
        "selected_test_indices": [0, 1, 2, 3],
        "time_reps": 7,
        "f64": True,
        "preconditioner": "helmholtz",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"unexpected {key}: {config.get(key)!r}")
    provenance = data["provenance"]
    if provenance.get("jax_backend") != "gpu":
        raise ValueError("confirmation did not use the GPU backend")
    if provenance.get("x64") is not True:
        raise ValueError("confirmation did not enable x64")
    if provenance.get("matmul_precision") != "highest":
        raise ValueError("confirmation did not use highest matmul precision")

    indexed = {
        (int(row["N"]), float(row["fom_tau"]), row["arm"]): row
        for row in data["rows"]
    }
    expected = {
        (n, tau, arm)
        for n in EXPECTED_NS
        for tau in EXPECTED_TAUS
        for arm in EXPECTED_ARMS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"row grid mismatch; missing={missing}, extra={extra}")

    for key, row in indexed.items():
        records = row["timed_records"]
        if len(records) != 28 or row["timing_shape"] != [4, 7]:
            raise ValueError(f"bad timing repetitions for {key}")
        if sorted(int(i) for i in row["timing_case_medians_s"]) != [0, 1, 2, 3]:
            raise ValueError(f"bad case medians for {key}")
        for record in records:
            if (
                not record["finite"]
                or record["flags_nonzero"] != 0
                or record["breakdowns"] != 0
            ):
                raise ValueError(f"solver-health failure for {key}")
        if row["max_timed_outer_residual"] > 1.001 * row["fom_tau"]:
            raise ValueError(f"outer residual exceeds tolerance for {key}")

    equivalence = [
        condition
        for mesh in data["equivalence"].values()
        for condition in mesh.values()
    ]
    if len(equivalence) != len(EXPECTED_NS) * len(EXPECTED_TAUS):
        raise ValueError("incomplete counting/JAX equivalence grid")
    for item in equivalence:
        if item["linear_solver"]["ours_flag"] != 0 or any(
            trajectory["flags_nonzero"] != 0
            or trajectory["breakdowns"] != 0
            for trajectory in item["per_trajectory"]
        ):
            raise ValueError("counting/JAX equivalence solve has flags")
    return indexed


def build_summary(data: dict, source: Path) -> dict:
    indexed = validate(data)
    comparisons = []
    for n in EXPECTED_NS:
        for tau in EXPECTED_TAUS:
            linear = indexed[n, tau, "linear"]
            cubic = indexed[n, tau, "cubic"]
            film = indexed[n, tau, "film_nmrom"]
            comparisons.append(
                {
                    "N": n,
                    "fom_tau": tau,
                    "linear_tol": linear["linear_tol"],
                    "linear": {
                        "time_ms": linear["timing_median_ms"],
                        "newton": linear["newton_total_median"],
                        "bicgstab": linear["linear_total_median"],
                    },
                    "cubic": {
                        "time_ms": cubic["timing_median_ms"],
                        "newton": cubic["newton_total_median"],
                        "bicgstab": cubic["linear_total_median"],
                        "saving_vs_linear_ms": cubic[
                            "paired_saving_vs_linear_median_ms"
                        ],
                        "saving_vs_linear_95ci_ms": cubic[
                            "paired_saving_vs_linear_median_95ci_ms"
                        ],
                    },
                    "guarded_film_nmrom": {
                        "time_ms": film["timing_median_ms"],
                        "newton": film["newton_total_median"],
                        "bicgstab": film["linear_total_median"],
                        "saving_vs_linear_ms": film[
                            "paired_saving_vs_linear_median_ms"
                        ],
                        "saving_vs_linear_95ci_ms": film[
                            "paired_saving_vs_linear_median_95ci_ms"
                        ],
                        "saving_vs_cubic_ms": film[
                            "paired_saving_vs_cubic_median_ms"
                        ],
                        "saving_vs_cubic_95ci_ms": film[
                            "paired_saving_vs_cubic_median_95ci_ms"
                        ],
                        "guard_accepted_count_median": film[
                            "film_guard_accepted_count_median"
                        ],
                        "guard_accepted_fraction_median": film[
                            "film_guard_accepted_fraction_median"
                        ],
                        "reduced_jacobians_median": film[
                            "reduced_jacobians_total_median"
                        ],
                    },
                    "health": {
                        "max_outer_residual": max(
                            indexed[n, tau, arm]["max_timed_outer_residual"]
                            for arm in EXPECTED_ARMS
                        ),
                        "max_trajectory_rel_l2": max(
                            indexed[n, tau, arm]["max_timed_trajectory_rel_l2"]
                            for arm in EXPECTED_ARMS
                        ),
                        "case_aware_tukey_outliers": sum(
                            indexed[n, tau, arm]["timing_outlier_count_tukey"]
                            for arm in EXPECTED_ARMS
                        ),
                    },
                }
            )

    all_records = [
        record
        for row in data["rows"]
        for record in row["timed_records"]
    ]
    equivalence = [
        condition
        for mesh in data["equivalence"].values()
        for condition in mesh.values()
    ]
    cubic_clear = sum(
        item["cubic"]["saving_vs_linear_95ci_ms"][0] > 0
        for item in comparisons
    )
    film_loses_cubic = sum(
        item["guarded_film_nmrom"]["saving_vs_cubic_95ci_ms"][1] < 0
        for item in comparisons
    )
    film_beats_linear = sum(
        item["guarded_film_nmrom"]["saving_vs_linear_95ci_ms"][0] > 0
        for item in comparisons
    )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "classification": "untouched-seed final confirmation",
        "provenance": data["provenance"],
        "cohort": {
            "seed": data["config"]["test_seed"],
            "canonical_draw_count": data["config"]["canonical_draw_count"],
            "selected_indices": data["config"]["selected_test_indices"],
            "repetitions_per_trajectory": data["config"]["time_reps"],
        },
        "method": {
            "dt": data["config"]["checkpoint_config"]["dt"],
            "steps": data["config"]["checkpoint_config"]["num_steps"],
            "preconditioner": data["config"]["preconditioner"],
            "outer_tolerances": data["config"]["fom_taus"],
            "inner_tolerances": data["config"]["linear_tols"],
            "timing_estimator": data["config"]["timing_estimator"],
            "confidence_interval": data["config"]["confidence_interval"],
            "film_contract": (
                "two-Jacobian weak FiLM NM-ROM candidate; exact full-FOM residual "
                "guard against live cubic at every step; all candidate, guard, "
                "fallback, and FOM costs charged in the same invocation"
            ),
        },
        "comparisons": comparisons,
        "audit": {
            "row_count": len(data["rows"]),
            "timed_record_count": len(all_records),
            "all_finite": all(record["finite"] for record in all_records),
            "flags_nonzero_total": sum(
                record["flags_nonzero"] for record in all_records
            ),
            "breakdowns_total": sum(record["breakdowns"] for record in all_records),
            "max_outer_residual_over_tau": max(
                row["max_timed_outer_residual"] / row["fom_tau"]
                for row in data["rows"]
            ),
            "max_reference_residual": max(
                row["reference_max_residual"] for row in data["rows"]
            ),
            "case_aware_tukey_outliers_total": sum(
                row["timing_outlier_count_tukey"] for row in data["rows"]
            ),
            "equivalence_condition_count": len(equivalence),
            "max_counting_vs_testbed_step_rel_diff": max(
                item["max_step_rel_difference"] for item in equivalence
            ),
            "max_counting_vs_testbed_trajectory_rel_diff": max(
                item["max_trajectory_rel_difference"] for item in equivalence
            ),
            "max_relative_solution_diff_vs_jax": max(
                item["linear_solver"]["relative_solution_difference_vs_jax"]
                for item in equivalence
            ),
        },
        "decision_counts": {
            "condition_count": len(comparisons),
            "cubic_clearly_faster_than_linear": cubic_clear,
            "guarded_film_clearly_slower_than_cubic": film_loses_cubic,
            "guarded_film_clearly_faster_than_linear": film_beats_linear,
        },
    }


def render_markdown(summary: dict) -> str:
    provenance = summary["provenance"]
    audit = summary["audit"]
    decisions = summary["decision_counts"]
    film_linear_conditions = decisions["guarded_film_clearly_faster_than_linear"]
    film_linear_noun = "condition" if film_linear_conditions == 1 else "conditions"
    lines = [
        "# Burgers-2D untouched-seed hybrid confirmation",
        "",
        (
            "Final same-invocation results generated from the checksummed run JSON. "
            "Positive paired savings mean the named candidate is faster. FiLM is the "
            "two-Jacobian genuine weak NM-ROM candidate guarded against the live cubic "
            "predictor; its listed time charges construction, both exact residual checks, "
            "the selected cubic fallback, and the finishing FOM."
        ),
        "",
        "## Result",
        "",
        (
            f"Cubic is clearly faster than linear in {decisions['cubic_clearly_faster_than_linear']} "
            f"of {decisions['condition_count']} mesh/tolerance conditions. The guarded FiLM "
            f"NM-ROM is clearly slower than cubic in "
            f"{decisions['guarded_film_clearly_slower_than_cubic']} of "
            f"{decisions['condition_count']} conditions, and clearly faster than linear in "
            f"{film_linear_conditions} {film_linear_noun}. Therefore "
            "the optimized cubic classical warm start is the final Burgers winner; the "
            "genuine NM-ROM remains an audited negative result."
        ),
        "",
        "## End-to-end time and solver work",
        "",
        (
            "`N/B` is the median total Newton/BiCGStab work over 50 steps. Confidence "
            "intervals cluster-bootstrap the four trajectory-level paired medians."
        ),
        "",
        "| N | FOM tau | inner tau | linear ms (N/B) | cubic ms (N/B) | cubic saving vs linear ms [95% CI] | guarded FiLM ms (N/B) | FiLM saving vs linear ms [95% CI] | FiLM saving vs cubic ms [95% CI] | accepted FiLM steps / 50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["comparisons"]:
        linear = item["linear"]
        cubic = item["cubic"]
        film = item["guarded_film_nmrom"]
        lines.append(
            "| "
            f"{item['N']} | {item['fom_tau']:.0e} | {item['linear_tol']:.0e} | "
            f"{fmt_ms(linear['time_ms'])} ({fmt_work(linear['newton'])}/{fmt_work(linear['bicgstab'])}) | "
            f"{fmt_ms(cubic['time_ms'])} ({fmt_work(cubic['newton'])}/{fmt_work(cubic['bicgstab'])}) | "
            f"{fmt_ms(cubic['saving_vs_linear_ms'])} {fmt_ci(cubic['saving_vs_linear_95ci_ms'])} | "
            f"{fmt_ms(film['time_ms'])} ({fmt_work(film['newton'])}/{fmt_work(film['bicgstab'])}) | "
            f"{fmt_ms(film['saving_vs_linear_ms'])} {fmt_ci(film['saving_vs_linear_95ci_ms'])} | "
            f"{fmt_ms(film['saving_vs_cubic_ms'])} {fmt_ci(film['saving_vs_cubic_95ci_ms'])} | "
            f"{fmt_work(film['guard_accepted_count_median'])} |"
        )

    lines += [
        "",
        "## Same-invocation accuracy and health",
        "",
        "| N | FOM tau | max returned outer residual | max trajectory relative L2 | case-aware Tukey outliers |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["comparisons"]:
        health = item["health"]
        lines.append(
            f"| {item['N']} | {item['fom_tau']:.0e} | "
            f"{health['max_outer_residual']:.6e} | "
            f"{health['max_trajectory_rel_l2']:.6e} | "
            f"{health['case_aware_tukey_outliers']} |"
        )

    lines += [
        "",
        "## Audit",
        "",
        f"- Source SHA-256: `{summary['source_sha256']}`",
        f"- Staged commit: `{provenance['commit']}`",
        f"- Slurm job: `{provenance['slurm_job_id']}` on `{provenance['gpu_kind']}`",
        f"- Backend/precision: `{provenance['jax_backend']}`, x64=`{provenance['x64']}`, matmul=`{provenance['matmul_precision']}`",
        f"- Timed records: {audit['timed_record_count']} across {audit['row_count']} rows; flags={audit['flags_nonzero_total']}, breakdowns={audit['breakdowns_total']}",
        f"- Maximum outer-residual/tolerance ratio: {audit['max_outer_residual_over_tau']:.9f}",
        f"- Maximum tight-reference residual: {audit['max_reference_residual']:.6e}",
        f"- Case-aware Tukey outliers retained and reported: {audit['case_aware_tukey_outliers_total']}",
        f"- Counting/JAX equivalence conditions: {audit['equivalence_condition_count']}; maximum relative solution difference vs JAX: {audit['max_relative_solution_diff_vs_jax']:.6e}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    summary = build_summary(data, args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "final_table.md").write_text(render_markdown(summary))


if __name__ == "__main__":
    main()
