"""Generate the experiment summary directly from audited run JSONs.

Usage: /absolute/python summarize.py OUTPUT.md runs/.../a.json [runs/.../b.json ...]
"""
from __future__ import annotations

import json
import os
import sys


def f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def sci(value, digits=2):
    return "—" if value is None else f"{value:.{digits}e}"


def ci(row, key):
    values = row.get(key)
    return "—" if not values else f"[{values[0]:.3f}, {values[1]:.3f}]"


def main():
    output, *inputs = sys.argv[1:]
    if not inputs:
        raise SystemExit("provide at least one run JSON")
    runs = []
    for path in inputs:
        with open(path) as fh:
            data = json.load(fh)
        if not data.get("complete"):
            raise SystemExit(f"incomplete run: {path}")
        runs.append((path, data))

    lines = [
        "# Poisson hybrid results (generated)", "",
        "This file is generated from the listed run JSONs; do not edit its numeric tables.", "",
        "## Provenance", "",
        "| run | job | GPU | commit | source hash |", "|---|---:|---|---|---|",
    ]
    for path, data in runs:
        p = data["provenance"]
        source = p.get("source_sha256", {}).get("feasibility.py", "—")
        lines.append(
            f"| {os.path.basename(path)} | {p.get('slurm_job_id', '—')} | "
            f"{p.get('gpu_kind', '—')} | {p.get('commit_short', '—')} | {source} |"
        )
    lines.extend([
        "", "## End-to-end rows", "",
        "Multi-arm rows use the stored mean of case medians and are mechanism evidence only for "
        "small learned-versus-zero differences because that rotation was not position-balanced. "
        "Balanced AB/BA rows use the median across case medians and are authoritative for the "
        "optimized K8-versus-zero comparison. Confidence intervals resample whole cases.", "",
        "| run | design | N | tolerance | arm | construct ms | total ms | total 95% CI ms | "
        "speedup/zero | speedup 95% CI | CG iters | guess A-error | final residual | "
        "meets tau | outliers | direct ms |",
        "|---|---|---:|---:|---|---:|---:|---|---:|---|---:|---:|---:|---|---:|---:|",
    ])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for row in data["rows"]:
            design = ("balanced AB/BA" if row.get("balanced_pair_authoritative")
                      else "multi-arm rotation")
            lines.append(
                f"| {label} | {design} | {row['N']} | {row['fom_tau']:.0e} | "
                f"`{row['arm']}` | "
                f"{f(row.get('construction_ms'))} | "
                f"{f(row.get('hybrid_total_ms'))} | {ci(row, 'hybrid_total_bootstrap_ci95_ms')} | "
                f"{f(row.get('speedup_vs_zero_cg'))} | "
                f"{ci(row, 'speedup_vs_zero_cg_bootstrap_ci95')} | "
                f"{f(row.get('iters_hybrid_timed_mean', row.get('iters_hybrid_mean')), 1)} | "
                f"{sci(row.get('guess_a_norm_ratio_mean'))} | "
                f"{sci(row.get('final_true_rel_residual_max'))} | "
                f"{'yes' if row.get('final_true_rel_residual_max', float('inf')) <= row['fom_tau'] else 'no'} | "
                f"{row.get('hybrid_timing_outlier_count', '—')} | "
                f"{f(row.get('exact_direct_ms'))} |"
            )
    lines.extend([
        "", "## Classical and native baselines", "",
        "These are from the same authoritative rotated blocks as the end-to-end rows.", "",
        "| run | N | tolerance | baseline | total ms | total 95% CI ms | iterations | "
        "outliers | max residual | meets tau | mean relative L2 |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---|---:|",
    ])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for mesh in data.get("mesh_checks", []):
            for tolerance, block in mesh.get("joint_timing", {}).items():
                means = block["mean_of_case_medians_ms"]
                summaries = block.get("timing_summaries", {})
                telemetry = block.get("timed_telemetry", {})
                names = [
                    name for name in block.get("order_base", [])
                    if name in ("zero_cg", "fft_dst_direct", "dense_dst_direct")
                    or name.startswith("native_")
                ]
                for name in names:
                    summary = summaries.get(name, {})
                    ci_s = summary.get("bootstrap_ci95_s")
                    ci_ms = None if ci_s is None else [1000.0 * value for value in ci_s]
                    grades = [grade for case in telemetry.get(name, []) for grade in case]
                    residual = max(
                        (grade["recomputed_true_rel_residual"] for grade in grades),
                        default=None,
                    )
                    rel_l2 = (
                        sum(grade["rel_l2_vs_exact_dst"] for grade in grades) / len(grades)
                        if grades else None
                    )
                    iterations = (
                        sum(grade["iterations"] for grade in grades) / len(grades)
                        if grades and "iterations" in grades[0] else None
                    )
                    lines.append(
                        f"| {label} | {mesh['N']} | {float(tolerance):.0e} | `{name}` | "
                        f"{f(means.get(name))} | "
                        f"{('—' if ci_ms is None else f'[{ci_ms[0]:.3f}, {ci_ms[1]:.3f}]')} | "
                        f"{f(iterations, 1)} | {summary.get('outlier_count', '—')} | "
                        f"{sci(residual)} | "
                        f"{'yes' if residual is not None and residual <= float(tolerance) else 'no'} | "
                        f"{sci(rel_l2)} |"
                    )
    lines.extend([
        "", "## Authoritative balanced learned-versus-zero confirmation", "",
        "Each adjacent pair is burn, learned-first/zero-second, reburn, "
        "zero-first/learned-second. Positive paired delta means the learned hybrid is slower. "
        "No timing outlier is removed.", "",
        "| run | N | tolerance | learned ms | zero ms | learned-zero ms | delta 95% CI ms | "
        "speedup | speedup 95% CI | case signs L/Z/T | repetition signs L/Z/T | "
        "learned/zero outliers | learned/zero iterations |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|---|---|---|",
    ])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for row in data["rows"]:
            if not row.get("balanced_pair_authoritative"):
                continue
            cs = row["paired_case_sign_counts"]
            rs = row["paired_repetition_sign_counts"]
            lines.append(
                f"| {label} | {row['N']} | {row['fom_tau']:.0e} | "
                f"{f(row['hybrid_total_ms'])} | {f(row['baseline_total_ms'])} | "
                f"{f(row['paired_delta_arm_minus_zero_ms'])} | "
                f"{ci(row, 'paired_delta_bootstrap_ci95_ms')} | "
                f"{f(row['speedup_vs_zero_cg'])} | "
                f"{ci(row, 'speedup_vs_zero_cg_bootstrap_ci95')} | "
                f"{cs['arm_faster']}/{cs['zero_faster']}/{cs['exact_tie']} | "
                f"{rs['arm_faster']}/{rs['zero_faster']}/{rs['exact_tie']} | "
                f"{row['hybrid_timing_outlier_count']}/{row['baseline_timing_outlier_count']} | "
                f"{f(row['iters_hybrid_timed_mean'], 1)}/"
                f"{f(row['iters_baseline_timed_mean'], 1)} |"
            )
    lines.extend([
        "", "### Balanced per-case medians", "",
        "`L1/L2` and `Z1/Z2` are the learned and zero medians when each method ran first/second.",
        "", "| run | N | tolerance | case | learned ms | zero ms | learned-zero ms | "
        "L1/L2 ms | Z1/Z2 ms | repetition signs L/Z/T | outliers L/Z |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for row in data["rows"]:
            if not row.get("balanced_pair_authoritative"):
                continue
            for case in row["balanced_pair_cases"]:
                lines.append(
                    f"| {label} | {row['N']} | {row['fom_tau']:.0e} | {case['case']} | "
                    f"{f(1000 * case['arm_median_s'])} | "
                    f"{f(1000 * case['zero_median_s'])} | "
                    f"{f(1000 * case['paired_delta_arm_minus_zero_median_s'])} | "
                    f"{f(1000 * case['arm_first_position_median_s'])}/"
                    f"{f(1000 * case['arm_second_position_median_s'])} | "
                    f"{f(1000 * case['zero_first_position_median_s'])}/"
                    f"{f(1000 * case['zero_second_position_median_s'])} | "
                    f"{case['arm_faster_repetition_count']}/"
                    f"{case['zero_faster_repetition_count']}/"
                    f"{case['exact_tie_repetition_count']} | "
                    f"{case['arm_outlier_count']}/{case['zero_outlier_count']} |"
                )
    lines.extend(["", "## Spectral-control paired comparisons", "",
                  "Positive delta means the candidate is slower than the named spectral control.", "",
                  "| run | N | tolerance | candidate | control | delta ms | 95% CI ms | "
                  "control/candidate |",
                  "|---|---:|---:|---|---|---:|---|---:|"])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for row in data["rows"]:
            controls = sorted(
                key.removeprefix("paired_delta_vs_")
                for key in row
                if key.startswith("paired_delta_vs_spectral_q")
            )
            for control in controls:
                pair = row[f"paired_delta_vs_{control}"]
                lines.append(
                    f"| {label} | {row['N']} | {row['fom_tau']:.0e} | `{row['arm']}` | "
                    f"`{control}` | "
                    f"{f(pair.get('mean_ms'))} | "
                    f"{ci(pair, 'bootstrap_ci95_ms')} | "
                    f"{f(pair.get('speedup_spectral_over_arm'))} |"
                )
    lines.extend(["", "## Run files", ""])
    lines.extend(f"- `{path}`" for path, _ in runs)
    with open(output, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
