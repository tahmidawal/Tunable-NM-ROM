"""Generate the experiment summary directly from audited run JSONs.

Usage: /absolute/python summarize.py OUTPUT.md runs/.../a.json [runs/.../b.json ...]
"""
from __future__ import annotations

import json
import os
import sys


def f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


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
        "Times are mean case medians from the authoritative rotated block. Confidence intervals "
        "are deterministic case-resampling bootstrap intervals when present.", "",
        "| run | N | arm | construct ms | total ms | total 95% CI ms | speedup/zero | "
        "speedup 95% CI | CG iters | guess A-error | outliers | direct ms |",
        "|---|---:|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for row in data["rows"]:
            lines.append(
                f"| {label} | {row['N']} | `{row['arm']}` | {f(row.get('construction_ms'))} | "
                f"{f(row.get('hybrid_total_ms'))} | {ci(row, 'hybrid_total_bootstrap_ci95_ms')} | "
                f"{f(row.get('speedup_vs_zero_cg'))} | "
                f"{ci(row, 'speedup_vs_zero_cg_bootstrap_ci95')} | "
                f"{f(row.get('iters_hybrid_timed_mean', row.get('iters_hybrid_mean')), 1)} | "
                f"{f(row.get('guess_a_norm_ratio_mean'), 4)} | "
                f"{row.get('hybrid_timing_outlier_count', '—')} | "
                f"{f(row.get('exact_direct_ms'))} |"
            )
    lines.extend(["", "## Spectral-control paired comparisons", "",
                  "Positive delta means the candidate is slower than the named spectral control.", "",
                  "| run | N | candidate | control | delta ms | 95% CI ms | control/candidate |",
                  "|---|---:|---|---|---:|---|---:|"])
    for path, data in runs:
        label = os.path.basename(path).removesuffix(".json")
        for row in data["rows"]:
            for control in ("spectral_q8", "spectral_q16", "spectral_q24",
                            "spectral_q32", "spectral_q48", "spectral_q64"):
                pair = row.get(f"paired_delta_vs_{control}")
                if pair is None:
                    continue
                lines.append(
                    f"| {label} | {row['N']} | `{row['arm']}` | `{control}` | "
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
