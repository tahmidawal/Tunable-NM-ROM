"""Generate measured pilot figures and a compact evidence file from raw JSON.

No projected curves are generated. Timing qualification is retained even when it
prevents a speed claim; the development cohort is never described as a final test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paired_measurement import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    output = run / "output"
    paired = output / "paired"
    result = json.loads((paired / "result.json").read_text())
    selection = json.loads((paired / "selection.json").read_text())
    contract = json.loads((paired / "contract.json").read_text())
    cold = json.loads((output / "cold_deployment.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    rows = [json.loads(line) for line in (paired / "invocations.jsonl").read_text().splitlines()]
    cases = contract["case_ids"]
    checks = {"all_ok_rows_same_call_metrics": all(
        row["status"] != "ok" or all(key in row for key in
            ("elapsed_ms", "relative_errors", "output_sha256", "solver_diagnostics")) for row in rows),
        "all_ok_rows_f64": all(row["status"] != "ok" or row["output_dtype"] == "float64" for row in rows),
        "expected_development_cases": cases == [0, 1],
        "within_same_allocation": bool(provenance.get("job_id") and provenance.get("gpu")),
        "all_recorded_costs_nonnegative": all(row.get("elapsed_ms", 0) >= 0 for row in rows),
    }
    recomputed = {}
    for name, summary in result["confirmation"].items():
        per_case = []
        for case in cases:
            rr = [r for r in rows if r["block"] == "confirmation" and r["subject"] == name and r["case_id"] == case]
            if len(rr) != contract["confirmation_repetitions"]:
                raise AssertionError("confirmation repetitions missing")
            successful = [r for r in rr if r["status"] == "ok"]
            if successful:
                per_case.append({"case": case,
                    "median_ms": float(np.median([r["elapsed_ms"] for r in successful])),
                    "worst_error": max(r["evolved_max_error"] for r in successful),
                    "failures": len(rr)-len(successful)})
        recomputed[name] = per_case
        if len(per_case) == len(cases):
            checks[f"{name}_median_cost_recomputed"] = bool(np.isclose(
                np.median([r["median_ms"] for r in per_case]), summary["median_case_ms"], rtol=1e-14))
            checks[f"{name}_worst_accuracy_recomputed"] = bool(np.isclose(
                max(r["worst_error"] for r in per_case), summary["worst_evolved_error"], rtol=1e-14))
    accounting = list(csv.DictReader((run / "sacct.txt").read_text().splitlines(), delimiter="|"))
    allocation = next(r for r in accounting if r["JobID"] == str(provenance["job_id"]))
    gpu_seconds = int(allocation["ElapsedRaw"])
    text = (run / "job.log").read_text()
    excluded = [line for line in text.splitlines() if any(x in line.lower() for x in
                ("out of memory", "resource_exhausted", "no space left", "captured constant", "large constant"))]
    affinity = [line for line in text.splitlines() if "hwloc_set_cpubind" in line]
    checks["scheduler_completed"] = allocation["State"] == "COMPLETED" and allocation["ExitCode"] == "0:0"
    checks["gpu_log_preflight"] = "jax_backend=gpu" in text
    checks["no_excluded_runtime_warnings"] = not excluded
    checks["full_checksum_verification"] = (run / "verification.txt").is_file() and all(
        line.endswith(": OK") for line in (run / "verification.txt").read_text().splitlines())
    analysis = {"schema": "jcp-paired-cost-analysis-v1", "job_id": provenance["job_id"],
        "source_commit": provenance["source_commit"], "gpu": provenance["gpu"],
        "gpu_seconds": gpu_seconds, "allocated_gpu_hours": gpu_seconds / 3600,
        "scope": "development cases 0 and 1; instrumented complete queries",
        "cold_scope": "model activation/compilation/first query in existing JAX/CUDA runtime; OS page cache retained",
        "checks": checks, "raw_audit_passed": all(checks.values()),
        "retained_cpu_affinity_messages": affinity, "excluded_runtime_messages": excluded,
        "comparisons": result["comparisons"], "confirmation_from_invocations": recomputed,
        "failure_count_all_blocks": sum(r["status"] != "ok" for r in rows),
        "timed_invocation_count": len(rows),
        "raw_sha256": hashlib.sha256((paired / "invocations.jsonl").read_bytes()).hexdigest()}
    write_json(output / "analysis.json", analysis)
    colors = {"target": "#d35c24", "pod": "#6c5aa5", "classical": "#238a8d"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for family in ("classical", "pod", "target"):
        selected = [(name, entry) for name, entry in selection.items()
                    if entry["family"] == family and entry["complete"]]
        for name, entry in selected:
            marker = "*" if family == "target" else ("s" if family == "pod" else ("o" if entry["mesh"] == 64 else "."))
            ax.scatter(entry["median_case_ms"], 100*entry["worst_evolved_error"],
                       color=colors[family], marker=marker, s=100 if family == "target" else 38, alpha=.8)
            if family == "target":
                ax.annotate(name.replace("learned_", ""), (entry["median_case_ms"], 100*entry["worst_evolved_error"]),
                            xytext=(5, 6), textcoords="offset points")
        ax.scatter([], [], color=colors[family], marker="*" if family == "target" else "o", label=family)
    ax.set(xscale="log", yscale="log", xlabel="Complete-query median (ms)", ylabel="Worst evolved relative error (%)",
           title="Development selection: full measured candidate set")
    ax.legend(frameon=False)
    names = list(result["confirmation"])
    labels = [name.replace("learned_", "").replace("fom_", "") for name in names]
    ax = axes[0, 1]
    for index, name in enumerate(names):
        entry = result["confirmation"][name]
        xx, yy = [], []
        for case in cases:
            cc = entry["cases"].get(str(case))
            if cc:
                xx.append(case)
                yy.append(100*cc["evolved_max_error"])
        ax.plot(xx, yy, "o-", label=labels[index])
    ax.set(yscale="log", xticks=cases, xlabel="Fixed development case", ylabel="Evolved maximum relative error (%)",
           title="Confirmation accuracy: each returned timed output")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 0]
    for index, name in enumerate(names):
        values = [r["elapsed_ms"] for r in rows if r["block"] == "confirmation"
                  and r["subject"] == name and r["status"] == "ok"]
        if values:
            ax.boxplot([values], positions=[index], widths=.55, showfliers=True,
                       patch_artist=True, boxprops={"facecolor": colors[result["confirmation"][name]["family"]], "alpha": .55})
    ax.set(yscale="log", xticks=np.arange(len(names)), xticklabels=labels,
           ylabel="Complete query (ms)", title="All confirmation repetitions; speed gates kept separately")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1, 1]
    successful = [r for r in cold if r["status"] == "ok"]
    cold_labels = [r["subject"].replace("learned_", "").replace("fom_", "") for r in successful]
    for i, row in enumerate(successful):
        warm = result["confirmation"][row["subject"]]["cases"][str(row["case_id"])]["median_ms"]
        ax.bar(i-.18, warm, width=.36, color="#238a8d", label="warm" if i == 0 else None)
        ax.bar(i+.18, row["elapsed_ms"], width=.36, color="#bd8d53", label="activation + compile + query" if i == 0 else None)
    ax.set(yscale="log", xticks=np.arange(len(successful)), xticklabels=cold_labels,
           ylabel="Elapsed time (ms)", title="Existing-runtime activation, one call per method")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("NS64 paired-cost pilot — measured development results", fontsize=16)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "paired-cost-pilot.png", dpi=170)
    fig.savefig(figures / "paired-cost-pilot.svg")
    print(json.dumps({"analysis": str(output / "analysis.json"), "raw_audit_passed": analysis["raw_audit_passed"],
                      "figure": str(figures / "paired-cost-pilot.png"), "comparisons": result["comparisons"]}))


if __name__ == "__main__":
    main()
