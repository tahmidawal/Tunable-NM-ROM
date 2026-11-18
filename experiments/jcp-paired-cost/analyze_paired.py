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


def finite_metrics(row):
    return (row.get("status") == "ok"
            and row.get("evolved_max_error") is not None
            and np.isfinite(row["evolved_max_error"])
            and np.isfinite(np.asarray(row.get("relative_errors", [None]), dtype=float)).all())


def finite_selection(entry):
    return (entry["complete"] and entry["worst_evolved_error"] is not None
            and np.isfinite(entry["worst_evolved_error"])
            and entry["median_case_evolved_max_error"] is not None
            and np.isfinite(entry["median_case_evolved_max_error"]))


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
    subject_info = {entry["name"]: entry for entry in contract["subjects"]}
    def display_name(name):
        entry = subject_info[name]
        meta = entry["metadata"]
        if entry["family"] == "classical":
            return f'N{entry["mesh"]}\n{meta["scheme"].upper()}\n{meta["steps"]} steps'
        is_pod = entry["family"] == "pod"
        if meta["kind"] == "span":
            return f'POD\nSpan R{meta["rank"]}' if is_pod else f'Span\nR{meta["rank"]}'
        settings = meta["settings"]
        heading = f'POD\nHead M{settings["rows"]}' if is_pod else f'Head\nM{settings["rows"]}'
        return f'{heading}\n$\\Delta t={settings["dt"]:g}$'
    parity = json.loads((output / "implementation_parity.json").read_text())
    frozen = json.loads((paired / "frozen_comparators.json").read_text())["comparators"]
    invalid_metrics = [r for r in rows if r["status"] == "ok" and not finite_metrics(r)]
    affected_names = {r["subject"] for r in invalid_metrics}
    eligible_names = {name for groups in frozen.values() for choice in groups.values()
                      for name in choice["eligible"]}
    chosen_names = {choice["selected"] for groups in frozen.values()
                    for choice in groups.values() if choice["selected"] is not None}
    checks = {"all_ok_rows_same_call_metrics": all(
        row["status"] != "ok" or all(key in row for key in
            ("elapsed_ms", "relative_errors", "output_sha256", "solver_diagnostics")) for row in rows),
        "all_ok_rows_f64": all(row["status"] != "ok" or row["output_dtype"] == "float64" for row in rows),
        "expected_development_cases": cases == provenance["config"]["case_ids"]
            and set(cases) <= set(range(16)) and provenance["config"]["eval_seed"] == 202609202,
        "within_same_allocation": bool(provenance.get("job_id") and provenance.get("gpu")),
        "all_recorded_costs_nonnegative": all(row.get("elapsed_ms", 0) >= 0 for row in rows),
        "declared_gpu_f64_highest": provenance["backend"] == "gpu"
            and provenance["x64"] and provenance["precision"] == "highest",
        "all_native_targets_match_admitted_states": len(parity) == len(contract["target_names"])*len(cases)
            and all(row["passed"] for row in parity)
            and {(row["subject"], row["case"]) for row in parity}
            == {(name, case) for name in contract["target_names"] for case in cases},
        "all_selection_repetitions_retained": all(
            sum(r["block"] == "selection" and r["subject"] == name and r["case_id"] == case for r in rows)
            == contract["selection_repetitions"] for name in selection for case in cases),
        "undefined_metric_candidates_never_eligible": not bool(affected_names & eligible_names),
        "undefined_metric_candidates_never_selected": not bool(affected_names & chosen_names),
        "undefined_metric_candidates_never_confirmed": not bool(affected_names & set(result["confirmation"])),
        "all_confirmation_metrics_finite": all(r["status"] != "ok" or finite_metrics(r)
            for r in rows if r["block"] == "confirmation"),
    }
    recomputed = {}
    for name, summary in result["confirmation"].items():
        per_case = []
        for case in cases:
            rr = [r for r in rows if r["block"] == "confirmation" and r["subject"] == name and r["case_id"] == case]
            if len(rr) != contract["confirmation_repetitions"]:
                raise AssertionError("confirmation repetitions missing")
            successful = [r for r in rr if finite_metrics(r)]
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
            checks[f"{name}_median_accuracy_recomputed"] = bool(np.isclose(
                np.median([r["worst_error"] for r in per_case]),
                summary["median_case_evolved_max_error"], rtol=1e-14))
    if (run / "fixture.json").is_file():
        fixture = json.loads((run / "fixture.json").read_text())
        checks["same_allocation_gpu_fixture"] = bool(fixture["passed"]
            and fixture["job_id"] == str(provenance["job_id"])
            and fixture["source_sha256"]["ns_query.py"] == provenance["query_sha256"])
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
        "scope": "original exposed development cases; instrumented complete queries",
        "case_ids": cases,
        "cold_scope": "model activation/compilation/first query in existing JAX/CUDA runtime; OS page cache retained",
        "checks": checks, "raw_audit_passed": all(checks.values()),
        "retained_cpu_affinity_messages": affinity, "excluded_runtime_messages": excluded,
        "comparisons": result["comparisons"], "confirmation_from_invocations": recomputed,
        "failure_count_all_blocks": sum(not finite_metrics(r) for r in rows),
        "as_recorded_failure_count_all_blocks": sum(r["status"] != "ok" for r in rows),
        "undefined_metric_candidate_ids": sorted(affected_names),
        "undefined_metric_invocations": [{k:r[k] for k in
            ("block", "repetition", "order", "subject", "case_id", "elapsed_ms", "output_sha256")}
            for r in invalid_metrics],
        "undefined_metric_handling": "Raw records are unchanged. Legacy ok/null-metric invocations are classified as failures only in this audit; frozen eligibility/selection/confirmation must contain none of their candidates.",
        "timed_invocation_count": len(rows),
        "raw_sha256": hashlib.sha256((paired / "invocations.jsonl").read_bytes()).hexdigest()}
    ratios = {}
    for target, groups in result["comparisons"].items():
        ratios[target] = {}
        for group, comparison in groups.items():
            name = comparison.get("comparator")
            if name is None:
                continue
            values = []
            for case in cases:
                a = result["confirmation"][target]["cases"].get(str(case))
                b = result["confirmation"][name]["cases"].get(str(case))
                if a is not None and b is not None:
                    values.append({"case": case, "target_ms": a["median_ms"],
                                   "comparator_ms": b["median_ms"],
                                   "comparator_over_target": b["median_ms"]/a["median_ms"]})
            ratios[target][group] = {"comparator": name, "qualified": comparison["qualified"], "cases": values}
    analysis["casewise_ratios"] = ratios
    write_json(output / "analysis.json", analysis)
    colors = {"target": "#d35c24", "pod": "#6c5aa5", "classical": "#238a8d"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    ax = axes[0, 0]
    for family in ("classical", "pod", "target"):
        selected = [(name, entry) for name, entry in selection.items()
                    if entry["family"] == family and finite_selection(entry) and entry["worst_evolved_error"] <= 1]
        for name, entry in selected:
            marker = "*" if family == "target" else ("s" if family == "pod" else ("o" if entry["mesh"] == 64 else "."))
            ax.scatter(entry["median_case_ms"], 100*entry["worst_evolved_error"],
                       color=colors[family], marker=marker, s=100 if family == "target" else 38, alpha=.8)
            if family == "target":
                target_index = contract["target_names"].index(name)
                offsets = [(-45, 25), (35, 48), (55, -4), (12, 14)]
                offset = offsets[target_index % len(offsets)] if len(contract["target_names"]) > 2 else (15, 12)
                ax.annotate(display_name(name), (entry["median_case_ms"], 100*entry["worst_evolved_error"]),
                            xytext=offset, textcoords="offset points", fontsize=8,
                            arrowprops={"arrowstyle":"-", "lw":.7, "color":"#777777"})
        ax.scatter([], [], color=colors[family], marker="*" if family == "target" else "o", label=family)
    ax.set(xscale="log", yscale="log", xlabel="Complete-query median (ms)", ylabel="Worst evolved relative error (%)",
           title="Development candidate accuracy–cost observations")
    ax.legend(frameon=False)
    above = sum(finite_selection(e) and e["worst_evolved_error"] > 1 for e in selection.values())
    failed = sum(not finite_selection(e) for e in selection.values())
    ax.text(.02, .02, f"{above} candidates above 100% error; {failed} failed candidates\nAll retained in raw selection table; outside plotted range.",
            transform=ax.transAxes, fontsize=8, va="bottom")
    names = list(result["confirmation"])
    labels = [display_name(name) for name in names]
    ax = axes[0, 1]
    for index, name in enumerate(names):
        entry = result["confirmation"][name]
        xx, yy = [], []
        for case in cases:
            cc = entry["cases"].get(str(case))
            if cc:
                xx.append(case)
                yy.append(100*cc["evolved_max_error"])
        ax.plot(xx, yy, "o-", label=labels[index].replace("\n", " "))
    ax.set(yscale="log", xticks=cases, xlabel="Fixed development case", ylabel="Evolved maximum relative error (%)",
           title="Confirmation accuracy: each returned timed output")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax = axes[1, 0]
    for index, name in enumerate(names):
        values = [r["elapsed_ms"] for r in rows if r["block"] == "confirmation"
                  and r["subject"] == name and r["status"] == "ok"]
        if values:
            ax.boxplot([values], positions=[index], widths=.55, showfliers=True,
                       patch_artist=True, boxprops={"facecolor": colors[result["confirmation"][name]["family"]], "alpha": .55})
    ax.set(yscale="log", xticks=np.arange(len(names)), xticklabels=labels,
           ylabel="Complete query (ms)", title="All confirmation repetitions; speed gates kept separately")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax = axes[1, 1]
    successful = [r for r in cold if r["status"] == "ok"]
    cold_labels = [display_name(r["subject"]) for r in successful]
    for i, row in enumerate(successful):
        warm = result["confirmation"][row["subject"]]["cases"][str(row["case_id"])]["median_ms"]
        ax.plot([i, i], [warm, row["elapsed_ms"]], color="#aaaaaa", lw=1)
        ax.scatter(i, warm, color="#238a8d", label="warm" if i == 0 else None)
        ax.scatter(i, row["elapsed_ms"], color="#bd8d53", marker="s",
                   label="activation + compile + query" if i == 0 else None)
    ax.set(yscale="log", xticks=np.arange(len(successful)), xticklabels=cold_labels,
           ylabel="Elapsed time (ms)", title="Existing-runtime activation, one call per method")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"NS64 paired-cost study — {len(cases)} existing development cases", fontsize=16)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "paired-cost-pilot.png", dpi=170)
    fig.savefig(figures / "paired-cost-pilot.svg")
    ratio_rows = (len(ratios)+1)//2
    figure2, panels = plt.subplots(ratio_rows, min(len(ratios), 2), figsize=(12, 4*ratio_rows),
                                  constrained_layout=True, squeeze=False)
    for axis, (target, groups) in zip(panels.flat, ratios.items()):
        for group, comparison in groups.items():
            rr = comparison["cases"]
            axis.plot([r["case"] for r in rr], [r["comparator_over_target"] for r in rr], "o-",
                      label=group.replace("_", " ") + ("" if comparison["qualified"] else " (gate failed)"))
        axis.axhline(1, color="#555555", lw=1, ls="--")
        axis.set(xticks=cases, xlabel="Fixed development case", ylabel="Comparator time / target time",
                 title=display_name(target).replace("\n", " ") + ": above one favors target")
        axis.legend(frameon=False, fontsize=8)
    for axis in list(panels.flat)[len(ratios):]:
        axis.set_visible(False)
    figure2.suptitle("Same-allocation confirmation: individual case ratios")
    figure2.savefig(figures / "paired-case-ratios.png", dpi=170)
    figure2.savefig(figures / "paired-case-ratios.svg")
    print(json.dumps({"analysis": str(output / "analysis.json"), "raw_audit_passed": analysis["raw_audit_passed"],
                      "figure": str(figures / "paired-cost-pilot.png"), "comparisons": result["comparisons"]}))


if __name__ == "__main__":
    main()
