"""Summarize retained campaign error arrays, weighting each physical case once.

CPU-only analysis of previously audited JSON; no model execution or field audit.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/2026-09-11-accuracy-improvements-and-wave-speed.json"
OUTPUT = ROOT / "reports/2026-09-11-accuracy-aggregates"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records):
    values = np.array([r["errors"] for r in records], dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    peaks, final = values.max(axis=1), values[:, -1]
    q1, q3 = np.quantile(peaks, [0.25, 0.75])
    return dict(
        cases=len(records), observations_per_case=values.shape[1],
        mean_case_max_percent=float(peaks.mean() * 100),
        median_case_max_percent=float(np.median(peaks) * 100),
        worst_percent=float(peaks.max() * 100),
        mean_final_percent=float(final.mean() * 100),
        median_final_percent=float(np.median(final) * 100),
        worst_final_percent=float(final.max() * 100),
        mean_case_and_observation_percent=float(values.mean() * 100),
        cases_above_5percent=int((peaks > 0.05).sum()),
        upper_tukey_outliers=int((peaks > q3 + 1.5 * (q3 - q1)).sum()),
        upper_tukey_threshold_percent=float((q3 + 1.5 * (q3 - q1)) * 100),
    )


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def main():
    campaign = json.loads(CAMPAIGN.read_text())
    sources = {str(CAMPAIGN.relative_to(ROOT)): digest(CAMPAIGN)}
    specifications = [
        ("Poisson", "poisson_correction", "result.json", "rows", "method", "r128_q32"),
        ("Heat", "heat", "archive/outputs/results.json", "rows", "method", "nmrom_initial_tail"),
        ("Burgers", "burgers_coverage", "archive/out/result.json", "invocations", "name", "frozen_stationary"),
        ("Reflective waves", "wave_confirmation", "cluster/out/pilot/result.json", "invocations", "method", "trained_nested40"),
    ]
    cases, repetition_checks = [], []
    expected = {}
    for pde, key, suffix, row_key, method_key, method in specifications:
        path = ROOT / campaign[key]["run"] / suffix
        source = str(path.relative_to(ROOT))
        sources[source] = digest(path)
        assert sources[source] == campaign["sources"][source], source
        raw = json.loads(path.read_text())
        assert raw["complete"]
        groups = defaultdict(list)
        for row in raw[row_key]:
            if row[method_key] == method:
                groups[row["intervals"], row["case"]].append(row)
        for (intervals, case), rows in sorted(groups.items()):
            cohort = rows[0].get("cohort", rows[0].get("group"))
            if pde == "Heat":
                assert len(rows) == 1
                repetitions = sorted(rows[0]["repetitions"], key=lambda r: r["repetition"])
                times = raw["config"]["times"]
                metrics = {"current_field": [r["vs_physical"]["relative_current"] for r in repetitions]}
                primary = "current_field"
            elif pde == "Poisson":
                repetitions = sorted(rows, key=lambda r: r["repetition"])
                times = None
                metrics = {"static_field": [[r["physical_error"]] for r in repetitions]}
                primary = "static_field"
            elif pde == "Burgers":
                repetitions = sorted(rows, key=lambda r: r["rep"])
                times = raw["output_times"]
                metrics = {
                    "initial_field": [r["error"]["fixed_initial_per_time"] for r in repetitions],
                    "current_field": [r["error"]["current_relative_per_time"] for r in repetitions],
                }
                primary = "initial_field"
            else:
                repetitions = sorted(rows, key=lambda r: r["repetition"])
                assert {r["setting"] for r in repetitions} == {raw["config"]["selection_frozen"]["dt"]}
                count = len(repetitions[0]["same_grid_discrepancy"]["energy_state"]["initial_normalized"])
                times = (np.arange(count) * raw["config"]["observation_dt"]).tolist()
                assert np.isclose(times[-1], raw["config"]["end_time"])
                metrics = {
                    "initial_energy_state": [r["same_grid_discrepancy"]["energy_state"]["initial_normalized"] for r in repetitions],
                    "initial_displacement": [r["same_grid_discrepancy"]["displacement"]["initial_normalized"] for r in repetitions],
                    "current_displacement": [r["same_grid_discrepancy"]["displacement"]["current_relative"] for r in repetitions],
                    "initial_energy_scaled_velocity": [r["same_grid_discrepancy"]["velocity"]["initial_normalized"] for r in repetitions],
                }
                primary = "initial_energy_state"
            rep_ids = [r.get("repetition", r.get("rep")) for r in repetitions]
            assert rep_ids == list(range(len(repetitions)))
            for metric, arrays in metrics.items():
                arrays = np.asarray(arrays, dtype=np.float64)
                # Do not hide run-to-run numerical changes by averaging repetitions.
                assert np.array_equal(arrays, np.broadcast_to(arrays[0], arrays.shape))
                assert times is None or arrays.shape[1] == len(times)
                repetition_checks.append(dict(pde=pde, intervals=intervals, case=case,
                                              metric=metric, repetitions=len(repetitions),
                                              maximum_error_array_difference=0.0))
                cases.append(dict(pde=pde, method=method, intervals=intervals, case=case,
                                  cohort=cohort, metric=metric, primary=metric == primary,
                                  times=times, errors=arrays[0].tolist(), source=source))
        if pde == "Poisson":
            panel = next(g for g in campaign[key]["groups"] if g["group"] == "all")
            for mesh in panel["meshes"]:
                expected[pde, mesh["intervals"]] = (panel["cases"], 100 * mesh["methods"][method]["worst_relative_error"])
        elif pde == "Heat":
            for row in campaign[key]["rows"]:
                if row["scope"] == "all_development" and row["method"] == method:
                    expected[pde, row["intervals"]] = (row["cases"], 100 * row["physical_error_worst"])
        elif pde == "Burgers":
            for row in campaign[key]["rows"]:
                if row["cohort"] == "all" and row["name"] == method:
                    expected[pde, row["intervals"]] = (row["cases"], 100 * row["worst_fixed_initial"])
        else:
            for row in campaign[key]["combined_panels"]:
                if row["method"] == method:
                    expected[pde, row["intervals"]] = (row["cases"], row["worst_initial_errors_percent"]["energy_state"])

    groups = defaultdict(list)
    for case in cases:
        groups[case["pde"], case["intervals"], case["metric"]].append(case)
    summaries = []
    for (pde, intervals, metric), records in groups.items():
        summary = dict(pde=pde, intervals=intervals, metric=metric,
                       method=records[0]["method"], primary=records[0]["primary"], **summarize(records))
        if summary["primary"]:
            count, worst = expected[pde, intervals]
            assert summary["cases"] == count
            np.testing.assert_allclose(summary["worst_percent"], worst, rtol=1e-13)
        summaries.append(summary)
    primary = [s for s in summaries if s["primary"]]
    assert len(primary) == len(expected)
    result = dict(status="Aggregation of accepted development results; no new solver run or field audit",
                  case_weighting="Each distinct physical case once; error arrays identical across every timing repetition",
                  summaries=summaries, cases=cases, repetition_checks=repetition_checks,
                  sources=sources, generator_sha256=digest(Path(__file__)))
    OUTPUT.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    lines = ["# Aggregate errors of the retained accuracy-campaign models", "",
             "Generated from the finalized September campaign's previously audited error arrays. These are development-cohort aggregates, not final publication validation or a new independent field audit.", "",
             "## Meaning of the aggregates", "",
             r"For case $i$ and requested output time $t_j$, let $e_{ij}$ be the recorded normalized physical error. The mean of case maxima is $n^{-1}\sum_i\max_j e_{ij}$; the mean final error is $n^{-1}\sum_i e_{iJ}$; the mean over cases and observations is $(nJ)^{-1}\sum_{i,j}e_{ij}$, where $J$ counts all saved observations including the initial state. The last statistic gives observations equal weight; it is not a continuous-time integral. Poisson has one static field per case.", "",
             "Each physical case receives equal weight. All timing repetitions have identical error arrays; only one copy enters these statistics. Source hashes, case counts and primary worst errors agree with the accepted campaign report. No cases or outliers are removed.", "",
             "The primary normalizations are static reference-field L2 for Poisson, current reference-field L2 for heat, initial reference-field L2 for Burgers, and initial energy-state scale for reflective waves. Their percentages are not interchangeable. Heat final time is recorded in its case arrays, as are the distinct Burgers and wave horizons.", "",
             "## Primary trajectory maxima, aggregated across cases", ""]
    lines += markdown_table(["PDE", "Intervals", "Cases", "Mean %", "Median %", "Worst %", "Cases above 5%", "Upper outliers"],
        [[s["pde"], s["intervals"], s["cases"], *[f"{s[k]:.6f}" for k in ("mean_case_max_percent", "median_case_max_percent", "worst_percent")],
          s["cases_above_5percent"], s["upper_tukey_outliers"]] for s in primary])
    lines += ["Cases above the displayed error threshold are a descriptive count, not a replacement for the campaign's full physical, reference and numerical acceptance gates. Upper outliers exceed the third quartile plus one-and-a-half interquartile ranges, using NumPy's linear quantiles; they remain included. Passing on average does not establish a worst-case pass.", "",
              "## Primary final-state and observation-averaged errors", ""]
    lines += markdown_table(["PDE", "Intervals", "Mean final %", "Median final %", "Worst final %", "Mean over cases and observations %"],
        [[s["pde"], s["intervals"], *[f"{s[k]:.6f}" for k in ("mean_final_percent", "median_final_percent", "worst_final_percent", "mean_case_and_observation_percent")]] for s in primary])
    lines += ["## Other normalizations and wave components", "",
              "Burgers current-field error uses the instantaneous reference norm, which differs from its primary initial-field normalization. Wave displacement is reported both relative to initial displacement and to current displacement; velocity uses the initial energy scale. Energy-state error includes displacement gradients as well as velocity and is not energy drift.", ""]
    lines += markdown_table(["PDE", "Intervals", "Metric", "Mean case maximum %", "Median case maximum %", "Worst %", "Mean final %"],
        [[s["pde"], s["intervals"], s["metric"], *[f"{s[k]:.6f}" for k in ("mean_case_max_percent", "median_case_max_percent", "worst_percent", "mean_final_percent")]] for s in summaries if not s["primary"]])
    lines += ["## Interpretation and provenance", "",
              "The older CP paper primarily used case aggregates and final heat states. Current heat final-state current-relative means are the closer statistical comparison, but benchmark families, horizons, meshes, training capacity and reference protocols differ. These tables alone cannot attribute a difference to the decoder architecture. The sealed final cohorts remain unopened, and no acceptance decision changes.", "",
              "[Accepted campaign](2026-09-11-accuracy-improvements-and-wave-speed.md) · [Per-case arrays, aggregates and source hashes](2026-09-11-accuracy-aggregates.json) · [Generator](generate_accuracy_aggregates.py)", "",
              "Regenerate with:", "", "```bash",
              "OPENBLAS_NUM_THREADS=1 /home/tahmid/Dev/.venv/bin/python reports/generate_accuracy_aggregates.py", "```", "",
              "## Plain-language glossary", "",
              "- **PDE / intervals / cases:** equation family / mesh subdivisions per axis / distinct physical inputs; timing repeats are not additional cases.",
              "- **Retained model / development cohort / sealed final cohort:** selected method / cases used during diagnosis and selection / untouched cases reserved for later final validation.",
              "- **Relative L2 / normalization:** field-error magnitude divided by the specified reference magnitude / choice of that divisor.",
              "- **Case maximum / mean / median / worst:** largest error over one case's saved times / arithmetic average across cases / middle case value / largest value across cases and saved times.",
              "- **Final / observation:** last requested output time / one saved output time, including the initial state.",
              "- **Mean over cases and observations:** equal-weight average of saved error scalars over both axes; different from an aggregate space-time norm.",
              "- **Upper outlier / quartile / interquartile range:** case beyond the stated upper statistical threshold / quarter-position of the sorted values / difference between upper and lower quartiles. Outliers are retained.",
              "- **Cases above 5% / acceptance gate:** count with a primary case maximum exceeding that physical error threshold / the fuller accuracy, reference and solver-validity requirements.",
              "- **Initial field / current field / static field:** reference at the initial time / reference at the evaluated time / stationary reference solution.",
              "- **Displacement / velocity / energy-state / energy drift:** wave field / its time derivative / combined displacement-gradient and velocity measure / change in conserved energy, which is a different diagnostic.",
              "- **CP / decoder / checkpoint:** canonical-polyadic tensor representation / map from reduced coordinates to a field / saved trained parameters.",
              "- **Repetition / SHA256 / source hash:** repeated timing invocation of the same case / content fingerprint / fingerprint linking the analysis to its accepted input file.", ""]
    OUTPUT.with_suffix(".md").write_text("\n".join(lines))
    print("Verified and wrote", OUTPUT.with_suffix(".md").relative_to(ROOT))
    for s in primary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
