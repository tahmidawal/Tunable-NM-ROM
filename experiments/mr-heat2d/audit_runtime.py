"""Audit runtime fields independently of experiment numerics; generate findings."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def restrict(a, n, observation):
    assert n % observation == 0
    stride = n//observation
    return np.pad(a, ((0, 0), (1, 1), (1, 1)))[:, stride:n:stride, stride:n:stride]


def metrics(fields, truth, n):
    a, b = fields.reshape(len(fields), -1), truth.reshape(len(truth), -1)
    diff = np.linalg.norm(a-b, axis=1)
    norms = np.linalg.norm(b, axis=1)
    return dict(relative_current=diff/norms, relative_initial=diff/norms[0], absolute_l2=diff/n,
                truth_norm_over_initial=norms/norms[0], energy=np.sum(a*a, axis=1)/(2*n*n),
                state_change_from_initial=np.linalg.norm(a-a[0], axis=1)/np.linalg.norm(a[0]))


def audit(archive):
    out = archive/"outputs"
    result = json.loads((out/"results.json").read_text())
    cfg, settings = result["config"], result["settings"]
    observation = settings["observation_intervals"]
    manifest = result["source_manifest"]
    for name, expected in manifest["sha256"].items():
        if name == "checkpoint.pkl":
            spec = f"{manifest['checkpoint_commit']}:{manifest['checkpoint_git_path']}"
        else:
            spec = f"{manifest['source_commit']}:{name}"
        blob = subprocess.check_output(["git", "show", spec])
        assert hashlib.sha256(blob).hexdigest() == expected, name
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest() == expected, name
    archive_count = 0
    for entry in (archive/"ARCHIVE.sha256").read_text().splitlines():
        expected, name = entry.split("  ", 1)
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest() == expected, name
        archive_count += 1
    assert result["complete"] and result["verification"]["passed"]
    assert result["metadata"]["backend"] == "gpu" and result["metadata"]["x64"]
    assert result["metadata"]["precision"] == "highest"
    log = (archive/f"job-{result['metadata']['job_id']}.log").read_text()
    assert "jax_backend=gpu" in log and "RUNTIME PILOT COMPLETE" in log
    assert not re.search(r"captured.*constant|Traceback|out of memory|RESOURCE_EXHAUSTED|No space left|failed call to cuInit", log, re.I)
    assert result["reference_evidence"]["rigorous_relative_bound"] is None

    cache = {}
    def field(record):
        path = record["path"]
        if path not in cache:
            a = np.load(out/path)["field"]
            digest = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
            assert digest == record["sha256_array"]
            assert list(a.shape) == record["shape"] and a.dtype.str == record["dtype"]
            assert a.dtype == np.float64 and np.all(np.isfinite(a))
            cache[path] = a
        return cache[path]

    refs = {row["case"]: (row["intervals"], field(row["field"])) for row in result["reference_fields"]}
    case_fields = {(row["intervals"], row["case"]): row for row in result["case_fields"]}
    index = {(row["intervals"], row["case"], row["method"]): row for row in result["rows"]}
    groups = {}
    max_disagreement = 0.
    parity_records = []
    delta = result["reference_evidence"]["empirical_relative_delta"]
    assert 0 <= delta < 1
    for row in result["rows"]:
        n, case, method = row["intervals"], row["case"], row["method"]
        local = case_fields[n, case]
        truth, discrete = field(local["physical"]), field(local["discrete"])
        np.testing.assert_array_equal(field(local["initial"]), discrete[0])
        nf, fine_reference = refs[case]
        np.testing.assert_array_equal(truth, restrict(fine_reference, nf, n))
        common_reference = restrict(truth, n, observation)
        group = groups.setdefault((n, method), dict(intervals=n, method=method, path=row["path"],
                                                   tolerance=row["gradient_tolerance"], cases=[]))
        derived = []
        for rep in row["repetitions"]:
            number = rep["repetition"]
            a = field(rep["field"])
            assert a.shape == (len(cfg["times"]), n-1, n-1)
            common = restrict(a, n, observation)
            for key, aa, bb, nn in (("vs_same_grid", a, discrete, n), ("vs_physical_per_grid", a, truth, n),
                                     ("vs_physical_common_grid", common, common_reference, observation)):
                for metric, computed in metrics(aa, bb, nn).items():
                    saved = np.asarray(rep[key][metric])
                    disagreement = float(np.max(abs(saved-computed)))
                    max_disagreement = max(max_disagreement, disagreement)
                    np.testing.assert_allclose(saved, computed, rtol=1e-12, atol=1e-14)
            phase = rep["phases"]
            assert all(value >= 0 for value in phase.values())
            assert sum(value for key, value in phase.items() if key != "query_seconds") <= phase["query_seconds"]+1e-8
            initial_bad = step_bad = budget = unused_bad = 0
            initial_attempts = step_attempts = 0.
            parity_ok = True
            field_drift = error_drift = 0.
            if rep["solver"]:
                tol = row["gradient_tolerance"]
                initial = np.asarray(rep["solver"]["initial_fits"])
                steps = np.asarray(rep["solver"]["steps"])
                best = initial[np.argmin(initial[:, 3])]
                initial_bad = int(best[4] > tol or best[2] == 4)
                unused_bad = int(np.sum(initial[:, 4] > tol))-int(best[4] > tol)
                step_bad = int(np.sum((steps[:, 4] > tol) | (steps[:, 2] == 4)))
                budget = int(best[2] == 0)+int(np.sum(steps[:, 2] == 0))
                initial_attempts = float(sum(initial[:, 0]))
                step_attempts = float(sum(steps[:, 0]))
                control = index[n, case, "rom_modular_gtol1e-09"]["repetitions"][number]
                strict = restrict(field(control["field"]), n, observation)
                strict_error = metrics(strict, common_reference, observation)["relative_current"]
                common_error = metrics(common, common_reference, observation)["relative_current"]
                field_drift = float(max(metrics(common, strict, observation)["absolute_l2"] /
                                         (np.linalg.norm(common_reference.reshape(len(common_reference), -1), axis=1)/observation)))
                error_drift = float(max(abs(common_error-strict_error)))
                if row["path"] == "compiled":
                    modular = index[n, case, f"rom_modular_gtol{tol:g}"]["repetitions"][number]
                    b = field(modular["field"])
                    im = np.asarray(modular["solver"]["initial_fits"])
                    sm = np.asarray(modular["solver"]["steps"])
                    zm = np.asarray(modular["solver"]["latent"])
                    zc = np.asarray(rep["solver"]["latent"])
                    field_parity = float(np.linalg.norm(a-b)/np.linalg.norm(b))
                    latent_parity = float(np.linalg.norm(zc-zm)/np.linalg.norm(zm))
                    counters = bool(np.array_equal(initial[:, :3], im[:, :3]) and np.array_equal(steps[:, :3], sm[:, :3]))
                    parity_ok = counters and field_parity <= settings["field_parity_tolerance"] and latent_parity <= settings["latent_parity_tolerance"]
                    parity_records.append(dict(intervals=n, case=case, tolerance=tol, repetition=number,
                                               fields=field_parity, latents=latent_parity, counters_equal=counters, passed=parity_ok))
            err = max(metrics(common, common_reference, observation)["relative_current"])
            adjusted = (err+delta)/(1-delta)
            derived.append(dict(repetition=number, query_seconds=phase["query_seconds"],
                                max_common_current_error=float(err), empirical_adjusted_error=float(adjusted),
                                rigorous_physical_error_bound=None, selected_initial_bad=initial_bad,
                                step_bad=step_bad, budget_exhausted=budget, unused_initial_bad=unused_bad,
                                initial_attempts=initial_attempts, step_attempts=step_attempts,
                                parity_passed=parity_ok, field_drift_from_strict=field_drift,
                                metric_drift_from_strict=error_drift,
                                solver_valid=not(initial_bad or step_bad or budget)))
        group["cases"].append(dict(case=case, repetitions=derived))
    for (n, method), group in groups.items():
        errors, latencies, paired_fom, paired_strict, paired_modular = [], [], [], [], []
        group["all_valid"] = True
        group["all_parity"] = True
        if group["path"] == "compiled":
            group["all_parity"] = all(gate["passed"] for gate in result["parity_gates"]
                                      if gate["intervals"] == n and gate["gradient_tolerance"] == group["tolerance"])
        group["drift_within_predeclared_ceiling"] = True
        for case in group["cases"]:
            number = case["case"]
            reps = case["repetitions"]
            errors.append(max(rep["max_common_current_error"] for rep in reps))
            latencies.append(np.median([rep["query_seconds"] for rep in reps]))
            fom = index[n, number, "fom_dst_exact_time"]["repetitions"]
            control = index[n, number, "rom_modular_gtol1e-09"]["repetitions"]
            paired_fom.append(np.median([f["phases"]["query_seconds"]/r["query_seconds"] for f, r in zip(fom, reps)]))
            paired_strict.append(np.median([f["phases"]["query_seconds"]/r["query_seconds"] for f, r in zip(control, reps)]))
            if group["path"] == "compiled":
                modular = index[n, number, f"rom_modular_gtol{group['tolerance']:g}"]["repetitions"]
                paired_modular.append(np.median([f["phases"]["query_seconds"]/r["query_seconds"] for f, r in zip(modular, reps)]))
            group["all_valid"] &= all(rep["solver_valid"] for rep in reps)
            group["all_parity"] &= all(rep["parity_passed"] for rep in reps)
            group["drift_within_predeclared_ceiling"] &= all(max(rep["field_drift_from_strict"], rep["metric_drift_from_strict"]) <= settings["max_error_drift_from_strict_control"] for rep in reps)
        group.update(error_mean=float(np.mean(errors)), error_median=float(np.median(errors)), error_worst=max(errors),
                     cases_above_005=sum(e>.05 for e in errors), median_query_seconds=float(np.median(latencies)),
                     primary_paired_fom_ratio=float(np.median(paired_fom)), primary_paired_strict_ratio=float(np.median(paired_strict)),
                     primary_paired_modular_ratio=float(np.median(paired_modular)) if paired_modular else None,
                     empirical_eligible_targets=[], rigorous_eligible_targets=None)
        if group["all_valid"] and group["all_parity"]:
            group["empirical_eligible_targets"] = [target for target in cfg["accuracy_targets"]
                if (max(errors)+delta)/(1-delta) <= target and delta <= target/10]
    return result, dict(passed=True, source_commit=manifest["source_commit"], archive_files=archive_count,
                        unique_fields_checked=len(cache), invocations=sum(len(row["repetitions"]) for row in result["rows"]),
                        max_saved_metric_disagreement=max_disagreement, observation_intervals=observation,
                        drift_ceiling=settings["max_error_drift_from_strict_control"],
                        empirical_reference_delta=delta, rigorous_reference_bound=None,
                        parity_checks=parity_records, groups=list(groups.values()))


def report(result, review, result_path):
    fmt = lambda v: f"{v:.6g}"
    groups = review["groups"]
    lines = ["# Frozen heat checkpoint: solver tolerance and compiled-query costs", "",
             "These provisional development findings compare execution paths and solver stopping "
             "on the unchanged heat checkpoint. They measure runtime without new training; the "
             "nonlinear-head initial-state approximation gap remains separate.", "",
             f"Generated from `{result_path}` and the native field audit. Scientific source "
             f"`{review['source_commit']}`, GPU job `{result['metadata']['job_id']}`, "
             f"`{result['metadata']['gpu']}`. Frozen checkpoint SHA-256: "
             f"`{result['checkpoint_sha256']}`.", "",
             "[Standalone runtime and field-drift figure](figures/heat-runtime.svg).", "",
             "The complete query transfers a full host initial field to the GPU, fits both initial "
             "starts, evolves the weak heat equations and returns all requested full fields to "
             "the host. The original modular strict control and efficient direct sine-transform "
             "FOM run in this same job. Timed order alternates, every paired block is preceded "
             "by clock burn-in, and compilation/reference/mesh setup are outside query costs.", "",
             "## Frozen settings and evidence checks", "",
             "| Setting | Value |", "|---|---|",
             f"| Solver meshes / shared observation intervals | {result['config']['evaluation_intervals']} / {review['observation_intervals']} |",
             f"| Validation cases / repetitions / timed invocations | {result['config']['n_validation']} / {result['settings']['repetitions']} / {review['invocations']} |",
             f"| CN timestep / tolerance ladder | {result['settings']['dt']} / {result['settings']['gradient_tolerances']} |",
             f"| Verified files / unique output and reference fields | {review['archive_files']} / {review['unique_fields_checked']} |",
             f"| Maximum saved-metric disagreement | {fmt(review['max_saved_metric_disagreement'])} |",
             f"| Empirical reference-refinement estimate | {fmt(review['empirical_reference_delta'])} |",
             "| Rigorous continuum error bound | Not established |", "",
             "Every field is restricted to the same observation grid. Current-field relative, "
             "initial-field relative and absolute L2 errors are independently recomputed for "
             r"every repetition. The empirical adjusted ratio is $(e+\delta)/(1-\delta)$; "
             "this refinement estimate is not a rigorous continuum bound. All original fresh "
             "reference and heat-advancement gates pass again.", "",
             "## Paired query costs and unchanged-model accuracy", "",
             "Times are cohort medians of within-case repetition medians. Ratios are the median "
             "across cases of each case's median paired-repetition ratio. The strict-control "
             "ratio compares the original modular strict arm with the listed method. All errors "
             "use the worst requested time and repetition for each case.", "",
             "| Intervals | Method | Query ms | FOM / method | Strict control / method | Current error median / worst | Cases above 5% |",
             "|---|---|---|---|---|---|---|"]
    for group in groups:
        lines.append(f"| {group['intervals']} | {group['method']} | {fmt(1000*group['median_query_seconds'])} | {fmt(group['primary_paired_fom_ratio'])} | {fmt(group['primary_paired_strict_ratio'])} | {fmt(group['error_median'])} / {fmt(group['error_worst'])} | {group['cases_above_005']} / {len(group['cases'])} |")
    lines += ["", "## Execution parity and solver validity", "",
              "Compiling the full query is an equivalent-execution claim only when fields, "
              "latent states and exact attempt/acceptance/reason counters agree for that "
              "tolerance. Every timed repetition is checked; a failed parity arm remains "
              "visible and cannot establish that claim.", "",
              "| Intervals | Method | All solver checks pass | All parity checks pass | Maximum field drift from strict | Drift ceiling passes | Empirical eligible targets |",
              "|---|---|---|---|---|---|---|"]
    for group in groups:
        reps = [rep for case in group["cases"] for rep in case["repetitions"]]
        lines.append(f"| {group['intervals']} | {group['method']} | {group['all_valid']} | {group['all_parity']} | {fmt(max(r['field_drift_from_strict'] for r in reps))} | {group['drift_within_predeclared_ceiling']} | {group['empirical_eligible_targets']} |")
    lines += ["", "Eligibility here requires finite fields, selected initial and rollout gradients "
              "meeting the arm's declared tolerance, no budget exhaustion, empirical reference "
              "margin and compiled-path parity. It remains development evidence. The drift "
              "ceiling is checked separately on both field differences and changes in error, "
              "so nearly unchanged error cannot conceal a changed trajectory. A method can "
              "meet the broad target yet fail the stricter accuracy-preservation check.", "",
              "| Intervals | Method | Initial fit ms | Evolution or direct FOM readout ms | Compiled device query ms | ROM readout ms | Median total initial / rollout attempts |",
              "|---|---|---|---|---|---|---|"]
    for group in groups:
        native = [row for row in result["rows"] if row["intervals"] == group["intervals"] and row["method"] == group["method"]]
        phases = [rep["phases"] for row in native for rep in row["repetitions"]]
        md = lambda key: fmt(1000*np.median([p[key] for p in phases])) if key in phases[0] else "—"
        reps = [rep for case in group["cases"] for rep in case["repetitions"]]
        evolution = 'evolve_and_readout_seconds' if group['path'] == 'direct' else 'evolution_seconds'
        lines.append(f"| {group['intervals']} | {group['method']} | {md('initial_fit_seconds')} | {md(evolution)} | {md('complete_device_seconds')} | {md('readout_seconds')} | {fmt(np.median([r['initial_attempts'] for r in reps]))} / {fmt(np.median([r['step_attempts'] for r in reps]))} |")
    lines += ["", "The direct FOM combines propagation and inverse-transform field readout; its "
              "phase is retained in raw JSON. Compiled paths have one inseparable device-query "
              "phase. A dash means an unavailable separate phase. Initial "
              "attempt counts sum both starts; rollout counts sum every actual timestep.", "",
              "## Scope and next accuracy question", "",
              "No network weights, latent or bank dimensions, training cases or spatial family "
              "changed. Runtime changes therefore cannot remedy the head's initial-field "
              "approximation gap. A next bounded proposal would freeze the spatial bank and "
              "existing head architecture while comparing head/code refinement on the original "
              "training cohort against expanded training coverage. This training experiment "
              "has not been launched. Broader multi-bump heat coverage and independent final "
              "confirmation remain open; no merge occurred.", "",
              "## Plain-language glossary", "",
              "- **FOM / ROM / CN / LM:** full-grid sine solver / learned reduced solver / "
              "Crank–Nicolson timestep / damped nonlinear least-squares iteration.",
              "- **Modular / compiled / checkpoint:** separate staged query calls / one compiled "
              "composition of those calls / saved unchanged neural weights.",
              "- **Tolerance / solver validity / parity:** required normalized gradient smallness / "
              "meeting declared finite/convergence checks / agreeing fields, latents and counters.",
              "- **Intervals / observation grid:** spatial cells along an axis / common locations "
              "where all meshes' errors are evaluated.",
              "- **Query ms / paired ratio:** full host-input to host-output milliseconds / "
              "same-case, same-repetition cost ratio summarized across cases.",
              "- **Current error / initial error / absolute L2:** discrepancy divided by current "
              "reference norm / initial norm / spatially integrated discrepancy alone.",
              "- **Median / worst / cases above:** middle cohort value / largest cohort value / "
              "number of cases missing the stated error threshold.",
              "- **Strict control / field drift:** original modular tight-tolerance arm / change "
              "in its field divided by current physical-reference norm.",
              "- **Empirical adjusted ratio / rigorous bound:** observed error with a refinement "
              "estimate and denominator correction / proven upper error limit, unavailable here.",
              "- **Readout / attempt / eligible target:** reconstructing requested fields / tried "
              "LM update / development target meeting the stated empirical and solver checks.",
              "- **Source hash / provisional:** content-verification identifier / development "
              "finding awaiting independent review and confirmation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result, review = audit(args.archive)
    args.audit.write_text(json.dumps(review, indent=2)+"\n")
    args.report.write_text(report(result, review, args.archive/"outputs/results.json"))
    print(json.dumps({k: v for k, v in review.items() if k not in ("groups", "parity_checks")}, indent=2))
