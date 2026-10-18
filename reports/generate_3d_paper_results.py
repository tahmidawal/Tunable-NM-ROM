"""Generate 3D campaign tables directly from audited invocation records.

Only explicit manifest entries are included. This script never selects a best
attempt, opens a final dataset, or writes to an experimental worktree.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
INPUT = REPORTS / "2026-09-20-3d-paper-inputs.json"
STEM = "2026-09-20-3d-paper-results"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return "—" if value is None else f"{100 * value:.4f}"


def number(value):
    return "—" if value is None else f"{value:.3f}"


def normalize(row, adapter, data):
    finite = bool(row.get("finite", True))
    if adapter == "ns_trajectory":
        same, physical = row["same_grid_errors"], row.get("fine_grid_errors")
        stationary = "steps" not in row or (row["cold"][2] == 4 and all(r == 4 for r in row["steps"][2]))
        return dict(mesh=data["config"]["n"], convention="periodic points per axis",
            method=row["method"], case=row["case"], repetition=row["repetition"],
            gpu_ms=1000*row["gpu_seconds"], total_ms=1000*row["with_host_seconds"], finite=finite,
            evolved=max(same[1:]) if finite else None, all_times=max(same) if finite else None,
            initial=same[0] if finite else None,
            physical=max(physical[1:]) if finite and physical is not None else None,
            stationary=stationary)
    if adapter == "burgers":
        return dict(
            mesh=data["config"]["nodes"], convention="nodes per axis",
            method=row["method"], case=row["case"], repetition=row["repetition"],
            gpu_ms=row["gpu_ms"], total_ms=row.get("total_ms"), finite=finite,
            evolved=row["worst_evolved"] if finite else None,
            all_times=row["worst_all"] if finite else None,
            initial=row["initial_error"] if finite else None,
            physical=row.get("physical_worst_evolved"),
            stationary=bool(row.get("stationary", row.get("converged", True))),
        )
    base = dict(
        mesh=row["intervals"], convention="intervals per axis",
        method=row["method"], case=row["case"], repetition=row["repetition"],
        gpu_ms=row["device_ms"], total_ms=row["total_ms"], finite=finite,
    )
    if adapter == "heat":
        return dict(base,
            evolved=row["same_grid"]["current_evolved"] if finite else None,
            all_times=row["same_grid"]["current_all"] if finite else None,
            initial=row["same_grid"]["initial_fit"] if finite else None,
            physical=row["physical"]["current_evolved"] if finite else None,
            stationary=row["nonstationary_solves"] == 0,
        )
    if adapter == "poisson":
        return dict(base,
            evolved=row["same_grid_error"] if finite else None,
            all_times=None, initial=None,
            physical=row["physical_error"] if finite else None,
            stationary=row["stationary"],
        )
    raise ValueError(f"Unknown invocation adapter: {adapter}")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["mesh"], row["method"]].append(row)
    result = []
    for (mesh, method), group in sorted(groups.items()):
        by_case = defaultdict(list)
        for row in group:
            by_case[row["case"]].append(row)
        # Use every invocation for failures and the worst value across repeats
        # for each case's accuracy; a bad repeated solve cannot disappear.
        per_case = []
        for case, repeated in sorted(by_case.items()):
            values = {}
            for metric in ("evolved", "all_times", "initial", "physical"):
                observed = [r[metric] for r in repeated if r[metric] is not None]
                values[metric] = max(observed) if observed else None
            per_case.append(dict(case=case, **values,
                finite=all(r["finite"] for r in repeated),
                stationary=all(r["stationary"] for r in repeated)))
        timings = [r["gpu_ms"] for r in group]
        assert all(math.isfinite(t) and t > 0 for t in timings)
        median = statistics.median(timings)
        metrics = {}
        for metric in ("evolved", "all_times", "initial", "physical"):
            values = [r[metric] for r in per_case if r[metric] is not None]
            assert all(math.isfinite(v) and v >= 0 for v in values)
            metrics[metric + "_worst"] = max(values) if values else None
            metrics[metric + "_median"] = statistics.median(values) if values else None
        total = [r["total_ms"] for r in group if r["total_ms"] is not None]
        result.append(dict(mesh=mesh, method=method, cases=len(by_case), invocations=len(group),
            gpu_ms_median=median, total_ms_median=statistics.median(total) if total else None,
            timing_outliers=sum(t > 1.5 * median for t in timings),
            nonfinite_cases=sum(not r["finite"] for r in per_case),
            nonstationary_cases=sum(not r["stationary"] for r in per_case),
            per_case=per_case, gpu_ms_repetitions=timings, **metrics))
    return result


def audit_invocation_coverage(rows, data, adapter):
    """No method may silently omit a case or repeated call from its panel."""
    cfg = data["config"]
    expected_repetitions = set(range(cfg["repetitions"]))
    if adapter == "burgers":
        count = len(cfg["validation_rows"])
    elif adapter == "ns_trajectory":
        count = cfg["timed_cases"]
    else:
        count = cfg["validation_count"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["mesh"], row["method"]].append((row["case"], row["repetition"]))
    checks = []
    for (mesh, method), calls in sorted(groups.items()):
        cases = {case for case, repetition in calls}
        assert len(cases) == count, (adapter, method, len(cases), count)
        expected = {(case, repetition) for case in cases for repetition in expected_repetitions}
        assert len(set(calls)) == len(calls), (adapter, method, "duplicate invocation identity")
        assert set(calls) == expected, (adapter, method, "missing/extra repetitions")
        checks.append(dict(mesh=mesh, method=method, cases=len(cases),
                           repetitions_per_case=len(expected_repetitions), invocations=len(calls)))
    return dict(passed=True, checks=checks)


def main():
    manifest = read(INPUT)
    provenance = {str(INPUT.relative_to(ROOT)): digest(INPUT),
                  str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    output = []
    lines = ["# Three-dimensional NM-ROM comparisons: generated results", "",
        "These tables contain retained, audited development experiments from the overnight campaign. "
        "They are provisional paper material: tuning, operator comparisons and independent final evaluation "
        "must be assessed per attempt before a row supports a manuscript claim.", "",
        "The explicit input manifest controls which attempts appear; no best run is selected automatically. "
        "Timed comparisons derive errors and costs from the same invocation records; untimed snapshot fits are separated explicitly. "
        "Every table retains unsuccessful methods and reports failure counts.", ""]
    for entry in manifest["runs"]:
        path = ROOT / entry["result"]
        audit_path = ROOT / entry["audit"]
        data, audit = read(path), read(audit_path)
        assert data["complete"] and audit["passed"], entry["attempt"]
        assert data["backend"] == "gpu" and data["x64"]
        assert data.get("precision", data.get("matmul_precision")) == "highest"
        assert not data.get("smoke", False)
        for p in (path, audit_path):
            provenance[str(p.relative_to(ROOT))] = digest(p)
        for extra_audit in entry.get("additional_audits", []):
            p = ROOT / extra_audit
            assert read(p)["passed"], extra_audit
            provenance[extra_audit] = digest(p)
        for failed_audit in entry.get("retained_failed_audits", []):
            p = ROOT / failed_audit
            assert read(p)["passed"] is False, failed_audit
            provenance[failed_audit] = digest(p)
        if entry["adapter"] == "ns_representation":
            import numpy as np
            artifact = path.parent / "representation_fields.npz"
            provenance[str(artifact.relative_to(ROOT))] = digest(artifact)
            with np.load(artifact) as f:
                truth = f["truth"].reshape(len(f["truth"]), -1)
                pred = f["prediction"].reshape(truth.shape)
                norms = np.linalg.norm(truth, axis=1)
                errors = {f"Learned bank R{data['config']['rank']} projection": f["bank_error"],
                          f"Neural head K{data['config']['k']} best-found fit": np.linalg.norm(pred-truth, axis=1)/norms}
                for rank in data['config']['pod_ranks']:
                    basis = f['pod_basis'][:, :rank]
                    errors[f"POD-{rank} projection"] = np.linalg.norm(truth-(truth@basis)@basis.T, axis=1)/norms
                rows = [dict(method=name, states=len(values), error_median=float(np.median(values)),
                             error_worst=float(np.max(values)), states_above_target=int(np.count_nonzero(values > .05)))
                        for name, values in errors.items()]
            run = dict(entry, rows=rows, source=data['source_commit'], job_id=data['job_id'], gpu=data['gpu'], audit=audit)
            output.append(run)
            lines += [f"## {entry['pde']} — {entry['attempt']}", "",
                      f"**Provisional:** {entry['qualification']}", "",
                      f"Source `{run['source']}`; job `{run['job_id']}`; GPU `{run['gpu']}`. "
                      f"[Records](../{entry['result']}) and [independent audit](../{entry['audit']}).", "",
                      f"The table contains {data['config']['dev_cases']} validation trajectories on a "
                      f"{data['config']['n']}³ periodic grid, with {len(truth)//data['config']['dev_cases']} snapshots per trajectory. "
                      "Snapshots from the same trajectory are correlated. Errors here use each snapshot's own velocity norm; "
                      "they must not be confused with the campaign's initial-normalized predicted-trajectory metric.", "",
                      "| Representation | Snapshots | Median error (%) | Worst error (%) | Snapshots above declared target | Status |",
                      "| --- | ---: | ---: | ---: | ---: | --- |"]
            for row in rows:
                lines.append(f"| {row['method']} | {row['states']} | {pct(row['error_median'])} | "
                             f"{pct(row['error_worst'])} | {row['states_above_target']} | failed representation target |")
            gate = data['reference_verification']['gates']['physical_reference_budget']
            lines += ["", f"The empirical reference refinement gate passes: worst discrepancy "
                      f"{pct(gate['worst_total_discrepancy'])}% against a declared {pct(gate['budget'])}% budget. "
                      "Passing the numerical reference checks does not remedy the representation failures above. "
                      "No rollout error, runtime or operator comparison is inferred from these snapshot fits.", ""]
            continue
        if entry.get("invocations"):
            records_path = ROOT / entry["invocations"]
            provenance[entry["invocations"]] = digest(records_path)
            invocations = read(records_path)
        else:
            invocations = list(data["invocations"])
        if data.get("operator_invocations"):
            assert data["operator_complete"] and audit["operator_complete"]
            invocations.extend(data["operator_invocations"])
        normalized = [normalize(r, entry["adapter"], data) for r in invocations]
        coverage = audit_invocation_coverage(normalized, data, entry["adapter"])
        rows = aggregate(normalized)
        for row in rows:
            flags = []
            for mesh in data.get("meshes", []):
                if mesh["intervals"] != row["mesh"]:
                    continue
                metadata = mesh.get("methods", {}).get(row["method"], {})
                if metadata.get("quadrature_certified") is False:
                    flags.append("failed EQ certificate")
            if row["nonstationary_cases"]:
                flags.append("stopping failures")
            if row["nonfinite_cases"]:
                flags.append("nonfinite output")
            if data.get("reference", {}).get("passed") is False:
                flags.append("reference refinement failed")
            row["qualification"] = "; ".join(flags) if flags else "development"
        run = dict(entry, rows=rows, source=data.get("commit", data.get("source_commit")),
                   job_id=data["job_id"], gpu=data["gpu"], audit=audit, invocation_coverage=coverage)
        output.append(run)
        lines += [f"## {entry['pde']} — {entry['attempt']}", "",
                  f"**Provisional:** {entry['qualification']}", "",
                  f"Source `{run['source']}`; job `{run['job_id']}`; GPU `{run['gpu']}`. "
                  f"[Invocation data](../{entry['result']}) and [independent audit](../{entry['audit']}).", ""]
        if entry["adapter"] == "burgers":
            lines += ["Errors use the initial-field norm. The evolved error excludes the initial compression; "
                      "all-times and initial errors are also shown. GPU timing begins with the dense input "
                      "already on device and ends with every requested dense output on device. Host transfers "
                      "are unmeasured in this attempt. Mesh size counts nodes per axis.", ""]
        elif entry["adapter"] == "heat":
            lines += ["Errors use each reference field's current norm. Evolved errors exclude the initial "
                      "state. Total timing includes host transfers; GPU timing includes initialization, "
                      "evolution and dense output. Mesh size counts intervals per axis.", ""]
        elif entry["adapter"] == "ns_trajectory":
            lines += ["Errors use the initial velocity-field norm, with all three components combined. "
                      "The evolved metric excludes time zero. " +
                      ("Physical error uses Fourier interpolation to the independently refined grid. "
                       if any(r["physical_worst"] is not None for r in rows) else
                       "This timed panel has a same-grid reference only; physical errors are unmeasured. ") +
                      "Total timing includes host transfers; device "
                      "timing includes initialization, evolution and every requested dense velocity field. "
                      "Mesh size counts periodic points per axis. The timed cohort is a declared subset "
                      "of the larger validation cohort; training-validation summaries are not substituted "
                      "for its measured query errors. " +
                      ("Saved latent histories support the linked independently sampled weak-gradient checks."
                       if entry.get("latent_history_audited") else
                       "Stopping records lack saved latent histories and support an internal-consistency check only."), ""]
            capacity = data.get("capacity")
            if capacity:
                records = [("Unrestricted learned bank", capacity["development_bank_initial_normalized"], None)]
                records += [(name, record["head_initial_normalized"], record["stationary_count"])
                            for name, record in capacity["heads"].items()]
                lines += ["Untimed representation diagnostics on the full development snapshots follow. "
                          "They fit known reference states and use the initial vector-field norm, including time zero. "
                          "Their errors are separate from predicted trajectory errors. " +
                          ("Every retained head fit also has an independently reconstructed analytic gradient."
                           if entry.get("representation_gradients_audited") else
                           "Stationarity counts are recorded solver exits, not independently reconstructed gradients."), "",
                          "| Representation | Snapshots | Median error (%) | Worst error (%) | Recorded stationary fits |",
                          "| --- | ---: | ---: | ---: | ---: |"]
                for name, values, stationary in records:
                    lines.append(f"| `{name}` | {values['count']} | {pct(values['median'])} | "
                                 f"{pct(values['worst'])} | {stationary if stationary is not None else '—'} |")
                gate = data["larger_rollout_gate"]
                lines += ["", f"The predeclared new NM-ROM rollout eligibility gate was "
                          f"**{'passed' if gate['eligible'] else 'failed'}**. "
                          "The timed methods below are exactly those measured in this attempt.", ""]
        else:
            lines += ["Errors use the reference solution norm. There is one stationary output field; "
                      "evolved, initial and all-times terminology does not apply. Total timing includes "
                      "host transfers. Mesh size counts intervals per axis.", ""]
        lines += ["| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['mesh']} | `{row['method']}` | {row['cases']} | "
                f"{pct(row['evolved_median'])} | {pct(row['evolved_worst'])} | "
                f"{pct(row['all_times_worst'])} | {pct(row['initial_worst'])} | "
                f"{pct(row['physical_worst'])} | {number(row['gpu_ms_median'])} | "
                f"{number(row['total_ms_median'])} | {row['nonfinite_cases']} / "
                f"{row['nonstationary_cases']} | {row['timing_outliers']} / {row['invocations']} | {row['qualification']} |")
        lines += ["", "A missing physical-error or total-time cell means unmeasured, not zero. "
                  "Physical errors require the attempt's separate reference-refinement qualification. "
                  "A passing numerical audit verifies the recorded experiment; it does not establish "
                  "good predictive accuracy, convergence of training or a competitive method.", ""]
    lines += ["## Glossary", "",
        "- **NM-ROM / ROM:** a neural-manifold reduced model / a model solving for a smaller state.",
        "- **Bank, head, rank:** learned spatial functions, their nonlinear coefficient map, and the number of spatial functions.",
        "- **q / K / R:** correction rank / nonlinear latent dimension / full learned-bank rank; values in method names identify the saved configuration.",
        "- **POD:** a linear reduced basis computed from training snapshots.",
        "- **FOM / DST:** the full-grid numerical solver / a discrete sine-transform direct solver.",
        "- **Dense / EQ:** full-grid contractions / sampled empirical quadrature; an EQ name alone does not mean its accuracy certificate passed.",
        "- **Free bank / Galerkin / weak:** unrestricted bank coefficients / projection against the basis / residual projection against smooth tests.",
        "- **Mesh / cases / calls:** grid size per spatial axis under the stated convention / distinct inputs / timed solver invocations including repetitions.",
        "- **Error median / worst:** median or maximum over distinct cases of each case's largest evolved error, or stationary solution error for Poisson; the largest value across repeated calls is retained.",
        "- **All-times / initial:** maximum including time zero / error from compressing the initial field.",
        "- **Physical error:** discrepancy against the independently refined reference; a refinement test is empirical, not a proved continuum bound.",
        "- **GPU / total median:** median elapsed milliseconds on the device / including recorded host transfers. Timings may only be compared within a job and matching output contract.",
        "- **Nonfinite / nonstationary cases:** inputs with an invalid output in any repeat / a failed declared numerical stopping check in any repeat.",
        "- **Timing outliers:** calls slower than one and a half times that method's median; every measured time remains in the machine-readable output.",
        "- **Status / certificate:** a row's remaining qualification / the declared held-out check that sampled quadrature reproduces the required moments accurately enough.",
        "- **Representation / snapshot / projection:** the fields a model can express / one saved state at one time / the nearest field in a linear basis under the stated norm. A best-found neural fit uses numerical optimization and is not a proof of global optimality.",
        "- **Snapshots above declared target:** saved states exceeding the representation accuracy threshold fixed in that experiment's design; these are not independent trajectory counts.",
        "- **Provisional / development / final cohort:** not accepted as a final paper claim / data available during selection / independent data reserved until configurations freeze.",
        "- **Source / job / audit:** pinned scientific code revision / cluster allocation identifier / independent validation record.", ""]
    (REPORTS / (STEM + ".md")).write_text("\n".join(lines))
    (REPORTS / (STEM + ".json")).write_text(json.dumps(dict(runs=output, provenance=provenance), indent=2) + "\n")
    print(f"Generated {len(output)} explicit attempts, {sum(len(r['rows']) for r in output)} rows.")


if __name__ == "__main__":
    main()
