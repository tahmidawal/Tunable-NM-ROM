"""Build the consolidated JCP initializer report from retained run records."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from audit_2026_11_23_jcp_ns_initializer import audit_run
from audit_2026_11_23_jcp_ns_phase import audit as audit_phase

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "2026-11-23-jcp-ns-initializer-assets"
EXPERIMENT = ROOT / "worktrees/2026-11-23-jcp-consolidated/experiments/jcp-ns-initializer"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def fmt(value):
    return f"{value:.6g}"


def latex_number(value):
    result = fmt(value)
    if "e" in result:
        mantissa, exponent = result.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return result


def plot_label(path):
    stem = path.stem
    if stem.startswith("initializer-stationarity-"):
        return "Initializer stationarity (" + stem.removeprefix("initializer-stationarity-") + ")"
    return stem.replace("-", " ").capitalize()


def build():
    audit_names = ["development-v1-audit", "development-v2-audit", "fresh-initializer-audit", "fresh-phase-audit", "freeze-audit"]
    audits = {name: read(ASSETS / (name + ".json")) for name in audit_names}
    assert all(a["integrity_passed"] for a in audits.values())
    phase = read(EXPERIMENT / "runs/f01/output/phase/summary.json")
    cfg = phase["config"]
    freeze = read(EXPERIMENT / "FROZEN_PROTOCOL.json")
    registration = read(EXPERIMENT / "FRESH_COHORT_REGISTRATION.json")
    integration = read(EXPERIMENT / "runs/f01/output/integration/summary.json")
    assert integration["passed"] and integration["query_qualified"] and integration["host_steps_stationary"]
    assert max(integration["per_output_gap"]) <= integration["bound"] and integration["returned_code_gap"] == 0
    retention = read(ASSETS / "retention-audit.json")
    assert retention["integrity_passed"]
    backup = read(ASSETS / "backup-audit.json")
    review = read(ASSETS / "independent-review.json")
    assert backup["passed"] and review["passed"] and retention["all_owner_closeouts_present"]
    for name, digest in review["audited_record_hashes"].items():
        assert sha(ASSETS / name) == digest
    consolidation = read(ASSETS / "consolidation.json")
    merge_audit = read(ASSETS / "merge-audit.json")
    assert consolidation["passed"] and merge_audit["passed"]
    initial_paths = {"Development: QR/SVD": "d01/output", "Development: fixed anchor": "d02/output",
                     "Untouched: fixed anchor": "f01/output/initializers"}
    run_rows = {}
    sources = []
    for audit in audits.values():
        for source in audit.get("sources", []):
            if "path" in source:
                assert sha(source["path"]) == source["sha256"]
            else:
                for key in ("detail", "arrays"):
                    assert sha(source[key]) == source[key + "_sha256"]
    # Reconstruct the saved decisions from current inputs, so a stale audit
    # cannot qualify altered record/provenance/agreement files.
    for name, relative in zip(audit_names[:3], initial_paths.values()):
        assert audit_run(EXPERIMENT / "runs" / relative) == audits[name], name
    assert audit_phase(EXPERIMENT / "runs/f01/output/phase") == audits["fresh-phase-audit"]
    plot_checks = []
    for relative in ("d01", "d02", "f01/initializers", "f01/phase"):
        directory = EXPERIMENT / "plots" / relative
        manifest = read(directory / "MANIFEST.json")
        for name, digest in manifest["files"].items():
            assert sha(directory / name) == digest
        builder = "build_fresh_phase_plots.py" if relative.endswith("/phase") else "build_initializer_plots.py"
        assert manifest["builder_sha256"] == sha(EXPERIMENT / builder)
        data = read(directory / "plotted-data.json")
        if relative.endswith("/phase"):
            assert data["registered_ids"] == data["observed_ids"] == list(range(cfg["eval_cases"]))
            assert data["complete_metrics"] and data["full_qualification_pass_count"] == sum(c["qualification"]["passed"] for c in phase["cases"])
            for key, audit_key in (("ordinary", "field_max"), ("shape", "aligned_max")):
                ratios = np.asarray(data["paired_case_evolved_max_span_over_head"][key])
                recorded = audits["fresh-phase-audit"]["paired_metrics"][audit_key]
                assert float(np.median(ratios)) == recorded["median_paired_span_over_head"]
                assert int(np.sum(ratios > 1)) == recorded["head_lower_count"]
        else:
            assert len(data["records"]) == data["expected_records"]
            assert len(data["case_order"]) == len(set(data["case_order"]))
        plot_checks.append({"plot_group": relative, "manifest_sha256": sha(directory / "MANIFEST.json"),
                            "verified_files": len(manifest["files"]), "source_builder_matches": True})
    (ASSETS / "plot-audit.json").write_text(json.dumps({"passed": True,
        "scope": "all plot and builder hashes, exact registered case coverage, displayed primary paired aggregation; visual review recorded separately",
        "groups": plot_checks}, indent=2) + "\n")

    def record_source(path, copied=None):
        entry = {"source": str(path.relative_to(ROOT)), "sha256": sha(path)}
        if copied is not None:
            entry["copy"] = str(copied.relative_to(REPORTS))
            assert sha(copied) == entry["sha256"]
        sources.append(entry)

    evidence = ASSETS / "evidence"
    evidence.mkdir(exist_ok=True)
    for label, relative in initial_paths.items():
        run = EXPERIMENT / "runs" / relative
        rows = read(run / "records.json")
        provenance = read(run / "provenance.json")
        run_rows[label] = rows
        stem = relative.split("/")[0]
        for name in ("records.json", "summary.json", "provenance.json"):
            src, dst = run / name, evidence / (stem + "-" + name)
            shutil.copy2(src, dst)
            record_source(src, dst)
        assert provenance["backend"] == "gpu" and provenance["x64"] and provenance["matmul_precision"] == "highest"
    for relative, name in [
        ("FROZEN_PROTOCOL.json", "frozen-protocol.json"),
        ("FRESH_COHORT_REGISTRATION.json", "fresh-cohort-registration.json"),
        ("PROTOCOL.md", "protocol.md"),
        ("runs/f01/output/phase/summary.json", "fresh-phase.json"),
        ("configs/fresh-initializers.json", "fresh-initializers-config.json"),
        ("configs/fresh-phase.json", "fresh-phase-config.json"),
        ("registration/cluster_replay.json", "cluster-metadata-replay.json"),
        ("runs/f01/output/integration/summary.json", "old-case-integration-fixture.json"),
    ]:
        src, dst = EXPERIMENT / relative, evidence / name
        shutil.copy2(src, dst)
        record_source(src, dst)
    for tag in ("d01", "d02", "f01"):
        for name in ("RESULTS.sha256", "INPUTS.sha256", "RESULT_ARCHIVE.sha256", "sacct.txt", "CLOSEOUT.json"):
            src = EXPERIMENT / "runs" / tag / name
            if src.exists():
                dst = evidence / (tag + "-" + name)
                shutil.copy2(src, dst)
                record_source(src, dst)
    for src in sorted((EXPERIMENT / "plots").rglob("*")):
        if src.is_file():
            dst = ASSETS / "figures" / src.relative_to(EXPERIMENT / "plots")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            record_source(src, dst)
    for name in ("build_initializer_plots.py", "build_fresh_phase_plots.py"):
        record_source(EXPERIMENT / name)
    for name in audit_names:
        record_source(ASSETS / (name + ".json"))
    for name in ("consolidation.json", "merge-audit.json", "retention-audit.json", "plot-audit.json", "backup-audit.json", "independent-review.json"):
        record_source(ASSETS / name)
    for name in ("audit_2026_11_23_jcp_ns_initializer.py", "audit_2026_11_23_jcp_ns_phase.py", "audit_2026_11_23_jcp_ns_freeze.py", "audit_2026_11_23_jcp_ns_retention.py", "audit_2026_11_23_jcp_ns_backup.py"):
        record_source(REPORTS / name)
    record_source(Path(__file__))

    summary_rows = []
    metrics = {}
    for label, rows in run_rows.items():
        values = {
            "checks": len(rows),
            "core_failures": sum(not r["repaired_pass"] for r in rows),
            "qualified": sum(r["initializer_qualified"] for r in rows),
            "all_field_agreement": all(r["old_new_initial_agreement_pass"] and r["initial_field_agreement_pass"] for r in rows),
            "max_old_new_gap": max(r["old_new_field_gap"] for r in rows),
        }
        metrics[label] = values
        summary_rows.append([label, values["checks"], values["core_failures"],
                             f'{values["qualified"]}/{values["checks"]}', fmt(values["max_old_new_gap"])])
    fresh = audits["fresh-initializer-audit"]
    pa = audits["fresh-phase-audit"]
    qualification_keys = [k for k in phase["cases"][0]["qualification"] if k != "passed"]
    check_labels = {"initialization": "Initializer", "common_state": "Matched initial fields", "operator": "Weak operators",
                    "rollout": "Trajectory solver", "jacobian": "Jacobian actions", "refinement": "Solver refinement", "reference": "Reference refinement"}
    phase_checks = [[check_labels[key], sum(c["qualification"][key] for c in phase["cases"]), len(phase["cases"]),
                     ", ".join(str(c["index"]) for c in phase["cases"] if not c["qualification"][key]) or "None"]
                    for key in qualification_keys]
    paired = pa["paired_metrics"]
    metric_labels = {"field_max": "Ordinary field error (%)", "aligned_max": "Centroid-aligned field error (%)",
                     "phase_max": "Centroid displacement (domain fraction)"}
    paired_rows = []
    for key, row in paired.items():
        scale = 1 if key == "phase_max" else 100
        paired_rows.append([metric_labels[key], fmt(scale * row["head_median"]), fmt(scale * row["head_worst"]),
                            fmt(scale * row["span_median"]), fmt(scale * row["span_worst"]),
                            fmt(row["median_paired_span_over_head"]), row["head_lower_count"]])
    flag_rows = []
    for label, audit in [("Development fixed anchor", audits["development-v2-audit"]), ("Untouched fixed anchor", fresh)]:
        flags = audit["preserved_arithmetic_flags"]
        flag_rows.append([label, flags["ordinary_direct_objective_guard_failures"],
                          flags["ordinary_direct_phase_objective_guard_failures"], flags["decimal_audited_records"]])
    merge_rows = [[x["branch"], "`" + x["commit"][:12] + "`", x["file_count"], "Unchanged"] for x in consolidation["lanes"]]
    init_status = "passes" if fresh["scientific_initializer_qualification_passed"] else "does not pass"
    phase_status = "passes" if pa["scientific_phase_confirmation_passed"] else "does not pass"
    failed_cases = [c["index"] for c in phase["cases"] if not c["qualification"]["passed"]]
    failure_text = "No case fails the complete phase contract." if not failed_cases else "Cases failing the complete phase contract: " + ", ".join(map(str, failed_cases)) + "."
    figures = sorted((ASSETS / "figures").rglob("*.png"))
    figure_links = "\n".join(f'- [{plot_label(p)} ({p.parent.relative_to(ASSETS / "figures")})]({p.relative_to(REPORTS)})' for p in figures)
    phase_figure = ASSETS / "figures/f01/phase/fresh-phase-shape.png"
    assert phase_figure.is_file()
    scientific_files = sum(x["file_count"] for x in consolidation["lanes"])
    frame_dimension = len(phase["cases"][0]["runs"][0]["metrics"][0]["phase_displacement"])
    advantage = (pa["scientific_phase_confirmation_passed"]
                 and paired["field_max"]["median_paired_span_over_head"] > 1
                 and paired["aligned_max"]["median_paired_span_over_head"] > 1)
    alignment_claim = ("The qualified median head advantage survives this declared alignment."
                       if advantage else "A qualified head advantage surviving alignment is not established by this run.")
    v1_missing_arrays = len(audits["development-v1-audit"]["phase_raw_array_missing_source_and_record_verified_only"])
    resource_rows = [[row["run"], row["job_id"], row["gpu"].split(",")[0], row["state"], row["allocated_gpu_seconds"]]
                     for row in retention["runs"]]
    cleanup_status = "complete" if retention["all_owner_closeouts_present"] else "pending final closeout"
    text = rf"""# JCP consolidation and Navier–Stokes initializer confirmation

This report records the authorized branch consolidation, the initializer repair, and its frozen validation on previously untouched original-family cases. The retained numerical results and independent audits are final for this registered configuration; broader accuracy–cost and generalization claims remain open.

The revised initializer **{init_status}** all registered untouched-case checks, and the fresh phase/shape campaign **{phase_status}** its complete numerical contract. {failure_text} There are no new timing or speedup claims.

## One consolidated JCP branch

The working branch is `exp/2026-11-23-jcp-consolidated`, backed up through `origin/codeonly/exp/2026-11-23-jcp-consolidated`. All {scientific_files} scientific Git files from the merged lanes match their original contents; their earlier unsuccessful results remain part of the record.

Final experiment source: `{backup['source_commit']}`. Verified code-only backup: `{backup['verified_remote_commit']}`. All {backup['eligible_experiment_files_byte_identical']} eligible initializer-experiment files match byte for byte, including all {backup['all_raw_manifest_files_tracked_and_mirrored']} raw result-manifest files across the attempts.

{table(['Merged branch', 'Original head', 'Scientific files', 'Content audit'], merge_rows)}

The frozen implementation source is `{freeze['execution_source_commit']}`; the accepted development solver source is `{freeze['accepted_solver_source_commit']}`. The confirmation allocation used package commit `{phase['source_commit']}`, which adds the frozen manifest and cohort registration without changing the frozen implementation. [Merge audit](2026-11-23-jcp-ns-initializer-assets/merge-audit.json), [freeze audit](2026-11-23-jcp-ns-initializer-assets/freeze-audit.json) and [registered protocol](2026-11-23-jcp-ns-initializer-assets/evidence/protocol.md) preserve the exact contracts. The original large cost arrays retain their previously documented local/archive scope; consolidation does not turn them into new GitHub-hosted numerical evidence.

## What changed in the initializer

The head, coordinate bank, training data, physical family, starting-code identities and original stationarity tolerances are unchanged. The first attempt added bounded QR and SVD corrections to the original solver and failed full qualification. Those outcomes remain visible below.

The accepted revision combines both legacy endpoint implementations with a fixed-anchor evaluation of the same mathematical residual. With a fixed code $z_*$, it evaluates

$$r(z)=H(z_*)-a_0+\Delta H(z_*,z-z_*).$$

Here $H$ is the frozen head, $a_0$ is the projected target coefficient vector, and $z$ is the head code.

Stable algebraic increments reduce cancellation near a fitted endpoint; the anchor is chosen once per fit and is never updated during polishing. Both ordinary-direct and anchored gradients must satisfy the original stationarity test. Separate higher-precision evaluations of the original small network check objective differences against the original objective band. They are arithmetic diagnostics on stored parameters, separate from the primary double-precision GPU solves.

This is a prospectively registered arithmetic amendment. Ordinary-direct objective-ranking flags remain failures of that older arithmetic comparison. They are neither erased nor described as passing the old contract. The revision changes multiple ingredients, so this campaign does not isolate the benefit of stable evaluation from adding the second legacy endpoint set.

{table(['Cohort / method', 'Selected fits', 'Core failures', 'Full qualification', 'Largest old/new physical gap'], summary_rows)}

Physical gaps are relative to the original initial-field norm. A small gap shows that the repair changes certification at nearly the same physical initial state; it does not by itself establish trajectory accuracy.

For {v1_missing_arrays} records in the unsuccessful first attempt, the raw-minimum phase-control residual/Jacobian arrays were not separately saved; that comparator's audit uses retained codes, freshly recorded objective values and source review. The accepted revision and new-case invocations retain every control's arrays. This limitation does not turn the first attempt into a qualified result.

{table(['Fixed-anchor cohort', 'Ordinary all-control objective flags', 'Ordinary phase-comparator flags', 'Decimal audits'], flag_rows)}

These counts concern selected-fit records, including repeated input paths and tolerances on the same physical cases; they are not counts of distinct independent cases.

## Untouched-case protocol and actual evolution

Seed `{registration['seed']}` defines {registration['count']} cases. Exact and rounded parameter comparisons found no overlap with the registered training, development and previously exposed cohorts. Source, configuration and cohort hashes were frozen before physical outcomes were generated. Historical sealed test cohorts remain closed.

The initialization matrix covers learned and POD heads, phase and query input paths, and every registered gradient tolerance. The trajectory comparison uses the learned coordinate bank; POD is covered by initializer checks only. The phase run independently audits its own {pa['actual_initial_fit_invocations']} actual initial-fit invocations rather than borrowing an initializer pass from another run. The common head/span comparison starts from the same reconstructed physical field; the native span starts from its own linear projection.

Those actual trajectory initializer calls separately retain {pa['actual_initial_ordinary_direct_objective_flags']} ordinary-direct objective flags and {pa['actual_initial_decimal_audits']} Decimal audits, all passing the amended independent objective contract. These are repeated invocations on the same cohort, not additional independent cases.

Before new physical cases were generated, a fixed exposed-case integration fixture passed: the shared initializer returned the same code through the phase and complete-query interfaces, and their trajectories agreed within the declared bound. That fixture is not a new cost measurement or a fresh POD rollout validation.

The trajectory setting is $N={cfg['n']}$, bank rank $R={cfg['rank']}$, head dimension $k={cfg['k']}$, scalar weak-test rows $M={cfg['rows']}$, $\Delta t={cfg['dt']}$ and horizon ${cfg['horizon']}$. Both head and span also solve for {frame_dimension} frame variables: {cfg['k'] + frame_dimension} versus {cfg['rank'] + frame_dimension} total online unknowns. The same-grid CNAB2 references use steps ${latex_number(cfg['reference']['dt'])}$ and ${latex_number(cfg['reference']['check_dt'])}$. These are discrete time-refinement checks, without spatial-refinement or continuum-error claims. All {pa['recorded_trajectory_steps']} trajectory steps are retained. Jacobian-action checks apply to the tighter trajectory tolerance only.

{table(['Phase check', 'Passing cases', 'Registered cases', 'Failed case IDs'], phase_checks)}

## What the new phase/shape results support

Each entry below starts from a maximum over evolved times for each case; time-zero fit errors are reported separately in the plot data. The ratio is the median of paired span/head ratios, not a ratio of medians. No failed case is removed.

These ratios compare the head and linear-span trajectory methods. The initializer's old/new physical differences are reported separately above.

{table(['Metric', 'Head median', 'Head worst', 'Common-span median', 'Common-span worst', 'Median paired span/head', 'Cases with lower head error'], paired_rows)}

The declared alignment translates fields using a physical energy centroid; it does not fit a translation to minimize error. Phase displacement uses wrapped domain fractions. The smallest recorded centroid concentration is {fmt(pa['minimum_centroid_concentration'])}, against the registered descriptive ambiguity threshold {fmt(cfg['centroid_concentration_threshold'])}; this threshold is not part of the trajectory qualification gate. The ordinary and aligned errors answer different questions: trajectory position and remaining shape both matter. {alignment_claim} This comparison does not establish a unique causal mechanism for drift.

## Figures and manuscript placement

![Measured untouched-case phase and shape comparison]({phase_figure.relative_to(REPORTS)})

Use the new phase/shape comparison with its qualification table in the main numerical-results section. Put initializer stationarity, field agreement, objective-arithmetic diagnostics and every-case panels in the solver-validation section or supplement. PNG, SVG and PDF versions and the plotted records are retained together.

{figure_links}

The [earlier measured JCP report](2026-11-19-jcp-physical-diagnostics-and-confirmation.md) still supplies the derivative-error, Burgers width–test and original accuracy–cost figures. Those datasets have not changed and must retain their historical qualifications. This report supersedes neither their measurements nor their failed gates.

## What still needs an experiment

- A complete-query accuracy–cost comparison using the revised initializer, paired against eligible classical and POD baselines in the same allocation. All initialization, physical output and audit work must be included. Historical POD rollout, reference and accuracy-target failures need their own resolution.
- A repaired sparse deployment that first passes admission, including the cold start and actual stencil closure. An initial-fit fix does not demonstrate sparse online scaling.
- A separate extension if the paper claims broader parameter ranges, longer horizons, or a causal mechanism: these fresh cases belong to the original family and use the registered horizon. Optional manifold-distance, tangent and reset studies were not repeated here.
- An ablation if the paper attributes the repair specifically to stable increments: the current accepted revision also includes both legacy endpoint sets.

These are distinct from the requested initializer confirmation. The strongest current manuscript argument is numerical qualification plus measured physical error, with cost claims restricted to their own valid evidence.

## Execution and retention

{table(['Run', 'Cluster job', 'GPU', 'Scheduler state', 'Allocated GPU-seconds'], resource_rows)}

The campaign used {retention['allocated_gpu_seconds']} allocated GPU-seconds in total. These are resource-accounting figures, not cross-job performance comparisons. Raw outputs and source archives pass exact hash verification, and archived source files match the recorded Git commits. Owned remote-directory cleanup is {cleanup_status} in the retained closeout records. Large aggregate archives have local retention and hashes; individual result-file coverage is verified separately against the code-only backup.

## Reproducibility and glossary

[Initializer audit](2026-11-23-jcp-ns-initializer-assets/fresh-initializer-audit.json), [phase audit](2026-11-23-jcp-ns-initializer-assets/fresh-phase-audit.json), [independent reviews](2026-11-23-jcp-ns-initializer-assets/independent-review.json), [backup and cleanup verification](2026-11-23-jcp-ns-initializer-assets/backup-audit.json), [data-source hashes](2026-11-23-jcp-ns-initializer-assets/sources.json), and [machine-readable report values](2026-11-23-jcp-ns-initializer-assets/measured-values.json) link every table to retained records. The report builder and independent audit scripts live beside this report. No measured value is manually entered into the report.

- **Head / learned head:** the trained low-dimensional nonlinear map into coefficients of the coordinate bank; its latent dimension is $k$.
- **Bank / rank $R$:** the fixed collection of spatial basis functions and its number of columns. **POD** is the data-derived orthogonal basis used by the comparator.
- **Span / common span / native span:** all linear combinations of a bank; the common span uses the head's initial field, while the native span uses its own initial linear projection.
- **Frame variables:** the {frame_dimension} translation coordinates evolved together with the head code or span coefficients; they describe the moving spatial frame.
- **Initializer / selected fit:** the optimization that maps an input physical field to a head code; one returned solution per case, bank, input path and tolerance is counted.
- **Phase / query input:** the original physical initial field versus the complete-query path's additional spectral projection.
- **Exposed / untouched cohort:** cases available during development versus cases whose physical outcomes were generated only after source and protocol freeze. Repeated tolerances are checks on the same cases, not independent samples.
- **Stationarity / gradient tolerance / core failure:** a sufficiently small normalized residual gradient, its acceptance threshold, and failure of the core numerical solver's gradient/objective checks.
- **QR / SVD / LM:** numerical linear-algebra factorizations and the damped least-squares solver; rank truncation alone does not certify convergence.
- **Fixed anchor / ordinary direct / amended arithmetic:** a frozen nearby code used for stable residual increments, the original direct residual calculation, and the prospectively specified revised evaluation contract.
- **Objective flag / Decimal audit:** a recorded failed cost-ranking comparison, and a separate high-precision original-network calculation used to check differences hidden by floating-point rounding.
- **Objective / allowance / endpoint:** the least-squares fit cost, its permitted numerical comparison difference, and a candidate returned code being compared.
- **Full qualification / phase check / passing cases:** conjunction of all declared numerical checks, an individual trajectory campaign gate, and the number of cases satisfying it. Integrity verification checks record consistency and is separate from scientific qualification.
- **Old/new physical gap:** relative difference between reconstructed initial fields before and after the repair; it is not their error against truth.
- **$N$, $M$, $\Delta t$, horizon:** spatial points per direction, number of weak test functions, trajectory time step, and final evolution time. **Weak tests** project the PDE residual onto prescribed smooth functions.
- **CNAB2 / reference / refinement:** the reference time-stepping scheme, its same-grid trajectory, and agreement under a finer step or tighter solve. No continuum truth is asserted.
- **Ordinary / aligned error:** field mismatch before or after the declared centroid translation. **Centroid concentration** measures whether that periodic centroid has a stable direction.
- **Median / worst / paired span/head / lower-error cases:** the middle case value, largest case value, casewise error ratio followed by its median, and the count where head error is smaller. **Evolved maximum** excludes time zero.
- **Jacobian action / operator / common state:** derivative consistency in a test direction, assembled weak operators, and agreement of reconstructed initial fields for the matched comparison.
- **Sparse admission / stencil closure / cold start:** checks allowing reduced spatial evaluation, every spatial site needed by its discretization, and work to form the initial state. **Complete query** includes initialization, evolution and requested output.
- **Commit / mirror / source hash:** recorded source version, its code-only remote backup, and a content fingerprint used to verify exact identity.
- **Scheduler state / allocated GPU-seconds:** whether the cluster job finished successfully, and its allocation duration times allocated GPU count. These include setup and validation and are not per-query speed measurements.
"""
    output = REPORTS / "2026-11-23-jcp-ns-initializer-repair.md"
    output.write_text(text)
    values = {"initializer": metrics, "phase": {k: v for k, v in pa.items() if k != "sources"},
              "ordinary_arithmetic_flags": {name: audits[name]["preserved_arithmetic_flags"] for name in ["development-v2-audit", "fresh-initializer-audit"]},
              "registered_config": cfg, "resource_accounting": retention,
              "frozen_protocol_sha256": sha(EXPERIMENT / "FROZEN_PROTOCOL.json")}
    (ASSETS / "measured-values.json").write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    (ASSETS / "sources.json").write_text(json.dumps({"sources": sources, "report_sha256": sha(output)}, indent=2) + "\n")
    for item in sources:
        assert sha(ROOT / item["source"]) == item["sha256"]
        if "copy" in item:
            assert sha(REPORTS / item["copy"]) == item["sha256"]
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert all((REPORTS / link).exists() for link in links)
    assert "fresh-phase-displacement-pairs.png" in text
    for path in figures:
        assert path.with_suffix(".pdf").exists() and path.with_suffix(".svg").exists()
    original_files = {p.relative_to(EXPERIMENT / "plots") for p in (EXPERIMENT / "plots").rglob("*") if p.is_file()}
    copied_files = {p.relative_to(ASSETS / "figures") for p in (ASSETS / "figures").rglob("*") if p.is_file()}
    assert original_files == copied_files
    verification = {"passed": True, "report_sha256": sha(output), "source_entries_verified": len(sources),
                    "local_links_verified": len(links), "figure_sets_png_pdf_svg": len(figures),
                    "all_copied_figures_match_final_source": True, "all_actual_phase_flags_reported": True,
                    "phase_counterexample_plot_indexed": True, "cleanup_complete": retention["all_owner_closeouts_present"]}
    (ASSETS / "report-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"report": str(output), "source_count": len(sources), "figure_count": len(figures),
                      "initializer_passed": fresh["scientific_initializer_qualification_passed"],
                      "phase_passed": pa["scientific_phase_confirmation_passed"]}))


if __name__ == "__main__":
    build()
