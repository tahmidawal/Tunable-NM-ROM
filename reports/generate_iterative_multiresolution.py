"""Consolidate audited, same-job iterative FOM resolution panels.

All numerical table entries are derived from retained run records.  The heat
adapter reuses the accepted September 10 experiment; the remaining adapters
read the new owner records when their audits and archive collection finish.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-09-11-iterative-fom-multiresolution"
SOURCES = {}


def read(path):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def same(a, b, label):
    if abs(a - b) > 1e-10 * max(1.0, abs(a), abs(b)):
        raise ValueError(f"Aggregate mismatch for {label}: {a} versus {b}")


def heat_panel():
    base = Path("worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/iterative_cg09")
    result = read(base / "archive/outputs/results.json")
    audit = read(base / "analysis/audit.json")
    collection = read(base / "COLLECTION-CHECK.json")
    cleanup = read(base / "REMOTE-CLEANUP.json")
    assert result["complete"] and audit["passed"]
    assert collection["archive_verified"] and cleanup["remote_absent"]
    assert audit["result_sha256"] == SOURCES[str(base / "archive/outputs/results.json")]
    settings = result["settings"]
    methods = {
        "nmrom_cholesky": "NMROM (Cholesky)",
        "fom_cg_cn": "CN-CG, tolerance 1e-6",
        "fom_cg_cn_tol1e3": "CN-CG, tolerance 1e-3",
        "fom_cg_cn_tol1e2": "CN-CG, tolerance 1e-2",
        "fom_same_grid": "Direct sine-transform FOM",
    }
    normalized = []
    for summary in audit["summaries"]:
        n, method = summary["intervals"], summary["method"]
        if method not in methods:
            continue
        groups = [x for x in result["rows"] if x["intervals"] == n and x["method"] == method]
        repetitions = [r for x in groups for r in x["repetitions"]]
        assert len(groups) == len(result["cases"])
        assert all(len(x["repetitions"]) == settings["timing_repetitions"] for x in groups)
        gpu_ms = median(r["phases"]["device_seconds"] for r in repetitions) * 1000
        host_ms = median(r["phases"]["host_seconds"] for r in repetitions) * 1000
        worst_error = max(max(r["vs_physical"]["relative_current"]) for r in repetitions)
        same(gpu_ms, summary["device_median_ms"], (n, method, "GPU"))
        same(host_ms, summary["host_median_ms"], (n, method, "host"))
        same(worst_error, summary["worst_physical_error"], (n, method, "error"))
        failed = {
            "initial_fits": summary["nonstationary_fit_count"],
            "nonlinear_steps": summary["nonstationary_step_count"],
            "linear_steps": summary.get("cg_nonconverged_steps", 0),
        }
        normalized.append(dict(
            intervals=n, method=method, label=methods[method],
            gpu_ms=gpu_ms, host_ms=host_ms, worst_error=worst_error,
            error_allowance=audit["maximum_reference_refinement"],
            convergence_failures=failed, converged=not any(failed.values()),
            gpu_outliers=summary["device_outliers"], host_outliers=summary["host_outliers"],
            invocations=len(repetitions),
        ))
    return dict(
        key="heat", label="Heat 2D", intervals=settings["requested_intervals"],
        primary_rom=settings["primary_method"], primary_fom=settings["primary_fom"],
        iterative_foms=[m for m, cfg in settings["cg_methods"].items() if cfg["theta"] == .5],
        diagnostics=["fom_same_grid"], rows=normalized,
        cases=len(result["cases"]), repetitions=settings["timing_repetitions"],
        target=settings["accuracy_target"], error_norm="current-relative field L2",
        reference="Refined continuum sine-series heat solution; empirical refinement allowance included.",
        model=f"Frozen $k={result['config']['k']}$, $r={result['config']['r']}$ separable decoder with Cholesky latent updates.",
        baseline="Matched Crank–Nicolson time grid with compiled CG; previous-state linear initialization.",
        notes="Initialization and all requested full fields are charged. This panel reuses the accepted heat allocation; other PDE times come from their own paired allocations.",
        source_commit=audit["source_commit"], metadata=result["metadata"],
        result_path=str(base / "archive/outputs/results.json"), audit_path=str(base / "analysis/audit.json"),
        archive_path=str(base / "ARCHIVE.json"),
        detail_report="2026-09-10-heat-iterative-cg-comparison.md",
    )


def poisson_panel():
    base = Path("worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/iterative_cg06")
    result = read(base / "result.json")
    read(base / "cluster/out/pilot/result.json")
    assert SOURCES[str(base / "result.json")] == SOURCES[str(base / "cluster/out/pilot/result.json")]
    audit = read(base / "audit.json")
    owner = read(base / "panel.json")
    cleanup = read(base / "cleanup.json")
    submission = read(base / "submission.json")
    assert result["complete"] and audit["passed"] and cleanup["remote_deleted_and_absence_checked"]
    assert audit["source_commit"] == submission["source_commit"] == result["provenance"]["commit"]
    assert audit["timed_invocations"] == len(result["rows"])
    cfg = result["config"]
    methods = {"nmrom": "Frozen QF NMROM", "dst": "Direct sine-transform FOM"}
    methods.update({f"cg_{tol:.0e}": f"CG, tolerance {tol:.0e}" for tol in cfg["cg_tolerances"]})
    normalized = []
    for n in cfg["intervals"]:
        owner_mesh = next(m for m in owner["meshes"] if m["intervals"] == n)
        for method, label in methods.items():
            rows = [r for r in result["rows"] if r["intervals"] == n and r["method"] == method]
            groups = [[r for r in rows if r["case"] == c] for c in range(owner["cohort_count"])]
            assert all(len(g) == cfg["repetitions"] for g in groups)
            native = owner_mesh["methods"][method]
            # Native owner tables use the median of per-case medians. Verify
            # them, then use the pooled repetition median consistently with
            # the accepted heat study in this cross-PDE presentation.
            same(median(median(r["fused_device_seconds"] for r in g) for g in groups) * 1000,
                 native["gpu_median_ms"], (n, method, "owner GPU"))
            same(median(median(r["total_seconds"] for r in g) for g in groups) * 1000,
                 native["host_median_ms"], (n, method, "owner host"))
            worst = max(r["physical_error"] for r in rows)
            same(worst, native["worst_relative_error"], (n, method, "error"))
            valid = all(r["solver_valid"] for r in rows)
            qualified = valid and all(
                r["conservative_physical_error"] <= cfg["development_target"]
                and r["reference_delta"] <= cfg["development_target"] * cfg["reference_fraction"]
                for r in rows)
            assert qualified == native["all_cases_pass_target"]
            failed = sum(not r["solver_valid"] for r in rows)
            normalized.append(dict(
                intervals=n, method=method, label=label,
                gpu_ms=median(r["fused_device_seconds"] for r in rows) * 1000,
                host_ms=median(r["total_seconds"] for r in rows) * 1000,
                worst_error=worst, adjusted_error=max(r["conservative_physical_error"] for r in rows),
                converged=valid, qualified=qualified, convergence_failures={"solves": failed},
                gpu_outliers=native["gpu_outlier_count"], host_outliers=native["host_outlier_count"],
                invocations=len(rows), stop_reason_counts=native["stop_reason_counts"],
                nonstationary_residual_target_invocations=native.get("nonstationary_residual_target_invocations", 0),
            ))
    return dict(
        key="poisson", label="Poisson 2D", intervals=cfg["intervals"], primary_rom="nmrom",
        primary_fom=f"cg_{cfg['primary_cg_tolerance']:.0e}",
        iterative_foms=[f"cg_{t:.0e}" for t in cfg["cg_tolerances"]], diagnostics=["dst"],
        rows=normalized, cases=owner["cohort_count"], repetitions=cfg["repetitions"],
        target=cfg["development_target"], error_norm="current-relative solution L2",
        reference=f"Full requested-mesh error against a restricted {max(cfg['reference_intervals'])}-interval finite-difference sine-transform reference. The gate uses $(e+\\delta)/(1-\\delta)$ and the declared reference-allowance check, where $e$ is measured error and $\\delta$ is reference refinement; this is empirical.",
        model="Frozen relative-loss separable decoder with exact reduced weak algebra, sine-product source projection and nearest-code initialization.",
        baseline="Compiled unpreconditioned CG starts from zero on the same requested grid; final true-residual verification is charged.",
        notes=f"All {owner['cohort_count']} previously opened development sources are retained. A solve may validly stop at its declared residual target without satisfying the separate stationarity threshold. Main tables here use pooled repetition medians; native owner tables additionally report medians of case medians.",
        source_commit=audit["source_commit"], metadata=result["provenance"],
        result_path=str(base / "result.json"), audit_path=str(base / "audit.json"), archive_path=str(base / "cleanup.json"),
    )


def wave_panels():
    base = Path("worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/iterative05")
    result_path = base / "cluster/out/pilot/result.json"
    result = read(result_path)
    audit = read(base / "analysis/summary.json")
    cleanup = read(base / "cleanup.json")
    submission = read(base / "submission.json")
    assert result["complete"] and audit["integrity_audit_passed"]
    assert cleanup["remote_deleted_and_absence_checked"] and cleanup["all_three_manifests_verified"]
    assert audit["result_sha256"] == SOURCES[str(result_path)]
    assert result["provenance"]["source_commit"] == submission["source_commit"]
    coordinator_path = Path("reports") / f"{STEM}.coordinator-audit.json"
    coordinator = read(coordinator_path)
    assert coordinator["passed"] and coordinator["result_sha256"] == SOURCES[str(result_path)]
    coordinator_script = Path("reports/audit_iterative_wave_fields.py")
    SOURCES[str(coordinator_script)] = hashlib.sha256((ROOT / coordinator_script).read_bytes()).hexdigest()
    assert coordinator["source_script_sha256"] == SOURCES[str(coordinator_script)]
    cfg = result["config"]
    assert .05 in cfg["accuracy_targets"]
    panels = []
    for boundary, label in (("dirichlet", "Reflective waves"), ("absorbing", "Absorbing waves")):
        diagnostic = "dst" if boundary == "dirichlet" else "rk4"
        methods = {cfg["primary_method"]: "Frozen MLP32 NMROM", diagnostic: "Direct sine-transform FOM" if boundary == "dirichlet" else "Explicit RK4 FOM"}
        methods.update({f"cg_{tol:g}": f"Midpoint CG, tolerance {tol:.0e}" for tol in cfg["cg_tolerances"]})
        normalized = []
        for n in cfg["meshes"]:
            for method, name in methods.items():
                rows = [r for r in result["invocations"] if (r["boundary"], r["intervals"], r["method"]) == (boundary, n, method)]
                native = next(g for g in audit["groups"] if (g["boundary"], g["intervals"], g["method"]) == (boundary, n, method))
                groups = [[r for r in rows if r["case"] == c] for c in cfg["validation_indices"]]
                assert all(len(g) == cfg["repetitions"] for g in groups)
                same(median(median(r["seconds"]["complete_device_query"] for r in g) for g in groups), native["query_median"], (boundary, n, method, "owner GPU"))
                components = {}
                for component, source in (("displacement", "displacement"), ("velocity", "velocity"), ("energy", "energy_state")):
                    current = max(r["same_grid_discrepancy"][source]["max_current_relative"] for r in rows)
                    initial = max(r["same_grid_discrepancy"][source]["max_initial_normalized"] for r in rows)
                    same(current, native["errors"][source]["current_max_worst"], (boundary, n, method, source, "current"))
                    same(initial, native["errors"][source]["initial_max_worst"], (boundary, n, method, source, "initial"))
                    components[component] = dict(current=current, initial=initial)
                failures = dict(
                    initial_fits=sum(not r.get("fit_stationary", True) for r in rows),
                    incomplete_rollouts=sum(not r["completed"] for r in rows),
                    linear_steps=sum(sum(v > 1.01 * r["setting"] for v in r.get("cg_true_relative", [])) for r in rows),
                )
                refinements = dict(
                    reference_refinement_cases=native["reference_refinement_failed_cases"],
                    rom_refinement_cases=native["time_refinement_failed_cases"],
                )
                gate = native["accuracy_qualification_by_target"][str(.05)]
                normalized.append(dict(
                    intervals=n, method=method, label=name,
                    gpu_ms=median(r["seconds"]["complete_device_query"] for r in rows) * 1000,
                    host_ms=None, gpu_plus_output_transfer_ms=median(r["device_plus_output_transfer_seconds"] for r in rows) * 1000,
                    worst_error=components["displacement"]["current"], component_errors=components,
                    converged=not any(failures.values()), convergence_failures=failures, refinement_failures=refinements,
                    qualified=gate["all_current_qualified"], initial_qualified=gate["all_initial_qualified"],
                    qualification_by_target=native["accuracy_qualification_by_target"],
                    gpu_outliers=native["timing_outliers"], host_outliers=None,
                    invocations=len(rows),
                ))
        panels.append(dict(
            key=f"wave_{boundary}", label=label, intervals=cfg["meshes"],
            primary_rom=cfg["primary_method"], primary_fom="cg_1e-06",
            iterative_foms=[f"cg_{t:g}" for t in cfg["cg_tolerances"]], diagnostics=[diagnostic],
            rows=normalized, cases=len(cfg["validation_indices"]), repetitions=cfg["repetitions"],
            target=.05, error_norm="current-relative displacement L2; gate checks all three wave components",
            wave_components=True,
            coordinator_check=dict(path=str(coordinator_path), **coordinator),
            reference="Same-grid semidiscrete wave reference: independent modal solution for reflective boundaries, refined explicit stepping for absorbing boundaries. No continuum accuracy claim.",
            model=f"Frozen nonlinear head with configuration dimension {result['invocations'][0]['configuration_dimension']} and spatial bank rank {cfg['frozen_bank_rank']}; the original RK4 latent dynamics are retained.",
            baseline=f"New implicit-midpoint CG control at the same {cfg['primary_dt']:.6g} time step and {round(cfg['end_time']/cfg['observation_dt'])+1} output times; this is an iterative control, not a replay of a historical wave algorithm.",
            notes="The target gate requires displacement, velocity and energy errors, reference/ROM refinement, initial-fit stationarity and successful evolution/linear solves. Initial-normalized qualification is reported separately. Only output transfers were measured; complete host times are unavailable. Native timing outliers exceed twice the same-case median, and all are retained.",
            source_commit=submission["source_commit"],
            metadata=dict(job_id=result["provenance"]["job_id"], gpu=", ".join(result["provenance"]["device_kind"])),
            result_path=str(result_path), audit_path=str(base / "analysis/summary.json"), archive_path=str(base / "cleanup.json"),
        ))
    return panels


def burgers_panel():
    base = Path("worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/iterative06")
    result = read(base / "archive/out/result.json")
    audit = read(base / "AUDIT.json")
    owner = read(base / "PANEL.json")
    collection = read(base / "COLLECTION-CHECK.json")
    cleanup = read(base / "SUBMISSION.json")
    primary_selection = read(base / "PRIMARY-SELECTION.json")
    assert result["complete"] and audit["passed"] and collection["archive_verified"]
    assert cleanup["remote_deleted"]
    assert result["commit"] == audit["source_commit"] == owner["source_commit"]
    assert len(result["invocations"]) == audit["invocations"]
    cfg = result["config"]
    settings = {s["name"]: s for s in cfg["fom_settings"]}
    methods = {"nmrom": "Frozen sampled-upwind NMROM"}
    methods.update({name: f"Newton {s['ntol']:.0e} / linear {s['ltol']:.0e}; {'FFT' if s['preconditioner']=='fft' else 'dense DST'}" for name, s in settings.items()})
    normalized = []
    for native in owner["rows"]:
        n, method = native["intervals"], native["name"]
        rows = [r for r in result["invocations"] if (r["intervals"], r["name"]) == (n, method)]
        groups = [[r for r in rows if r["case"] == c] for c in range(cfg["cases"])]
        assert all(len(g) == cfg["reps"] for g in groups)
        same(median(median(r["gpu_seconds"] for r in g) for g in groups) * 1000, native["gpu_ms"], (n, method, "owner GPU"))
        same(median(median(r["host_seconds"] for r in g) for g in groups) * 1000, native["host_ms"], (n, method, "owner host"))
        worst = max(r["error"]["fixed_initial_max"] for r in rows)
        same(worst, native["worst_fixed_initial"], (n, method, "error"))
        physical_pass = native["reference_pass"] and native["error_plus_margin_pass"]
        if method == "nmrom":
            failures = dict(
                initial_budget_or_nonfinite=sum(r["ic_reason"] in (0, 3) for r in rows),
                step_budget_or_nonfinite=sum(sum(s in (0, 3) for s in r["stop_reasons"]) for r in rows),
            )
            stalled = dict(
                initial_fits=sum(r["ic_reason"] == 2 for r in rows),
                steps=sum(sum(s == 2 for s in r["stop_reasons"]) for r in rows),
            )
            # A stalled fit can be physically accurate, but its gradient was
            # not part of this stopping contract. Do not call it converged.
            converged = None if any(stalled.values()) and not any(failures.values()) else not any(failures.values())
            gate_label = "accuracy pass; stalled" if physical_pass and any(stalled.values()) and not any(failures.values()) else None
        else:
            failures = dict(
                nonlinear_queries=sum(not r["nonlinear_converged"] for r in rows),
                linear_queries=sum(not r["linear_converged"] for r in rows),
            )
            converged, stalled, gate_label = not any(failures.values()), {}, None
        normalized.append(dict(
            intervals=n, method=method, label=methods[method],
            gpu_ms=median(r["gpu_seconds"] for r in rows) * 1000,
            host_ms=median(r["host_seconds"] for r in rows) * 1000,
            worst_error=worst, worst_current_error=native["worst_current_relative"],
            converged=converged, physical_pass=physical_pass, qualified=native['eligible_development_5percent'],
            stationarity_status=native.get('nonlinear_stationarity_status', 'unmeasured for stalled ROM exits' if stalled else 'not applicable'),
            convergence_failures=failures, stalled_events=stalled, gate_label=gate_label,
            original_development_eligibility=native["eligible_development_5percent"],
            gpu_outliers=native["timing_outliers_gt2x_case_median"],
            host_outliers=sum(r["host_seconds"] > 2 * median(q["host_seconds"] for q in group) for group in groups for r in group),
            invocations=len(rows),
        ))
    return dict(
        key="burgers", label="Burgers 2D", intervals=cfg["meshes"], primary_rom="nmrom",
        primary_fom=primary_selection["primary_fom"], iterative_foms=list(settings), diagnostics=[],
        rows=normalized, cases=cfg["cases"], repetitions=cfg["reps"],
        target=cfg["target_fixed_initial"], error_norm="fixed-initial field L2",
        reference=f"Regenerated {cfg['reference_mesh']}-interval implicit upwind reference with spatial and temporal refinement checks; the gate adds the empirical refinement margin.",
        model=f"Frozen $k={result['K']}$, $r={result['R']}$ decoder with {result['M']} weak modes and {result['m']} fitted quadrature samples, refitted for each mesh. Linear weak terms are preassembled; sign-upwind advection is sampled and is not the historical small-bank polynomial tensor.",
        baseline=f"Compiled Newton–BiCGStab with FFT sine-transform preconditioning at the same {cfg['dt']:.6g} step size. Tight tolerances are the primary control; looser tolerances and the historical dense-transform preconditioner are retained.",
        notes="Physical accuracy and nonlinear stationarity are distinct. The original ROM stopping contract accepts a small-step/improvement stall; such exits have unmeasured stationarity and are labeled explicitly here. Timing charges inner true-residual checks and diagnostic output-step predecessor fields on the GPU. Complete host timing transfers public fields; remaining diagnostic transfers occur afterward. Outliers exceed twice the same-case median; none are excluded.",
        source_commit=audit["source_commit"], metadata=dict(job_id=result["job_id"], gpu=result["gpu"]),
        result_path=str(base / "archive/out/result.json"), audit_path=str(base / "AUDIT.json"), archive_path=str(base / "ARCHIVE.json"),
    )


def lookup(panel, n, method):
    return next(row for row in panel["rows"] if row["intervals"] == n and row["method"] == method)


def qualifies(panel, row):
    if "qualified" in row:
        return row["qualified"]
    return row["converged"] and row["worst_error"] + row.get("error_allowance", 0) <= panel["target"]


def fastest_passing(panel, n):
    candidates = [lookup(panel, n, name) for name in panel["iterative_foms"]]
    return min((r for r in candidates if qualifies(panel, r)), key=lambda r: r["gpu_ms"], default=None)


def plot(panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(2, len(panels), figsize=(3.25 * len(panels), 6.4), squeeze=False, layout="constrained")
    for col, panel in enumerate(panels):
        ns = panel["intervals"]
        for method, label, color in [
            (panel["primary_rom"], "NMROM", "#0072B2"),
            (panel["primary_fom"], "Primary iterative FOM", "#D55E00"),
        ]:
            rows = [lookup(panel, n, method) for n in ns]
            axes[0, col].plot(ns, [r["gpu_ms"] for r in rows], "o-", color=color, label=label)
            axes[1, col].plot(ns, [100 * r["worst_error"] for r in rows], "o-", color=color, label=label)
        selected = [fastest_passing(panel, n) for n in ns]
        axes[0, col].plot(ns, [r["gpu_ms"] if r else float("nan") for r in selected], "s--", color="#009E73", label="Fastest passing tested FOM")
        axes[1, col].axhline(100 * panel["target"], color="0.4", linestyle="--", label="Development error target")
        axes[0, col].set_title(panel["label"])
        axes[1, col].set_title(panel["error_norm"].split(";")[0], fontsize=9)
        axes[1, col].set_xlabel("Intervals per axis")
        if panel.get("wave_components"):
            axes[1, col].text(.02, .60, "Gate also checks velocity + energy", transform=axes[1, col].transAxes, fontsize=7, color="0.35", bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=1))
        elif panel["key"] == "burgers":
            axes[1, col].text(.02, .02, "ROM stationarity unmeasured", transform=axes[1, col].transAxes, fontsize=7, color="0.35")
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(ns)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, alpha=.2)
    axes[0, 0].set_ylabel("Median complete GPU query (ms)")
    axes[1, 0].set_ylabel("Worst relative error (%)")
    fig.suptitle("Development comparison: frozen ROMs and same-grid iterative FOMs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(axes[1, 0].get_lines()[-1])
    labels.append(f"{100*panels[0]['target']:.6g}% error threshold")
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    paths = [ROOT / f"reports/{STEM}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths


def render(panels):
    lines = [
        "# Multiresolution NMROM versus iterative FOM comparisons", "",
        "Audited development results for the current frozen separable models, using same-grid iterative full-order solvers. Paper claims remain provisional because these are existing development cohorts, with one frozen trained model per panel and independent final cases unopened.", "",
        "GPU times include input projection or initial fitting, the solve or full rollout, and every requested full-field device output. Where available, complete input/output transfers are measured separately from the same invocation; wave records measure output transfer only, so their complete host times are unavailable. Each PDE ladder uses a single allocation; compare each ROM with its paired FOM, not absolute wall times across PDEs.", "",
        f"![Resolution scaling of runtime and field error]({STEM}.png)", "",
        "## Primary iterative comparisons", "",
        "The FOM tolerance is fixed across each resolution ladder. Times are medians of all retained repetitions, with the same repetition count for every case. Ratios divide those FOM and ROM medians; values above unity favor the ROM. The error columns retain every case, requested time and repetition. A fast result that fails accuracy or its declared numerical gates is not an accepted speedup. Burgers' declared ROM stopping contract accepts stalls; these are labeled explicitly and do not establish stationarity.", "",
        "| Problem | Intervals/axis | ROM GPU ms | FOM GPU ms | FOM/ROM | ROM / FOM worst error (%) | ROM / FOM target + declared gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for panel in panels:
        for n in panel["intervals"]:
            rom, fom = [lookup(panel, n, panel[k]) for k in ("primary_rom", "primary_fom")]
            status = " / ".join(row.get("gate_label") or ("pass" if qualifies(panel, row) else "fail") for row in (rom, fom))
            lines.append(f"| {panel['label']} | {n} | {rom['gpu_ms']:.3f} | {fom['gpu_ms']:.3f} | {fom['gpu_ms']/rom['gpu_ms']:.3f} | {100*rom['worst_error']:.6g} / {100*fom['worst_error']:.6g} | {status} |")
    lines += ["", "## Effect of relaxing the full solver tolerance", "",
        "This table selects the fastest tested iterative FOM that meets the panel's physical error target and stopping rules, using the same development cohort. This is a disclosed development selection, not a global optimum or independent final confirmation. A ratio remains diagnostic if the ROM fails its own target or convergence check.", "",
        "| Problem | Intervals/axis | Fastest passing tested FOM | FOM GPU ms | FOM worst error (%) | FOM/ROM GPU | FOM/ROM including transfers |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for panel in panels:
        for n in panel["intervals"]:
            rom, fom = lookup(panel, n, panel["primary_rom"]), fastest_passing(panel, n)
            if fom is None:
                lines.append(f"| {panel['label']} | {n} | none | — | — | — | — |")
                continue
            host = f"{fom['host_ms']/rom['host_ms']:.3f}" if fom.get("host_ms") and rom.get("host_ms") else "unavailable"
            lines.append(f"| {panel['label']} | {n} | {fom['label']} | {fom['gpu_ms']:.3f} | {100*fom['worst_error']:.6g} | {fom['gpu_ms']/rom['gpu_ms']:.3f} | {host} |")
    lines += ["", "## What each panel measures", "",
        "The percentages below use different explicitly named norms. They should not be ranked across PDEs. Error relative to the current wave state can become large after energy leaves an absorbing domain; initial-normalized wave errors are additional diagnostics, not substitutes for the current-relative result.", "",
        "| Problem | Development cases × timing repeats | Error norm | Error target (%) | Primary ROM | Primary iterative FOM |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for panel in panels:
        rom = lookup(panel, panel["intervals"][0], panel["primary_rom"])
        fom = lookup(panel, panel["intervals"][0], panel["primary_fom"])
        lines.append(f"| {panel['label']} | {panel['cases']} × {panel['repetitions']} | {panel['error_norm']} | {100*panel['target']:.6g} | {rom['label']} | {fom['label']} |")
    for panel in panels:
        lines += ["", f"**{panel['label']}.** {panel['model']} {panel['baseline']} {panel['reference']} {panel['notes']}"]
        if panel.get("detail_report"):
            lines += ["", f"[Additional archived controls and detailed audit]({panel['detail_report']})."]
    wave_panels = [p for p in panels if p.get("wave_components")]
    if wave_panels:
        lines += ["", "## Additional wave accuracy diagnostics", "",
            r"These errors use the same timed outputs as the main table. The displacement column above does not by itself establish accuracy of velocity or the full energy state. Wave initial-normalized displacement uses $\|u_{\mathrm{ref}}(0)\|_M$; velocity and energy-state errors both use $\sqrt{2E_{\mathrm{ref}}(0)}$. The mass-weighted spatial norm is $\|\cdot\|_M$, and $E$ is the sum of kinetic and potential wave energy. Current-relative entries whose reference norm is zero or below the declared zero threshold are undefined and explicitly marked in the native records.", "",
            "| Boundary | Intervals/axis | Method | Current displacement / velocity / energy error (%) | Initial-normalized displacement / velocity / energy error (%) | All-state current / initial target pass |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
        for panel in wave_panels:
            for row in panel["rows"]:
                component = row["component_errors"]
                current = " / ".join(f"{100*component[k]['current']:.6g}" for k in ("displacement", "velocity", "energy"))
                initial = " / ".join(f"{100*component[k]['initial']:.6g}" for k in ("displacement", "velocity", "energy"))
                status = " / ".join("yes" if row[key] else "no" for key in ("qualified", "initial_qualified"))
                lines.append(f"| {panel['label']} | {row['intervals']} | {row['label']} | {current} | {initial} | {status} |")
    lines += ["", "## Controls included in this comparison and solver exits", "",
        "The listed failures are counts of the named fit or solve events, including all repetitions; their units differ across algorithms. Outliers are retained in all medians. Their exact rules and the raw repetition arrays are linked in the native audits.", "",
        "| Problem | Intervals/axis | Method | GPU / host ms | Worst error (%) | Solver exits and refinement failures | GPU / host outliers |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for panel in panels:
        for row in sorted(panel["rows"], key=lambda r: (r["intervals"], r["method"])):
            failures = "; ".join(f"{key}: {value}" for key, value in row["convergence_failures"].items() if value) or "none"
            stalls = "; ".join(f"stalled {key}: {value}" for key, value in row.get("stalled_events", {}).items() if value)
            if stalls:
                failures += "; " + stalls + " (stationarity unmeasured)"
            refinements = "; ".join(f"{key}: {value}" for key, value in row.get("refinement_failures", {}).items() if value)
            if refinements:
                failures += "; " + refinements
            host = f"{row['host_ms']:.3f}" if row.get("host_ms") is not None else "unavailable"
            host_outliers = row['host_outliers'] if row.get('host_outliers') is not None else 'unavailable'
            lines.append(f"| {panel['label']} | {row['intervals']} | {row['label']} | {row['gpu_ms']:.3f} / {host} | {100*row['worst_error']:.6g} | {failures} | {row['gpu_outliers']} / {host_outliers} |")
    lines += ["", "## Reproducibility and scope", "",
        "All results require GPU preflight, float64, highest matrix precision, a private run directory, GPU burn-in, retained timing repetitions, and paired outputs from the timed invocation. Accepted archives have checksum collection and exact-directory cleanup records. Offline training, mesh assembly and compilation are excluded from online timing and retained in the native records. Frozen weights transfer across each panel's meshes; this study does not include retraining at each resolution. Native Poisson, Burgers and wave panels additionally report medians of per-case medians; this report recomputes pooled repetition medians consistently with the accepted heat table, so displayed times and ratios can differ between those aggregations.", "",
        "| Problem | Job | GPU | Scientific source | Native results / audit / archive |",
        "| --- | --- | --- | --- | --- |",
    ]
    for panel in panels:
        metadata = panel["metadata"]
        links = " / ".join(f"[{label}](../{panel[key]})" for key, label in [("result_path", "results"), ("audit_path", "audit"), ("archive_path", "archive")])
        lines.append(f"| {panel['label']} | {metadata['job_id']} | {metadata['gpu']} | `{panel['source_commit']}` | {links} |")
    if wave_panels:
        check = wave_panels[0]["coordinator_check"]
        lines += ["", f"A separate coordinator implementation recomputed displacement errors for {check['distinct_displacement_fields']} full largest-grid predictions, covering {check['timed_invocation_identities']} timed-output identities, with maximum metric difference {check['maximum_metric_difference']:.6g}. This is an additional first-case cross-check of both boundaries and the NMROM/tight-CG pair; the owner audit covers the complete cohort, velocity and energy. [Coordinator evidence](../{check['path']})."]
    lines += ["", "Direct sine-transform or explicit wave solvers remain labeled controls; a win against the selected iterative algorithm does not imply a win against every FOM. Pre-reset wave experiments remain excluded. The historical smaller Burgers tensor checkpoint and the former ViT + CP model are not silently substituted for the current model. All experiment worktrees remain separate.", "",
        f"The adjacent [{STEM}.json]({STEM}.json) contains every normalized row, source hash and panel note. The [generator](generate_iterative_multiresolution.py) reads native JSON evidence and checks its aggregates before writing tables and plots.", "",
        "## Glossary", "",
        "- **Intervals/axis:** number of spatial cells along each coordinate direction; boundary conventions determine the number of stored nodes.",
        "- **ROM / NMROM / FOM:** reduced model / nonlinear-manifold reduced model / numerical solver that evolves or solves on the full spatial grid.",
        "- **GPU / host ms:** median blocked device-query time / same invocation including transfer of supplied inputs and requested outputs, in milliseconds.",
        "- **FOM/ROM:** FOM median time divided by ROM median time; a larger-than-unity value favors the ROM under that particular timing contract.",
        "- **Current-relative field L2:** spatial root-sum-square error divided by the reference field norm at that same time; quadrature weights are included where the panel requires them.",
        r"- **Fixed-initial / initial-normalized error:** error divided by a declared fixed initial scale. Heat/Burgers use the initial field norm; wave displacement uses $\|u_{\mathrm{ref}}(0)\|_M$, while wave velocity and energy-state errors use $\sqrt{2E_{\mathrm{ref}}(0)}$.",
        "- **Worst error:** largest stated relative error across every retained case, output time and repetition; not the error of a typical case.",
        "- **Target + declared gates:** the field-error allowance and the panel's original numerical checks pass. Burgers' accepted stall exits are not a proof of stationarity; wave gates additionally require reference and time-step refinement.",
        "- **CG / BiCGStab / Newton:** iterative linear-system solvers / a linear solver that can handle nonsymmetric systems / repeated linearized updates for a nonlinear equation.",
        "- **Tolerance / residual / stationarity:** numerical stopping threshold / discrepancy in a solved equation / sufficiently small objective gradient.",
        "- **CN / midpoint / Newmark / RK4:** Crank–Nicolson / implicit midpoint / an implicit wave time integrator / fourth-order explicit Runge–Kutta stepping.",
        "- **Preconditioner / DST / Cholesky:** an operation that makes iterative equations easier to solve / discrete sine transform / factorization for a symmetric positive-definite matrix.",
        "- **k / r / bank / weak modes:** latent coordinate count / spatial-bank size / learned spatial functions / smooth functions used to test the PDE rather than its pointwise residual.",
        "- **QF / sampled upwind / tensor:** quadrature-free reduced algebra / evaluation of the flow-direction-dependent spatial operator at fitted sample points / precomputed coefficient contractions; these are different operator implementations.",
        "- **Frozen checkpoint / mesh transfer:** unchanged learned parameters / evaluating those same learned functions on a different mesh.",
        "- **Development cohort / final cases:** inputs used in method development / separate inputs reserved for independent confirmation.",
        "- **Reference / refinement allowance:** numerical solution used for grading / measured disagreement between refined reference calculations; empirical agreement is not a rigorous continuum error bound.",
        "- **Fit or solve event / invocation / outlier:** an individual optimization or equation solve / one complete timed query / an unusually slow retained timing under the native stated rule.",
        "- **Iterative / direct / diagnostic:** repeated equation-solving updates / an algebraic transform or factorization solution / a control reported to interpret the main comparison.",
        "- **Source hash / checksum archive:** content-based identification of the executed code / preserved run files whose collected bytes were verified.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heat-check", action="store_true", help="Verify only the reused heat adapter; do not publish a partial report.")
    args = parser.parse_args()
    heat = heat_panel()
    if args.heat_check:
        print(json.dumps({"panel": heat["label"], "checked_rows": len(heat["rows"]), "sources": SOURCES}, indent=2))
        return
    # Missing native files or failed gates stop generation before any report
    # is written. Never publish a partial panel as a completed study.
    panels = [poisson_panel(), heat, burgers_panel(), *wave_panels()]
    assert all(p["intervals"] == heat["intervals"] for p in panels)
    assert len({p["key"] for p in panels}) == len(panels)
    for panel in panels:
        for n in panel["intervals"]:
            lookup(panel, n, panel["primary_rom"])
            lookup(panel, n, panel["primary_fom"])
    figures = plot(panels)
    output = ROOT / "reports" / STEM
    script = Path(__file__).resolve()
    SOURCES[str(script.relative_to(ROOT))] = hashlib.sha256(script.read_bytes()).hexdigest()
    payload = dict(
        schema="iterative-fom-multiresolution-v1", status="audited development results",
        date="2026-09-11", aggregation="Median of all retained repetitions, with equal repeats per case; ratios of those medians.",
        target_status="Scientific integrity is audited separately from physical accuracy and solver stopping rules.",
        source_sha256=SOURCES, panels=panels,
    )
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".md").write_text(render(panels))
    columns = ["problem", "intervals", "method", "gpu_ms", "host_ms", "worst_error", "error_norm", "target", "qualified", "solver_converged", "stationarity_status", "stalled_initial_fits", "stalled_steps", "gpu_outliers", "host_outliers", "invocations", "job_id"]
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for panel in panels:
            for row in panel["rows"]:
                writer.writerow(dict(
                    problem=panel["label"], error_norm=panel["error_norm"], target=panel["target"],
                    qualified=qualifies(panel, row), job_id=panel["metadata"]["job_id"],
                    solver_converged=row["converged"], stationarity_status=row.get("stationarity_status", "see native solver audit"),
                    stalled_initial_fits=row.get("stalled_events", {}).get("initial_fits", 0),
                    stalled_steps=row.get("stalled_events", {}).get("steps", 0),
                    **{key: row[key] for key in ["intervals", "method", "gpu_ms", "host_ms", "worst_error", "gpu_outliers", "host_outliers", "invocations"]},
                ))
    artifacts = [output.with_suffix(ext) for ext in (".md", ".json", ".csv")] + figures
    manifest = dict(
        source_sha256=SOURCES,
        generated_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
    )
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
