"""Audited Poisson online-tuning adapter for the unified results report."""

from collections import Counter
import hashlib
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/online_tuning07")


def close(a, b):
    assert abs(a-b) <= 1e-10*max(1., abs(a), abs(b)), (a, b)


def short_label(method, preset, retained):
    if method == "nmrom_baseline":
        return f"Baseline; M{retained}"
    if method.startswith("budget_"):
        return f"Budget {preset['budget']}; M{retained}"
    if method.startswith("stationarity_"):
        return f"Stop {preset['stationarity_stop']:.0e}; M{retained}"
    if method.startswith("residual_"):
        return f"Residual {preset['tau']:g}; M{retained}"
    if method.startswith("multistart"):
        return f"{preset['starts']} starts; M{retained}"
    return f"Strict; M{retained}"


def load_tuning(read, sources):
    # Missing evidence stops a requested final build. A draft must explicitly
    # opt out; never silently drop an incomplete experiment from its report.
    raw = read(BASE / "result.json")
    read(BASE / "cluster/out/pilot/result.json")
    assert sources[str(BASE / "result.json")] == sources[str(BASE / "cluster/out/pilot/result.json")]
    owner, audit, cleanup, submission = [read(BASE / name) for name in ("panel.json", "audit.json", "cleanup.json", "submission.json")]
    coordinator_path = Path("reports/2026-09-11-poisson-online-tuning.coordinator-audit.json")
    coordinator = read(coordinator_path)
    assert raw["complete"] and audit["passed"] and coordinator["passed"]
    assert cleanup["remote_deleted_and_absence_checked"]
    assert cleanup["all_three_manifests_verified"] and cleanup["source_hashes_verified"]
    assert audit["archive_sha256"] == cleanup["archive_sha256"]
    assert audit["timed_invocations"] == len(raw["rows"])
    assert audit["result_sha256"] == sources[str(BASE / "result.json")]
    assert raw["provenance"]["commit"] == submission["source_commit"] == owner["source_commit"]
    assert coordinator["result_sha256"] == sources[str(BASE / "result.json")]
    script = Path("reports/audit_poisson_online_tuning.py")
    sources[str(script)] = hashlib.sha256((ROOT / script).read_bytes()).hexdigest()
    assert coordinator["source_script_sha256"] == sources[str(script)]
    cfg, cohort = raw["config"], len(raw["cohort"]["parameters"])
    assert cohort == owner["cohort_count"]
    assert owner["checkpoint_sha256"] == cfg["checkpoint_sha256"]
    presets = {p["id"]: p for p in cfg["presets"]}
    rows, comparisons, selected = [], [], []
    for phase in owner["phases"]:
        n, phase_name = phase["intervals"], phase["phase"]
        normalized = {}
        for method, item in phase["methods"].items():
            rs = [r for r in raw["rows"] if (r["phase"], r["intervals"], r["method"]) == (phase_name, n, method)]
            repeats = cfg["screen_repetitions"] if phase_name == "screen" else cfg["repetitions"]
            assert Counter(r["case"] for r in rs) == Counter({c: repeats for c in range(cohort)})
            gpu, host = median(r["fused_device_seconds"] for r in rs)*1000, median(r["total_seconds"] for r in rs)*1000
            error, adjusted = max(r["physical_error"] for r in rs), max(r["conservative_physical_error"] for r in rs)
            for a, b in ((gpu, item["gpu_median_ms"]), (host, item["host_median_ms"]), (error, item["worst_relative_error"]), (adjusted, item["worst_adjusted_relative_error"])):
                close(a, b)
            assert [r["fused_device_seconds"]*1000 for r in rs] == item["gpu_repetitions_ms"]
            assert [r["total_seconds"]*1000 for r in rs] == item["host_repetitions_ms"]
            qualified = all(r["solver_valid"] and r["conservative_physical_error"] <= cfg["development_target"] and r["reference_delta"] <= cfg["development_target"]*cfg["reference_fraction"] for r in rs)
            assert qualified == item["all_cases_pass_target"]
            assert dict(Counter(str(r["reason"]) for r in rs)) == item["stop_reason_counts"]
            for field, key in (("fused_device_seconds", "gpu_outlier_count"), ("total_seconds", "host_outlier_count")):
                per_case = {c: median(r[field] for r in rs if r["case"] == c) for c in range(cohort)}
                assert sum(r[field] > 1.5*per_case[r["case"]] for r in rs) == item[key]
            nr = dict(phase=phase_name, intervals=n, method=method, gpu_ms=gpu, host_ms=host,
                      worst_error=error, adjusted_error=adjusted, qualified=qualified,
                      converged=all(r["solver_valid"] for r in rs), invocations=len(rs),
                      convergence_failures={"solves": sum(not r["solver_valid"] for r in rs)},
                      gpu_outliers=item["gpu_outlier_count"], host_outliers=item["host_outlier_count"],
                      stop_reason_counts=item["stop_reason_counts"], worst_error_case=item["worst_error_case"],
                      failed_target_cases=item["failed_target_cases"])
            if method in presets:
                stationary = sum(r["stationary"] for r in rs)
                assert stationary == item["stationary_invocations"]
                assert all(r["preset"] == presets[method] and r["retained_modes"] == item["retained_modes"] for r in rs)
                nr.update(preset=presets[method], requested_modes=item["requested_modes"], retained_modes=item["retained_modes"],
                          stationary_invocations=stationary, all_stationary=stationary == len(rs) and all(r["all_start_linear_valid"] for r in rs),
                          total_attempts_median=item["total_attempts_median"], short_label=short_label(method, presets[method], item["retained_modes"]))
                assert nr["all_stationary"] == item["all_stationary"]
            else:
                nr["short_label"] = "Direct DST" if method == "dst" else f"CG {method.removeprefix('cg_')}"
            rows.append(nr)
            normalized[method] = nr
        # Independently reproduce cohort-wide ranking (no per-query truth oracle).
        candidates = [r for m, r in normalized.items() if m in presets and phase["methods"][m]["all_finite"]]
        stationary = [r for r in candidates if r["all_stationary"]]
        def fast(items):
            return min(items, key=lambda r:(r["gpu_ms"], r["method"]), default=None)
        def accurate(items):
            return min(items, key=lambda r:(r["worst_error"], r["gpu_ms"], r["method"]), default=None)
        ranks = dict(fastest_gpu=fast(candidates), most_accurate=accurate(stationary), unrestricted_accuracy=accurate(candidates),
                     fastest_passing=fast([r for r in candidates if r["qualified"]]), fastest_stationary=fast(stationary))
        assert {k: r["method"] if r else None for k, r in ranks.items()} == phase["selections"]
        if phase_name != "confirmation":
            continue
        passing_fom = fast([r for m, r in normalized.items() if m.startswith("cg_") and r["qualified"]])
        assert passing_fom and passing_fom["method"] == phase["fastest_passing_cg"]
        tight = normalized[f"cg_{cfg['primary_cg_tolerance']:.0e}"]
        comparisons.append(dict(intervals=n, tight_ms=tight["gpu_ms"], tight_error=tight["worst_error"],
                                fast_ms=passing_fom["gpu_ms"], fast_error=passing_fom["worst_error"], fast_label=passing_fom["short_label"],
                                direct_ms=normalized["dst"]["gpu_ms"], direct_error=normalized["dst"]["worst_error"]))
        role_rows = [("Baseline", normalized["nmrom_baseline"]), ("Fastest tested", ranks["fastest_gpu"]),
                     ("Fastest stationary", ranks["fastest_stationary"]),
                     ("Most accurate*", ranks["most_accurate"]), ("Fastest passing", ranks["fastest_passing"])]
        if ranks["unrestricted_accuracy"] is not ranks["most_accurate"]:
            role_rows.append(("Accuracy diagnostic", ranks["unrestricted_accuracy"]))
        for role, row in role_rows:
            selected.append(dict(intervals=n, role=role, row=row, tight_fom_ms=tight["gpu_ms"], fastest_fom_ms=passing_fom["gpu_ms"]))
    successes = [s for s in selected if s["role"] == "Fastest passing" and s["row"] is not None]
    finding = ("No tested configuration meets the original physical and numerical target on every development case. " if not successes else "A tested setting reaches the original physical and numerical target on at least one mesh. ")
    finding += "*Most accurate is restricted to settings with every selected solve stationary at the original threshold; an unrestricted accuracy winner is shown separately when different."
    improvements = []
    for n in cfg["intervals"]:
        mesh = {s["role"]: s["row"] for s in selected if s["intervals"] == n}
        if mesh["Most accurate*"]:
            improvements.append(100*(mesh["Baseline"]["worst_error"]-mesh["Most accurate*"]["worst_error"]))
    if improvements:
        finding += f" The largest reduction in worst error from baseline is {max(improvements):.8f} percentage points across the confirmed meshes."
    return dict(audited=True, job_id=owner["job_id"], gpu=raw["provenance"]["gpu"], source_commit=owner["source_commit"],
                checkpoint_sha256=cfg["checkpoint_sha256"], cases=cohort, target=cfg["development_target"],
                repetitions=cfg["repetitions"], rows=rows, selections=selected, comparators=comparisons,
                scope=f"{len(presets)} frozen-checkpoint settings screened at {cfg['screen_intervals']} intervals on all {cohort} Gaussian-source development cases. The fixed shortlist is confirmed at every requested mesh with {cfg['repetitions']} repeats; endpoint selection remains developmental.",
                finding=finding,
                settings_note=f"Frozen k{raw['checkpoint_config']['k']}/r{raw['checkpoint_config']['r']} decoder; no retraining or online empirical quadrature. M is the actual retained weak-mode count. These are preassembled, precompiled presets: operators and code caches are prepared for each mesh and M; budgets and gradient thresholds select compiled kernels. New unprepared settings incur offline work. The original stationarity gate is {cfg['stationarity_tolerance']:.0e}; in-loop checks are charged, post-query auditing is excluded for every method.",
                configuration=cfg, frozen_selection=raw["selection"], exit_labels=raw["contract"]["stopping_reasons"],
                result_path=str(BASE / "result.json"), audit_path=str(BASE / "audit.json"), archive_path=str(BASE / "cleanup.json"),
                coordinator_audit_path=str(coordinator_path), coordinator_check=coordinator)


def markdown(tuning):
    t = tuning
    lines = ["## Poisson: fastest and most accurate online settings", "", t["scope"], "", t["finding"], "",
             "Timings and CG comparators in this section come from the new tuning job. They are not combined with FOM timings from the earlier Poisson allocation.", "",
             "| Intervals/axis | Selection | Setting | ROM GPU ms | Worst / adjusted error (%) | Stationary invocations | Target + numerical gates | Tight / fastest passing CG to ROM ratio |",
             "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for s in t["selections"]:
        r = s["row"]
        if r is None:
            lines.append(f"| {s['intervals']} | {s['role']} | none | — | — | — | — | — |")
            continue
        lines.append(f"| {s['intervals']} | {s['role']} | `{r['method']}` (actual M={r['retained_modes']}) | {r['gpu_ms']:.3f} | {100*r['worst_error']:.6g} / {100*r['adjusted_error']:.6g} | {r['stationary_invocations']}/{r['invocations']} | {'pass' if r['qualified'] else 'fail'} | {s['tight_fom_ms']/r['gpu_ms']:.3f} / {s['fastest_fom_ms']/r['gpu_ms']:.3f} |")
    lines += ["", t["settings_note"], "", "### Complete Poisson tuning screen", "",
              "These are the complete coarse-screen measurements, kept separate from confirmation timings. Settings are ranked over the whole cohort; online initialization and selection among starting guesses use the weak objective, never the true field error.", "",
              "| Setting | Actual / requested M | Iteration budget | Residual reduction threshold | In-loop stationarity threshold | Starts | GPU / host ms | Worst error (%) | Stationary / invocations | Failed target cases | Exit reasons | GPU / host outliers |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for r in t["rows"]:
        if r["phase"] != "screen" or "preset" not in r:
            continue
        p = r["preset"]
        stops = "; ".join(f"{t['exit_labels'][reason]}: {count}" for reason, count in r["stop_reason_counts"].items())
        threshold = f"{p['stationarity_stop']:.0e}" if p["stationarity_stop"] is not None else "disabled"
        lines.append(f"| `{r['method']}` | {r['retained_modes']} / {r['requested_modes']} | {p['budget']} | {p['tau']:g} | {threshold} | {p['starts']} | {r['gpu_ms']:.3f} / {r['host_ms']:.3f} | {100*r['worst_error']:.6g} | {r['stationary_invocations']} / {r['invocations']} | {r['failed_target_cases']} | {stops} | {r['gpu_outliers']} / {r['host_outliers']} |")
    lines += ["", "### Poisson confirmation configurations and exits", "",
              "| Intervals/axis | Setting | Actual M | GPU / host ms | Worst error (%) | Stationary / invocations | Failed target cases | Exit reasons | GPU / host outliers |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for r in t["rows"]:
        if r["phase"] != "confirmation" or "preset" not in r:
            continue
        stops = "; ".join(f"{t['exit_labels'][reason]}: {count}" for reason, count in r["stop_reason_counts"].items())
        lines.append(f"| {r['intervals']} | `{r['method']}` | {r['retained_modes']} | {r['gpu_ms']:.3f} / {r['host_ms']:.3f} | {100*r['worst_error']:.6g} | {r['stationary_invocations']} / {r['invocations']} | {r['failed_target_cases']} | {stops} | {r['gpu_outliers']} / {r['host_outliers']} |")
    links = " / ".join(f"[{label}](../{t[key]})" for key, label in (("result_path", "full records"), ("audit_path", "owner audit"), ("coordinator_audit_path", "independent endpoint audit"), ("archive_path", "checksum collection and cleanup")))
    lines += ["", f"Tuning job `{t['job_id']}`, {t['gpu']}, scientific source `{t['source_commit']}`. {links}.", ""]
    return "\n".join(lines)
