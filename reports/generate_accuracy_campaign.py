"""Generate the campaign report from checksum-linked, independently audited runs.

Only completed owner audits enter tables. Timing ratios use one run, GPU,
mesh and cohort. Full repetition arrays remain in the normalized JSON.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/2026-09-11-accuracy-improvements-and-wave-speed"
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = ROOT / path
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def heat():
    run = Path("worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/accuracy10")
    panel = read(run/"analysis/panel.json")
    audit = read(run/"analysis/audit.json")
    raw = read(run/"archive/outputs/results.json")
    coordinator = read("reports/2026-09-11-heat-accuracy.coordinator-audit.json")
    cleanup = read(run/"REMOTE-CLEANUP.json")
    assert audit["passed"] and coordinator["passed"] and raw["complete"]
    assert audit["result_sha256"] == panel["result_sha256"] == coordinator["source_result_sha256"]
    assert panel["result_sha256"] == SOURCES[str(run/"archive/outputs/results.json")]
    assert panel["audit_sha256"] == SOURCES[str(run/"analysis/audit.json")]
    scopes = {"all_development": lambda x: True,
              "opened_development": lambda x: x != "confirmation_development",
              "confirmation_development": lambda x: x == "confirmation_development"}
    selected = []
    for row in panel["rows"]:
        if row["scope"] not in scopes:
            continue
        native = [r for r in raw["rows"] if r["intervals"] == row["intervals"]
                  and r["method"] == row["method"] and scopes[row["scope"]](r["cohort"])]
        reps = [p for r in native for p in r["repetitions"]]
        times = [p["phases"]["device_seconds"]*1000 for p in reps]
        host = [p["phases"]["host_seconds"]*1000 for p in reps]
        errors = [max(p["vs_physical"]["relative_current"]) for p in reps]
        assert np.isclose(np.median(times), row["device_median_ms"], rtol=1e-14)
        assert np.isclose(max(errors), row["physical_error_worst"], rtol=1e-14)
        selected.append(dict(row, gpu_repetitions_ms=times, host_repetitions_ms=host))
    return dict(rows=selected, metadata=panel["metadata"], primary=panel["primary_method"],
                models=panel["models"], bank_gate=panel["bank_gate"], run=str(run), cleanup=cleanup)


def wave(run_name):
    run = Path("worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs")/run_name
    panel = read(run/"panel.json")
    audit = read(run/"audit.json")
    raw = read(run/"cluster/out/pilot/result.json")
    selection = read(run/"selection.json") if (ROOT/run/"selection.json").exists() else None
    cleanup = read(run/"cleanup.json")
    assert audit["passed"] and raw["complete"] and not raw["final_test_opened"]
    assert panel["source_result_sha256"] == audit["result_sha256"] == SOURCES[str(run/"cluster/out/pilot/result.json")]
    assert str(audit["job_id"]) == str(raw["provenance"]["job_id"]) == str(cleanup["job_id"])
    assert audit["source_commit"] == raw["provenance"]["source_commit"]
    assert audit["timed_invocations"] == len(raw["invocations"])
    assert cleanup["remote_deleted_and_absence_checked"] and cleanup["all_three_manifests_verified"]
    for row in panel["panels"]:
        invocations = [r for r in raw["invocations"] if (r["intervals"], r["cohort"], r["method"], r["setting"])
                       == (row["intervals"], row["cohort"], row["method"], row["setting"])]
        times = [r["seconds"]["complete_device_query"]*1000 for r in invocations]
        np.testing.assert_allclose(times, row["timing_ms"], rtol=1e-14)
        assert np.isclose(np.median(times), row["median_gpu_ms"], rtol=1e-14)
        for key in ("displacement", "velocity", "energy_state"):
            initial = 100*max(r["same_grid_discrepancy"][key]["max_initial_normalized"] for r in invocations)
            current = 100*max(r["same_grid_discrepancy"][key]["max_current_relative"] for r in invocations)
            assert np.isclose(initial, row["worst_initial_errors_percent"][key], rtol=1e-14, atol=1e-14)
            assert np.isclose(current, row["worst_current_errors_percent"][key], rtol=1e-14, atol=1e-14)
        assert row["all_state_5percent_pass"] == (max(row["worst_initial_errors_percent"].values()) <= 5)
    combined = []
    keys = sorted({(p["intervals"], p["method"], p["setting"]) for p in panel["panels"]})
    for n, method, setting in keys:
        parts = [p for p in panel["panels"] if (p["intervals"], p["method"], p["setting"]) == (n, method, setting)]
        reps = [r for r in raw["invocations"] if (r["intervals"], r["method"], r["setting"]) == (n, method, setting)]
        times = [r["seconds"]["complete_device_query"]*1000 for r in reps]
        errors = {key: max(p["worst_initial_errors_percent"][key] for p in parts)
                  for key in ("displacement", "velocity", "energy_state")}
        combined.append(dict(intervals=n, method=method, setting=setting, cohort="all", cases=len({r["case"] for r in reps}),
                             median_gpu_ms=float(np.median(times)), timing_ms=times,
                             outliers_above_twice_median=int(sum(t > 2*np.median(times) for t in times)),
                             worst_initial_errors_percent=errors,
                             worst_current_errors_percent={key: max(p["worst_current_errors_percent"][key] for p in parts) for key in errors},
                             all_state_5percent_pass=max(errors.values()) <= 5,
                             numerical_gate_pass=all(p["numerical_gate_pass"] for p in parts),
                             time_refinement_pass=all(p["time_refinement_pass"] is not False for p in parts),
                             physical_and_numerical_pass=all(p["physical_and_numerical_pass"] for p in parts)))
    for row in combined:
        choices = [p for p in combined if p["intervals"] == row["intervals"] and p["method"].startswith("cg") and p["physical_and_numerical_pass"]]
        if choices:
            best = min(choices, key=lambda p: p["median_gpu_ms"])
            row["fastest_tested_passing_cg"] = {k: best[k] for k in ("method", "setting", "median_gpu_ms")}
            row["fastest_passing_cg_over_method"] = best["median_gpu_ms"]/row["median_gpu_ms"]
    return dict(**panel, combined_panels=combined, provenance=raw["provenance"], selection=selection,
                run=str(run), cleanup=cleanup, case_count=len({r["case"] for r in raw["invocations"]}),
                repetitions=raw["config"]["repetitions"])


def poisson(run_name, coordinator_file):
    run = Path("worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs")/run_name
    panel = read(run/"panel.json")
    audit = read(run/"audit.json")
    raw = read(run/"result.json")
    coordinator = read(coordinator_file)
    cleanup = read(run/"cleanup.json")
    assert audit["passed"] and coordinator["passed"] and raw["complete"] and audit["remote_deleted"]
    assert audit["result_sha256"] == coordinator["source_result_sha256"] == SOURCES[str(run/"result.json")]
    for group in panel["groups"]:
        for mesh in group["meshes"]:
            for method, row in mesh["methods"].items():
                reps = [r for r in raw["rows"] if r["method"] == method and r["intervals"] == mesh["intervals"]
                        and (group["group"] == "all" or r["group"] == group["group"])]
                times = [r["fused_device_seconds"]*1000 for r in reps]
                assert np.isclose(np.median(times), row["gpu_median_ms"], rtol=1e-14)
                assert np.isclose(max(r["physical_error"] for r in reps), row["worst_relative_error"], rtol=1e-14)
                np.testing.assert_allclose(times, row["gpu_repetitions_ms"], rtol=1e-14)
    return dict(panel, run=str(run), cleanup=cleanup)


def burgers(run_name, coordinator_file):
    run = Path("worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs")/run_name
    panel = read(run/"PANEL.json")
    audit = read(run/"AUDIT.json")
    raw = read(run/"archive/out/result.json")
    coordinator = read(coordinator_file)
    collection = read(run/"COLLECTION-CHECK.json")
    cleanup = read(run/"CLEANUP.json")
    assert audit["passed"] and coordinator["passed"] and raw["complete"] and raw["final_cohort_unopened"]
    assert audit["result_sha256"] == coordinator["source_result_sha256"] == SOURCES[str(run/"archive/out/result.json")]
    assert coordinator["owner_audit_sha256"] == SOURCES[str(run/"AUDIT.json")]
    assert str(audit["job_id"]) == str(raw["job_id"]) == str(collection["job_id"])
    assert audit["source_commit"] == raw["commit"] == collection["source_commit"]
    assert collection["archive_verified"]
    assert cleanup["collected_archive_verified"] and cleanup["remote_absent"]
    assert str(cleanup["job_id"]) == str(raw["job_id"])
    for row in panel["rows"]:
        reps = [r for r in raw["invocations"] if r["name"] == row["name"] and r["intervals"] == row["intervals"]
                and (row["cohort"] == "all" or r["cohort"] == row["cohort"])]
        times = [r["gpu_seconds"]*1000 for r in reps]
        host = [r["host_seconds"]*1000 for r in reps]
        assert len(reps) == row["invocations"]
        assert np.isclose(np.median(times), row["gpu_ms"], rtol=1e-14)
        assert np.isclose(np.median(host), row["host_ms"], rtol=1e-14)
        assert np.isclose(max(r["error"]["fixed_initial_max"] for r in reps), row["worst_fixed_initial"], rtol=1e-14)
        assert np.isclose(max(r["error"]["current_relative_max"] for r in reps), row["worst_current_relative"], rtol=1e-14)
        assert np.isclose(max(r["error"]["fixed_initial_per_time"][0] for r in reps), row["worst_initial_error"], rtol=1e-14)
        row["gpu_repetitions_ms"], row["host_repetitions_ms"] = times, host
    cases = [dict(intervals=r["intervals"], method=r["name"], case=r["case"], cohort=r["cohort"],
                  initial_error=r["error"]["fixed_initial_per_time"][0], worst_error=r["error"]["fixed_initial_max"])
             for r in raw["invocations"] if r["rep"] == 0]
    return dict(panel, run=str(run), collection=collection, cleanup=cleanup, case_errors=cases)


def wave_confirmation_lines(w):
    lines = ["## Reflective waves: frozen multiresolution confirmation", "",
             "The selected nested decoder, initializer and integration step were frozen before introducing the new development cases. The original head, its accelerated implementation, named CG controls and direct DST are retimed together in this confirmation job. The phase-only head and independently retrained larger head were not confirmed across these meshes.", ""]
    lookup = {(p["intervals"], p["method"]): p for p in w["combined_panels"]}
    meshes = sorted({p["intervals"] for p in w["combined_panels"]})
    rows = []
    for n in meshes:
        selected = lookup[n, "trained_nested40"]
        cg = selected["fastest_tested_passing_cg"]["method"]
        for method in ("baseline", "chol_guard", "trained_nested40", cg, "dst"):
            p = lookup[n, method]
            e = p["worst_initial_errors_percent"]
            rows.append([n, method, f"{p['median_gpu_ms']:.6f}",
                         " / ".join(f"{e[key]:.6f}" for key in ("displacement", "velocity", "energy_state")),
                         p["outliers_above_twice_median"], "Pass" if p["physical_and_numerical_pass"] else "Fail"])
    lines += table(["Intervals", "Method", "GPU ms", "Worst initial-scaled u / v / energy-state error %", "GPU outliers", "Physical and numerical criteria"], rows)
    fine = lookup[meshes[-1], "trained_nested40"]
    original = lookup[meshes[-1], "baseline"]
    cg = fine["fastest_tested_passing_cg"]
    lines += [f"At {meshes[-1]} intervals on all {fine['cases']} cases, the selected ROM takes {fine['median_gpu_ms']:.6f} GPU ms versus {original['median_gpu_ms']:.6f} ms for the same-job original ROM ({original['median_gpu_ms']/fine['median_gpu_ms']:.6f}× acceleration). The fastest tested passing CG setting takes {cg['median_gpu_ms']:.6f} ms, a {cg['median_gpu_ms']/fine['median_gpu_ms']:.6f}× timing ratio. The ROM still misses the all-state target at {fine['worst_initial_errors_percent']['energy_state']:.6f}% energy-state error, so this is not an accuracy-qualified speedup at that target. Direct DST remains faster.", "",
              "The pooled comparator must pass over the complete cohort. A fast CG setting that fails physical accuracy on an intermediate mesh is excluded even when its algebraic residual test passes. Retained repetitions and all rejected CG settings remain in the normalized JSON.", ""]
    rows = []
    for p in w["panels"]:
        if p["intervals"] == meshes[-1] and p["method"] in ("baseline", "trained_nested40"):
            e, current = p["worst_initial_errors_percent"], p["worst_current_errors_percent"]
            rows.append([p["cohort"], p["method"], " / ".join(f"{e[key]:.6f}" for key in ("displacement", "velocity", "energy_state")),
                         f"{current['displacement']:.6f} / {current['velocity']:.6f}", "Pass" if p["physical_and_numerical_pass"] else "Fail"])
    lines += table(["Fine-grid cohort", "Method", "Initial-scaled u / v / energy-state error %", "Current-relative u / v error %", "Physical and numerical criteria"], rows)
    lines += ["The new development cohort stays separate from the opened selection cases; neither cohort is the paper's sealed final test. Every returned trajectory and refinement comparison passed the recorded numerical checks. A configuration-key error stopped the first confirmation attempt before any query cases were generated; the retry changed only that operational lookup and retained the frozen scientific configuration.", ""]
    return lines


def build():
    h, w6, w7, w8 = heat(), wave("accel06"), wave("accel07"), wave("accel08")
    w9, w10 = wave("accel09"), wave("accel10")
    wf = wave("accel12")
    wf_coordinator = read("reports/2026-09-11-wave-accuracy.coordinator-audit.json")
    assert wf_coordinator["passed"] and wf_coordinator["source_result_sha256"] == wf["source_result_sha256"]
    assert wf_coordinator["owner_audit_sha256"] == SOURCES[str(Path(wf["run"])/"audit.json")]
    w10_coordinator = read("reports/2026-09-11-wave-correction-screen.coordinator-audit.json")
    assert w10_coordinator["passed"] and w10_coordinator["source_result_sha256"] == w10["source_result_sha256"]
    ps = poisson("staged_accuracy08", "reports/2026-09-11-poisson-staged-accuracy.coordinator-audit.json")
    pc = poisson("capacity_accuracy09", "reports/2026-09-11-poisson-capacity-accuracy.coordinator-audit.json")
    pc_diagnostic = read(Path(pc["run"])/"head-correction-diagnostic.json")
    assert pc_diagnostic["source_result_sha256"] == SOURCES[str(Path(pc["run"])/"result.json")]
    assert pc_diagnostic["source_audit_sha256"] == SOURCES[str(Path(pc["run"])/"audit.json")]
    b = burgers("accuracy08", "reports/2026-09-11-burgers-accuracy.coordinator-audit.json")
    bc = burgers("accuracy09", "reports/2026-09-11-burgers-coverage.coordinator-audit.json")
    normalized = dict(status="Heat, Burgers and reflective waves complete and audited; staged/capacity Poisson audited; only the final nested Poisson family remains active.",
                      heat=h, poisson_staged=ps, poisson_capacity=pc, poisson_training_correction_diagnostic=pc_diagnostic,
                      burgers_initial=b, burgers_coverage=bc, wave_confirmation=wf, wave_confirmation_coordinator_audit=wf_coordinator,
                      wave_screens={"accel06": w6, "accel07": w7, "accel08": w8, "accel09": w9, "accel10": w10},
                      wave_correction_coordinator_audit=w10_coordinator)
    lines = ["# Accuracy improvements and reflective-wave speed", "",
             "This report records the controlled accuracy-training and wave-acceleration campaign. The included numbers have passed independent development audits; the overall campaign and publication validation are still incomplete.", "",
             "Absorbing waves are excluded. Final paper cohorts remain unopened, and the existing experiment branches remain separate. Tables are generated from saved invocation records; no wall-clock ratio crosses jobs or GPUs.", "",
             "## Heat: targeted training improves accuracy", "",
             "The spatial bank, latent dimension and online solver are unchanged. Uniform continuation, extra initial-field sampling, and initial-field sampling plus a difficult-example loss are matched training arms. Initial-plus-tail was declared primary before evaluation.", ""]
    labels = {"nmrom_frozen": "Original head", "nmrom_uniform": "Uniform continuation",
              "nmrom_initial_only": "Initial emphasis", "nmrom_initial_tail": "Initial + tail (primary)",
              "fom_cg_cn": "CG, tolerance 1e-6", "fom_cg_cn_tol1e2": "CG, tolerance 1e-2",
              "linear_weak_exact": "Free linear bank (different ROM)", "fom_same_grid": "Direct DST FOM"}
    lookup = {(r["scope"], r["intervals"], r["method"]): r for r in h["rows"]}
    largest = max(r["intervals"] for r in h["rows"])
    all_primary = lookup["all_development", largest, h["primary"]]
    old = lookup["all_development", largest, "nmrom_frozen"]
    loose = lookup["all_development", largest, "fom_cg_cn_tol1e2"]
    lines += [f"At {largest} intervals, worst current-relative error on all {all_primary['cases']} development cases falls from {100*old['physical_error_worst']:.6f}% to {100*all_primary['physical_error_worst']:.6f}%. GPU query time changes from {old['device_median_ms']:.6f} to {all_primary['device_median_ms']:.6f} ms; the primary is {loose['device_median_ms']/all_primary['device_median_ms']:.6f}× faster than the fastest tested passing CG setting in this job.", "",
              "The new cohort contains harder cases. Compare old and new heads on the same cohort: the larger original-head error in the expanded table is not a retraction of the earlier table. Host-inclusive time and direct-DST comparisons are separate below.", ""]
    rows = []
    for scope, scope_label in [("opened_development", "Earlier cases"), ("confirmation_development", "New development"), ("all_development", "Combined")]:
        for method in ("nmrom_frozen", "nmrom_uniform", "nmrom_initial_only", "nmrom_initial_tail"):
            r = lookup[scope, largest, method]
            rows.append([scope_label+f" ({r['cases']})", labels[method], f"{100*r['physical_error_median']:.4f} / {100*r['physical_error_worst']:.4f}",
                         f"{r['device_median_ms']:.4f}", f"{r['host_median_ms']:.4f}",
                         f"{r['initial_nonstationary_fits']} / {r['step_nonstationary_fits']}", "Pass" if r["qualified"] else "Fail"])
    lines += table(["Cohort (cases)", "Model", "Median / worst error %", "GPU ms", "Host ms", "Nonstationary initial / steps", "5% and solve criteria"], rows)
    rows = []
    for n in sorted({r["intervals"] for r in h["rows"]}):
        for method in ("nmrom_frozen", "nmrom_initial_tail", "fom_cg_cn", "fom_cg_cn_tol1e2", "fom_same_grid", "linear_weak_exact"):
            r = lookup["all_development", n, method]
            rows.append([n, labels[method], f"{r['device_median_ms']:.6f}", f"{r['host_median_ms']:.6f}",
                         f"{100*r['physical_error_worst']:.6f}", f"{r['device_outliers']} / {r['host_outliers']}"])
    lines += table(["Intervals per axis", "Method", "GPU ms", "Host ms", "Worst error %", "GPU / host outliers"], rows)
    lines += ["The small GPU runtime improvement does not imply a host-inclusive improvement. The direct DST FOM remains faster than the nonlinear ROM. The free linear-bank control uses unrestricted bank coefficients and is a different reduced model.", "",
              "## Poisson: staged training at unchanged capacity did not solve the problem", "",
              "The learned spatial bank is trained first with free training coefficients, then the nonlinear head is fitted in the full field metric, followed by joint refinement. Ordinary joint continuation is matched to the staged optimizer time. All procedures keep the original bank size, latent dimension, source-input contract and exact weak solver.", ""]
    rows = []
    plabels = {"original_relative": "Original", "joint_matched": "Matched joint",
               "staged_head": "Staged bank + head", "staged_joint": "Staged + joint"}
    for group in ps["groups"]:
        mesh = max(group["meshes"], key=lambda x: x["intervals"])
        for method in ps["config"]["trained_model_ids"]:
            r = mesh["methods"][method]
            bank = mesh["bank_diagnostics"][method]
            rows.append([f"{group['group']} ({group['cases']})", mesh["intervals"], plabels[method],
                         f"{100*r['worst_relative_error']:.6f}", f"{100*bank['worst_same_grid_relative_error']:.6f}",
                         f"{r['gpu_median_ms']:.6f}", r["gpu_outlier_count"], r["invalid_invocations"], "Pass" if r["all_cases_pass_target"] else "Fail"])
    lines += table(["Development cohort (cases)", "Intervals", "Model", "Worst physical error %", "Bank projection error %", "GPU ms", "GPU outliers", "Invalid solves", "5% target"], rows)
    lines += ["Matched joint continuation slightly improves the earlier cases but worsens the later development cases. Neither staged endpoint improves the expanded-cohort worst error. The nonlinear solves are stationary; the remaining error is not resolved by simply allowing more online iterations.", "",
              "Bank projection uses the full same-grid field norm; the online physical error uses a refined-grid reference. These columns are related diagnostics, not an additive error decomposition. A larger learned-bank experiment is now separate from this unsuccessful fixed-capacity comparison. Normalized training-snapshot POD projections motivate that experiment but do not prove a worst-case lower bound for every possible bank.", "",
              "## Poisson: more spatial features help the bank, but not yet the complete solver", "",
              "The capacity arm widens the learned spatial bank without changing its initial decoded function or latent dimension. The added head outputs start at zero. Both sizes undergo bank, head and joint stages, with the smaller control matched to each larger-model phase's measured optimizer time. This isolates bank capacity from a simultaneous latent increase or difficult-case reweighting.", "",
              "All cases in this comparison were already opened during development. The head-projection diagnostic uses stationary full-field fits and is a best-found value, not a proof of the global nonlinear minimum. It was run only on the middle mesh; dashes on other meshes mean it was not evaluated there. All recorded online solves are stationary.", ""]
    rows = []
    all_capacity = next(g for g in pc["groups"] if g["group"] == "all")
    for mesh in all_capacity["meshes"]:
        for method in pc["config"]["trained_model_ids"]:
            r, bank = mesh["methods"][method], mesh["bank_diagnostics"][method]
            head = mesh["head_diagnostics"].get(method, {})
            head_error = head.get("worst_stationary_same_grid_error")
            rows.append([mesh["intervals"], method, f"{r['latent_dimension']} / {r['feature_count']}",
                         f"{100*bank['worst_same_grid_relative_error']:.6f}", "—" if head_error is None else f"{100*head_error:.6f}",
                         f"{100*r['worst_relative_error']:.6f}", f"{r['gpu_median_ms']:.6f}", r["gpu_outlier_count"],
                         "Pass" if r["all_cases_pass_target"] else "Fail"])
    lines += table(["Intervals", "Model", "Latent / bank size", "Bank projection error %", "Best-found head error %", "Online physical error %", "GPU ms", "GPU outliers", "5% physical target"], rows)
    last_capacity = max(all_capacity["meshes"], key=lambda m: m["intervals"])
    capacity_old = last_capacity["methods"]["original_relative"]
    capacity_new = last_capacity["methods"]["r128_joint"]
    banks = last_capacity["bank_diagnostics"]
    lines += [f"At {last_capacity['intervals']} intervals on all {all_capacity['cases']} cases, the expanded joint endpoint lowers the bank projection error from {100*banks['original_relative']['worst_same_grid_relative_error']:.6f}% to {100*banks['r128_joint']['worst_same_grid_relative_error']:.6f}%, but complete online error changes from {100*capacity_old['worst_relative_error']:.6f}% to {100*capacity_new['worst_relative_error']:.6f}%. GPU query time also rises from {capacity_old['gpu_median_ms']:.6f} to {capacity_new['gpu_median_ms']:.6f} ms. Thus this endpoint is not accepted as an online accuracy or speed improvement.", "",
              f"The proposed bank target is {100*pc_diagnostic['proposed_bank_target']:.0f}%, separately from the {100*pc_diagnostic['online_physical_target']:.0f}% online physical target; both remain missed. A subsequent training-only residual decomposition motivates fixed correction directions. It uses saved optimized training codes, not independently certified stationary training fits, and its truth-assisted corrections are reconstruction diagnostics rather than online PDE results.", "",
              "## Burgers: stricter solves work; initial-field retraining regressed", "",
              "The fixed-bank comparison crosses the original and refined heads with the earlier stall-based optimizer and explicit stationarity stopping. The refined head trains on regenerated initial fields in the fine-grid physical norm, while replay preserves original decoded outputs at old training codes. Replay targets are not new PDE trajectories.", "",
              f"The combined cohort has {b['config']['cases']+b['config']['fresh_cases']} development cases: {b['config']['cases']} previously opened and {b['config']['fresh_cases']} introduced in this comparison. Every method has {b['config']['reps']} timed repetitions per case and mesh.", ""]
    rows = []
    for r in b["rows"]:
        if r["cohort"] != "all":
            continue
        states = "—" if r["stationary_invocations"] is None else f"{r['stationary_invocations']}/{r['invocations']}"
        rows.append([r["intervals"], r["name"], f"{100*r['median_case_fixed_initial']:.6f} / {100*r['worst_fixed_initial']:.6f}",
                     f"{100*r['worst_initial_error']:.6f}", f"{r['gpu_ms']:.6f}", f"{r['host_ms']:.6f}",
                     states, r["timing_outliers_gt2x_case_median"], "Pass" if r["physical_and_numerical_pass"] else "Fail"])
    lines += table(["Intervals", "Method", "Median / worst fixed-initial error %", "Worst initial error %", "GPU ms", "Host ms", "Stationary queries", "GPU outliers", "Physical and numerical criteria"], rows)
    fine = {r["name"]: r for r in b["rows"] if r["cohort"] == "all" and r["intervals"] == max(b["config"]["meshes"])}
    strict, accepted, trained, bfom = (fine[k] for k in ("frozen_stationary", "frozen_accepted", "trained_stationary", "fft_loose"))
    lines += [f"On the largest mesh, strict stopping changes worst fixed-initial error from {100*accepted['worst_fixed_initial']:.6f}% to {100*strict['worst_fixed_initial']:.6f}%, with GPU time {accepted['gpu_ms']:.6f} → {strict['gpu_ms']:.6f} ms. It is {bfom['gpu_ms']/strict['gpu_ms']:.6f}× faster than the same-job loose iterative FOM, and both meet the declared criteria. The trained strict head reaches {100*trained['worst_fixed_initial']:.6f}% error and is rejected as an accuracy improvement.", "",
              "The old stopping contract accepted small-progress exits; those saved results are preserved. The added gradient audit reveals that they were not stationary under the stricter criterion. Stricter solving makes convergence explicit but barely changes the physical error. The retraining improves its training initial fields while worsening development trajectories, motivating a separate broader-training-coverage comparison.", "",
              "An earlier attempt failed an overly tight diagnostic equality between two floating-point gradient evaluations. Its training checkpoint and references were preserved with audited lineage, then reused unchanged in this replay. All reported costs and predictions were regenerated together in the replay job. The stationarity threshold was unchanged; charged, posthoc and independent CPU gradients must all pass, and no threshold classification disagreement occurred.", "",
              "Burgers errors are divided by the reference initial-field norm and maximized over saved times. Its physical criterion includes the empirical reference-refinement allowance; the coarse mesh remains unqualified. The FFT-labeled FOM is an iterative Newton–BiCGStab solve using an FFT preconditioner, not a direct nonlinear solution.", "",
              "## Burgers: broader initial coverage does not repair trajectory accuracy", "",
              "The final bounded training arm expands the initial-condition draw while retaining the spatial bank, head size, replay weight, update count, batch sizes and strict solver. More training codes also enlarge the initial-guess library and optimizer state, so this compares complete training procedures rather than isolated head weights. EQ is refitted using the same old training-code indices; the added initial codes are not substituted into its fitting set.", ""]
    rows = []
    for r in bc["rows"]:
        if r["cohort"] == "all":
            rows.append([r["intervals"], r["name"], f"{100*r['worst_fixed_initial']:.6f}",
                         f"{100*r['worst_initial_error']:.6f}", f"{r['gpu_ms']:.6f}", f"{r['host_ms']:.6f}",
                         r["timing_outliers_gt2x_case_median"], "Pass" if r["physical_and_numerical_pass"] else "Fail"])
    lines += table(["Intervals", "Strict method", "Worst trajectory error %", "Worst initial error %", "GPU ms", "Host ms", "GPU outliers", "Physical and numerical criteria"], rows)
    n = max(bc["config"]["meshes"])
    fine_coverage = {r["name"]: r for r in bc["rows"] if r["cohort"] == "all" and r["intervals"] == n}
    old_b, small_b, broad_b = (fine_coverage[k] for k in ("frozen_stationary", "trained576_stationary", "trained4608_stationary"))
    worst = max((r for r in bc["case_errors"] if r["intervals"] == n and r["method"] == "trained4608_stationary"), key=lambda r: r["worst_error"])
    same = next(r for r in bc["case_errors"] if (r["intervals"], r["method"], r["case"]) == (n, "frozen_stationary", worst["case"]))
    lines += [f"At {n} intervals, the broader head improves the unsuccessful smaller-training arm from {100*small_b['worst_fixed_initial']:.6f}% to {100*broad_b['worst_fixed_initial']:.6f}% worst trajectory error, but the original strict head remains better at {100*old_b['worst_fixed_initial']:.6f}%. All online strict solves meet stationarity. Both retrained heads are rejected as replacements for the original.", "",
              f"On case {worst['case']}, initial reconstruction improves from {100*same['initial_error']:.6f}% to {100*worst['initial_error']:.6f}%, while trajectory error worsens from {100*same['worst_error']:.6f}% to {100*worst['worst_error']:.6f}%. This identifies a limitation of improving initial-field reconstruction alone. Because the decoder, initial-guess library and refitted EQ weights change together, the experiment does not uniquely attribute the later error to one mechanism.", "",
              "Both inherited controls reproduce their saved parent trajectories and solver counters. Training and reference bytes have explicit inherited provenance; all displayed query costs and errors were rerun in this job. No runtime ratio crosses GPU allocations. Initial code-fit failures are retained in the training audit; passing online stationarity does not certify every offline training fit. The next accuracy direction would require trajectory-aware training and a controlled quadrature-fidelity comparison; neither was added after seeing these results.", "",
              "## Reflective waves: geometry and time-step screens", "",
              f"These screens use the same {w6['case_count']} opened reflective Dirichlet cases at {w6['panels'][0]['intervals']} intervals, with {w6['repetitions']} timed repetitions per case. They are development screens, not a large independent test set.", "",
              "Shared analytic decoder derivatives and guarded Cholesky solves remove repeated work in latent evolution. At the original step, these preserve the mathematical trajectory to audited floating-point parity. Larger steps are a separate integration change and require refinement checks.", ""]
    p6 = {(p["method"], p["setting"]): p for p in w6["panels"]}
    base, same, faster = p6["baseline", .0025], p6["chol_guard", .0025], p6["chol_guard", .01]
    lines += [f"On the opened {base['intervals']}-interval screen, the unchanged-step implementation is {base['median_gpu_ms']/same['median_gpu_ms']:.6f}× faster than the original ROM. Including the retained larger step gives {base['median_gpu_ms']/faster['median_gpu_ms']:.6f}×. These are ROM implementation ratios; an accuracy-qualified FOM advantage is not established by this screen.", ""]
    rows = []
    for p in w6["panels"]:
        errs = p["worst_initial_errors_percent"]
        rows.append([p["method"], p["setting"], f"{p['median_gpu_ms']:.6f}",
                     " / ".join(f"{errs[k]:.6f}" for k in ("displacement", "velocity", "energy_state")),
                     p["outliers_above_twice_median"], "Pass" if p["all_state_5percent_pass"] else "Fail"])
    lines += table(["Method", "Step / CG tolerance", "GPU ms", "Worst u / v / energy-state error %", "Timing outliers", "All-state 5%"], rows)
    lines += [r"Wave displacement error is divided by the initial displacement norm. Velocity and energy-state errors are divided by $\sqrt{2E_0}$, where $E_0$ is initial physical energy; initial velocity can be zero, and its norm is not the denominator. The energy-state error measures displacement-gradient and velocity error, not energy-conservation drift. Current-relative displacement and velocity errors are separately retained in the JSON and exclude the recorded zero/vanishing reference times. DST is the same-grid semidiscrete reference, so its zero discrepancy is not zero continuum error.", "",
              "The follow-up also tested unrestricted bank evolution and nonlinear output projection. Those methods evolve a larger linear state and are labeled separately from the original nonlinear latent dynamics. Both projected-output variants missed the all-state target; the unrestricted linear control passed on the opened cases.", ""]
    rows = []
    for p in w7["panels"]:
        e = p["worst_initial_errors_percent"]
        rows.append([p["method"], p["setting"], f"{p['median_gpu_ms']:.6f}",
                     " / ".join(f"{e[k]:.6f}" for k in ("displacement", "velocity", "energy_state")),
                     p["outliers_above_twice_median"], "Pass" if p["all_state_5percent_pass"] else "Fail"])
    lines += table(["Follow-up method", "Step / CG tolerance", "GPU ms", "Worst u / v / energy-state error %", "GPU outliers", "All-state 5%"], rows)
    failed = [r for r in w7["time_refinement"] if not r["passed"]]
    for r in failed:
        lines += [f"Rejected time step: `{r['method']}` at {r['dt']} on `{r['case']}` has a {100*max(r['maxima'].values()):.6f}% half-step discrepancy. Its faster timing is diagnostic only.", ""]
    lines += ["The CG control was also allowed to use larger time steps; any FOM speed ratio must use a tested setting that passes its physical and solve criteria.", "",
              "## Reflective waves: training the original latent dimension", "",
              "Two matched training arms use a fixed encoder with consistent latent velocities. One trains displacement reconstruction; the other adds displacement-energy and tangent-velocity losses. Their comparison isolates those extra losses. The original head used independently optimized snapshot codes, so comparison with that checkpoint also changes the training-code procedure.", ""]
    rows = []
    for p in w8["panels"]:
        e = p["worst_initial_errors_percent"]
        rows.append([p["method"], p["setting"], f"{p['median_gpu_ms']:.6f}",
                     " / ".join(f"{e[k]:.6f}" for k in ("displacement", "velocity", "energy_state")),
                     p["outliers_above_twice_median"], "Pass" if p["all_state_5percent_pass"] else "Fail"])
    lines += table(["Training-screen method", "Step / CG tolerance", "GPU ms", "Worst u / v / energy-state error %", "GPU outliers", "All-state 5%"], rows)
    wave_original = next(p for p in w8["panels"] if p["method"] == "chol_guard")
    wave_phase = next(p for p in w8["panels"] if p["method"] == "trained_phase")
    lines += ["The combined training improves the three displayed initial-scaled errors compared with both the original and new field-only heads. It does not improve every error normalization: "
              f"current-relative displacement changes from {wave_original['worst_current_errors_percent']['displacement']:.6f}% to {wave_phase['worst_current_errors_percent']['displacement']:.6f}%, while current-relative velocity improves from {wave_original['worst_current_errors_percent']['velocity']:.6f}% to {wave_phase['worst_current_errors_percent']['velocity']:.6f}%. "
              "All nonlinear heads in this screen still miss the all-state target. Initial fitting, numerical rank and half-step checks pass. These separately trained endpoints are not online changes to one trained network.", "",
              "## Reflective waves: correction coordinates help more than retraining a larger head", "",
              "Increasing the latent dimension and retraining with the same phase-aware loss worsened both runtime and accuracy on the opened screen. The next arm preserves the trained nonlinear head and adds fixed linear correction directions from training data. This enlarges the nonlinear manifold while containing the original one exactly; it is a separately prepared decoder.", "",
              r"The enriched decoder is $h_{40}(z,y)=h_{32}(z)+B_8y$. The extra directions are frozen training principal components. All initial coordinates are fitted from the supplied fields, and the full enlarged state evolves through nonlinear latent dynamics. The initial-guess library uses appended principal-component coordinates without a residual correction; that limitation is frozen for confirmation.", ""]
    for label, screen in [("Larger retrained head", w9), ("Nested correction directions", w10)]:
        rows = []
        for p in screen["panels"]:
            e = p["worst_initial_errors_percent"]
            rows.append([p["method"], p["setting"], f"{p['median_gpu_ms']:.6f}",
                         " / ".join(f"{e[k]:.6f}" for k in ("displacement", "velocity", "energy_state")),
                         p["outliers_above_twice_median"], "Pass" if p["all_state_5percent_pass"] else "Fail"])
        lines += [f"{label}: all comparisons below use this screen's own paired timings.", ""]
        lines += table(["Method", "Step / CG tolerance", "GPU ms", "Worst u / v / energy-state error %", "GPU outliers", "All-state 5%"], rows)
    nested = next(p for p in w10["panels"] if p["method"] == "trained_nested40")
    nested_original = next(p for p in w10["panels"] if p["method"] == "baseline")
    lines += [f"The nested head reaches {nested['worst_initial_errors_percent']['energy_state']:.6f}% worst energy-state error and {nested['median_gpu_ms']:.6f} GPU ms on the opened screen. Its {nested_original['median_gpu_ms']/nested['median_gpu_ms']:.6f}× acceleration is relative to the same-job original ROM, not a qualified FOM speedup. It still misses the all-state target. Its weights, initializer, step and solver were frozen before multiresolution evaluation on additional development cases.", ""]
    lines += wave_confirmation_lines(wf)
    lines += ["## Work still in progress", "",
              "The final nested Poisson correction family is queued or running. Heat, Burgers and reflective-wave experiments are complete and audited. The remaining accepted result will be added here, including any unsuccessful prefixes.", "",
              "## Reproduction and evidence", "",
              f"Heat job `{h['metadata']['job_id']}` contains the paired GPU measurements; scientific source and checkpoint hashes are in the linked owner panel. Independent coordinator checks cover each model's worst saved trajectory on every mesh.", "",
              f"- [Heat complete panel](../{h['run']}/analysis/summary.md)",
              f"- [Poisson fixed-capacity panel](../{ps['run']}/panel.json)",
              f"- [Poisson larger-bank panel](../{pc['run']}/panel.json)",
              f"- [Burgers training and solver panel](../{b['run']}/PANEL.json)",
              f"- [Burgers broader-coverage panel](../{bc['run']}/PANEL.json)",
              f"- [Wave geometry audit](../{w6['run']}/audit.json)",
              f"- [Wave follow-up audit](../{w7['run']}/audit.json)",
              f"- [Wave training audit](../{w8['run']}/audit.json)",
              f"- [Wave larger-head audit](../{w9['run']}/audit.json)",
              f"- [Wave correction-head audit](../{w10['run']}/audit.json)",
              f"- [Wave multiresolution confirmation audit](../{wf['run']}/audit.json)",
              "- [Normalized values, repetition arrays and source hashes](2026-09-11-accuracy-improvements-and-wave-speed.json)", "",
              "Run `reports/generate_accuracy_campaign.py` with the repository Python environment to rebuild. All source hashes and generator identity are embedded in the adjacent JSON. These are single-training-seed development studies on the recorded families; they do not establish broad PDE generalization or final paper performance.", "",
              "## Plain-language glossary", "",
              "- **Intervals / mesh:** subdivisions along each spatial axis; larger values request more output points.",
              "- **Bank / head / latent:** learned spatial functions / network choosing their coefficients / compressed coordinates solved online.",
              "- **POD / bank projection:** a span built from training-snapshot singular vectors / the closest unrestricted bank combination in the stated field norm. These are diagnostic controls, not deployed nonlinear networks.",
              "- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained to a nonlinear decoder.",
              "- **GPU / host ms:** blocked complete GPU input-to-output query time / the same heat invocation including input and output transfers.",
              "- **Relative error:** error magnitude divided by the specified reference magnitude. Heat uses the current true field at each time; the displayed wave screen uses initial physical scales.",
              "- **Median / worst:** middle case error / largest case error, with each case scored at its worst saved output time. Runtime uses the median of all retained repetitions.",
              "- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. A CG tolerance is its stopping threshold, not its measured field error.",
              "- **Stationarity:** sufficiently small gradient of the reduced solve objective. This does not itself guarantee physical accuracy.",
              "- **Tail emphasis:** training loss that assigns more influence to large reconstruction errors within a training batch.",
              "- **EQ / replay / coverage:** fitted quadrature weights approximating weak sums / preserving old decoded outputs during training / the range and number of training initial conditions.",
              "- **All-state / energy-state:** checking displacement, velocity and their combined energy norm / the norm combining velocity and spatial-gradient error.",
              "- **Guard / parity / refinement:** a numerical check with a more robust fallback / agreement with unchanged equations / agreement after reducing the integration step.",
              "- **Cholesky / tangent velocity:** a factorization for solving a positive-definite small matrix system / the decoder Jacobian multiplied by latent velocity.",
              "- **Outlier:** heat and Poisson repetition above one-and-a-half times its case median; Burgers repetition above twice its case median; wave repetition above twice its panel median. Counts and every duration are retained.",
              "- **Development / sealed final:** cases used in diagnosis and method selection / untouched cases reserved for the paper's later final evaluation.", ""]
    normalized["sources"] = SOURCES
    normalized["generator_sha256"] = sha(Path(__file__))
    REPORT.with_suffix(".json").write_text(json.dumps(normalized, indent=2)+"\n")
    REPORT.with_suffix(".md").write_text("\n".join(lines))
    print(REPORT.with_suffix(".md"))


if __name__ == "__main__":
    build()
