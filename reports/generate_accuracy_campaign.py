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
    return dict(**panel, provenance=raw["provenance"], selection=selection,
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


def build():
    h, w6, w7, w8 = heat(), wave("accel06"), wave("accel07"), wave("accel08")
    w9, w10 = wave("accel09"), wave("accel10")
    w10_coordinator = read("reports/2026-09-11-wave-correction-screen.coordinator-audit.json")
    assert w10_coordinator["passed"] and w10_coordinator["source_result_sha256"] == w10["source_result_sha256"]
    ps = poisson("staged_accuracy08", "reports/2026-09-11-poisson-staged-accuracy.coordinator-audit.json")
    normalized = dict(status="Audited heat, fixed-capacity Poisson and wave screens/training; follow-up capacity tests, Burgers and wave confirmation remain active.",
                      heat=h, poisson_staged=ps, wave_screens={"accel06": w6, "accel07": w7, "accel08": w8, "accel09": w9, "accel10": w10},
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
    lines += [f"The nested head reaches {nested['worst_initial_errors_percent']['energy_state']:.6f}% worst energy-state error and {nested['median_gpu_ms']:.6f} GPU ms on the opened screen. Its {nested_original['median_gpu_ms']/nested['median_gpu_ms']:.6f}× acceleration is relative to the same-job original ROM, not a qualified FOM speedup. It still misses the all-state target. Its weights, initializer, step and solver are frozen before multiresolution evaluation on additional development cases.", "",
              "## Work still in progress", "",
              "Poisson bank-capacity training and Burgers initial-field training plus stationarity-aware solving are still underway. Their completed audits will be added here, including unsuccessful arms. Frozen wave multiresolution confirmation is also outstanding.", "",
              "## Reproduction and evidence", "",
              f"Heat job `{h['metadata']['job_id']}` contains the paired GPU measurements; scientific source and checkpoint hashes are in the linked owner panel. Independent coordinator checks cover each model's worst saved trajectory on every mesh.", "",
              f"- [Heat complete panel](../{h['run']}/analysis/summary.md)",
              f"- [Poisson fixed-capacity panel](../{ps['run']}/panel.json)",
              f"- [Wave geometry audit](../{w6['run']}/audit.json)",
              f"- [Wave follow-up audit](../{w7['run']}/audit.json)",
              f"- [Wave training audit](../{w8['run']}/audit.json)",
              f"- [Wave larger-head audit](../{w9['run']}/audit.json)",
              f"- [Wave correction-head audit](../{w10['run']}/audit.json)",
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
              "- **All-state / energy-state:** checking displacement, velocity and their combined energy norm / the norm combining velocity and spatial-gradient error.",
              "- **Guard / parity / refinement:** a numerical check with a more robust fallback / agreement with unchanged equations / agreement after reducing the integration step.",
              "- **Cholesky / tangent velocity:** a factorization for solving a positive-definite small matrix system / the decoder Jacobian multiplied by latent velocity.",
              "- **Outlier:** heat repetition above one-and-a-half times its case median; wave repetition above twice its panel median. Counts and every duration are retained.",
              "- **Development / sealed final:** cases used in diagnosis and method selection / untouched cases reserved for the paper's later final evaluation.", ""]
    normalized["sources"] = SOURCES
    normalized["generator_sha256"] = sha(Path(__file__))
    REPORT.with_suffix(".json").write_text(json.dumps(normalized, indent=2)+"\n")
    REPORT.with_suffix(".md").write_text("\n".join(lines))
    print(REPORT.with_suffix(".md"))


if __name__ == "__main__":
    build()
