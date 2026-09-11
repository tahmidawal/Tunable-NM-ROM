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
    selection = read(run/"selection.json")
    cleanup = read(run/"cleanup.json")
    assert audit["passed"] and raw["complete"] and not raw["final_test_opened"]
    assert panel["source_result_sha256"] == SOURCES[str(run/"cluster/out/pilot/result.json")]
    for row in panel["panels"]:
        invocations = [r for r in raw["invocations"] if (r["intervals"], r["cohort"], r["method"], r["setting"])
                       == (row["intervals"], row["cohort"], row["method"], row["setting"])]
        times = [r["seconds"]["complete_device_query"]*1000 for r in invocations]
        np.testing.assert_allclose(times, row["timing_ms"], rtol=1e-14)
        assert np.isclose(np.median(times), row["median_gpu_ms"], rtol=1e-14)
    return dict(**panel, provenance=raw["provenance"], selection=selection,
                run=str(run), cleanup=cleanup)


def build():
    h, w6, w7 = heat(), wave("accel06"), wave("accel07")
    normalized = dict(status="Audited heat and wave screens; Poisson, Burgers and wave training/confirmation remain active.",
                      heat=h, wave_screens={"accel06": w6, "accel07": w7})
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
              "## Reflective waves: geometry and time-step screens", "",
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
    lines += ["Wave errors in this table use fixed initial physical scales. The energy-state error measures the error in displacement gradients and velocity; it is not energy-conservation drift. Current-relative displacement and velocity errors are separately retained in the JSON. DST is the same-grid semidiscrete reference, so its zero discrepancy is not zero continuum error.", "",
              "The follow-up also tested unrestricted bank evolution and nonlinear output projection. Those methods evolve a larger linear state and are labeled separately from the original nonlinear latent dynamics. Both projected-output variants missed the all-state target; the unrestricted linear control passed on the opened cases.", ""]
    rows = []
    for p in w7["panels"]:
        e = p["worst_initial_errors_percent"]
        rows.append([p["method"], p["setting"], f"{p['median_gpu_ms']:.6f}",
                     " / ".join(f"{e[k]:.6f}" for k in ("displacement", "velocity", "energy_state")),
                     "Pass" if p["all_state_5percent_pass"] else "Fail"])
    lines += table(["Follow-up method", "Step / CG tolerance", "GPU ms", "Worst u / v / energy-state error %", "All-state 5%"], rows)
    failed = [r for r in w7["time_refinement"] if not r["passed"]]
    for r in failed:
        lines += [f"Rejected time step: `{r['method']}` at {r['dt']} on `{r['case']}` has a {100*max(r['maxima'].values()):.6f}% half-step discrepancy. Its faster timing is diagnostic only.", ""]
    lines += ["The CG control was also allowed to use larger time steps; any FOM speed ratio must use a tested setting that passes its physical and solve criteria. The next wave stage trains matched field-only and field/energy/tangent-velocity heads, then confirms frozen settings across meshes and new development cases.", "",
              "## Work still in progress", "",
              "Poisson staged bank/head training and Burgers initial-field training plus stationarity-aware solving are not yet accepted in this report. Their completed audits will be added here, including unsuccessful arms. Wave head training and multiresolution confirmation are also outstanding.", "",
              "## Reproduction and evidence", "",
              f"Heat source `{h['metadata']['job_id']}` is the paired GPU job; scientific source and checkpoint hashes are in the linked owner panel. Independent coordinator checks cover each model's worst saved trajectory on every mesh.", "",
              f"- [Heat complete panel](../{h['run']}/analysis/summary.md)",
              f"- [Wave geometry audit](../{w6['run']}/audit.json)",
              f"- [Wave follow-up audit](../{w7['run']}/audit.json)",
              "- [Normalized values, repetition arrays and source hashes](2026-09-11-accuracy-improvements-and-wave-speed.json)", "",
              "Run `reports/generate_accuracy_campaign.py` with the repository Python environment to rebuild. All source hashes and generator identity are embedded in the adjacent JSON. These are single-training-seed development studies on the recorded families; they do not establish broad PDE generalization or final paper performance.", "",
              "## Plain-language glossary", "",
              "- **Intervals / mesh:** subdivisions along each spatial axis; larger values request more output points.",
              "- **Bank / head / latent:** learned spatial functions / network choosing their coefficients / compressed coordinates solved online.",
              "- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained to a nonlinear decoder.",
              "- **GPU / host ms:** blocked complete GPU input-to-output query time / the same heat invocation including input and output transfers.",
              "- **Relative error:** error magnitude divided by the specified reference magnitude. Heat uses the current true field at each time; the displayed wave screen uses initial physical scales.",
              "- **Median / worst:** middle case error / largest case error, with each case scored at its worst saved output time. Runtime uses the median of all retained repetitions.",
              "- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. A CG tolerance is its stopping threshold, not its measured field error.",
              "- **Stationarity:** sufficiently small gradient of the reduced solve objective. This does not itself guarantee physical accuracy.",
              "- **Tail emphasis:** training loss that assigns more influence to large reconstruction errors within a training batch.",
              "- **All-state / energy-state:** checking displacement, velocity and their combined energy norm / the norm combining velocity and spatial-gradient error.",
              "- **Guard / parity / refinement:** a numerical check with a more robust fallback / agreement with unchanged equations / agreement after reducing the integration step.",
              "- **Outlier:** heat repetition above one-and-a-half times its case median; wave repetition above twice its panel median. Counts and every duration are retained.",
              "- **Development / sealed final:** cases used in diagnosis and method selection / untouched cases reserved for the paper's later final evaluation.", ""]
    normalized["sources"] = SOURCES
    normalized["generator_sha256"] = sha(Path(__file__))
    REPORT.with_suffix(".json").write_text(json.dumps(normalized, indent=2)+"\n")
    REPORT.with_suffix(".md").write_text("\n".join(lines))
    print(REPORT.with_suffix(".md"))


if __name__ == "__main__":
    build()
