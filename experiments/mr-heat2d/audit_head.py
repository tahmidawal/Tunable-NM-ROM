"""Independent NumPy artifact audit and generated fixed-bank findings."""
import argparse
import hashlib
import json
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np
from audit_runtime import metrics, restrict


def same_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same_tree(x, y)
    else:
        np.testing.assert_array_equal(a, b)


def audit(archive):
    out = archive/"outputs"
    r = json.loads((out/"results.json").read_text())
    cfg, settings = r["config"], r["settings"]
    source = r["source_manifest"]
    expected_models = {"frozen"} | {f"{cohort}_seed{seed}" for cohort in ("original", "expanded") for seed in settings["minibatch_seeds"]}
    assert {row["name"] for row in r["models"]} == expected_models
    assert {row["model"] for row in r["reconstruction"]} == expected_models
    for path, digest in source["sha256"].items():
        spec = f"{source['checkpoint_commit']}:{source['checkpoint_git_path']}" if path == "checkpoint.pkl" else f"{source['source_commit']}:{path}"
        assert hashlib.sha256(subprocess.check_output(["git", "show", spec])).hexdigest() == digest
        assert hashlib.sha256((archive/path).read_bytes()).hexdigest() == digest
    nfiles = 0
    for entry in (archive/"ARCHIVE.sha256").read_text().splitlines():
        digest, name = entry.split("  ", 1)
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest() == digest
        nfiles += 1
    assert r["complete"] and r["verification"]["passed"]
    assert r["metadata"]["backend"] == "gpu" and r["metadata"]["x64"] and r["metadata"]["precision"] == "highest"
    log = (archive/f"job-{r['metadata']['job_id']}.log").read_text()
    assert "jax_backend=gpu" in log and "HEAD PILOT COMPLETE" in log
    assert not re.search(r"captured.*constant|Traceback|out of memory|RESOURCE_EXHAUSTED|No space left|failed call to cuInit", log, re.I)
    base = pickle.loads((archive/"checkpoint.pkl").read_bytes())
    models = {}
    for row in r["models"]:
        blob = (out/row["checkpoint_path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row["checkpoint_sha256"]
        model = pickle.loads(blob); models[row["name"]] = model
        for key in ("B", "g", "out_scale"): same_tree(base["params"][key], model["params"][key])
        for key in ("h", "h_lin"):
            a, b = base["params"][key], model["params"][key]
            if key == "h":
                assert [[x.shape for x in layer] for layer in a] == [[x.shape for x in layer] for layer in b]
            else: assert a.shape == b.shape
        if row["name"] == "frozen": same_tree(base["params"], model["params"])
        else:
            assert row["details"]["sampled_snapshots"] == settings["updates"]*settings["batch_size"]
            assert len(model["codes"]) == row["details"]["trajectories"]*len(cfg["times"])
    original, added, val = [np.asarray(r["cohorts"][key]) for key in ("original_train", "additional_train", "validation")]
    assert not any(np.array_equal(a, b) for a in np.concatenate((original, added)) for b in val)
    assert r["cohorts"]["validation_in_training_or_lookup"] is False and r["cohorts"]["final_cohort_opened"] is False
    compression = np.load(out/"training_compression.npz")
    count = int(compression["original_snapshot_count"])
    idx = compression["new_code_nearest_original_snapshot"]
    assert min(idx) >= 0 and max(idx) < count
    np.testing.assert_array_equal(compression["expanded_codes"][:count], base["codes"])
    np.testing.assert_array_equal(compression["expanded_codes"][count:], base["codes"][idx])
    cache = {}
    def field(item):
        path = item["path"]
        if path not in cache:
            a = np.load(out/path)["field"]
            digest = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
            assert digest == item["sha256_array"] and list(a.shape) == item["shape"] and a.dtype.str == item["dtype"]
            assert a.dtype == np.float64 and np.all(np.isfinite(a))
            cache[path] = a
        return cache[path]
    references = {row["case"]: (row["intervals"], field(row["field"])) for row in r["reference_fields"]}
    truth = field(r["bank_projection"]["truth"])
    bank_projection = field(r["bank_projection"]["reconstructed"])
    assert r["bank_projection"]["rank"] == cfg["r"]
    mismatch = 0.
    def check(saved, predicted, reference, n):
        nonlocal mismatch
        for key, actual in metrics(predicted, reference, n).items():
            expected = np.asarray(saved[key]); mismatch = max(mismatch, float(max(abs(expected-actual))))
            np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-14)
    reconstruction = []
    for row in r["reconstruction"]:
        fields = field(row["reconstructed"])
        info = np.asarray(row["fits"]); selected = np.asarray(row["selected_starts"])
        np.testing.assert_array_equal(selected, np.argmin(info[:, :, 3], axis=1))
        best = info[np.arange(len(info)), selected]
        lookup = np.asarray(row["nearest_training_snapshot_indices"])
        assert min(lookup.ravel()) >= 0 and max(lookup.ravel()) < len(models[row["model"]]["codes"])
        errors, later, bank_errors = [], [], []
        for record in row["cases"]:
            case = record["case"]
            check(record["vs_same_grid"], fields[case], truth[case], row["intervals"])
            check(record["bank_projection"], bank_projection[case], truth[case], row["intervals"])
            nf, ref = references[case]
            check(record["vs_physical"], fields[case], restrict(ref, nf, row["intervals"]), row["intervals"])
            errors.append(record["vs_same_grid"]["relative_current"][0])
            later.append(max(record["vs_same_grid"]["relative_current"][1:]))
            bank_errors.append(max(record["bank_projection"]["relative_current"]))
        initial_info = best[::len(cfg["times"])]
        initial_bad = int(np.sum((initial_info[:, 4] > settings["fit_gradient_tolerance"]) | (initial_info[:, 2] != 1)))
        gate = row["model"] != "frozen" and max(errors) <= settings["rollout_gate"]["initial_worst_max"] and initial_bad == 0
        assert gate == row["rollout_gate_passed"]
        reconstruction.append(dict(model=row["model"], initial_errors=errors, later_time_max_errors=later,
            initial_mean=float(np.mean(errors)), initial_median=float(np.median(errors)), initial_worst=max(errors),
            initial_cases_above005=sum(e>.05 for e in errors), initial_nonstationary=initial_bad,
            later_median=float(np.median(later)), later_worst=max(later), bank_worst=max(bank_errors),
            all_selected_nonstationary=int(np.sum(best[:, 4] > settings["fit_gradient_tolerance"])),
            unused_start_nonstationary=int(np.sum(info[:, :, 4] > settings["fit_gradient_tolerance"])-np.sum(best[:, 4] > settings["fit_gradient_tolerance"])),
            rollout_gate=bool(gate)))
    qualified = [row["model"] for row in reconstruction if row["rollout_gate"]]
    assert qualified == r["qualified_refinements"] and bool(qualified) == r["rollout_performed"]
    groups = {}; index = {(row["intervals"], row["case"], row["method"]): row for row in r["rows"]}
    cases = {(row["intervals"], row["case"]): row for row in r["case_fields"]}
    invocations = 0; delta = r["reference_evidence"]["empirical_relative_delta"]
    assert 0 <= delta < 1 and r["reference_evidence"]["rigorous_relative_bound"] is None
    if qualified:
        methods = {"fom_dst_exact_time"} | {f"{name}_gtol{tol:g}" for name in ["frozen"]+qualified for tol in settings["rollout_gradient_tolerances"]}
        assert set(index) == {(n, case, name) for n in settings["rollout_intervals"] for case in range(cfg["n_validation"]) for name in methods}
    for row in r["rows"]:
        assert [rep["repetition"] for rep in row["repetitions"]] == list(range(settings["timing_repetitions"]))
        n, case, label = row["intervals"], row["case"], row["method"]
        local = cases[n, case]; ref = field(local["physical"]); discrete = field(local["discrete"])
        np.testing.assert_array_equal(field(local["initial"]), discrete[0])
        nf, fine = references[case]; np.testing.assert_array_equal(ref, restrict(fine, nf, n))
        common_ref = restrict(ref, n, settings["observation_intervals"])
        group = groups.setdefault((n, label), dict(intervals=n, method=label, model=row["model"], tolerance=row["gradient_tolerance"], cases=[]))
        records = []
        for rep in row["repetitions"]:
            fields = field(rep["field"]); invocations += 1
            check(rep["vs_same_grid"], fields, discrete, n); check(rep["vs_physical_per_grid"], fields, ref, n)
            check(rep["vs_physical_common_grid"], restrict(fields, n, settings["observation_intervals"]), common_ref, settings["observation_intervals"])
            phases = rep["phases"]; assert all(v >= 0 for v in phases.values())
            assert sum(v for k, v in phases.items() if k != "query_seconds") <= phases["query_seconds"]+1e-8
            initial_bad = step_bad = budget = 0
            if rep["solver"]:
                ii = np.asarray(rep["solver"]["initial_fits"]); best = ii[np.argmin(ii[:, 3])]
                si = np.asarray(rep["solver"]["steps"]); tol = row["gradient_tolerance"]
                initial_bad = int(best[4] > tol or best[2] != 1)
                step_bad = int(np.sum((si[:, 4] > tol) | (si[:, 2] != 1)))
                budget = int(best[2] == 0)+int(np.sum(si[:, 2] == 0))
            error = max(rep["vs_physical_common_grid"]["relative_current"])
            records.append(dict(repetition=rep["repetition"], seconds=phases["query_seconds"], error=error,
                                empirical_adjusted_error=(error+delta)/(1-delta), rigorous_physical_bound=None,
                                initial_bad=initial_bad, step_bad=step_bad, budget_exits=budget,
                                valid=not(initial_bad or step_bad or budget)))
        group["cases"].append(dict(case=case, repetitions=records))
    for group in groups.values():
        errors = [max(rep["error"] for rep in case["repetitions"]) for case in group["cases"]]
        times = [np.median([rep["seconds"] for rep in case["repetitions"]]) for case in group["cases"]]
        ratios = []
        for case in group["cases"]:
            fom = index[group["intervals"], case["case"], "fom_dst_exact_time"]["repetitions"]
            ratios.append(np.median([f["phases"]["query_seconds"]/rep["seconds"] for f, rep in zip(fom, case["repetitions"])]))
        valid = all(rep["valid"] for case in group["cases"] for rep in case["repetitions"])
        parity = all(g["passed"] for g in r.get("parity_gates", []) if g["intervals"] == group["intervals"] and g["method"] == group["method"])
        if group["model"] != "fom":
            assert sum(g["intervals"] == group["intervals"] and g["method"] == group["method"] for g in r["parity_gates"]) == cfg["n_validation"]
        group.update(error_median=float(np.median(errors)), error_worst=max(errors), cases_above005=sum(e>.05 for e in errors),
                     median_query_seconds=float(np.median(times)), primary_paired_fom_ratio=float(np.median(ratios)),
                     all_solver_valid=valid, all_compilation_parity=parity,
                     empirical_targets=[t for t in cfg["accuracy_targets"] if valid and parity and (max(errors)+delta)/(1-delta) <= t and delta <= t/10],
                     rigorous_targets=None)
    return r, dict(passed=True, source_commit=source["source_commit"], archive_files=nfiles, unique_arrays=len(cache),
                   invocations=invocations, maximum_metric_disagreement=mismatch, bank_rank=r["bank_projection"]["rank"],
                   frozen_spatial_arrays_equal_all_checkpoints=True, validation_leakage_detected=False,
                   empirical_reference_delta=delta, rigorous_reference_bound=None,
                   reconstruction=reconstruction, rollout_groups=list(groups.values()))


def generate(r, a, path):
    f = lambda x: f"{x:.6g}"
    originals = [row for row in a["reconstruction"] if row["model"].startswith("original_")]
    expanded = [row for row in a["reconstruction"] if row["model"].startswith("expanded_")]
    lines = ["# Heat head refinement: additional optimization versus training coverage", "",
             "These provisional development results hold the spatial bank and neural architecture "
             "fixed while refining the coefficient head and training codes. Every scheduled "
             "endpoint and failed reconstruction/rollout gate is retained; this is not final "
             "confirmation or broader multi-bump heat coverage.", "",
             f"Generated from `{path}`. Source `{a['source_commit']}`, GPU job "
             f"`{r['metadata']['job_id']}`, `{r['metadata']['gpu']}`. Native audit verified "
             f"{a['archive_files']} files and {a['unique_arrays']} arrays, with maximum recomputed "
             f"metric difference {f(a['maximum_metric_disagreement'])}. Frozen spatial parameters "
             f"match exactly across every checkpoint; bank rank remains {a['bank_rank']}.", "",
             f"Expanded-coverage worst initial errors are {[f(row['initial_worst']) for row in expanded]}, "
             f"compared with {[f(row['initial_worst']) for row in originals]} after matched "
             "refinement on the original cohort. The spatial bank and head architecture did "
             "not change. [Accuracy and complete-query figure](figures/heat-head.svg).", "",
             "## Controlled training and offline cost", "",
             f"Each refinement uses {r['settings']['updates']} updates and minibatches of "
             f"{r['settings']['batch_size']} snapshots. The two paired sampling seeds are "
             f"{r['settings']['minibatch_seeds']}; additional training uses seed "
             f"{r['settings']['additional_training_seed']}. Validation is unchanged and absent "
             "from the loss and training-code initialization library. The final cohort stays unopened.", "",
             "| Model | Training trajectories | Observed refinement wall s | Final training relative MSE |",
             "|---|---|---|---|"]
    for row in r["models"]:
        d = row["details"]
        lines.append(f"| {row['name']} | {d['trajectories']} | {f(d['seconds']) if 'seconds' in d else '—'} | {f(d['final_full_relative_mse']) if 'final_full_relative_mse' in d else '—'} |")
    lines += ["", "Refinement seconds are observed wall durations including first-call compilation "
              "and host work. No dedicated warm-GPU timing block preceded each training arm; "
              "these are offline costs, not a paired steady-state training-speed comparison.", "",
              "The QR-compressed objective equals full-field relative reconstruction loss, "
              "including the constant error outside the bank. A GPU test checks both loss and "
              "gradient parity. More training on the original cohort is the control for "
              "additional optimization; improvement over the frozen head alone does not prove "
              "a benefit from added coverage.", "",
              "## Strict reconstruction and rollout gate", "",
              "Initial errors are separate from time-maximum errors after the initial state. "
              "The same nearest-training-code/mean-code multistart fitting budget is applied "
              "to every model. All starts and selected gradients/stops are preserved.", "",
              "| Model | Initial error median / worst | Later error median / worst | Initial cases above 5% | Initial / all-time selected nonstationary fits | Rollout gate |",
              "|---|---|---|---|---|---|"]
    for row in a["reconstruction"]:
        lines.append(f"| {row['model']} | {f(row['initial_median'])} / {f(row['initial_worst'])} | {f(row['later_median'])} / {f(row['later_worst'])} | {row['initial_cases_above005']} | {row['initial_nonstationary']} / {row['all_selected_nonstationary']} | {row['rollout_gate']} |")
    lines += ["", f"The unchanged unrestricted bank's worst reconstruction error is "
              f"{f(a['reconstruction'][0]['bank_worst'])}. The predeclared rollout gate requires "
              f"every selected initial fit to be stationary and below "
              f"{r['settings']['rollout_gate']['initial_worst_max']} relative error. This gate "
              "does not certify autonomous dynamics. A stationary fit is not proof of a global minimum.", ""]
    if r["rollout_performed"]:
        lines += ["## Paired complete-query results", "",
                  "Every cost and error comes from the same saved invocation. Full host input, "
                  "online initial fitting, evolution, requested dense fields and host output "
                  "are charged. The frozen original head and direct DST FOM are timed in this "
                  "same job; no cross-job raw times are compared. Both solver meshes use one "
                  "shared observation grid. Query times are median case-medians; ratios are "
                  "medians across cases of median paired-repetition FOM/ROM ratios.", "",
                  "| Intervals | Method | Query ms | FOM / method | Current error median / worst | All solver / parity checks pass | Empirical targets |",
                  "|---|---|---|---|---|---|---|"]
        for row in a["rollout_groups"]:
            lines.append(f"| {row['intervals']} | {row['method']} | {f(1000*row['median_query_seconds'])} | {f(row['primary_paired_fom_ratio'])} | {f(row['error_median'])} / {f(row['error_worst'])} | {row['all_solver_valid']} / {row['all_compilation_parity']} | {row['empirical_targets']} |")
    else:
        lines += ["No refined arm passed the declared reconstruction gate. Consequently no new "
                  "rollout/timing claim is made from this job; all failed arms and checkpoints remain saved."]
    lines += ["", f"Physical-reference refinement evidence is empirical, with delta "
              f"{f(a['empirical_reference_delta'])}. Eligibility uses "
              r"$(e+\delta)/(1-\delta)$ and the reference-budget check; rigorous physical bounds "
              "remain null. These are development findings, not a mathematical continuum guarantee.", "",
              "## Plain-language glossary", "",
              "- **Bank / head / code:** frozen spatial functions / nonlinear coefficient map / "
              "compressed coordinate fitted to a training snapshot.",
              "- **Original / expanded / frozen:** original training cohort / added independent "
              "training draws / unchanged initial checkpoint.",
              "- **QR / MSE / rank:** exact orthonormal field compression / mean squared error / "
              "number of independent bank directions.",
              "- **Median / worst / cases above:** middle cohort error / largest cohort error / "
              "count missing the stated threshold.",
              "- **Multistart / stationary / rollout gate:** fitting from several training-derived "
              "starts / sufficiently small gradient / predeclared condition for testing evolution.",
              "- **FOM / ROM / DST:** full-grid solver / reduced solver / fast sine transform.",
              "- **Query ms / paired ratio / parity:** complete input-to-output milliseconds / "
              "same-case repetition cost ratio / matching compiled and modular fields and counters.",
              "- **Current error / empirical target / rigorous bound:** error divided by current "
              "reference magnitude / target supported by refinement evidence and solver checks / "
              "proved physical error limit, unavailable here.",
              "- **Offline / validation / final cohort:** work before online queries / development "
              "evaluation cases / untouched independent confirmation cases.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("archive", type=Path)
    p.add_argument("--audit", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(); result, review = audit(args.archive)
    args.audit.write_text(json.dumps(review, indent=2)+"\n")
    args.report.write_text(generate(result, review, args.archive/"outputs/results.json"))
    print(json.dumps({k: v for k, v in review.items() if k not in ("reconstruction", "rollout_groups")}, indent=2))
