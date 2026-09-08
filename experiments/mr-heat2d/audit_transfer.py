"""Independent NumPy saved-field audit and generated transfer findings; no JAX."""
import argparse
import hashlib
import json
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from audit_runtime import metrics, restrict
from restore_transfer import sha256


def array_hash(a):
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()


def draws(seed, count, cfg):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(*cfg["center_range"], (count, 2)), rng.uniform(*cfg["width_range"], count), rng.uniform(*cfg["amplitude_range"], count)))


def audit(archive):
    out = archive/"outputs"; result = json.loads((out/"results.json").read_text())
    cfg, settings = result["config"], result["settings"]
    source = result["source_manifest"]
    for name, expected in source["sha256"].items():
        origin = source["provenance"][name]
        assert hashlib.sha256(subprocess.check_output(["git", "show", origin["commit"]+":"+origin["git_path"]])).hexdigest() == expected
        assert sha256(archive/name) == expected, name
    checked_files = 0
    for line in (archive/"ARCHIVE.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha256(archive/name) == expected, name
        checked_files += 1
    assert result["complete"] and result["verification"]["passed"]
    assert result["metadata"]["backend"] == "gpu" and result["metadata"]["x64"] and result["metadata"]["precision"] == "highest"
    log = (archive/f"job-{result['metadata']['job_id']}.log").read_text()
    assert "jax_backend=gpu" in log and "TRANSFER PILOT COMPLETE" in log
    assert not re.search(r"captured.*constant|Traceback|out of memory|RESOURCE_EXHAUSTED|No space left|failed call to cuInit", log, re.I)
    assert result["reference_evidence"]["rigorous_relative_bound"] is None
    assert not result["final_cohort_opened"] and not result["validation_in_training_or_lookup"]
    assert result["query_contract"] == settings["initial_output_policy"]
    expected_cases = []
    training = np.concatenate((draws(790710, 32, cfg), draws(790713, 128, cfg)))
    for cohort in settings["cohorts"]:
        for index, draw in enumerate(draws(cohort["seed"], cohort["count"], cfg)):
            case = result["cases"][len(expected_cases)]
            assert case["cohort"] == cohort["name"] and case["seed"] == cohort["seed"] and case["cohort_index"] == index
            np.testing.assert_array_equal(case["draw"], draw)
            assert not any(np.array_equal(draw, d) for d in training)
            expected_cases.append(case)
    assert len(result["cases"]) == len(expected_cases)
    for model in result["models"]:
        assert sha256(out/model["output_path"]) == model["sha256"]
        assert sha256(archive/(model["name"]+".pkl")) == model["sha256"]
        checkpoint = pickle.loads((out/model["output_path"]).read_bytes())
        assert checkpoint["config"] == cfg and checkpoint["codes"].shape == (960, cfg["k"])
    assert all(g["passed"] and g["exact_counter_reason_parity"] for g in result["parity_gates"])
    assert len(result["parity_gates"]) == len(expected_cases)*len(settings["requested_intervals"])*len(settings["checkpoints"])
    disagreement = 0.; fields_checked = set(); timed = 0; case_rows = []; reference_delta = []
    fom_kernel_disagreement = 0.; reference_kernel_disagreement = 0.
    def independent_evolution(u0, n, continuum):
        spectrum = dstn(u0, type=1, norm="ortho")
        modes = np.arange(1, n, dtype=np.float64)
        one = (np.pi*modes)**2 if continuum else 4*n*n*np.sin(np.pi*modes/(2*n))**2
        lam = one[:, None]+one[None, :]
        for t in cfg["times"][1:]:
            yield idstn(spectrum*np.exp(-cfg["diffusivity"]*t*lam), type=1, norm="ortho")
    def field(record):
        a = np.load(out/record["path"])["field"]
        assert array_hash(a) == record["sha256_array"]
        assert list(a.shape) == record["shape"] and a.dtype.str == record["dtype"] == "<f8"
        assert np.all(np.isfinite(a))
        fields_checked.add(record["path"])
        return a
    def check(saved, actual):
        nonlocal disagreement
        for key, values in actual.items():
            error = float(np.max(abs(np.asarray(saved[key])-values)))
            disagreement = max(disagreement, error)
            np.testing.assert_allclose(saved[key], values, rtol=1e-12, atol=1e-14)
        if "vanished_below_1e-3_initial" in saved:
            np.testing.assert_array_equal(saved["vanished_below_1e-3_initial"], actual["truth_norm_over_initial"] < 1e-3)
    local_index = {(r["intervals"], r["case"]): r for r in result["case_fields"]}
    for cid, reference in enumerate(result["reference_fields"]):
        fine, coarse = field(reference["fine"]), field(reference["coarse"])
        nf, nc = reference["fine_intervals"], reference["coarse_intervals"]
        for nref, trajectory in ((nf, fine), (nc, coarse)):
            for observed, expected in zip(trajectory[1:], independent_evolution(trajectory[0], nref, True)):
                error = float(np.linalg.norm(observed-expected)/np.linalg.norm(expected))
                reference_kernel_disagreement = max(reference_kernel_disagreement, error)
                assert error < 1e-12
        refined = metrics(coarse, restrict(fine, nf, nc), nc)
        check(reference["refinement"], refined)
        reference_delta.append(float(max(refined["relative_current"])))
        for n in settings["requested_intervals"]:
            local = local_index[n, cid]
            physical = field(local["physical"]); initial = field(local["initial"]); discrete = field(local["discrete"])
            np.testing.assert_array_equal(physical, restrict(fine, nf, n))
            np.testing.assert_array_equal(initial, physical[0]); np.testing.assert_array_equal(initial, discrete[0])
            # Independently regenerate the supplied analytic field from preserved draw.
            x = np.arange(1, n)/n; xx, yy = np.meshgrid(x, x, indexing="ij")
            cx, cy, width, amp = result["cases"][cid]["draw"]
            analytic = amp*16*xx*(1-xx)*yy*(1-yy)*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*width**2))
            np.testing.assert_allclose(initial, analytic, rtol=1e-14, atol=1e-15)
            observation = settings["observation_intervals"]
            common_ref = restrict(physical, n, observation)
            delta_full = metrics(restrict(coarse, nc, n), physical, n)
            delta_common = metrics(restrict(coarse, nc, observation), common_ref, observation)
            check(local["reference_refinement_full"], delta_full); check(local["reference_refinement_common"], delta_common)
            check(local["spatial_discrete_vs_physical"], metrics(discrete, physical, n))
            for row in (r for r in result["rows"] if r["case"] == cid and r["intervals"] == n):
                record = dict(intervals=n, case=cid, cohort=row["cohort"], method=row["method"], model=row["model"],
                    solver_intervals=row["solver_intervals"], repetitions=[], valid=True, stationary=True, nonstationary_initial=0, nonstationary_steps=0,
                    unused_initial_nonstationary=0, initial_attempts=[], step_attempts=[])
                assert len(row["repetitions"]) == settings["timing_repetitions"]
                stored = None; stored_path = None
                for number, rep in enumerate(row["repetitions"]):
                    assert rep["repetition"] == number
                    if rep["field"]["path"] != stored_path:
                        stored = field(rep["field"]); stored_path = rep["field"]["path"]
                    a = stored
                    assert array_hash(a) == rep["full_field_sha256"]
                    full = metrics(a, physical, n); common = metrics(restrict(a, n, observation), common_ref, observation)
                    check(rep["vs_physical_per_grid"], full); check(rep["vs_physical_common_grid"], common)
                    check(rep["vs_same_grid"], metrics(a, discrete, n))
                    phase = rep["phases"]
                    assert all(value >= 0 for value in phase.values())
                    assert sum(v for k, v in phase.items() if k != "query_seconds") <= phase["query_seconds"]+1e-8
                    valid = bool(np.all(np.diff(full["energy"]) <= 1e-12) and np.all(full["state_change_from_initial"][1:] > 1e-8))
                    if row["model"] == "fom":
                        np.testing.assert_array_equal(a[0], initial)
                        assert not rep["solver"]
                        if number == 0:
                            solver = row["solver_intervals"]; stride = n//solver
                            coarse_initial = initial[stride-1::stride, stride-1::stride]
                            for observed, expected in zip(a[1:], independent_evolution(coarse_initial, solver, False)):
                                if solver != n:
                                    positions = np.arange(1, n, dtype=np.float64)/stride
                                    lower = np.floor(positions).astype(int); w = positions-lower
                                    padded = np.pad(expected, 1)
                                    along_x = padded[lower, :]*(1-w)[:, None]+padded[lower+1, :]*w[:, None]
                                    expected = along_x[:, lower]*(1-w)[None, :]+along_x[:, lower+1]*w[None, :]
                                error = float(np.linalg.norm(observed-expected)/np.linalg.norm(expected))
                                fom_kernel_disagreement = max(fom_kernel_disagreement, error)
                                assert error < 1e-12
                    else:
                        info = np.asarray(rep["solver"]["initial_fits"]); steps = np.asarray(rep["solver"]["steps"])
                        best_index = int(np.argmin(info[:, 3])); selected = info[best_index]; tol = row["gradient_tolerance"]
                        initial_bad = int(selected[2] != 1 or selected[4] > tol)
                        steps_bad = int(np.sum((steps[:, 2] != 1)|(steps[:, 4] > tol)))
                        record["nonstationary_initial"] += initial_bad; record["nonstationary_steps"] += steps_bad
                        record["unused_initial_nonstationary"] += int(np.sum((info[:, 2] != 1)|(info[:, 4] > tol)))-initial_bad
                        record["initial_attempts"].append(int(np.sum(info[:, 0]))); record["step_attempts"].append(int(np.sum(steps[:, 0])))
                        record["stationary"] &= not (initial_bad or steps_bad)
                        valid &= not (initial_bad or steps_bad)
                    record["valid"] &= valid
                    record["repetitions"].append(dict(phases=phase, valid=valid,
                        full_error=float(max(full["relative_current"])), common_error=float(max(common["relative_current"])),
                        full_initial_error=float(full["relative_current"][0]), common_initial_error=float(common["relative_current"][0]),
                        full_initial_normalized_error=float(max(full["relative_initial"])), common_initial_normalized_error=float(max(common["relative_initial"])),
                        full_empirical_adjusted=float(max((full["relative_current"]+delta_full["relative_current"])/(1-delta_full["relative_current"]))),
                        common_empirical_adjusted=float(max((common["relative_current"]+delta_common["relative_current"])/(1-delta_common["relative_current"])))))
                    timed += 1
                record["query_median_seconds"] = float(np.median([r["phases"]["query_seconds"] for r in record["repetitions"]]))
                record["timing_outliers_over_3x_case_median"] = sum(r["phases"]["query_seconds"] > 3*record["query_median_seconds"] for r in record["repetitions"])
                case_rows.append(record)
        del fine, coarse
    np.testing.assert_allclose(result["reference_evidence"]["empirical_relative_delta"], max(reference_delta), rtol=0, atol=1e-14)
    assert max(reference_delta) < cfg["reference_uncertainty_budget"]
    assert timed == result["timed_invocations"] == settings["estimated_timed_calls"]
    groups, selections = [], []
    for cohort in [c["name"] for c in settings["cohorts"]]+["union"]:
        for n in settings["requested_intervals"]:
            rows = [r for r in case_rows if r["intervals"] == n and (cohort == "union" or r["cohort"] == cohort)]
            current = []
            for method in sorted({r["method"] for r in rows}):
                cases = [r for r in rows if r["method"] == method]
                g = dict(cohort=cohort, intervals=n, method=method, model=cases[0]["model"], cases=[r["case"] for r in cases],
                    cost_seconds=float(np.median([r["query_median_seconds"] for r in cases])), valid=all(r["valid"] for r in cases),
                    nonstationary_initial=sum(r["nonstationary_initial"] for r in cases), nonstationary_steps=sum(r["nonstationary_steps"] for r in cases),
                    timing_outliers=sum(r["timing_outliers_over_3x_case_median"] for r in cases),
                    case_times={r["case"]: r["query_median_seconds"] for r in cases})
                for key in ("full_error", "common_error", "full_initial_error", "common_initial_error", "full_initial_normalized_error", "common_initial_normalized_error", "full_empirical_adjusted", "common_empirical_adjusted"):
                    g[key] = max(rep[key] for case in cases for rep in case["repetitions"])
                for phase in ("input_restriction_transfer_seconds", "device_solve_readout_seconds", "full_host_output_seconds"):
                    g[phase] = float(np.median([np.median([r["phases"][phase] for r in case["repetitions"]]) for case in cases]))
                groups.append(g); current.append(g)
            for norm in ("full", "common"):
                for target in cfg["accuracy_targets"]:
                    eligible = [g for g in current if g["valid"] and g[norm+"_empirical_adjusted"] <= target]
                    foms = [g for g in eligible if g["model"] == "fom"]
                    roms = [g for g in eligible if g["model"] != "fom"]
                    f = min(foms, key=lambda g: g["cost_seconds"]) if foms else None
                    r = min(roms, key=lambda g: g["cost_seconds"]) if roms else None
                    selections.append(dict(cohort=cohort, intervals=n, norm=norm, target=target,
                        fom=f["method"] if f else None, rom=r["method"] if r else None,
                        fom_seconds=f["cost_seconds"] if f else None, rom_seconds=r["cost_seconds"] if r else None,
                        paired_fom_over_rom=float(np.median([f["case_times"][i]/r["case_times"][i] for i in f["cases"]])) if f and r else None))
    return dict(passed=True, results_sha256=sha256(out/"results.json"), archive_files_checked=checked_files,
                full_fields_checked=len(fields_checked), timed_invocations=timed, max_metric_disagreement=disagreement,
                independent_fom_kernel_relative_disagreement=fom_kernel_disagreement,
                independent_spectral_reference_relative_disagreement=reference_kernel_disagreement,
                reference_empirical_delta=max(reference_delta), rigorous_relative_bound=None,
                nonstationary_initial=sum(r["nonstationary_initial"] for r in case_rows), nonstationary_steps=sum(r["nonstationary_steps"] for r in case_rows),
                case_rows=case_rows, groups=groups, selections=selections)


def report(result, audited):
    union = [g for g in audited["groups"] if g["cohort"] == "union" and g["model"] != "fom"]
    qualified = [s for s in audited["selections"] if s["cohort"] == "union" and s["norm"] == "full" and s["target"] == .05 and s["paired_fom_over_rom"] is not None]
    advantage = sum(s["paired_fom_over_rom"] > 1 for s in qualified)
    text = ["# Frozen heat transfer and efficient full-output cost", "",
        "These bounded development results are checked against preserved full fields. They use both frozen expanded-coverage heads and unchanged initializer libraries on original and fresh inputs from the restricted single-bump family; final confirmation remains unopened.", "",
        f"Across the union cohort and all requested meshes, the largest head current-relative physical error was {max(g['full_error'] for g in union)*100:.6f}%. At the empirically adjusted full-grid 5% target, a head qualified at {len(qualified)} meshes and beat the selected efficient FOM at {advantage} of those meshes. This describes this restricted development family only.", "",
        f"The native independent NumPy audit checked {audited['timed_invocations']} invocations and {audited['full_fields_checked']} field files; maximum metric disagreement was {audited['max_metric_disagreement']:.12g}. The empirical spectral refinement discrepancy was {audited['reference_empirical_delta']:.12g}. This is observed convergence evidence, with no rigorous physical error certificate.", "",
        f"Independent SciPy propagation and NumPy physical interpolation reproduce the stored FOM fields to relative discrepancy {audited['independent_fom_kernel_relative_disagreement']:.12g}, and the continuum-spectral reference trajectories to {audited['independent_spectral_reference_relative_disagreement']:.12g}.", "",
        "[Accuracy and complete-query figure](heat-transfer.png) · [Input, device and output cost figure](heat-transfer-components.png)", "",
        "Every FOM returns the exact supplied initial field. Host coarse restriction, GPU propagation, physically aligned interpolation and complete contiguous host output construction are charged. ROM outputs include its actual fitted initial state; full input projection, nonlinear fitting, evolution and full readout are charged. Both use the same requested host output grids and times.", "",
        f"The run took {result['elapsed_seconds']/60:.6f} minutes. The audit found {audited['nonstationary_initial']} selected nonstationary initial fits and {audited['nonstationary_steps']} nonstationary evolution steps across all repetitions. Failures remain in raw records and cannot qualify for target selection.", "",
        f"There were {sum(r['timing_outliers_over_3x_case_median'] for r in audited['case_rows'])} timing repetitions above three times their own case/configuration median. Every repetition remains in the archive and in timing summaries.", "",
        "Costs below are medians of per-case timing medians. All errors are maxima over every case, repetition and output time in the named cohort. Each head remains separately visible; this experiment changes neither its representation nor its training coverage.", "",
        "| Cohort | Output intervals | Method | Query ms | Full current error % | Common current error % | Initial full error % | Invalid cases |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for g in audited["groups"]:
        bad = sum(not c["valid"] for c in audited["case_rows"] if c["intervals"] == g["intervals"] and c["method"] == g["method"] and c["case"] in g["cases"])
        text.append(f"| {g['cohort']} | {g['intervals']} | {g['method']} | {g['cost_seconds']*1e3:.6f} | {g['full_error']*100:.6f} | {g['common_error']*100:.6f} | {g['full_initial_error']*100:.6f} | {bad} |")
    text += ["", r"Selection uses every repetition's validity and empirical adjusted physical error $(e+\delta)/(1-\delta)$, where $e$ is observed current-relative error and $\delta$ is the paired reference-refinement discrepancy in the same norm. Full requested-grid and common-grid selections remain separate. The ratio is the median of per-case FOM/ROM ratios of case timing medians; values above one would favor ROM. A missing ROM means no head qualified, not zero runtime.", "", "| Cohort | Output | Norm | Target % | Selected FOM | Selected ROM | FOM ms | ROM ms | Paired FOM/ROM |", "|---|---:|---|---:|---|---|---:|---:|---:|"]
    number = lambda value, scale=1: "—" if value is None else f"{value*scale:.6f}"
    for s in audited["selections"]:
        text.append(f"| {s['cohort']} | {s['intervals']} | {s['norm']} | {s['target']*100:g} | {s['fom'] or '—'} | {s['rom'] or '—'} | {number(s['fom_seconds'], 1000)} | {number(s['rom_seconds'], 1000)} | {number(s['paired_fom_over_rom'])} |")
    text += ["", "The coordinate bank, full-input QR projection and weak operator were rebuilt at every requested mesh. The following setup durations are observed wall times including first compilation and host work; they are outside paired online query cost and do not include the separate operator-verification work. Bank plus projection bytes are array storage, not peak device allocation.", "",
        "| Output intervals | Interior unknowns | Bank rank | Bank + projection MiB | Bank evaluation s | QR/operator s | Operator discrepancy | Jacobian discrepancy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for setup in result["setups"]:
        text.append(f"| {setup['intervals']} | {setup['interior_unknowns']} | {setup['rank']} | {(setup['bank_bytes']+setup['initialization_projection_bytes'])/1024**2:.6f} | {setup['bank_evaluation_seconds']:.6f} | {setup['operator_qr_seconds']:.6f} | {setup['exact_weak_operator_relative_error']:.6g} | {setup['exact_weak_jacobian_relative_error']:.6g} |")
    text += ["", "The reference and supplied initial field are preserved at complete requested resolution. Same-grid semidiscrete discrepancies are separate from discrete-versus-spectral spatial errors in the raw results. ROM-versus-discrete error includes representation, nonlinear solve and Crank–Nicolson time error; it is not a pure solver error. Prior timestep validation is retained, but this transfer study does not independently separate those three contributions. Current and initial normalization, absolute errors, state advancement and energy/decay diagnostics remain in each raw invocation.", "",
        "Mesh setup, compilation and reference generation are outside online query cost. GPU capacity figures in the planning config are live-array estimates; peak device allocation was not profiled, and the configured JAX allocator reserves a fraction of device memory. Detailed Slurm host-memory accounting is preserved separately. All checkpoints and code libraries are byte-identical to the declared inputs. Large extracted fields can be restored from the checked tracked archive chunks using `restore_transfer.py`; metadata and generated audit are also tracked directly.", "",
        "## Plain-language glossary", "",
        "- **Cohort / original / fresh / union:** a group of physical inputs / repeated earlier development inputs / new independently seeded development inputs / both groups together.",
        "- **Output intervals / common grid:** cells per axis on the fully returned grid / fixed shared physical observation nodes for mesh comparison.",
        "- **Method / head / FOM / ROM:** solver configuration / frozen latent-to-coefficient neural mapping / full-order heat solver / reduced nonlinear solver.",
        "- **DST / semidiscrete / spectral:** sine-transform propagation / exact time solution of fixed-grid equations / continuum sine-series physical reference.",
        "- **Query ms / case median / paired FOM/ROM:** charged full-input-to-full-output milliseconds / middle repeated time for one input / median across inputs of full-model time divided by reduced-model time.",
        "- **Full current error / common current error / initial full error:** discrepancy divided by the current reference norm on all returned nodes / the same on shared observation nodes / actual fitted initial discrepancy on all returned nodes.",
        "- **Initial-normalized / absolute / decay:** discrepancy divided by the initial truth norm / area-weighted unnormalized discrepancy / the reference field becoming smaller in time.",
        "- **Invalid cases / stationarity / multistart:** inputs with any failed validity check / the configured small-gradient stopping condition / fitting from both a nearby library code and the mean code.",
        "- **Empirical adjustment / target / selected envelope:** reference allowance inferred from observed refinement / required physical error / cheapest configuration satisfying the same cohort and target.",
        "- **QR / weak operator / Crank–Nicolson:** exact full-field least-squares compression / heat equations tested against smooth functions / fixed second-order time formula.",
        "- **Interior unknowns / bank rank / MiB / operator and Jacobian discrepancies:** scalar field values excluding known boundaries / number of independent spatial bank columns / binary megabytes of array storage / relative differences from the explicit discrete stencil and its derivative.",
        "- **Frozen / checkpoint / code library / final:** unchanged model parameters / complete saved model / unchanged training starting codes / reserved independent evaluation not used here.", ""]
    return "\n".join(text)


def compact_summary(audited):
    summary = dict(heads=[], union_full_5pct=[s for s in audited["selections"] if s["cohort"] == "union" and s["norm"] == "full" and s["target"] == .05])
    names = sorted({r["method"] for r in audited["groups"] if r["model"] != "fom"})
    for name in names:
        rows = [r for r in audited["groups"] if r["cohort"] == "union" and r["method"] == name]
        summary["heads"].append(dict(method=name, worst_full=max(r["full_error"] for r in rows),
            worst_common=max(r["common_error"] for r in rows), worst_initial=max(r["full_initial_error"] for r in rows), all_valid=all(r["valid"] for r in rows)))
    summary["native_audit"] = {k: v for k, v in audited.items() if k not in ("case_rows", "groups", "selections")}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("archive", type=Path); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    audited = audit(args.archive)
    (args.out/"audit.json").write_text(json.dumps(audited, indent=2, allow_nan=False)+"\n")
    result = json.loads((args.archive/"outputs/results.json").read_text())
    (args.out/"HEAT-TRANSFER-NOTES.md").write_text(report(result, audited))
    (args.out/"SUMMARY.json").write_text(json.dumps(compact_summary(audited), indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in audited.items() if k not in ("case_rows", "groups", "selections")}))
