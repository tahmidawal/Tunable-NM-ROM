"""Frozen-choice confirmation on newly drawn physical cases; no selector is called."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import run_paired as R
from paired_measurement import Case, block, eligible, output_metrics, summarize, write_json

np, jax, jnp, F, SR, BI = R.np, R.jax, R.jnp, R.F, R.SR, R.BI
HERE = Path(__file__).resolve().parent


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def drift(summary, before, after, name):
    ratios = []
    for case in summary[name]["cases"].values():
        x = case["repetitions_ms"]
        if len(x) != 5:
            return None
        a, b = np.median(x[:2]), np.median(x[-2:])
        ratios.append(float(max(a / b, b / a)))
    if not ratios or not before[name]["complete"] or not after[name]["complete"]:
        return None
    a, b = before[name]["median_case_ms"], after[name]["median_case_ms"]
    return {"worst_case_early_late": max(ratios),
            "sentinel_start_end": max(a / b, b / a)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert cfg["eval_seed"] == 2026111901 and cfg["case_ids"] == list(range(16))
    assert cfg["eval_seed"] not in cfg["blocked_existing_seeds"]
    assert cfg["confirmation_repetitions"] == 5 and cfg["n"] == 64 and cfg["horizon"] == .2
    assert jax.default_backend() == "gpu" and jax.config.jax_enable_x64
    assert os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((HERE / "PROVENANCE.json").read_text())
    for rel, entry in frozen["files"].items():
        assert R.sha(HERE / rel) == entry["sha256"], rel
    provenance = {"config": cfg, "config_sha256": R.sha(args.config),
        "driver_sha256": R.sha(__file__), "source_provenance": frozen,
        "source_commit": os.environ.get("SOURCE_COMMIT"), "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                         "--format=csv,noheader"], text=True).strip(),
        "backend": jax.default_backend(), "x64": jax.config.jax_enable_x64,
        "precision": str(jax.config.jax_default_matmul_precision)}
    write_json(args.out / "provenance.json", provenance)
    parameters = F.parameters(cfg["eval_seed"], 16)
    assert sha_array(parameters) == "ea4c10c63cf423da985b0452fdf7810b7f59428f54ad46cbdd0ac9e9664df9d6"
    parameter_record = {"seed": cfg["eval_seed"], "parameters": parameters.tolist(),
                        "sha256": sha_array(parameters), "case_ids": cfg["case_ids"]}
    write_json(args.out / "parameters.json", parameter_record)
    dt = cfg["truth_dt"]
    steps = round(cfg["horizon"] / dt)
    truth_run, fine_run = F.make_solver(dt, steps, steps // 5), F.make_solver(dt / 2, 2 * steps, 2 * steps // 5)
    geometry = F.geometry(cfg["n"])
    cases, references = [], []
    for i, par in enumerate(parameters):
        initial = F.initial(cfg["n"], par)
        truth = np.asarray(truth_run(jnp.asarray(initial), float(par[-1]), geometry))
        fine = np.asarray(fine_run(jnp.asarray(initial), float(par[-1]), geometry))
        norm = np.linalg.norm(initial)
        delta = float(np.max(np.linalg.norm((truth - fine).reshape(6, -1), axis=1)) / norm)
        init_gap = float(np.linalg.norm(truth[0] - initial) / norm)
        references.append({"case": i, "parameter": par.tolist(), "truth_sha256": sha_array(truth),
                           "finer_sha256": sha_array(fine), "input_sha256": sha_array(truth[0]),
                           "dt_refinement_max": delta, "initial_projection_gap": init_gap,
                           "absolute_reference_pass": delta <= cfg["reference_absolute_gate"] and init_gap <= 1e-12})
        cases.append(Case(i, truth[0].copy(), float(par[-1]), truth))
        write_json(args.out / "references.json", references)
    R.log("fresh references generated, frozen numerical checks next")
    setup = []
    start = time.perf_counter()
    bank_file = HERE / "assets/bank_selected.npz"
    bank_data = np.load(bank_file)
    G, bank_info = BI.mesh_bank(BI.load_params(bank_file), bank_data["T"], cfg["n"], chunk=32768, abort=.05)
    bank_info.pop("lowdin_factor")
    operators, checks = SR.build_operators_fast(G, cfg["n"], cfg["rom"]["rows"], check_modes=8)
    params, codes = R.load_head(HERE / "assets/head_k8_n64.npz")
    learned_assets = args.out / "deployed_assets/learned"
    R.save_assets(learned_assets, G, operators, params, codes)
    setup.append({"phase": "frozen learned mesh bank and operators", "wall_seconds": time.perf_counter() - start,
                  "checks": checks, "bank_info": bank_info, "scope": "offline setup, not complete query"})
    original = json.loads((HERE / "configs/original_dev64.json").read_text())
    start = time.perf_counter()
    R.log("regenerating frozen POD basis from recorded training seed")
    pod_G, pod_info = R.rebuild_pod(original, cfg["n"], cfg["horizon"])
    pod_ops, pod_checks = SR.build_operators_fast(pod_G, cfg["n"], 292, check_modes=8)
    pod_params, pod_codes = R.load_head(HERE / "assets/pod_head_k8_n64.npz")
    pod_assets = args.out / "deployed_assets/pod"
    R.save_assets(pod_assets, pod_G, pod_ops, pod_params, pod_codes)
    setup.append({"phase": "frozen POD recipe regeneration and operators", "wall_seconds": time.perf_counter() - start,
                  "checks": pod_checks, "pod_info": pod_info, "scope": "offline setup, not complete query"})
    write_json(args.out / "offline_setup.json", setup)
    specs = [(t["name"], "target", t, learned_assets) for t in cfg["targets"]]
    specs += [(f"pod_head_m{s['rows']}_dt{s['dt']:g}", "pod", s, pod_assets) for s in cfg["pod_head_settings"]]
    subjects, qualifications = [], []
    for name, family, spec, assets in specs:
        settings = {"rows": spec["rows"], "dt": spec["dt"]}
        subject = R.native_subject(name, family, "head", 64, cfg, assets, settings)
        tighter = R.native_subject(name + "_tighter", family, "head", 64, cfg, assets,
            {**settings, "gtol": cfg["solver_refinement"]["gtol"], "fit_gtol": cfg["solver_refinement"]["fit_gtol"]})
        subjects.append(subject)
        R.log("compile and tight/tighter check: " + name)
        for case in cases:
            value = subject.query(case.initial, case.viscosity)
            fine = tighter.query(case.initial, case.viscosity)
            metrics = output_metrics(value.fields, case)
            gap = float(np.max(np.linalg.norm((value.fields - fine.fields).reshape(6, -1), axis=1)) / np.linalg.norm(case.initial))
            bound = min(cfg["solver_refinement"]["absolute_gate"], cfg["solver_refinement"]["fraction_of_error"] * metrics["evolved_max_error"])
            ref_bound = min(cfg["reference_absolute_gate"], cfg["reference_fraction_of_target_error"] * metrics["evolved_max_error"])
            ref = references[case.case_id]
            passed = bool(value.numerical_valid and fine.numerical_valid and gap <= bound and
                          ref["absolute_reference_pass"] and ref["dt_refinement_max"] <= ref_bound)
            qualifications.append({"subject": name, "case": case.case_id, "passed": passed,
                "solver_refinement_difference": gap, "solver_refinement_bound": bound,
                "reference_refinement_bound": ref_bound, "reference_refinement_difference": ref["dt_refinement_max"],
                "nominal_valid": value.numerical_valid, "tighter_valid": fine.numerical_valid,
                "nominal_diagnostics": value.diagnostics, "tighter_diagnostics": fine.diagnostics,
                "nominal_metrics": metrics, "tighter_metrics": output_metrics(fine.fields, case)})
            write_json(args.out / "numerical_qualification.json", qualifications)
    for candidate in cfg["classical_candidates"]:
        subject = R.classical_subject(candidate, cfg)
        subjects.append(subject)
        for case in cases:
            value = subject.query(case.initial, case.viscosity)
            output_metrics(value.fields, case)
    out = args.out / "paired"
    write_json(out / "contract.json", {"config": cfg, "subjects": [dict(name=s.name, family=s.family,
        mesh=s.mesh, metadata=s.metadata) for s in subjects], "selection_performed": False,
        "frozen_before_truth": True, "scope": "fresh physical cohort; frozen models and comparators"})
    rows_path = out / "invocations.jsonl"
    R.log("paired measurements begin")
    seed = cfg["timing_order_seed"]
    before_rows = block(subjects, cases[:1], 3, seed + 100, "sentinel_before", rows_path, R.burn)
    rows = block(subjects, cases, 5, seed, "fresh_confirmation", rows_path, R.burn)
    after_rows = block(subjects, cases[:1], 3, seed + 101, "sentinel_after", rows_path, R.burn)
    summary = summarize(rows, subjects, cfg["case_ids"])
    before, after = summarize(before_rows, subjects, [0]), summarize(after_rows, subjects, [0])
    drifts = {s.name: drift(summary, before, after, s.name) for s in subjects}
    numerical_pass = {s.name: (all(r["passed"] for r in qualifications if r["subject"] == s.name)
        if s.family != "classical" else all(r["absolute_reference_pass"] for r in references)) for s in subjects}
    for s in subjects:
        if s.family == "classical":
            numerical_pass[s.name] = numerical_pass[s.name] and all(
                references[int(i)]["dt_refinement_max"] <= min(cfg["reference_absolute_gate"],
                    cfg["reference_fraction_of_target_error"] * c["evolved_max_error"])
                for i, c in summary[s.name]["cases"].items())
    thresholds, comparisons = {}, {}
    for target, threshold in cfg["frozen_thresholds"].items():
        names = [target] + list(dict.fromkeys(cfg["frozen_comparators"][target].values()))
        thresholds[target] = {}
        for name in names:
            c = summary[name]
            thresholds[target][name] = bool(c["complete"] and numerical_pass[name] and
                c["median_case_evolved_max_error"] <= threshold["median_case_evolved_max_error"] and
                c["worst_evolved_error"] <= threshold["worst_evolved_error"])
        comparisons[target] = {}
        for group, name in cfg["frozen_comparators"][target].items():
            dpass = all(drifts[a] is not None and max(drifts[a].values()) <= cfg["drift_gate_ratio"] for a in [target, name])
            nvalid = bool(numerical_pass[target] and numerical_pass[name])
            match = eligible(summary[name], summary[target])
            ids = sorted(set(summary[name]["cases"]) & set(summary[target]["cases"]), key=int)
            ratios = [summary[name]["cases"][i]["median_ms"] / summary[target]["cases"][i]["median_ms"] for i in ids]
            comparisons[target][group] = {"comparator": name, "fresh_head_accuracy_match": bool(match),
                "frozen_target_eligible": thresholds[target][target], "frozen_comparator_eligible": thresholds[target][name],
                "numerical_pass": nvalid, "timing_drift_pass": bool(dpass),
                "qualified_fixed_threshold": bool(nvalid and dpass and thresholds[target][target] and thresholds[target][name]),
                "qualified_fresh_head_match": bool(nvalid and dpass and match),
                "observed_ratio_of_median_times": summary[name]["median_case_ms"] / summary[target]["median_case_ms"],
                "paired_case_ids": [int(i) for i in ids], "paired_case_ratios": ratios,
                "median_paired_case_ratio": float(np.median(ratios)),
                "head_faster_case_count": sum(r > 1 for r in ratios),
                "ratio_direction": "comparator / head; above one means head faster; qualification is explicit"}
    result = {"schema": "fresh-frozen-complete-query-result-v1", "confirmation": summary,
              "comparisons": comparisons, "frozen_threshold_eligibility": thresholds,
              "numerical_pass": numerical_pass, "timing_drifts": drifts,
              "sentinels_before": before, "sentinels_after": after,
              "selection_performed": False, "cold_deployment": "not repeated; warm complete query only",
              "source_provenance": "../provenance.json", "all_reference_absolute_pass": all(r["absolute_reference_pass"] for r in references),
              "all_numerical_checks_pass": all(numerical_pass.values())}
    write_json(out / "result.json", result)
    R.log("fresh frozen confirmation complete; all eligibility outcomes retained")


if __name__ == "__main__":
    main()
