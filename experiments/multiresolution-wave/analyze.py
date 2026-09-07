"""Generate numerical findings and review artifacts from checked native outputs.

Repetitions measure timing variability, not independent trajectory draws.
Reference contraction estimates are conditional diagnostics, never proven bounds.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import numpy as np

METRICS = ("displacement", "velocity", "energy_state")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return value is not None and np.isfinite(value)


def aggregate(values):
    vv = [float(v) for v in values if finite(v)]
    return dict(count=len(values), nonfinite=len(values)-len(vv), median=float(np.median(vv)) if vv else None,
                mean=float(np.mean(vv)) if vv else None, worst=max(vv) if vv else None)


def max_error(metrics):
    values = [metrics[k]["max_initial_normalized"] for k in METRICS]
    return max(values) if all(finite(v) for v in values) else None


def write(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


def normalized_pairs(data, result_path, summary, out):
    """Light adapter for coordinator accounting; no absent evidence defaults true."""
    cfg, prov = data["config"], data["provenance"]
    hashes = {os.path.relpath(result_path, out): digest(result_path)}
    pairs = []
    for bc in cfg["boundaries"]:
        for n in cfg["meshes"]:
            baseline_settings = [("dst", 0.)] if bc == "dirichlet" else [("rk4", x) for x in cfg["fom_cfls"]]
            for dt in cfg["rom_dts"]:
                for fom, setting in baseline_settings:
                    label = f"pair_{bc}_{n}_{dt}_{fom}_{setting}"
                    p = dict(schema_version=1, stage="development", reference_kind="refined_physical",
                        configuration_selection="predeclared", source_artifact_sha256=hashes,
                        provenance=dict(backend=prov["jax_backend"], x64=prov["x64"], matmul_precision=prov["matmul_precision"],
                            job_id=prov["job_id"], gpu_identity=prov["host"]+":"+prov["device_kind"][0], source_commit=prov["source_commit"]),
                        case_ids=[str(i) for i in cfg["validation_indices"]], repetitions=cfg["repetitions"],
                        required_error_metrics=list(METRICS), output_contract_id=f"host_phase_fields_{n}_all_observations",
                        reference_uncertainty_bound=None, reference_uncertainty_fraction=cfg["reference_budget_fraction"],
                        reference_uncertainty_note="Empirical self-differences and conditional contraction estimates are supplied in summary.json; no proven uncertainty bound is asserted.",
                        targets=cfg["accuracy_targets"], independent_final_cohort=False, invocations=[])
                    for row in data["invocations"]:
                        if row["boundary"] != bc or row["intervals"] != n:
                            continue
                        is_rom = row["method"] == "rom" and row["setting"] == dt
                        is_fom = row["method"] == fom and row["setting"] == setting
                        if not (is_rom or is_fom):
                            continue
                        case_dt = next((x for x in summary["rom_time_refinement"] if x["boundary"] == bc and x["intervals"] == n and x["case"] == row["case"]), None)
                        # The historical doubled-fit-budget gate has not been checked.
                        valid = bool(row["completed"] and (not is_rom or (row["fit_stationary"] and case_dt and case_dt["within_original_refinement_target"])))
                        p["invocations"].append(dict(case_id=str(row["case"]), repeat=row["repetition"], method="nmrom" if is_rom else "fom",
                            invocation_id=row["invocation_id"], metric_invocation_id=row["invocation_id"],
                            job_id=prov["job_id"], gpu_identity=p["provenance"]["gpu_identity"],
                            output_contract_id=p["output_contract_id"], query_includes_input_and_output=True,
                            query_seconds=row["seconds"]["complete_query"], warmup_and_burnin_complete=True,
                            device_synchronized=True, completed=row["completed"], numerically_valid=valid,
                            fit_budget_stability_rechecked=False if is_rom else None,
                            errors={k: row["physical_reference_error"][k]["max_initial_normalized"] for k in METRICS}))
                    path = out/(label+".json")
                    write(path, p)
                    pairs.append(path.name)
    return pairs


def analyze(record):
    cluster = record/"cluster"
    native = cluster/"out/pilot"
    result_path = native/"result.json"
    data = json.loads(result_path.read_text())
    cfg = data["config"]
    if not data.get("complete"):
        raise RuntimeError("Pilot incomplete")
    cleanup = json.loads((record/"cleanup.json").read_text())
    if not cleanup["remote_deleted_and_absence_checked"] or not cleanup["all_three_manifests_verified"]:
        raise RuntimeError("Checked collection/cleanup required")
    error_log = (cluster/"logs"/(str(data["provenance"]["job_id"])+".err")).read_text()
    for bad in ("captured constant", "Captured constant", "out of memory", "RESOURCE_EXHAUSTED", "Traceback", "No space left"):
        if bad in error_log:
            raise RuntimeError("Inspect prohibited runtime warning/error: "+bad)
    if (cluster/"EXIT_CODE").read_text().strip() != "0":
        raise RuntimeError("Nonzero runtime status")
    out = record/"analysis"
    out.mkdir(exist_ok=True)
    summary = dict(status="Completed development pilot; accuracy and efficiency conclusions are scoped to declared cases.",
        provenance=data["provenance"], result_sha256=digest(result_path), config=cfg,
        cleaned_remote=True, checkpoint_hashes=data["input_sha256"], groups=[], rom_time_refinement=[], reference_uncertainty=[],
        final_cohort_opened=False, fit_budget_stability_rechecked=False,
        offline_training_repeated=False, mesh_assembly=data["meshes"], bank_projection=data.get("bank_projection_diagnostic", []),
        reference_warning="Self-refinement is empirical. Absorber continuation estimates assume the observed contraction persists; neither is a proven continuum bound.")
    groups = defaultdict(list)
    for row in data["invocations"]:
        groups[row["boundary"], row["intervals"], row["method"], row["setting"]].append(row)
    for (bc, n, method, setting), rows in groups.items():
        cases = []
        for ci in cfg["validation_indices"]:
            rr = [r for r in rows if r["case"] == ci]
            if len(rr) != cfg["repetitions"]:
                raise RuntimeError("Missing timing repetitions")
            cases.append(dict(case=ci, completed=all(r["completed"] for r in rr),
                stationary=None if method != "rom" else all(r["fit_stationary"] for r in rr),
                deterministic_output_hashes=len({json.dumps(r["output_sha256"], sort_keys=True) for r in rr}) == 1,
                query_seconds=[r["seconds"]["complete_query"] for r in rr],
                query_median=float(np.median([r["seconds"]["complete_query"] for r in rr])),
                phase_medians={k: float(np.median([r["seconds"][k] for r in rr])) for k in rr[0]["seconds"]},
                physical_reference_error={k: max(r["physical_reference_error"][k]["max_initial_normalized"] for r in rr) for k in METRICS},
                initial_physical_reference_error={k: rr[0]["physical_reference_error"][k]["initial_normalized"][0] for k in METRICS},
                same_grid_discrepancy={k: max(r["same_grid_discrepancy"][k]["max_initial_normalized"] for r in rr) for k in METRICS},
                worst_required_physical_error=max(max_error(r["physical_reference_error"]) for r in rr),
                maximum_current_relative={k: max((r["physical_reference_error"][k]["max_current_relative"] for r in rr if r["physical_reference_error"][k]["max_current_relative"] is not None), default=None) for k in METRICS},
                final_current_relative={k: rr[0]["physical_reference_error"][k]["current_relative"][-1] for k in METRICS},
                final_absolute={k: rr[0]["physical_reference_error"][k]["absolute"][-1] for k in METRICS},
                final_reference_norm={k: rr[0]["physical_reference_error"][k]["reference_norm"][-1] for k in METRICS},
                final_vanishing={k: rr[0]["physical_reference_error"][k]["reference_vanishing"][-1] for k in METRICS},
                final_zero={k: rr[0]["physical_reference_error"][k]["reference_zero"][-1] for k in METRICS},
                final_truth_energy_fraction=rr[0]["physical_reference_error"]["truth_energy_fraction"][-1],
                final_predicted_energy_fraction=rr[0]["physical_reference_error"]["energy_fraction"][-1]))
        summary["groups"].append(dict(boundary=bc, intervals=n, method=method, setting=setting, cases=cases,
            complete_query_seconds=aggregate([x["query_median"] for x in cases]),
            physical_reference_error=aggregate([x["worst_required_physical_error"] for x in cases]),
            failures_by_target={str(t): sum(not x["completed"] or x["worst_required_physical_error"] > t for x in cases) for t in cfg["accuracy_targets"]}))
    for bc in cfg["boundaries"]:
        for n in cfg["meshes"]:
            for ci in cfg["validation_indices"]:
                coarse, fine = cfg["rom_dts"]
                names = [f"{bc}_{n}_{ci}_0_rom_{dt}.npz" for dt in (coarse, fine)]
                with np.load(native/names[0]) as a, np.load(native/names[1]) as b, np.load(native/f"mesh_{bc}_{n}.npz") as mesh:
                    da, db = a["coefficients"]-b["coefficients"], a["velocity_coefficients"]-b["velocity_coefficients"]
                    matched = next(r for r in data["invocations"] if r["invocation_id"] == names[0][:-4])
                    c = matched["parameters"][5]
                    mm = matched["same_grid_discrepancy"]
                    displacement = np.linalg.norm(da, axis=-1)/mm["displacement"]["initial_scale"]
                    velocity = np.linalg.norm(db, axis=-1)/mm["velocity"]["initial_scale"]
                    state = np.sqrt(np.maximum(0., np.sum(db*db, axis=-1)+np.einsum("tr,rs,ts->t", da, c*c*mesh["stiffness"], da)))/mm["energy_state"]["initial_scale"]
                    maxima = dict(displacement=float(displacement.max()), velocity=float(velocity.max()), energy_state=float(state.max()))
                    complete = bool(np.all(a["rollout_completed"]) and np.all(b["rollout_completed"]))
                    summary["rom_time_refinement"].append(dict(boundary=bc, intervals=n, case=ci, coarse_dt=coarse, fine_dt=fine,
                        completed=complete, maxima=maxima, within_original_refinement_target=complete and max(maxima.values()) <= .01))
        for ci in cfg["validation_indices"]:
            fine = next(r for r in data["references"] if r["boundary"] == bc and r["case"] == ci and "fine_intervals" in r)
            if bc == "dirichlet":
                summary["reference_uncertainty"].append(dict(boundary=bc, case=ci,
                    empirical_self_difference=max_error(fine["self_refinement"]), proven_bound=None))
            else:
                nested = next(r for r in data["references"] if r["boundary"] == bc and r["case"] == ci and "adjacent_coarse_to_this_mesh" in r)
                dcoarse = max_error(nested["adjacent_coarse_to_this_mesh"])
                dfine = max_error(nested["same_grid_to_physical_reference"])
                temporal = max_error(fine["temporal_refinement"])
                ratio = dfine/dcoarse
                estimate = dfine*ratio/(1-ratio)+temporal if 0 <= ratio < 1 else None
                summary["reference_uncertainty"].append(dict(boundary=bc, case=ci,
                    adjacent_coarse_difference=dcoarse, adjacent_fine_difference=dfine,
                    temporal_difference=temporal, observed_spatial_ratio=ratio,
                    conditional_fine_reference_estimate=estimate, proven_bound=None))
    summary["normalized_pair_files"] = normalized_pairs(data, result_path, summary, out)
    write(out/"summary.json", summary)
    render(summary, out/"FINDINGS.md")
    print(json.dumps(dict(summary=str(out/"summary.json"), native_sha256=summary["result_sha256"], completed_queries=len(data["invocations"])), indent=2))


def render(s, path):
    cfg = s["config"]
    lines = ["# Frozen fresh-wave decoder across spatial meshes", "",
        "This generated report covers the bounded development pilot of actual frozen-decoder wave evolution and complete-query FOM comparisons. Numerical findings are provisional development evidence; the final cohort remains sealed.", "",
        f"Source `{s['provenance']['source_commit']}`, job `{s['provenance']['job_id']}`, device `{s['provenance']['device_kind'][0]}`. Native result SHA-256: `{s['result_sha256']}`.", "",
        f"The declared validation seed is `{cfg['validation_seed']}`, with indices `{cfg['validation_indices']}`; both boundaries use the same physical parameter draws across meshes. The frozen MLP has latent dimension `{cfg['latent']}` and bank/weak-test dimension `{cfg['rank']}`. The physical horizon is `{cfg['end_time']}` with observation interval `{cfg['observation_dt']}`. These are development cases, not independent-cohort confirmation.", "",
        "The input contract is one host displacement field, one host velocity field and scalar wave speed. The output contract is both physical fields on the requested mesh at every observation time, copied to the host. Cold fitting, speed rescaling, evolution, and dense field output are included. There is no external forcing. QR conversion changes coefficient coordinates exactly and preserves the frozen physical decoder; all mesh-specific operator tables are rebuilt.", "",
        "| Boundary | Intervals | Method | Step/CFL | Query median (s) | Physical error median | Physical error worst | Cases above target |",
        "|---|---:|---|---:|---:|---:|---:|---|"]
    for g in s["groups"]:
        failures = "; ".join(f"{t}: {n}/{len(g['cases'])}" for t,n in g["failures_by_target"].items())
        lines.append(f"| {g['boundary']} | {g['intervals']} | {g['method']} | {g['setting']:.6g} | {g['complete_query_seconds']['median']:.6g} | {g['physical_reference_error']['median']:.6g} | {g['physical_reference_error']['worst']:.6g} | {failures} |")
    lines += ["", "Each error is the maximum over displacement, velocity and energy-state errors, themselves maximized over observation times. Displacement uses its initial norm; velocity and energy state use the initial phase-state energy scale. Case timings are medians of retained repetitions; the table then takes their cohort median. Repetitions do not increase the number of physical cases.", "",
        "Physical-reference errors above use a common observation grid. Separate native-grid FOM discrepancies remain in the JSON. No qualified speedup is inferred from a raw time ratio when accuracy fails. Reference self-differences and absorber contraction are empirical, so normalized pair records leave a proven uncertainty bound unspecified.", "",
        "| Boundary | Intervals | ROM step | Input (s) | Cold fit / speed (s) | Evolution (s) | Field output (s) | Nonstationary cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in s["groups"]:
        if g["method"] != "rom":
            continue
        med = lambda key: statistics.median(c["phase_medians"][key] for c in g["cases"])
        lines.append(f"| {g['boundary']} | {g['intervals']} | {g['setting']:.6g} | {med('input_transfer'):.6g} | {med('initialization_and_parameter_projection'):.6g} | {med('evolution'):.6g} | {med('field_output_and_transfer'):.6g} | {sum(not c['stationary'] for c in g['cases'])} |")
    lines += ["", f"Cold fitting uses `{cfg['fit_iterations']}` iterations per multistart budget and checks selected-fit stationarity. The historical doubled-budget fitting-stability gate has not been rechecked. Dense cold fitting and requested field output grow with the mesh. Previous training costs are not remeasured, so no break-even query count is asserted.", "",
        "| Boundary | Intervals | Case | Displacement step difference | Velocity step difference | Energy-state step difference | Refinement pass |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for r in s["rom_time_refinement"]:
        m = r["maxima"]
        lines.append(f"| {r['boundary']} | {r['intervals']} | {r['case']} | {m['displacement']:.6g} | {m['velocity']:.6g} | {m['energy_state']:.6g} | {r['within_original_refinement_target']} |")
    lines += ["", "The refinement verdict uses the original numerical-refinement target and compares the two declared ROM steps, with displacement and physical tangent velocity in the mesh's exact bank coordinates. This is separate from physical accuracy.", "",
        "| Boundary | Case | Reference self / finest adjacent difference | Temporal difference | Spatial ratio | Conditional finest-reference estimate |",
        "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "unavailable" if x is None else f"{x:.6g}"
    for r in s["reference_uncertainty"]:
        lines.append(f"| {r['boundary']} | {r['case']} | {fmt(r.get('empirical_self_difference', r.get('adjacent_fine_difference')))} | {fmt(r.get('temporal_difference'))} | {fmt(r.get('observed_spatial_ratio'))} | {fmt(r.get('conditional_fine_reference_estimate'))} |")
    lines += ["", "For fixed walls, the reference is independently implemented continuum sine propagation, with spectral mesh self-refinement. For absorption, the reference solves the declared absorbing-boundary PDE, not an infinite-domain replacement. The conditional finest-mesh estimate is", "", "$$\\widehat e_f = \\frac{r}{1-r}\\,d_f + d_t,$$", "", "where $d_f$ is the finest adjacent spatial difference, $r$ is the ratio of fine to coarse adjacent differences, and $d_t$ is the temporal pair difference. This assumes continued contraction and is not a rigorous bound.", "",
        "| Absorbing intervals | ROM step | Case | Final displacement relative | Final velocity relative | Final energy-state relative | Final displacement absolute | Final reference energy fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in s["groups"]:
        if g["boundary"] == "absorbing" and g["method"] == "rom":
            for c in g["cases"]:
                r = c["final_current_relative"]
                lines.append(f"| {g['intervals']} | {g['setting']:.6g} | {c['case']} | {fmt(r['displacement'])} | {fmt(r['velocity'])} | {fmt(r['energy_state'])} | {fmt(c['final_absolute']['displacement'])} | {fmt(c['final_truth_energy_fraction'])} |")
    lines += ["", "Current-state normalization exposes errors when absorption leaves a small physical field. Absolute errors and the explicit zero/vanishing flags remain available in `summary.json` and native invocation records. Undefined zero-reference relative errors remain null. Phase and mean-field diagnostics are retained separately in the native result.", "",
        "| Boundary | Intervals | New-mesh assembly including first compilation (s) | Stored bank/mass/operators (bytes) |",
        "|---|---:|---:|---:|"]
    for m in s["mesh_assembly"]:
        lines.append(f"| {m['boundary']} | {m['intervals']} | {m['assembly_seconds_including_first_compile']:.6g} | {m['storage_bytes']} |")
    lines += ["", "The assembly timer includes checkpoint loading, coordinate-network evaluation, QR conversion and discrete mass/stiffness/boundary-damping construction. It recurs for a new mesh. Physical-speed rescaling recurs inside each measured query. Storage above excludes neural weights and requested output arrays, whose hashes and sizes are recorded per invocation.", "",
        "| Boundary | Intervals | Case | Unrestricted projected displacement error | Primary ROM initial displacement error | Primary ROM evolved displacement error |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in s["bank_projection"]:
        g = next(g for g in s["groups"] if g["boundary"] == row["boundary"] and g["intervals"] == row["intervals"] and g["method"] == "rom" and g["setting"] == cfg["rom_dts"][0])
        case = next(c for c in g["cases"] if c["case"] == row["case"])
        lines.append(f"| {row['boundary']} | {row['intervals']} | {row['case']} | {row['metrics']['displacement']['max_initial_normalized']:.6g} | {case['initial_physical_reference_error']['displacement']:.6g} | {case['physical_reference_error']['displacement']:.6g} |")
    lines += ["", "The focused unrestricted-bank projection diagnostic measures the learned spatial span's mass projection of same-grid truth. It is not a dynamical trajectory or an equal-latent-dimensional comparator. Its energy error is not an optimal energy-norm lower bound. Primary ROM columns use the independently refined physical reference. No architecture search or new training was performed.", "",
        "## Plain-language glossary", "",
        "**Intervals** count grid cells per axis; fixed-wall boundary values are prescribed, while absorbing boundary values are evolved. **Boundary** identifies either fixed-zero reflecting walls (`dirichlet`) or the declared local absorbing condition. **Case** is a predetermined validation trajectory. **Validation** means development data excluded from training; the separate final cohort is sealed. **Frozen** means the network weights are reused without training.", "",
        "**ROM** is the reduced wave solve using latent coordinates. **FOM** is the full spatially discretized wave solve. **DST** is an exact discrete sine transform propagator for the semidiscrete fixed-wall problem. **RK4** is a fourth-order explicit time integrator. **Step** is the ROM time-step size. **CFL** controls the FOM step relative to mesh spacing and wave speed. **Query** includes the complete stated input, initialization, evolution, output and transfer work. **Median** is the middle value; **worst** is the largest value. **Cases above target** counts failed or over-threshold trajectories, not timing repetitions.", "",
        "**Latent dimension** counts reduced coordinates. **Bank** is the spatial feature family generated by the coordinate network. **Weak tests** project the wave equation onto bank features. **QR** is a matrix factorization used here only to change coefficient coordinates. **Tangent velocity** is the decoder derivative applied to latent velocity; **curvature** is its second derivative contribution. **Cold fit** initializes the latent state from the supplied physical fields. **Stationary** means the recorded local optimization derivative tests passed, not a proven global best fit.", "",
        "**Displacement** is the wave field; **velocity** is its time derivative. **Energy-state error** combines displacement-gradient and velocity errors. **Initial normalized** divides by a fixed initial scale. **Current relative** divides by the field's norm at the observation time. **Absolute** is the unnormalized physical error. **Vanishing** flags a reference norm below the configured fraction of its initial scale; **zero** flags a numerically absent reference and makes its relative error undefined. **Energy fraction** divides current energy by its initial value. **Phase** compares modal oscillation angles. **Mean field** is the spatially integrated field.", "",
        "**Same-grid discrepancy** compares ROM or FOM output with a tightly resolved solve of the identical spatial discretization. **Physical reference** uses an independently refined approximation to the declared continuum PDE. **Self-refinement** compares reference discretizations. **Contraction ratio** measures whether successive differences shrink. **Conditional estimate** assumes that shrinkage persists; **proven bound** would require stronger evidence. **Refinement pass** checks time-step sensitivity, not full physical accuracy. **Assembly** builds cached mesh-dependent operators. **Compilation** prepares the JAX executable. **Burn-in** stabilizes GPU activity before timing. **SHA-256** identifies exact artifact bytes.", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    analyze(args.record.resolve())
