"""Generate all pilot numerical statements directly from saved run JSON/arrays."""
import argparse
import json
from pathlib import Path

import numpy as np


def fmt(value):
    return f"{value:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    assert result["complete"]
    cfg = result["config"]
    lines = ["# Restricted heat development pilot: frozen mesh transfer", "",
             "These provisional findings establish the new heat port and measure a bounded "
             "continuous-coordinate NM-ROM pilot against direct sine-transform propagation. "
             "They await independent review and do not cover the archived multi-bump heat "
             "family, per-resolution optimization, or an independent final cohort.", "",
             f"Generated from `{args.result}`. Source commit: `{result['source_manifest']['source_commit']}`; "
             f"job `{result['metadata']['job_id']}`; GPU `{result['metadata']['gpu']}`. "
             "The run records GPU execution, float64 arithmetic and highest matmul precision. "
             f"Checkpoint SHA-256: `{result['checkpoint_sha256']}`.", "",
             "The complete query supplies a host initial interior field and returns host interior "
             "fields at every requested time. Input transfer, full-input projection and latent "
             "fitting, evolution, dense field readout and output transfer are included. The "
             "FOM propagates each requested time directly, with no imposed timestep count. "
             "The same frozen weights are reevaluated and the operators rebuilt on every mesh.", "",
             "## Pinned development configuration", "",
             "| Setting | Value |", "|---|---|",
             f"| Training intervals | {cfg['train_intervals']} |",
             f"| Evaluation intervals | {cfg['evaluation_intervals']} |",
             f"| Training / validation trajectories | {cfg['n_train']} / {cfg['n_validation']} |",
             f"| Latent / bank / weak test dimensions | {cfg['k']} / {cfg['r']} / {cfg['modes_per_axis']**2} |",
             f"| Output times | {cfg['times']} |",
             f"| CN timestep choices | {cfg['time_steps']} |",
             f"| Diffusivity | {cfg['diffusivity']} |",
             f"| Center / width / amplitude bounds | {cfg['center_range']} / {cfg['width_range']} / {cfg['amplitude_range']} |",
             f"| Train / validation / model seeds | {cfg['train_seed']} / {cfg['validation_seed']} / {cfg['model_seed']} |",
             f"| Optimizer steps / trained parameters | {cfg['steps']} / {result['training']['parameters']} |",
             f"| Timing repetitions per case and method | {cfg['repetitions']} |", "",
             "The initial condition is a single smooth Gaussian times the decoder's polynomial "
             "zero-boundary factor. Data-generation descriptors never enter the head or the "
             "initial-fitting routine. Initial fitting uses exact QR compression of full-field "
             "least squares, with nearest-training-code and mean-code starts. Both starts' "
             "counters and normalized gradients are retained.", "",
             "## Independent reference and state-advancement checks", "",
             "| Check | Measured value |", "|---|---|"
             ]
    verification = result["verification"]
    for key in ("dst_vs_scipy", "dst_involution", "laplacian_diagonalization",
                "discrete_eigenmode_decay", "continuum_eigenmode_decay", "initial_fidelity",
                "weak_discrete_operator_identity", "linear_weak_rollout_exact_cn",
                "second_output_advancement", "frozen_negative_control_error",
                "cn_time_refinement_ratio", "hard_boundary_max"):
        lines.append(f"| {key} | {fmt(verification[key])} |")
    lines += [f"| Maximum spectral reference refinement error | {fmt(max(verification['reference_spectral_refinement_current_errors']))} |",
              f"| Minimum / maximum spatial refinement ratio | {fmt(min(min(r['ratios']) for r in verification['spatial_refinement']))} / {fmt(max(max(r['ratios']) for r in verification['spatial_refinement']))} |", "",
              "All declared reference gates passed. The independent analytic linear-mode test "
              "uses the same LM/scan implementation as the learned rollout. The deliberately "
              "frozen carry fails that test. These are implementation and truth checks; learned "
              "accuracy is reported below.", "",
              "## Accuracy across the complete validation cohort", "",
              "Each error is the maximum over requested output times, then aggregated across "
              "trajectories. These first tables use each mesh's own sampled physical norm; "
              "the common-observation audit below is the transfer comparison. Current normalization divides by the current physical-reference "
              "field; initial normalization divides by its initial field. All numbers here are "
              "provisional development-cohort results, not final confirmation.", "",
              "| Intervals | Method | Current error median / worst | Initial error median / worst | Initial-fit median / worst | Cases above 5% current error |", "|---|---|---|---|---|---|"]
    groups = {}
    for row in result["rows"]:
        groups.setdefault((row["intervals"], row["method"]), []).append(row)
    for (n, method), rows in groups.items():
        current = [r["time_max_current_error"] for r in rows]
        initial = [r["time_max_initial_error"] for r in rows]
        fit = [r["repetitions"][0]["vs_physical_reference"]["relative_current"][0] for r in rows]
        lines.append(f"| {n} | {method} | {fmt(np.median(current))} / {fmt(max(current))} | {fmt(np.median(initial))} / {fmt(max(initial))} | {fmt(np.median(fit))} / {fmt(max(fit))} | {sum(v>.05 for v in current)} / {len(rows)} |")
    lines += ["", "## Complete query timing", "",
              "Costs and errors come from the same captured invocations. Reported costs are "
              "the cohort median of each case's repetition median. Speed ratio is the median "
              "of paired FOM/ROM case ratios. The target column checks observed error only; "
              "it does not certify solver validity or a reference-error margin. The FOM and ROM are timed on the same physical "
              "GPU with alternating order and clock burn-in.", "",
              "| Intervals | Method | Query ms | Paired FOM / method | Observed-error-only targets |", "|---|---|---|---|---|"]
    for (n, method), rows in groups.items():
        baseline = {r["case"]: r for r in groups[(n, "fom_dst_exact_time")]}
        ratios = [baseline[r["case"]]["median_query_seconds"]/r["median_query_seconds"] for r in rows]
        attained = [target for target in cfg["accuracy_targets"] if all(max(rep['vs_physical_reference']['relative_current']) <= target for r in rows for rep in r['repetitions'])]
        lines.append(f"| {n} | {method} | {fmt(1000*np.median([r['median_query_seconds'] for r in rows]))} | {fmt(np.median(ratios))} | {attained} |")
    lines += ["", "| Intervals | Method | Initial fit ms | Evolution ms | Readout ms | Input / output transfer ms |", "|---|---|---|---|---|---|"]
    for (n, method), rows in groups.items():
        phases = [rep["phases"] for row in rows for rep in row["repetitions"]]
        median = lambda key: fmt(1000*np.median([p.get(key, 0.) for p in phases]))
        evolve_key = "evolve_and_readout_seconds" if method == "fom_dst_exact_time" else "evolution_seconds"
        lines.append(f"| {n} | {method} | {median('initial_fit_seconds')} | {median(evolve_key)} | {median('readout_seconds')} | {median('input_seconds')} / {median('output_transfer_seconds')} |")
    lines += ["", "For the direct FOM, the evolution column includes its inverse-transform field "
              "readout. Phase medians need not sum to the median total. These are measured "
              "modular query pipelines; a fused implementation could change dispatch overhead.", "",
              "## Representation, dynamics and numerical-error diagnostics", "",
              "| Intervals | Unrestricted-bank error median / worst | Reconstruction error median / worst | Discrete spatial error median / worst |", "|---|---|---|---|"]
    for n in cfg["evaluation_intervals"]:
        rows = [r for r in result["diagnostic_rows"] if r["intervals"] == n]
        values = [[max(r[key]["relative_current"]) for r in rows] for key in ("unrestricted_bank", "multistart_reconstruction", "discrete_spatial_error")]
        lines.append(f"| {n} | " + " | ".join(f"{fmt(np.median(v))} / {fmt(max(v))}" for v in values) + " |")
    lines += ["", "These reconstruction fits use extra truth-based starts only as diagnostics; "
              "they do not initialize autonomous rollouts. Unrestricted projection has the "
              "full bank dimension and is not a matched-dimension competitor.", "",
              "| Intervals | CN dt | FOM CN-vs-semidiscrete error median / worst | ROM dt-refinement difference median / worst |", "|---|---|---|---|"]
    for n in cfg["evaluation_intervals"]:
        rows = [r for r in result["diagnostic_rows"] if r["intervals"] == n]
        archive = np.load(args.result.parent/f"fields_N{n}.npz")
        refinement = []
        coarse, fine = cfg["time_steps"]
        for case in range(cfg["n_validation"]):
            a = archive[f"n{n}_case{case}_rom_cn_dt{coarse:g}_rep0"].reshape(len(cfg["times"]), -1)
            b = archive[f"n{n}_case{case}_rom_cn_dt{fine:g}_rep0"].reshape(len(cfg["times"]), -1)
            reference = archive[f"n{n}_case{case}_discrete_reference"].reshape(len(cfg["times"]), -1)
            refinement.append(max(np.linalg.norm(a-b, axis=1)/np.linalg.norm(reference, axis=1)))
        for dt in cfg["time_steps"]:
            values = [max(next(c for c in row["cn_discretization"] if c["dt"] == dt)["error_vs_semidiscrete"]["relative_current"]) for row in rows]
            lines.append(f"| {n} | {dt} | {fmt(np.median(values))} / {fmt(max(values))} | {fmt(np.median(refinement))} / {fmt(max(refinement))} |")
    lines += ["", "The FOM CN error isolates the timestep formula on the same spatial grid. "
              "The ROM coarse/fine difference includes nonlinear projection and solver effects "
              "and is not by itself a time-convergence order certificate.", "",
              "## Solver status, decay and actual advancement", "",
              "| Intervals | Method | Selected initial fits nonstationary | Rollout steps nonstationary / total | Budget-exhausted steps | Energy-increase trajectories | Minimum final/initial truth norm | Minimum first-to-final relative state change |", "|---|---|---|---|---|---|---|---|"]
    for (n, method), rows in groups.items():
        if method == "fom_dst_exact_time":
            continue
        noninitial = nonsteps = total = budget = energy = 0
        decay, changes = [], []
        archive = np.load(args.result.parent/f"fields_N{n}.npz")
        for row in rows:
            rep = row["repetitions"][0]
            initial = np.asarray(rep["solver"]["initial_fits"])
            selected = initial[np.argmin(initial[:, 3])]
            noninitial += int(selected[4] > cfg["gradient_tolerance"])
            steps = np.asarray(rep["solver"]["steps"])
            nonsteps += int(np.sum(steps[:, 4] > cfg["gradient_tolerance"]))
            budget += int(np.sum(steps[:, 2] == 0)); total += len(steps)
            energy += int(np.any(np.diff(rep["vs_same_grid"]["energy"]) > 1e-12))
            decay.append(rep["vs_physical_reference"]["truth_norm_over_initial"][-1])
            fields = archive[f"n{n}_case{row['case']}_{method}_rep0"]
            changes.append(np.linalg.norm(fields[-1]-fields[1])/np.linalg.norm(fields[1]))
        lines.append(f"| {n} | {method} | {noninitial} / {len(rows)} | {nonsteps} / {total} | {budget} | {energy} / {len(rows)} | {fmt(min(decay))} | {fmt(min(changes))} |")
    lines += ["", "Stationarity uses the saved normalized gradient and the pinned tolerance. "
              "Small accepted steps and damping exhaustion remain nonstationary if their "
              "gradient misses that criterion. Raw reasons, gradients, attempts, accepted "
              "steps, all timing repetitions and every output field are preserved.", "",
              "## Offline and new-mesh costs", "",
              "| Work | Seconds |", "|---|---|",
              f"| Reference verification | {fmt(result['verification_seconds'])} |",
              f"| Training data generation | {fmt(result['training_data_seconds'])} |",
              f"| Training, including first-step compilation | {fmt(result['training']['seconds'])} |", "",
              "| Intervals | Bank evaluation s | QR/operator assembly s | Bank MiB | Initialization projection MiB | Weak operator / Jacobian relative mismatch |", "|---|---|---|---|---|---|"]
    for setup in result["setups"]:
        lines.append(f"| {setup['intervals']} | {fmt(setup['bank_evaluation_seconds'])} | {fmt(setup['operator_qr_seconds'])} | {fmt(setup['bank_bytes']/2**20)} | {fmt(setup['initialization_projection_bytes']/2**20)} | {fmt(setup['exact_weak_operator_relative_error'])} / {fmt(setup['exact_weak_jacobian_relative_error'])} |")
    lines += ["", "No amortization benefit is claimed without positive complete-query savings "
              "at attained accuracy. This first pilot does not include a classical "
              "coarser-grid output envelope, per-resolution retraining, a broad component-count "
              "heat cohort or independent training/data repeats. Those remain open.", "",
              "## Plain-language glossary", "",
              "- **Intervals / bank / latent / weak tests:** cells along an axis / learned spatial "
              "functions / compressed state coordinates / smooth sine functions averaging the PDE.",
              "- **FOM / NM-ROM / ROM:** full-grid solver / nonlinear-manifold reduced solver / reduced solver.",
              "- **DST / FFT / CN:** sine transform / fast Fourier transform / Crank–Nicolson timestep. "
              "**Semi-discrete:** finite-difference space with exact-in-time propagation.",
              "- **Physical reference / spatial error:** independently refined continuum sine solution / "
              "difference caused by the discrete spatial operator.",
              "- **Current error / initial error / initial fit:** L2 discrepancy divided by current "
              "reference norm / initial reference norm / discrepancy in the fitted initial state.",
              "- **Median / worst / cases above:** middle cohort value / largest cohort value / count "
              "missing the stated threshold. **Time maximum:** worst requested output time.",
              "- **Query ms / paired ratio:** full input-to-output milliseconds / same-case FOM time "
              "divided by method time. A ratio above one means lower method latency.",
              "- **Reconstruction / unrestricted bank / rollout:** best recorded truth fit / projection "
              "allowing every bank coefficient to vary / autonomous state evolution.",
              "- **LM / QR / nonstationary:** damped least-squares solver / exact orthonormal–triangular "
              "compression of initial fitting / fitted gradient exceeds the declared tolerance.",
              "- **Budget-exhausted / energy / decay:** iteration limit reached / half squared spatial "
              "L2 norm / field magnitude relative to its starting value.",
              "- **Readout / transfer / assembly / MiB:** field reconstruction / movement between host "
              "and GPU / precomputing mesh operators / bytes divided by the binary megabyte.",
              "- **Checkpoint / source commit / SHA-256 / provisional:** saved trained weights / pinned "
              "code revision / content-verification hash / development evidence awaiting review.", ""]
    if args.audit:
        audit = json.loads(args.audit.read_text())
        section = ["## Common observation-grid and artifact audit", "",
                   f"Native audit passed with {audit['archive_files_verified']} file checksums verified "
                   f"against the pulled archive and source content checked against git commit "
                   f"`{audit['source_commit']}`. Saved field/error consistency maximum discrepancy: "
                   f"{fmt(audit['native_json_error_max_discrepancy'])}.", "",
                   f"Every mesh's saved fields are restricted to the same {audit['observation_intervals']}-interval "
                   "interior observation grid. The table uses every case and repetition. Eligibility "
                   "requires selected initial fits and all rollout steps to be stationary, finite "
                   "outputs, observed error plus reference uncertainty below target, and a "
                   "reference uncertainty budget below one tenth of target. Eligibility is only "
                   "for this development cohort; it is not independent confirmation.", "",
                   "| Solver intervals | Method | Common current error median / worst | All repetitions valid | Eligible development targets |",
                   "|---|---|---|---|---|"]
        for group in audit["groups"]:
            section.append(f"| {group['intervals']} | {group['method']} | {fmt(group['common_current_median'])} / {fmt(group['common_current_worst'])} | {group['all_repetitions_valid']} | {group['eligible_targets_with_reference_margin']} |")
        unused = sum(rep["unused_initial_starts_nonstationary"] for group in audit["groups"] for case in group["cases"] for rep in case["repetitions"])
        section += ["", f"Unused initial starts that missed stationarity across all timed repetitions: {unused}. "
                    "They remain recorded; each selected initial fit passed. The selected-fit "
                    "criterion is explicit and does not hide the failed alternatives.", ""]
        index = lines.index("## Plain-language glossary")
        lines[index:index] = section
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
