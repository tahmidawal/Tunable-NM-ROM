"""Generate the direct linear heat-bank comparison from accepted run artifacts."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/"worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/linear05"
NAMES = {"linear_weak_exact": "Linear bank, exact reduced time evolution",
         "linear_weak_cn": "Linear bank, original discrete weak step",
         "nmrom": "Current nonlinear ROM", "fom_same_grid": "Same-grid direct FOM",
         "fom_coarse16": "Coarse direct FOM, interpolated"}


def main():
    native = RECORD/"archive/outputs/results.json"
    audit_path = RECORD/"analysis/audit.json"
    audit = json.loads(audit_path.read_text()); result = json.loads(native.read_text())
    cleanup = json.loads((RECORD/"REMOTE-CLEANUP.json").read_text())
    assert audit["passed"] and result["complete"] and cleanup["remote_absent"]
    assert audit["result_sha256"] == hashlib.sha256(native.read_bytes()).hexdigest()
    rows = audit["summaries"]; ns = result["settings"]["requested_intervals"]; finest = max(ns)
    cfg = result["config"]; settings = result["settings"]
    lookup = {(r["intervals"], r["method"]): r for r in rows}
    coarse = lookup[finest, "fom_coarse16"]
    exact = lookup[finest, "linear_weak_exact"]; nm = lookup[finest, "nmrom"]; fom = lookup[finest, "fom_same_grid"]
    lines = ["# Direct linear heat evolution in the learned spatial bank", "",
        "Completed and audited development comparison of a linear ROM using the frozen learned bank, the current nonlinear ROM, and direct heat FOMs. These are provisional scientific findings because the cohort is used for development; final paper cases remain unopened.", "",
        f"At the finest requested mesh, freeing the bank coefficients changes worst current-relative error from {100*nm['worst_physical_error']:.6f}% to {100*exact['worst_physical_error']:.6f}% and GPU query time from {nm['device_median_ms']:.6f} ms to {exact['device_median_ms']:.6f} ms. The measured speed ratio is {nm['device_median_ms']/exact['device_median_ms']:.3f}× versus the current NMROM and {fom['device_median_ms']/exact['device_median_ms']:.3f}× versus the same-grid direct FOM. These ratios compare medians within this allocation; they do not imply matched FOM error.", "",
        f"The coarse direct FOM takes {coarse['device_median_ms']:.6f} ms on the GPU with worst error {100*coarse['worst_physical_error']:.6f}%. The linear ROM and this coarse FOM both meet the declared {100*settings['accuracy_target']:.6g}% development target. This run therefore does not establish a linear-ROM GPU advantage over the coarse-FOM control, although the linear ROM is more accurate.", "",
        f"The linear ROM evolves {cfg['r']} free bank coefficients; the NMROM evolves {cfg['k']} nonlinear latent variables. Both use the same frozen learned spatial weights and {cfg['modes_per_axis']**2} smooth sine test moments. This is a linear learned-basis ROM, not the original nonlinear-manifold method and not a matched-dimension comparison.", "",
        f"## Finest mesh: {finest} intervals per axis", "",
        "| Method | GPU query median (ms) | Host query median (ms) | Worst physical relative error (%) | Worst initial error (%) | GPU / host timing outliers |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in NAMES:
        r = lookup[finest, name]
        lines.append(f"| {NAMES[name]} | {r['device_median_ms']:.6f} | {r['host_median_ms']:.6f} | {100*r['worst_physical_error']:.6f} | {100*r['worst_initial_error']:.6f} | {r['device_outliers']} / {r['host_outliers']} |")
    lines += ["", f"Each row includes all {len(result['cases'])} fixed development cases and {settings['timing_repetitions']} timed repetitions per case. Errors cover every requested output time, including the actual ROM initial fit/projection. The physical reference uses a continuum spectral operator with an empirical refinement check; the FOM solves its discrete spatial problem exactly in time.", "",
        "## Complete mesh ladder", "",
        "| Intervals | Method | GPU query median (ms) | Host query median (ms) | Worst physical relative error (%) | Worst same-grid relative error (%) | GPU / host outliers |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in ns:
        for name in NAMES:
            r = lookup[n, name]
            lines.append(f"| {n} | {NAMES[name]} | {r['device_median_ms']:.6f} | {r['host_median_ms']:.6f} | {100*r['worst_physical_error']:.6f} | {100*r['worst_same_grid_error']:.6f} | {r['device_outliers']} / {r['host_outliers']} |")
    lines += ["", "## What changed and what was charged", "",
        r"Factor the learned spatial bank as $G=QR$. If $B$ maps bank coefficients to sine moments, define $C=BR^{-1}$. Free bank coordinates obey the continuous weak least-squares system", "",
        r"$$\dot y=-\nu C^+\Lambda C y,\qquad u=Qy.$$", "",
        "Small reduced propagation maps are precomputed for the requested times. Online evaluation projects the supplied full field, applies those maps, and reconstructs every requested full field. There is no iterative nonlinear fit or timestep solve. This removes the nonlinear head restriction and uses more independent state coefficients.", "",
        "The discrete linear control precomputes powers of the linear least-squares step obtained by replacing the original head with free coefficients. It retains the old time formula. A smaller-step control is recorded for accuracy only.", "",
        "GPU timings start with the supplied full initial field on the GPU and finish with all full outputs ready on the GPU. Projection, initialization, evolution and full reconstruction are charged. The host column adds actual full input/output transfers from the same invocation. Offline learned-bank/operator assembly, checkpoint training and compilation are excluded and retained in the raw JSON. No data-generation descriptors enter any solver.", "",
        f"The coarse FOM uses {settings['coarse_solver_intervals']} intervals and physically aligned interpolation to every requested output node. Its full input is restricted on the GPU in this run, so its host protocol differs from the earlier heat transfer report. Ratios to earlier allocations must not be computed.", "",
        "These linear methods retain a full-grid initial projection and full-grid reconstruction. Their small evolution maps are independent of mesh size; total full-query cost is not constant with resolution.", "",
        "## Validation and provenance", "",
        f"GPU job `{result['metadata']['job_id']}` on `{result['metadata']['gpu']}`, node `{result['metadata']['node']}`, scientific source `{audit['source_commit']}`. Frozen checkpoint `{settings['checkpoint']}`; family `{cfg['family']}`; diffusivity `{cfg['diffusivity']}`; cohort seeds `{[c['seed'] for c in settings['cohorts']]}`. Float64 and highest matrix precision were verified, with GPU preflight and no rejected runtime warnings.", "",
        f"The independent NumPy/SciPy audit checked {audit['unique_fields_checked']} unique preserved fields, {audit['timed_invocations_checked']} paired timing/error invocations, and {audit['metric_entries_checked']} metric entries. Maximum recomputation discrepancy: {audit['maximum_metric_difference']:.12g}. Independent QR-based reduced-operator checks have maximum relative discrepancy {max(r['relative_error'] for r in audit['operator_errors']):.12g}.", "",
        f"Reference meshes: `{settings['reference_pair']}`. Maximum current-relative restricted reference difference: {audit['maximum_reference_refinement']:.12g}, below the declared empirical budget {cfg['reference_uncertainty_budget']:.12g}. This is not a certified continuum error bound. Restricted reference pairs are preserved; finer source arrays retain hashes and regeneration seeds.", "",
        f"Maximum linear discrete-step half-step discrepancy: {max(r.get('worst_half_step_current_delta',0) for r in rows):.12g}. Nonstationary nonlinear initial-start groups / steps: {sum(r['nonstationary_fit_count'] for r in rows)} / {sum(r['nonstationary_step_count'] for r in rows)}. Counts include repeated invocations and both attempted initial starts; stationarity is distinct from field accuracy.", "",
        f"All {len(audit['prior_nmrom_field_parity'])} NMROM case/mesh fields match the previously archived frozen-head rollout to a maximum relative difference of {max(r['relative_difference'] for r in audit['prior_nmrom_field_parity']):.12g}. This verifies the comparator fields; its timings come entirely from the new allocation.", "",
        audit["outlier_rule"], "",
        "Transport and archived members passed checksums, and the exact remote job directory was removed. The heat branch remains separate as previously requested. No earlier measurement is retracted; this is a new reduced method and a new paired comparison.", "",
        f"[Raw results]({'../'+native.relative_to(ROOT).as_posix()}) · [Independent audit]({'../'+audit_path.relative_to(ROOT).as_posix()}) · [Configuration]({'../'+(RECORD/'archive/experiments/mr-heat2d/config-linear.json').relative_to(ROOT).as_posix()})", "",
        "## Glossary", "",
        "- **Intervals:** cells per spatial axis; the boundary values are identically zero and interior arrays are returned.",
        "- **ROM / NMROM / FOM:** reduced model / nonlinear-manifold reduced model / full-grid solver.",
        "- **Learned bank:** frozen spatial functions shared by the linear and nonlinear reduced models.",
        "- **Coefficient / latent variable:** independent linear weight / coordinate passed through the nonlinear head.",
        "- **Weak test moment:** a smooth sine-weighted average used to measure the PDE residual.",
        "- **GPU query / host query:** full accelerator input-to-output time / the same invocation including input and output transfers.",
        "- **Median:** middle timing value over all retained cases and repetitions at that mesh and method.",
        "- **Worst physical relative error:** largest full-field Euclidean error divided by the current reference norm, across all cases, repetitions and output times; reported as a percentage.",
        "- **Worst same-grid relative error:** the same statistic against the exact-in-time discrete solver on the requested mesh.",
        "- **Worst initial error:** largest relative error of the fitted/projected initial field.",
        "- **Outlier:** a timing exceeding the stated within-case threshold; retained in the reported medians.",
        "- **Exact reduced time evolution:** matrix exponential of the fixed reduced linear differential equation; it still has spatial approximation and floating-point error.",
        "- **Discrete weak step / half-step:** the original Crank–Nicolson weak least-squares formula / its smaller-timestep control.",
        "- **Coarse FOM:** a direct solver on a smaller grid followed by interpolation to the requested output grid.",
        "- **Reference refinement:** agreement between finer spectral grids; empirical evidence, not a rigorous bound.",
        "- **Stationarity:** satisfying the nonlinear objective's gradient stopping rule, not proof of global optimality or physical accuracy.",
        "- **Development cohort / checkpoint:** examples used for method development / saved frozen learned weights and codes."]
    target = ROOT/"reports/2026-09-10-heat-linear-bank-comparison.md"
    target.write_text("\n".join(lines)+"\n")
    manifest = dict(source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (native,audit_path,RECORD/"REMOTE-CLEANUP.json")}, report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(target)


if __name__ == "__main__": main()
