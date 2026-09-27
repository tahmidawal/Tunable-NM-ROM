"""Summarize saved evidence and proposed solver/EQ tests; performs no GPU work."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burgers-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.burgers_tree / "experiments/neural-operator-burgers"
    index_path = experiment / "runs/refinement02/live-diagnosis/index.json"
    audit_path = experiment / "checks/refinement02-diagnosis-audit.json"
    index = json.loads(index_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert index["complete"] and audit["passed"]
    assert sha(index_path) == audit["diagnosis_index_sha256"]
    steps, initial, source_fields = [], [], []
    for row in index["invocations"]:
        if row["method"] != "rom" or row["rep"] != 0:
            continue
        path = index_path.parent / row["path"]
        assert sha(path) == row["sha256"]
        with np.load(path) as arrays:
            steps.extend(arrays["iterations"].tolist())
            initial.append(int(arrays["initial_iterations"]))
        source_fields.append({"path": str(path), "sha256": row["sha256"]})
    assert len(initial) == len(index["cases"]) == 8
    errors = {}
    for name in ("bank", "nonlinear_best_found", "online"):
        values = np.asarray([c["errors"][name]["per_time"] for c in index["cases"]])
        peaks, evolved = values.max(axis=1), values[:, 1:].max(axis=1)
        errors[name] = {
            "median_case_peak_percent": float(100 * np.median(peaks)),
            "worst_case_peak_percent": float(100 * peaks.max()),
            "median_evolved_case_peak_percent": float(100 * np.median(evolved)),
            "worst_evolved_case_peak_percent": float(100 * evolved.max()),
        }
    report = {
        "status": "proposed tests; saved-evidence summary only; no new solver invocation",
        "scope": "classic fixed-checkpoint solver and EQ tuning; no nested capacity model",
        "sources": {str(p): sha(p) for p in (index_path, audit_path)},
        "iteration_source_fields": source_fields,
        "fixed": {
            "checkpoint_sha256": index["checkpoint_sha256"],
            "latent_dimension": index["K"], "bank_rank": index["R"],
            "weak_modes": index["M"], "intervals": index["intervals"],
            "dt": 0.005, "native_solver_config": index["config"]["strict"],
            "initialization_policy": "same supplied-field Gaussian fit and candidate library",
        },
        "saved_evidence": {
            "case_count": len(initial), "errors": errors,
            "step_iteration_quantiles": dict(zip(
                ("min", "q25", "median", "q75", "q95", "max"),
                np.quantile(steps, (0, .25, .5, .75, .95, 1)).tolist())),
            "initial_fit_iterations_by_case": initial,
            "qualification": "inherited-checkpoint calibration diagnostic; best-found fits are not certified global optima; errors are not additive",
        },
        "proposed_stages": [
            {"name": "converged full-upwind weak sentinel control",
             "case_ids": ["burgers-calibration-00002", "burgers-calibration-00003"],
             "purpose": "verify output stability under tighter solver settings at fixed timestep before a broader screen",
             "hold_fixed": "checkpoint, weak modes, timestep and Gauss cold start",
             "qualification": "numerical convergence of the reduced equations is not physical truth",
             "separate_initializer_check": "compare sampled and full supplied-field initial fits diagnostically; do not mix initializers in the EQ screen"},
            {"name": "quadrature control", "eq_points": [256, 512, 1024],
             "additional_control": "full-grid FOM-exact upwind advection projected onto the same weak modes",
             "hold_fixed": "checkpoint, modes, cold fit, timestep, solver tolerances and budgets",
             "fitting": "decoder-output training snapshots only; record each NNLS support, weights and fit error",
             "purpose": "determine whether EQ changes the complete trajectory or the high-effort accuracy endpoint"},
            {"name": "iteration screen", "step_budgets": [2, 4, 8],
             "reference": "converged control; retain its actual budget and stopping records",
             "hold_fixed": "one selected EQ rule, cold fit, residual threshold and stationarity threshold",
             "purpose": "measure deliberately early-stopped outputs and a converged reference"},
            {"name": "tolerance screen", "evolution_normalized_gradient_tolerances": [1e-3, 1e-5, 1e-6],
             "step_budget": "same sufficiently large budget as the converged control",
             "hold_fixed": "same EQ rule, initial fitting and residual threshold",
             "normalization": "norm(J.T r)/(norm(J) norm(r)); not a physical-error tolerance",
             "optional_followup": "expose and screen normalized weak-residual thresholds separately if the initial screen warrants it"},
            {"name": "confirmation", "rule": "freeze a small shortlist on calibration cases, then evaluate on common validation cases",
             "timing": "complete query with FNO and tolerance-matched efficient FOM controls in one GPU job; all repetitions retained",
             "outputs": "return supplied initial field exactly for every deployable method; retain native initial compression error separately"},
        ],
        "readiness_gaps": [
            "eq_ablation.py implements the three EQ sizes but has not run",
            "its budget pre-check does not bound NNLS fit runtime; enforce per-stage walltime and preserve partial evidence",
            "full-grid weak advection exists as a saved-state diagnostic, not a complete rollout arm",
            "make_rom hard-codes the evolution residual threshold; expose it without changing initial fitting",
            "make_rom shares gtol between initial fitting and evolution; separate those controls before a stationarity-tolerance sweep",
            "keep original numerical acceptance gates; label early-stopped approximations separately rather than treating them as stationary solves",
            "compare physical trajectory errors, convergence records and paired query cost; do not select using residual alone",
        ],
        "conditional_followup": [
            "If representation diagnostics miss the target, improve one checkpoint's reconstruction/trajectory training, then freeze and repeat this same tuning protocol.",
            "If best-found fits are accurate but the converged full-weak rollout is poor, isolate timestep and weak-projection/dynamics errors before attributing the gap to training.",
            "If EQ causes the trajectory gap, improve the decoder-output-trained rules while keeping the neural checkpoint fixed.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "saved_cases": len(initial),
                      "step_iteration_quantiles": report["saved_evidence"]["step_iteration_quantiles"]}))


if __name__ == "__main__":
    main()
