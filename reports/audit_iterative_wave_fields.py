"""Coordinator cross-check of largest-grid wave displacement, independent of owner helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/iterative05"
OUTPUT = ROOT / "reports/2026-09-11-iterative-fom-multiresolution.coordinator-audit.json"


def main():
    native = RECORD / "cluster/out/pilot"
    result_path = native / "result.json"
    raw = result_path.read_bytes()
    result = json.loads(raw)
    cleanup = json.loads((RECORD / "cleanup.json").read_text())
    assert result["complete"] and cleanup["all_three_manifests_verified"]
    assert cleanup["remote_deleted_and_absence_checked"]
    cfg = result["config"]
    n, case = max(cfg["meshes"]), min(cfg["validation_indices"])
    methods = [cfg["primary_method"], "cg_1e-06"]
    checks = []
    for boundary in cfg["boundaries"]:
        with np.load(native / f"reference_{boundary}_{n}_{case}.npz") as f:
            truth = f["u"]
        weights = np.ones(truth.shape[-1], dtype=np.float64)
        if boundary == "absorbing":
            weights[[0, -1]] = .5
        # Uniform h^2 cancels from the relative norm. Trapezoid endpoint
        # weights remain necessary on absorbing grids with boundary nodes.
        spatial_weights = weights[:, None] * weights[None, :]
        norm = np.array([np.sqrt(np.sum(frame * frame * spatial_weights)) for frame in truth])
        for method in methods:
            rows = [r for r in result["invocations"] if
                    (r["boundary"], r["intervals"], r["case"], r["method"]) == (boundary, n, case, method)]
            assert len(rows) == cfg["repetitions"]
            with np.load(native / rows[0]["field_artifact"]) as f:
                predicted = f["u"]
            assert predicted.dtype == np.float64 and predicted.shape == truth.shape
            assert np.isfinite(predicted).all()
            digest = hashlib.sha256(predicted.tobytes()).hexdigest()
            discrepancy = np.array([
                np.sqrt(np.sum((u - ref) ** 2 * spatial_weights))
                for u, ref in zip(predicted, truth)
            ])
            defined = norm > 1e-14 * norm[0]
            current = np.divide(discrepancy, norm, out=np.full_like(norm, np.nan), where=defined)
            initial = discrepancy / norm[0]
            errors = []
            for row in rows:
                assert row["output_sha256"]["u"] == digest
                saved = row["same_grid_discrepancy"]["displacement"]
                saved_current = np.array([np.nan if x is None else x for x in saved["current_relative"]])
                saved_initial = np.asarray(saved["initial_normalized"])
                np.testing.assert_allclose(current, saved_current, rtol=1e-11, atol=1e-11, equal_nan=True)
                np.testing.assert_allclose(initial, saved_initial, rtol=1e-11, atol=1e-11)
                errors.extend(np.abs(current[defined] - saved_current[defined]))
                errors.extend(np.abs(initial - saved_initial))
            checks.append(dict(
                boundary=boundary, intervals=n, case=case, method=method,
                timed_invocations=len(rows), field_sha256=digest,
                worst_current_relative=float(np.nanmax(current)),
                worst_initial_normalized=float(initial.max()),
                maximum_metric_difference=float(max(errors)),
            ))
            print(boundary, method, checks[-1]["maximum_metric_difference"], flush=True)
    report = dict(
        passed=True, result_sha256=hashlib.sha256(raw).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="First declared development case, largest requested mesh, both boundaries, frozen NMROM and tight CG; all output times and all retained repetition identities. Displacement only; owner audit covers every field and velocity/energy diagnostics.",
        checks=checks, distinct_displacement_fields=len(checks),
        timed_invocation_identities=sum(c["timed_invocations"] for c in checks),
        maximum_metric_difference=max(c["maximum_metric_difference"] for c in checks),
    )
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
