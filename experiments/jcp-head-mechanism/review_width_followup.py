"""Read-only NumPy audit of retained W0 arrays; no simulation or GPU work."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEAK = HERE.parents[2] / "2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility"
JOB = WEAK / "runs/w00"
OUT = JOB / "output"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


hashes = []
for line in (JOB / "RESULTS.sha256").read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = JOB / name
    actual = sha(path)
    assert expected == actual, name
    hashes.append({"relative_path": name, "sha256": actual})

summary = load(OUT / "summary.json")
env = load(OUT / "environment.json")
cohort = load(OUT / "cohort-drawing.json")
assert set(summary["cases"]) == set(map(str, range(6)))
assert cohort["guard_passed"] and cohort["legacy_same_host_bitwise"]
assert env["backend"] == "gpu" and env["x64"] and env["matmul_precision"] == "highest"
assert env["checkpoint_sha256"] == sha(WEAK / "inputs/frozen.pkl")
assert env["rotation_sha256"] == sha(WEAK / "inputs/rotation_R512.npz")

G = np.concatenate([np.load(p)["bank"] for p in sorted(OUT.glob("bank-columns-*.npz"))], axis=1)
n = env["config"]["mesh_intervals"]
assert G.shape == ((n-1)**2, 512) and G.dtype == np.float64 and np.isfinite(G).all()
R = np.load(OUT / "qr-factor.npz")["Rmass"]
gram = G.T @ G / n**2
gram_factor_gap = float(np.linalg.norm(gram-R.T @ R) / np.linalg.norm(gram))
assert gram_factor_gap <= 1e-10
ranks = {}
for width in (128, 256, 512):
    eig = np.linalg.eigvalsh(gram[:width, :width])
    sing = np.linalg.svd(R[:width, :width], compute_uv=False)
    ranks[str(width)] = {
        "gram": {str(t): int(np.sum(eig > t*eig[-1])) for t in (1e-10, 1e-12, 1e-14)},
        "triangular": {str(t): int(np.sum(sing**2 > t*sing[0]**2)) for t in (1e-10, 1e-12, 1e-14)},
        "physical_condition": float(sing[0]/sing[-1]),
    }
    assert all(k == width for kind in ("gram", "triangular") for k in ranks[str(width)][kind].values())

records, reference_records = [], []
for case_id in range(6):
    folder = OUT / f"case-{case_id:02d}"
    refs = []
    controls = load(folder / "reference-controls.json")
    for level in (0, 1):
        z = np.load(folder / f"reference-{level}.npz")
        ref = z["fields"]
        old = np.load(folder / f"original-reference-{level}.npz")["fields"]
        arithmetic = np.load(folder / f"reference-arithmetic-{level}.npz")
        old_gap = float(np.max(np.linalg.norm(ref-old, axis=1))/np.linalg.norm(ref[0]))
        assert np.isfinite(ref).all() and int(z["iterations"].max()) < 20
        assert float(z["residual"].max()) <= 1e-10
        assert float(arithmetic["original_relative_residual"].max()) <= 1e-10
        assert float(arithmetic["primal_difference"].max()) <= 1e-12 and old_gap <= 1e-8
        assert controls["levels"][level]["passed"]
        refs.append(ref)
    truth_gap = float(np.max(np.linalg.norm(refs[0]-refs[1], axis=1))/np.linalg.norm(refs[0][0]))
    assert truth_gap <= env["config"]["reference_max_disagreement"]
    reference_records.append({"case": case_id, "tight_tighter_gap": truth_gap})
    ref = refs[1]
    norm0 = np.linalg.norm(ref[0])
    previous = None
    for width in (128, 256, 512):
        data = np.load(folder / f"projection-r{width}.npz")
        basis = G[:, :width]
        field = data["coeff"] @ basis.T
        error = np.linalg.norm(field-ref, axis=1)/norm0
        metric_gap = float(np.max(np.abs(error-data["error"])))
        ortho = float(np.linalg.norm((field-ref) @ basis)/(np.linalg.norm(ref)*np.linalg.norm(basis)))
        nested_gap = 0.0 if previous is None else float(np.max(error-previous))
        assert metric_gap <= 1e-10 and ortho <= 1e-9 and nested_gap <= 1e-10
        row = {"case": case_id, "width": width, "initial_error": float(error[0]),
               "maximum_all_times_error": float(error.max()), "metric_gap": metric_gap,
               "orthogonality_defect": ortho, "maximum_nesting_violation": nested_gap,
               "errors": error.tolist()}
        if width == 128:
            old_path = WEAK / "runs" / ("p03" if case_id < 2 else "p04b") / "output" / f"case-{case_id:02d}" / "projection.npz"
            old_error = np.load(old_path)["error"]
            row["accepted_baseline_error_curve_gap"] = float(np.max(np.abs(error-old_error)))
            row["accepted_baseline_sha256"] = sha(old_path)
            assert row["accepted_baseline_error_curve_gap"] <= 1e-10
        records.append(row)
        previous = error

aggregates = {}
for width in (128, 256, 512):
    rec = [r for r in records if r["width"] == width]
    aggregates[str(width)] = {
        "worst_initial": max(r["initial_error"] for r in rec),
        "worst_all_times": max(r["maximum_all_times_error"] for r in rec),
    }
base = aggregates["128"]
admitted = [w for w in (256, 512) if all(aggregates[str(w)][k] <= .8*base[k] for k in base)]
assert admitted == [int(w) for w, allowed in summary["provisional_admission"].items() if allowed]
log = (JOB / "job.log").read_text()
bad_terms = [x for x in ("captured constant", "out of memory", "oom", "disk full", "traceback", "warning", "failed to", "error:") if x in log.lower()]
assert "jax_backend=gpu" in log and "projection_completed=True" in log and not bad_terms, bad_terms
collection_path = JOB / "collection.json"
collection = load(collection_path) if collection_path.exists() else None
if collection is not None:
    assert collection["scheduler_state"] == "COMPLETED" and collection["exit_code"] == "0:0"
    assert collection["checksum_verified"] and collection["gpu_log_valid"] and collection["no_environmental_warnings"]
result = {
    "reviewer": "head_mechanism_run", "job_id": env["job_id"],
    "scope": "Independent CPU postprocessing of retained arrays only; no new simulation, training, cohort or GPU job.",
    "numerical_W0_screen_accepted": True, "admitted_widths_from_raw_arrays": admitted,
    "scheduler_collection_record_available": collection is not None,
    "collection": collection, "summary_sha256": sha(OUT / "summary.json"),
    "results_manifest_sha256": sha(JOB / "RESULTS.sha256"), "verified_file_count": len(hashes),
    "rehashed_files": hashes, "qr_gram_factor_relative_gap": gram_factor_gap,
    "physical_ranks": ranks, "reference_checks": reference_records, "aggregates": aggregates,
    "maximum_saved_metric_gap": max(r["metric_gap"] for r in records),
    "maximum_R128_reproduction_gap": max(r.get("accepted_baseline_error_curve_gap", 0.) for r in records),
    "records": records,
    "interpretation": "Both wider frozen prefixes pass the registered projection screen. This establishes approximation capacity on these six discrete reference trajectories, not evolved PDE accuracy. W1 requires separate implementation/admission clearance and parent release.",
}
destination = HERE / "reviews/p2-W0-independent-result-review.json"
destination.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("numerical_W0_screen_accepted", "admitted_widths_from_raw_arrays", "verified_file_count", "maximum_saved_metric_gap", "maximum_R128_reproduction_gap", "aggregates", "scheduler_collection_record_available")}, indent=2))
