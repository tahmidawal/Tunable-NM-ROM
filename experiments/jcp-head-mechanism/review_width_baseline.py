"""Independent retained-array audit of the R128 width-study baseline replay."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
WEAK = HERE.parents[2]/"2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility"
JOB = WEAK/"runs/w128"
OUT = JOB/"output"
load = lambda p: json.loads(p.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bank(folder):
    return np.concatenate([np.load(p)["bank"] for p in sorted(folder.glob("bank-columns-*.npz"))], axis=1)


verified = []
for line in (JOB/"RESULTS.sha256").read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    assert sha(JOB/name) == expected, name
    verified.append({"relative_path": name, "sha256": expected})
collection = load(JOB/"collection.json")
assert collection["scheduler_state"] == "COMPLETED" and collection["exit_code"] == "0:0"
assert collection["checksum_verified"] and collection["gpu_log_valid"] and collection["no_environmental_warnings"]
env, summary = load(OUT/"environment.json"), load(OUT/"summary.json")
cfg = env["config"]
assert env["backend"] == "gpu" and env["x64"] and env["matmul_precision"] == "highest"
assert cfg["bank_width"] == 128 and cfg["solve_rows"] == [2048] and cfg["case_ids"] == list(range(6))
assert env["buffer_residual_rows"] == 9216 and env["observer_start_row_zero_based"] == 8192
assert all(r == 128 for r in [*env["tangent_ranks"].values(), *env["initialization_qr_ranks"].values()])
assert env["scaled_tangent_rank"] == 128
G = bank(OUT)
old_banks = {tag: bank(WEAK/"runs"/tag/"output") for tag in ("p03", "p04b")}
new_ops = np.load(OUT/"operators.npz")
operator_gaps = {}
for tag in old_banks:
    old_ops = np.load(WEAK/"runs"/tag/"output/operators.npz")
    assert np.array_equal(new_ops["modes"][:2048], old_ops["modes"][:2048])
    assert np.array_equal(new_ops["lam"][:2048], old_ops["lam"][:2048])
    gap = float(np.linalg.norm(new_ops["A"][:2048]-old_ops["A"][:2048])/np.linalg.norm(old_ops["A"][:2048]))
    assert gap <= 1e-12
    operator_gaps[tag] = gap

records = []
for case_id in range(6):
    tag = "p03" if case_id < 2 else "p04b"
    old_root = WEAK/"runs"/tag/"output"/f"case-{case_id:02d}"
    folder = OUT/f"case-{case_id:02d}"
    cs = summary["cases"][str(case_id)]
    arm = cs["arms"]["2048"]
    assert cs["reference_eligible"] and cs["initialization_control"]["passed"] and arm["numerically_eligible"]
    assert all(p["passed"] for p in arm["parity"])
    assert all(s["same_rank"] and s["rescaling_invariance_relative"] <= 1e-8 for s in arm["reached_spectra"])
    reference = np.load(folder/"reference-1.npz")["fields"]
    norm0 = np.linalg.norm(reference[0])
    refs = []
    for level in (0, 1):
        z = np.load(folder/f"reference-{level}.npz")
        assert np.isfinite(z["fields"]).all() and z["iterations"].max() < 20 and z["residual"].max() <= 1e-10
        assert cs["reference_controls"][level]["arithmetic_parity"]["passed"]
        refs.append(z["fields"])
    assert np.max(np.linalg.norm(refs[0]-refs[1], axis=1))/norm0 <= cfg["reference_max_disagreement"]
    fields = {}
    for level, gtol in cfg["solver_gtols"].items():
        reps = cfg["timing_repetitions"] if level == "tight" else 1
        for rep in range(reps):
            name = f"m2048-{level}-rep{rep}.npz"
            z, old = np.load(folder/name), np.load(old_root/name)
            stats = z["stats"]
            assert np.isfinite(z["coeff"]).all() and np.all(stats[:, 1] == 1) and np.all(stats[:, 2] <= gtol)
            assert np.all(stats[:, 0] <= cfg["solver_max_iterations"])
            field = z["coeff"] @ G.T
            old_field = old["coeff"] @ old_banks[tag].T
            err = np.linalg.norm(field-reference, axis=1)/norm0
            saved_metric_gap = float(np.max(np.abs(err-z["metrics"][:, 0])))
            curve_gap = float(np.max(np.abs(err-old["metrics"][:, 0])))
            field_gap = float(np.max(np.linalg.norm(field-old_field, axis=1))/norm0)
            assert saved_metric_gap <= 1e-10 and curve_gap <= 1e-8 and field_gap <= arm["tight_tighter_threshold"]
            records.append({"case": case_id, "level": level, "repetition": rep,
                "maximum_error": float(err.max()), "maximum_stationarity": float(stats[:, 2].max()),
                "saved_metric_gap": saved_metric_gap, "accepted_baseline_error_curve_gap": curve_gap,
                "accepted_baseline_field_gap": field_gap, "source_file_sha256": sha(folder/name),
                "accepted_baseline_file_sha256": sha(old_root/name)})
            fields[level] = field
    pair_gap = float(np.max(np.linalg.norm(fields["tight"]-fields["tighter"], axis=1))/norm0)
    assert pair_gap <= arm["tight_tighter_threshold"] and arm["solver_pair_eligible"]
    residuals = np.load(folder/"m2048-residuals.npz")
    assert residuals["scaled"].shape == (50, 9216)
    for key, values in (("independent_scaled_shell_max", residuals["scaled"]), ("independent_unscaled_shell_max", residuals["unscaled"])):
        assert np.isclose(np.max(np.linalg.norm(values[:, 8192:], axis=1)), arm[key], rtol=1e-12, atol=1e-15)

review = {"reviewer": "head_mechanism_run", "decision": "R128 baseline replay accepted; registered wider jobs may proceed under parent's conditional release, with all future gates unchanged.",
    "job_id": env["job_id"], "verified_file_count": len(verified), "rehashed_files": verified,
    "collection": {k: v for k, v in collection.items() if k != "files"},
    "summary_sha256": sha(OUT/"summary.json"), "operator_prefix_relative_gaps": operator_gaps,
    "maximum_saved_metric_gap": max(r["saved_metric_gap"] for r in records),
    "maximum_original_error_curve_gap": max(r["accepted_baseline_error_curve_gap"] for r in records),
    "maximum_original_physical_field_gap": max(r["accepted_baseline_field_gap"] for r in records),
    "worst_primary_error": max(r["maximum_error"] for r in records if r["level"] == "tighter"),
    "records": records,
    "scope": "Independent CPU inspection/reconstruction of retained arrays only, all cases and every saved repetition. This validates baseline reproduction and the common observer, not wider-prefix accuracy or timing."}
path = HERE/"reviews/p2-W1-baseline-independent-review.json"
path.write_text(json.dumps(review, indent=2)+"\n")
print(json.dumps({k:v for k,v in review.items() if k not in ("rehashed_files", "records", "collection")}, indent=2))
