"""Independent NumPy recomputation of a mesh job's reported errors (no JAX, none of the
driver's error code). Runs in the job, in a separate process, before the field files are
deleted.

The required field sets are derived from summary.json: every finite reduced-model arm,
the frame-frozen control, and every CNAB2 row that has errors. A missing file, a wrong
shape, a nonfinite value or a disagreement fails the audit. Each set is reduced with an
einsum of squared differences and compared with summary.json: per-case per-time errors,
evolved worst / median, the count over 5 %. Bounds: 1e-12 (float64 sets) and 2e-6 (float32
CNAB2 sets), relative to max(1, error).

Must-fail control: a copy of the first required set with one case perturbed by 1e-3 of
its initial norm is passed through the SAME accept function and must be rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOUND = {"float64": 1e-12, "float32": 2e-6}


def relative(fields, truth):
    diff = fields.astype(np.float64) - truth
    num = np.sqrt(np.einsum("ctijkl,ctijkl->ct", diff, diff))
    den = np.sqrt(np.einsum("cijkl,cijkl->c", truth[:, 0], truth[:, 0]))
    return num / den[:, None]


def required(report):
    """(key, claimed errors, claimed stats or None, dtype) for every set that must exist."""
    out = []
    for name, row in report["rollouts"].items():
        if row["finite"]:
            out.append((name, row["errors"], row["stats"], "float64"))
    out.append(("control_frame_zero", report["control_frame_zero_errors"],
                report["control_frame_zero"], "float64"))
    for st, row in report["cnab2"].items():
        if row["errors"] is not None:
            out.append((f"cnab2_s{st}", row["errors"], row["stats"], "float32"))
    return out


def accept(fields, truth, claimed_errors, stats, dtype, target):
    """Returns (ok, gap, evolved, reason)."""
    if fields.shape != truth.shape:
        return False, float("inf"), None, f"shape {fields.shape} != {truth.shape}"
    if str(fields.dtype) != dtype:
        return False, float("inf"), None, f"dtype {fields.dtype} != {dtype}"
    if not np.all(np.isfinite(fields)):
        return False, float("inf"), None, "nonfinite field values"
    errors = relative(fields, truth)
    claimed_errors = np.asarray(claimed_errors, dtype=np.float64)
    if claimed_errors.shape != truth.shape[:2]:
        return False, float("inf"), None, f"claimed error shape {claimed_errors.shape}"
    for key in ("evolved_worst", "evolved_median"):
        if not np.isfinite(stats[key]):
            return False, float("inf"), None, f"nonfinite claimed {key}"
    if not (np.all(np.isfinite(errors)) and np.all(np.isfinite(claimed_errors))):
        return False, float("inf"), None, "nonfinite errors"
    evolved = errors[:, 1:].max(axis=1)
    scale = max(1.0, float(np.max(np.abs(claimed_errors))))
    gap = float(np.max(np.abs(errors - claimed_errors))) / scale
    gap = max(gap, abs(float(evolved.max()) - stats["evolved_worst"]) / scale,
              abs(float(np.median(evolved)) - stats["evolved_median"]) / scale)
    if int(np.sum(evolved > target)) != stats["cases_evolved_over_target"]:
        return False, gap, evolved, "count over target disagrees"
    if not gap <= BOUND[dtype]:
        return False, gap, evolved, f"gap {gap:.3e} > {BOUND[dtype]}"
    return True, gap, evolved, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = json.loads((args.out / "summary.json").read_text())
    target = float(report["config"]["target_relative"])
    truth = np.load(args.out / "dev_truth.npy")
    if not np.all(np.isfinite(truth)):
        raise RuntimeError("nonfinite truth")
    sets = required(report)
    key0, err0, st0, dt0 = sets[0]
    bad = np.load(args.out / f"fields_{key0}.npy")
    bad = bad.copy()
    bad[0, 1:] += (1e-3 * np.linalg.norm(truth[0, 0]) / np.sqrt(truth[0, 0].size)).astype(bad.dtype)
    ok_bad, gap_bad, _, why_bad = accept(bad, truth, err0, st0, dt0, target)
    control_rejected = not ok_bad
    print(f"control: perturbed copy of {key0} gap {gap_bad:.3e} rejected={control_rejected} ({why_bad})",
          flush=True)
    del bad
    checked, failed = {}, {}
    for key, errs, stats, dtype in sets:
        path = args.out / f"fields_{key}.npy"
        if not path.exists():
            failed[key] = "missing field file"
            continue
        ok, gap, evolved, why = accept(np.load(path), truth, errs, stats, dtype, target)
        checked[key] = dict(dtype=dtype, gap=gap, ok=ok, reason=why,
                            evolved_worst=None if evolved is None else float(evolved.max()))
        if not ok:
            failed[key] = why
        print(key, json.dumps(checked[key]), flush=True)
    extra = sorted({p.stem.replace("fields_", "") for p in args.out.glob("fields_*.npy")}
                   - {k for k, *_ in sets})
    passed = bool(control_rejected and not failed and not extra)
    payload = dict(schema="ns3d-coordnet-verify-v2", passed=passed, control_rejected=control_rejected,
                   control_gap=gap_bad, settings=checked, failed=failed, unexpected_files=extra,
                   job_id=report.get("job_id"), source_commit=report.get("source_commit"))
    (args.out / "verify.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(f"audit failed: control_rejected={control_rejected} failed={failed} extra={extra}")
    print("verify ok", flush=True)


if __name__ == "__main__":
    main()
