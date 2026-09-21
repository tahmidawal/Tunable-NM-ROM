"""Recompute diag01 relative errors from saved fields with NumPy only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def relative(pred, ref, initial):
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    num = np.sqrt(np.sum(diff * diff, axis=tuple(range(1, diff.ndim))))
    den = np.sqrt(np.sum(np.asarray(initial, dtype=np.float64) ** 2))
    return num / max(float(den), 1e-300)


def file_dt(value):
    return f"{float(value):.4f}".replace(".", "p")


def check(stored, recomputed, label, failures):
    stored = np.asarray(stored, dtype=np.float64)
    recomputed = np.asarray(recomputed, dtype=np.float64)
    if stored.shape != recomputed.shape or not np.all(np.isfinite(stored)) or not np.all(np.isfinite(recomputed)):
        failures.append(dict(label=label, max_abs_gap=None, reason="shape or nonfinite"))
        return None
    gap = float(np.max(np.abs(stored - recomputed)))
    if not np.isfinite(gap) or gap > 1e-12:
        failures.append(dict(label=label, max_abs_gap=gap))
    return gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.out / "summary.json").read_text())
    if summary.get("final_cohort_opened"):
        raise SystemExit("final cohort was opened")
    truth = np.load(args.out / "dev_truth.npy")
    failures = []
    checked = []
    for rank, row in summary["ranks"].items():
        projection = np.load(args.out / f"projection_r{rank}.npy")
        truth_dt = row["galerkin_dt_truth"]["dt"]
        galerkin = np.load(args.out / f"galerkin_r{rank}_dt{file_dt(truth_dt)}.npy")
        onestep = np.load(args.out / f"onestep_r{rank}.npy")
        floor = np.empty((len(truth), truth.shape[1]))
        rollout = np.empty_like(floor)
        for case in range(len(truth)):
            floor[case] = relative(projection[case], truth[case], truth[case, 0])
            rollout[case] = relative(galerkin[case], truth[case], truth[case, 0])
        checked.append(dict(
            rank=int(rank),
            floor_gap=check(row["floor"]["errors"], floor, f"floor-{rank}", failures),
            rollout_gap=check(row["galerkin_dt_truth"]["errors"], rollout, f"rollout-{rank}", failures),
        ))
        local = np.empty((len(truth), truth.shape[1] - 1))
        for case in range(len(truth)):
            for instant in range(truth.shape[1] - 1):
                num = np.linalg.norm(onestep[case, instant] - truth[case, instant + 1])
                local[case, instant] = num / np.linalg.norm(truth[case, 0])
        checked[-1]["onestep_gap"] = check(row["one_interval"]["errors"], local, f"onestep-{rank}", failures)
        if "galerkin_dt_rom" in row:
            extra = np.load(args.out / f"galerkin_r{rank}_dt{file_dt(row['galerkin_dt_rom']['dt'])}.npy")
            extra_errors = np.empty_like(rollout)
            for case in range(len(truth)):
                extra_errors[case] = relative(extra[case], truth[case], truth[case, 0])
            checked[-1]["rom_dt_gap"] = check(
                row["galerkin_dt_rom"]["errors"], extra_errors, f"romdt-{rank}", failures)
    for path in sorted(args.out.glob("oracle_shift_r*.npy")):
        rank = path.stem.split("oracle_shift_r", 1)[1]
        recon = np.load(path)
        errors = np.empty((len(truth), truth.shape[1]))
        for case in range(len(truth)):
            errors[case] = relative(recon[case], truth[case], truth[case, 0])
        checked.append(dict(rank=int(rank), oracle_gap=check(
            summary["oracle_shift"]["ranks"][rank]["errors"], errors, f"oracle-{rank}", failures)))
    for path in sorted(args.out.glob("pca_k*_r*.npy")):
        stem = path.stem
        width, rank = stem.split("_r")
        width = width.split("pca_k", 1)[1]
        recon = np.load(path)
        errors = np.empty((len(truth), truth.shape[1]))
        for case in range(len(truth)):
            errors[case] = relative(recon[case], truth[case], truth[case, 0])
        stored = summary["ranks"][rank]["affine_pca_head"][width]["errors"]
        checked.append(dict(rank=int(rank), pca_k=int(width), pca_gap=check(
            stored, errors, f"pca-{width}-{rank}", failures)))
    report = dict(passed=not failures, checked=checked, failures=failures,
                  final_cohort_opened=False, truth_shape=list(truth.shape))
    (args.out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], failures=len(failures))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
