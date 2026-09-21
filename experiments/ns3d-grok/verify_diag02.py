"""Recompute diag02 relative errors from saved fields with NumPy only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify_diag import check, file_dt, relative


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
        for kind, filename in (
            ("per_time_oracle", f"projection_oracle_r{rank}.npy"),
            ("frozen_c0", f"projection_frozen_r{rank}.npy"),
        ):
            recon = np.load(args.out / filename)
            errors = np.empty((len(truth), truth.shape[1]))
            for case in range(len(truth)):
                errors[case] = relative(recon[case], truth[case], truth[case, 0])
            checked.append(dict(rank=int(rank), kind=kind, gap=check(
                row[kind]["errors"], errors, f"{kind}-{rank}", failures)))
        for tag, block in row["galerkin"].items():
            recon = np.load(args.out / f"galerkin_r{rank}_dt{file_dt(block['dt'])}.npy")
            if tag != file_dt(block["dt"]):
                failures.append(dict(label=f"tag-{rank}-{tag}", max_abs_gap=None, reason="dt tag"))
            errors = np.empty((len(truth), truth.shape[1]))
            for case in range(len(truth)):
                errors[case] = relative(recon[case], truth[case], truth[case, 0])
            checked.append(dict(rank=int(rank), dt=block["dt"], gap=check(
                block["errors"], errors, f"galerkin-{rank}-{tag}", failures)))
    report = dict(passed=not failures, checked=checked, failures=failures,
                  final_cohort_opened=False)
    (args.out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], failures=len(failures))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
