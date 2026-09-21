"""Recompute diag03 output-time errors from saved fields with NumPy only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify_diag import check, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.out / "summary.json").read_text())
    if summary.get("final_cohort_opened"):
        raise SystemExit("final cohort was opened")
    truth = np.load(args.out / "dev_output.npy")
    failures = []
    checked = []
    for rank, row in summary["ranks"].items():
        for mode, by_dt in row.items():
            for tag, block in by_dt.items():
                recon = np.load(args.out / f"{mode}_r{rank}_dt{tag}.npy")
                errors = np.empty((len(truth), truth.shape[1]))
                for case in range(len(truth)):
                    errors[case] = relative(recon[case], truth[case], truth[case, 0])
                checked.append(dict(rank=int(rank), mode=mode, dt=block["dt"], gap=check(
                    block["errors"], errors, f"{mode}-{rank}-{tag}", failures)))
    report = dict(passed=not failures, checked=checked, failures=failures, final_cohort_opened=False)
    (args.out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], failures=len(failures))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
