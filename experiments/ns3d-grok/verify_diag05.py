"""Recompute diag05 relative errors from saved fields with NumPy only."""
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
    truth = np.load(args.out / "eval_truth.npy")
    failures = []
    checked = []
    tracked = np.load(args.out / "tracked.npy")
    errors = np.empty((len(truth), truth.shape[1]))
    for case in range(len(truth)):
        errors[case] = relative(tracked[case], truth[case], truth[case, 0])
    checked.append(dict(kind="rom", gap=check(summary["rom"]["errors"], errors, "rom", failures)))
    for tag, block in summary["fom"].items():
        fields = np.load(args.out / f"fom_dt{tag}.npy")
        fom_errors = np.empty_like(errors)
        for case in range(len(truth)):
            fom_errors[case] = relative(fields[case], truth[case], truth[case, 0])
        checked.append(dict(kind=tag, gap=check(block["errors"], fom_errors, f"fom-{tag}", failures)))
    report = dict(passed=not failures, checked=checked, failures=failures,
                  final_cohort_opened=bool(summary.get("final_cohort_opened")))
    (args.out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], failures=len(failures),
                          final=report["final_cohort_opened"])), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
