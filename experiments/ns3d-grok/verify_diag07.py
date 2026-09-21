"""Recompute diag07 errors and the parity gap from saved fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify_diag import check, relative


def gaps(predicted, reference):
    values = []
    for case in range(len(reference)):
        denominator = np.linalg.norm(reference[case])
        values.append(float(np.linalg.norm(predicted[case] - reference[case]) / max(denominator, 1e-300)))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.out / "summary.json").read_text())
    truth = np.load(args.out / "eval_truth.npy")
    tracker = np.load(args.out / "tracker.npy")
    coeff = np.load(args.out / "coeff.npy")
    failures = []
    checked = []
    for name, fields, record in (
        ("tracker", tracker, summary["tracker"]),
        ("coeff", coeff, summary["coeff"]),
    ):
        recomputed = np.empty((len(truth), truth.shape[1]))
        for case in range(len(truth)):
            recomputed[case] = relative(fields[case], truth[case], truth[case, 0])
        checked.append(dict(kind=name, gap=check(record["errors"], recomputed, name, failures)))
    parity = gaps(coeff, tracker)
    stored = np.asarray(summary["coeff"]["parity_gaps"], dtype=np.float64)
    gap = float(np.max(np.abs(stored - np.asarray(parity))))
    if gap > 1e-12:
        failures.append(dict(label="coeff-parity", max_abs_gap=gap))
    checked.append(dict(kind="coeff-parity", gap=gap))
    for tag, block in summary["fom"].items():
        fields = np.load(args.out / f"fom_dt{tag}.npy")
        recomputed = np.empty((len(truth), truth.shape[1]))
        for case in range(len(truth)):
            recomputed[case] = relative(fields[case], truth[case], truth[case, 0])
        checked.append(dict(kind=tag, gap=check(block["errors"], recomputed, f"fom-{tag}", failures)))
    report = dict(passed=not failures, checked=checked, failures=failures,
                  final_cohort_opened=bool(summary.get("final_cohort_opened")))
    (args.out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], failures=len(failures),
                          final=report["final_cohort_opened"])), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
