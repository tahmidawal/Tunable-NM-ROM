"""Recompute diag06 errors and parity gaps from saved fields."""
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


def truth_errors(predicted, truth):
    errors = np.empty((len(truth), truth.shape[1]))
    for case in range(len(truth)):
        errors[case] = relative(predicted[case], truth[case], truth[case, 0])
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.out / "summary.json").read_text())
    if summary.get("final_cohort_opened") or summary.get("sealed_seed_opened"):
        raise SystemExit("a sealed cohort was opened")
    truth = np.load(args.out / "dev_truth.npy")
    tracker = np.load(args.out / "tracker.npy")
    failures = []
    checked = []
    arms = [("tracker", "tracker.npy", summary["tracker"], tracker)]
    arms.append(("coeff_full", "coeff_full.npy", summary["coeff_full"], tracker))
    for record in summary["truncations"]:
        arms.append((record["name"], record["file"], record, tracker))
    arms.append(("f32", "coeff_f32.npy", summary["f32"], np.load(args.out / "coeff_full.npy")))
    if summary["fourier"] is not None:
        arms.append(("fourier", "fourier.npy", summary["fourier"], np.load(args.out / "fourier_grid.npy")))
    for name, filename, record, reference in arms:
        fields = np.load(args.out / filename)
        recomputed = truth_errors(fields, truth)
        checked.append(dict(kind=name, gap=check(record["errors"], recomputed, name, failures)))
        if name != "tracker":
            got = gaps(fields, reference)
            stored = np.asarray(record["parity_gaps"], dtype=np.float64)
            gap = float(np.max(np.abs(stored - np.asarray(got))))
            if gap > 1e-12:
                failures.append(dict(label=f"{name}-parity", max_abs_gap=gap))
            checked.append(dict(kind=f"{name}-parity", gap=gap))
    for tag, block in summary["fom"].items():
        fields = np.load(args.out / f"fom_dt{tag}.npy")
        recomputed = truth_errors(fields, truth)
        checked.append(dict(kind=tag, gap=check(block["errors"], recomputed, f"fom-{tag}", failures)))
    report = dict(passed=not failures, checked=checked, failures=failures, final_cohort_opened=False)
    (args.out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], failures=len(failures))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
