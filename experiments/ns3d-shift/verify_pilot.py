"""Independent NumPy recomputation of the pilot's reported errors.

Reads only the saved fields and the truth, recomputes the relative L2 with a
different reduction than the run used, and compares against summary.json. No
JAX, no GPU, no reuse of the pilot's error code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def relative(pred, truth):
    """(cases, times): sqrt of summed squares, normalized by the initial field."""
    out = np.empty(truth.shape[:2], dtype=np.float64)
    for case in range(truth.shape[0]):
        den = np.sqrt(np.einsum("...,...->", truth[case, 0], truth[case, 0]))
        for instant in range(truth.shape[1]):
            diff = pred[case, instant] - truth[case, instant]
            out[case, instant] = np.sqrt(np.einsum("...,...->", diff, diff)) / max(den, 1e-300)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.out / "summary.json").read_text())
    truth = np.load(args.out / "dev_truth.npy")
    checked = {}
    worst_gap = 0.0
    for path in sorted(args.out.glob("fields_*.npy")) + [args.out / "tracker_fields.npy"]:
        if not path.exists():
            continue
        label = path.stem.replace("fields_", "").replace("tracker_fields", "C_tracker")
        pred = np.load(path)
        errors = relative(pred, truth)
        evolved = errors[:, 1:].max(axis=1)
        mine = dict(evolved_worst=float(evolved.max()),
                    evolved_median=float(np.median(evolved)),
                    cases_over_target=int(np.sum(evolved > 0.05)))
        if label == "C_tracker":
            claimed = report["C_tracker"]["stats"]
        else:
            claimed = report["arms"][label]["stats"]
        gap = max(abs(mine["evolved_worst"] - claimed["evolved_worst"]),
                  abs(mine["evolved_median"] - claimed["evolved_median"]))
        worst_gap = max(worst_gap, gap)
        checked[label] = dict(recomputed=mine, reported=dict(
            evolved_worst=claimed["evolved_worst"],
            evolved_median=claimed["evolved_median"],
            cases_over_target=claimed["cases_evolved_over_target"]), gap=float(gap))
        if mine["cases_over_target"] != claimed["cases_evolved_over_target"]:
            raise RuntimeError(f"{label}: case count over target disagrees")
        print(label, json.dumps(checked[label]), flush=True)
    if worst_gap > 1e-12:
        raise RuntimeError(f"independent recomputation disagrees by {worst_gap}")
    # The must-fail control and the mechanism must still read as designed.
    base = report["config"]["base_rank"]
    assert report["A0_fixed_bank_floor"][str(base)]["cases_evolved_over_target"] > 0, \
        "A0 must-fail control did not fail"
    payload = dict(schema="ns3d-shift-verify-v1", worst_gap=float(worst_gap),
                   arms=checked, job_id=report.get("job_id"),
                   source_commit=report.get("source_commit"))
    (args.out / "verify.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("verify ok", worst_gap, flush=True)


if __name__ == "__main__":
    main()
