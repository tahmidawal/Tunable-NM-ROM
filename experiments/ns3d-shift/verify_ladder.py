"""Independent NumPy recomputation of a ladder job's reported errors.

Runs in the job, before the large field files are deleted. No JAX, no reuse of the
driver's error code: a different reduction, read back from disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify_pilot import relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.out / "summary.json").read_text())
    truth = np.load(args.out / "dev_truth.npy")
    checked = {}
    worst_gap = 0.0
    for path in sorted(args.out.glob("fields_*.npy")):
        key = path.stem.replace("fields_", "")
        errors = relative(np.load(path), truth)
        evolved = errors[:, 1:].max(axis=1)
        claimed = report["frontier"][key]["stats"]
        gap = max(abs(float(evolved.max()) - claimed["evolved_worst"]),
                  abs(float(np.median(evolved)) - claimed["evolved_median"]))
        worst_gap = max(worst_gap, gap)
        if int(np.sum(evolved > 0.05)) != claimed["cases_evolved_over_target"]:
            raise RuntimeError(f"{key}: case count over target disagrees")
        checked[key] = dict(evolved_worst=float(evolved.max()),
                            evolved_median=float(np.median(evolved)), gap=float(gap))
        print(key, json.dumps(checked[key]), flush=True)
    if worst_gap > 1e-12:
        raise RuntimeError(f"independent recomputation disagrees by {worst_gap}")

    # the parity gate on the driver fix must have been recorded and met
    parity = [v.get("parity_vs_reference_lm") for v in report["frontier"].values()
              if "parity_vs_reference_lm" in v]
    if not parity:
        raise RuntimeError("no frontier setting was parity-checked against the LM arm")
    if max(parity) > 1e-8 and min(parity) > 1e-8:
        raise RuntimeError(f"no setting met the 1e-8 parity gate: {parity}")
    payload = dict(schema="ns3d-shift-ladder-verify-v1", worst_gap=float(worst_gap),
                   settings=checked, parity_vs_reference_lm=parity,
                   job_id=report.get("job_id"), source_commit=report.get("source_commit"))
    (args.out / "verify.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("verify ok", worst_gap, "parity", min(parity), flush=True)


if __name__ == "__main__":
    main()
