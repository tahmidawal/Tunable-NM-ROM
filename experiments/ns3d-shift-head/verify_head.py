"""Independent NumPy recomputation of a head-ladder job's reported errors.

Runs in the job, in a separate process, before the large field files are deleted.
No JAX and none of the driver's error code: every saved field set is read back
from disk, reduced with a different formula (einsum of squared differences), and
compared against the numbers in summary.json. The reduced-model rows are saved
in float64 (bound 1e-12); the FOM rows in float32 (bound 2e-6 absolute).

A must-fail control is run first: a copy of one field set with one case
perturbed by 1e-3 of its initial norm must be REJECTED by the same comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def relative(fields, truth):
    diff = fields.astype(np.float64) - truth
    num = np.sqrt(np.einsum("ctijkl,ctijkl->ct", diff, diff))
    den = np.sqrt(np.einsum("cijkl,cijkl->c", truth[:, 0], truth[:, 0]))
    return num / den[:, None]


def claimed(report, key):
    if key.startswith("lin_"):
        return report["linear"][key]
    if key.startswith("cnab2_s"):
        return report["cnab2"][key[len("cnab2_s"):]]
    if key.startswith("span"):
        return report["span"][key]
    if key.startswith("fd_"):
        return report["fd_cg"][key]
    return report["frontier"][key]


def compare(errors, entry, target=0.05):
    evolved = errors[:, 1:].max(axis=1)
    stats = entry["stats"]
    gap = max(abs(float(evolved.max()) - stats["evolved_worst"]),
              abs(float(np.median(evolved)) - stats["evolved_median"]),
              float(np.max(np.abs(errors - np.asarray(entry["errors"])))))
    count_ok = int(np.sum(evolved > target)) == stats["cases_evolved_over_target"]
    return gap, count_ok, evolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.out / "summary.json").read_text())
    truth = np.load(args.out / "dev_truth.npy")
    paths = sorted(args.out.glob("fields_*.npy"))
    if not paths:
        raise RuntimeError("no saved fields to verify")

    # must-fail control on real data
    first = paths[0]
    key0 = first.stem.replace("fields_", "")
    bad = np.load(first).astype(np.float64)
    bad[0, 1:] += 1e-3 * np.linalg.norm(truth[0, 0]) / np.sqrt(truth[0, 0].size)
    gap_bad, _, _ = compare(relative(bad, truth), claimed(report, key0))
    control_rejected = gap_bad > 2e-6
    print(f"control: perturbed copy of {key0} gap {gap_bad:.3e} rejected={control_rejected}",
          flush=True)
    if not control_rejected:
        raise RuntimeError("the audit cannot detect a perturbed field set")
    del bad

    checked = {}
    worst64 = worst32 = 0.0
    for path in paths:
        key = path.stem.replace("fields_", "")
        fields = np.load(path)
        errors = relative(fields, truth)
        gap, count_ok, evolved = compare(errors, claimed(report, key))
        if not count_ok:
            raise RuntimeError(f"{key}: case count over target disagrees")
        if fields.dtype == np.float64:
            worst64 = max(worst64, gap)
        else:
            worst32 = max(worst32, gap)
        checked[key] = dict(dtype=str(fields.dtype), evolved_worst=float(evolved.max()),
                            evolved_median=float(np.median(evolved)), gap=float(gap))
        print(key, json.dumps(checked[key]), flush=True)
    if worst64 > 1e-12 or worst32 > 2e-6:
        raise RuntimeError(f"independent recomputation disagrees: f64 {worst64} f32 {worst32}")
    payload = dict(schema="ns3d-shift-head-verify-v1", worst_gap_float64=float(worst64),
                   worst_gap_float32=float(worst32), control_gap=float(gap_bad),
                   control_rejected=bool(control_rejected), settings=checked,
                   job_id=report.get("job_id"), source_commit=report.get("source_commit"))
    (args.out / "verify.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("verify ok", worst64, worst32, flush=True)


if __name__ == "__main__":
    main()
