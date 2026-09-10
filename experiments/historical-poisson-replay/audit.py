"""Reconstruct historical replay errors and timing aggregates from saved artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path):
    report = json.loads(path.read_text())
    assert report["complete"]
    cfg = report["config"]
    assert cfg["x64"] and cfg["backend"] == "gpu"
    assert cfg["matmul_precision"] == "highest"
    assert cfg["slurm_job"], "local smoke artifacts are not research results"
    assert report["gates"]["truth_residual_max"] < cfg.get("fom_res_tol", 1e-10)
    if cfg["N"] == 1024:
        assert report["gates"]["truth_vs_direct_relative_max"] < 1e-11
    fields = path.parent / report["fields_file"]
    assert sha(fields) == report["fields_sha256"]
    maxima = dict(error=0.0, timing_ms=0.0)
    reconstructed = 0
    with np.load(fields) as saved:
        truth = saved["truth"]
        assert truth.dtype == np.float64
        norms = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
        for row in report["rows"]:
            captured = saved[row["field_key"]]
            assert captured.dtype == np.float64
            ids = row["source_ids"]
            actual = np.linalg.norm((captured[ids]-truth[ids]).reshape(len(ids), -1), axis=1)/norms[ids]
            maxerr = float(np.max(np.abs(actual-row["err_rel_l2_all"])))
            assert maxerr < 1e-12, (row["field_key"], maxerr)
            maxima["error"] = max(maxima["error"], maxerr)
            assert abs(float(np.mean(actual))-row["err_rel_l2"]) < 1e-12
            assert abs(float(np.median(actual))-row["err_rel_l2_median"]) < 1e-12
            assert abs(float(np.max(actual))-row["err_rel_l2_max"]) < 1e-12
            raw = np.asarray([row["time_raw_s"][str(i)] for i in ids])
            assert raw.shape == (len(ids), cfg["reps"])
            assert np.all(np.isfinite(raw)) and np.all(raw > 0)
            time_ms = 1000*np.median(np.median(raw, axis=1))
            diff = abs(float(time_ms)-row["time_ms"])
            assert diff < 1e-12
            maxima["timing_ms"] = max(maxima["timing_ms"], diff)
            outliers = int(np.count_nonzero(raw > 3*np.median(raw, axis=1)[:, None]))
            assert outliers == row["time_outliers_gt_3x_source_median"]
            if row["method"] in ["full", "qf"]:
                reasons = [x["reason"] for x in row["solver_per_source"]]
                assert row["stop_reasons"] == {str(r):reasons.count(r) for r in set(reasons)}
                assert row["censored_frac"] == float(np.mean(np.asarray(reasons) != 2))
                assert all(0 <= x["attempts"] <= cfg["gn_iters"] for x in row["solver_per_source"])
            reconstructed += len(ids)
    for comparison in report["comparisons"]:
        rom = next(r for r in report["rows"] if r["method"] == "qf"
                   and r["cohort"] == comparison["cohort"] and r["tau"] == comparison["tau"])
        eligible = [r for r in report["rows"] if r["method"] == "cg"
                    and r["cohort"] == rom["cohort"] and r["err_rel_l2"] <= rom["err_rel_l2"]]
        chosen = min(eligible, key=lambda r:r["time_ms"])
        assert chosen["fom_tol"] == comparison["cg_tol"]
        assert abs(chosen["time_ms"]/rom["time_ms"]-comparison["cg_over_rom"]) < 1e-12
    return dict(result=str(path), result_sha256=sha(path),
                fields_sha256=sha(fields), N=cfg["N"], job=cfg["slurm_job"],
                reconstructed_timed_field_errors=reconstructed, maximum_differences=maxima,
                checks="full float64 timed fields, every raw timing, error aggregates, stop reasons, CG selection",
                passed=True)


def main():
    run = Path(sys.argv[1])
    audits = [audit(p) for p in sorted((run / "out").glob("poisson_n*.json"))]
    expected_meshes = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    assert len(audits) == expected_meshes
    result = dict(meshes=audits,
                  complete_four_mesh_ladder=bool(len(audits) == 4),
                  reconstructed_timed_field_errors=sum(x["reconstructed_timed_field_errors"] for x in audits),
                  passed=True)
    (run / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
