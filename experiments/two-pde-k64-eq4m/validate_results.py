"""Fail-fast schema and measurement audit for the paired k sweep."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

KS = (4, 6, 8, 12, 16, 24, 32, 48, 64)
TAUS = (1e-3, 1e-2)


def finite_array(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape:
        raise AssertionError(f"{label}: shape {a.shape}, expected {shape}")
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{label}: non-finite timing")
    return a


def outliers(per_source_reps):
    med = np.median(np.asarray(per_source_reps, dtype=float), axis=1)
    center = float(np.median(med))
    mad = float(np.median(np.abs(med - center)))
    cutoff = center + 5.0 * max(mad, np.finfo(float).eps)
    return int(np.sum(med > cutoff)), center, mad


def audit(path, pde, expected_tr):
    with open(path) as f:
        d = json.load(f)
    assert d["complete"] is True
    cfg = d["config"]
    assert cfg["backend"] == "gpu"
    assert cfg["x64"] is True
    assert cfg["matmul_precision"] == "highest"
    assert cfg["time_reps"] == 9 and cfg["time_warm"] == 3
    assert cfg["direct_component_timing"] is True
    assert math.isclose(float(cfg["tr_factor"]), expected_tr)
    rows = d["rows"]
    assert len(rows) == len(KS) * len(TAUS)
    assert {(int(r["k"]), float(r["tau"])) for r in rows} == {
        (k, tau) for k in KS for tau in TAUS}
    summaries = []
    for r in rows:
        k = int(r["k"])
        tag = f"{pde}/k{k}/tau{r['tau']}"
        assert r["pde"] == pde and r["method"] == "coord" and r["N"] == 64
        assert r["M"] == 4 * k and r["m"] == 16 * k == 4 * r["M"]
        assert r["n_sources"] == 16 and r["jax_backend"] == "gpu"
        assert math.isclose(float(r["trust_factor"]), expected_tr)
        assert np.isfinite(float(r["trust_delta"])) and float(r["trust_delta"]) > 0
        e2e = finite_array(r["time_ms_e2e_repetitions_per_source"], (16, 9), tag + "/whole")
        solve = finite_array(r["time_ms_solve_repetitions_per_source"], (16, 9), tag + "/solve")
        decode = finite_array(r["time_ms_decode_repetitions"], (9,), tag + "/decode")
        if pde == "burgers2d":
            finite_array(r["time_ms_cold_start_repetitions_per_source"],
                         (16, 9), tag + "/cold")
        assert np.isfinite(float(r["eq_info"]["rel_fit"]))
        assert np.isfinite(float(r["eq_info"]["row_rel_max"]))
        assert len(r["err_rel_l2_per_source"]) == 16
        n_out, med, mad = outliers(e2e)
        summaries.append(dict(k=k, tau=float(r["tau"]),
                              whole_median_ms=med, whole_mad_ms=mad,
                              whole_outliers_5mad=n_out,
                              solve_median_ms=float(np.median(solve)),
                              decode_median_ms=float(np.median(decode)),
                              censored=bool(r["censored"]),
                              n_blowup=int(r.get("n_blowup", 0))))
    return d, summaries


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: validate_results.py <poisson.json> <burgers.json> <audit.json>")
    p, b, out = sys.argv[1:]
    pd, ps = audit(p, "poisson2d", 1.0)
    bd, bs = audit(b, "burgers2d", 0.01)
    identity = lambda d: (d["config"]["node"], d["config"]["gpu"],
                          d["config"]["slurm_job"])
    if identity(pd) != identity(bd):
        raise AssertionError(f"PDEs did not share one device/job: {identity(pd)} vs {identity(bd)}")
    result = dict(complete=True, same_device=True,
                  node=identity(pd)[0], gpu=identity(pd)[1], slurm_job=identity(pd)[2],
                  outlier_rule="source median > across-source median + 5*MAD",
                  poisson=ps, burgers=bs)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(f"VALIDATION-PASSED rows={len(ps) + len(bs)} device={identity(pd)}", flush=True)


if __name__ == "__main__":
    main()
