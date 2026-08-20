"""Independent integrity audit for the locked Poisson N=2048 smoke/final JSON.

Usage: /absolute/python audit_2048.py <smoke-license|final> INPUT.json [AUDIT.json]
"""
from __future__ import annotations

import json
import math
import sys


TAUS = [1e-6, 1e-8, 1e-10]
ARMS = ["lmtrmean_c64_q0", "spectral_q1024", "spectral_q2048"]
CONTROLS = [
    "zero_cg", "dense_dst_direct", "fft_dst_direct",
    "spectral_q1024", "spectral_q2048",
]


def fail(message):
    raise SystemExit(f"N2048 AUDIT FAIL: {message}")


def check(condition, message):
    if not condition:
        fail(message)


def finite_nested(values):
    if isinstance(values, list):
        return all(finite_nested(value) for value in values)
    return math.isfinite(float(values))


def audit(mode, path):
    with open(path) as fh:
        data = json.load(fh)
    expected_kind = "smoke" if mode == "smoke-license" else "final"
    check(data.get("complete") is True, "run is incomplete")
    cfg = data["config"]
    prov = data["provenance"]
    expected = (
        dict(seed=13579, n_test=1, n_time=1, reps=2, warm=1, burn=0.0, boot=100)
        if expected_kind == "smoke" else
        dict(seed=20260826, n_test=16, n_time=8, reps=12, warm=3, burn=3.0, boot=10000)
    )
    check(cfg["poisson_2048_kind"] == expected_kind, "wrong locked mode")
    check(cfg["ns"] == [2048], "wrong mesh")
    check(cfg["fom_taus"] == TAUS, "wrong tolerance panel")
    check([arm["name"] for arm in cfg["arms"]] == ARMS, "wrong arm panel")
    check(cfg["native_cg_sensitivity_arms"] == [], "unexpected native arms")
    check(cfg["test_seed"] == expected["seed"], "wrong seed")
    check(cfg["n_test"] == expected["n_test"] and cfg["n_time"] == expected["n_time"],
          "wrong cohort sizes")
    check(cfg["time_reps"] == expected["reps"] and cfg["time_warm"] == expected["warm"],
          "wrong learned timing contract")
    check(cfg["burn_s"] == expected["burn"], "wrong burn duration")
    check(cfg["bootstrap_reps"] == expected["boot"], "wrong clustered bootstrap count")
    check(cfg["balanced_pair_arm"] == "lmtrmean_c64_q0", "wrong AB/BA arm")
    check(cfg["balanced_control_reps"] == 10 and not cfg["pairwise_diagnostic"],
          "wrong production-control timing contract")
    check(cfg["M"] == 64 and cfg["m"] == 256 and cfg["gn_iters"] == 60,
          "wrong frozen NM-ROM configuration")
    check(cfg["lm_rom_tau"] == 0.01 and cfg["trust_region_scale"] == 1.0,
          "wrong NM-ROM stopping/trust configuration")
    check(cfg["dtype"] == "f64" and cfg["matmul_precision"] == "highest",
          "wrong dtype/matmul precision")
    check(prov["jax_backend"] == "gpu" and prov["x64"], "not GPU/f64")
    check("A100" in prov["gpu_kind"] and "80GB" in prov["gpu_kind"],
          f"wrong GPU class: {prov['gpu_kind']}")
    check(not prov.get("dirty"), "dirty staged provenance")

    check(len(data["mesh_checks"]) == 1, "expected one mesh record")
    mesh = data["mesh_checks"][0]
    check(mesh["N"] == 2048 and mesh["n_dof"] == 2046**2, "wrong mesh metadata")
    check(len(mesh["test_parameters"]) == expected["n_test"], "wrong persisted cohort")
    check(mesh["device_memory_gate"]["passed"] is True, "memory gate not recorded pass")
    peak = mesh["device_memory_gate"]["peak_bytes"]
    limit = mesh["device_memory_gate"]["limit_bytes"]
    check(peak > 0 and limit > 0 and peak / limit <= 0.80, "peak memory exceeds 80%")
    check(mesh["device_memory_peak_fraction"] == peak / limit, "memory fraction mismatch")

    pair_blocks = mesh["balanced_pair_timing"]
    prod_blocks = mesh["balanced_production_controls"]
    for tau in TAUS:
        key = str(tau)
        check(key in pair_blocks and key in prod_blocks, f"missing tau block {key}")
        pair = pair_blocks[key]
        check(pair["exact_equal_position_asserted"] is True, f"AB/BA balance absent {key}")
        check(pair["same_invocation_cost_accuracy_work"] is True,
              f"AB/BA same-invocation contract absent {key}")
        check(len(pair["cases"]) == expected["n_time"], f"wrong AB/BA cases {key}")
        for case in pair["cases"]:
            check(case["arm_positions"] == ["first", "second"] * (expected["reps"] // 2),
                  f"wrong AB/BA positions {key}")
            check(len(case["arm_all_s"]) == expected["reps"]
                  and len(case["zero_all_s"]) == expected["reps"],
                  f"wrong AB/BA raw repetition count {key}")
            check(finite_nested(case["arm_all_s"]) and finite_nested(case["zero_all_s"]),
                  f"nonfinite AB/BA time {key}")
            for grade in case["arm_timed_telemetry"] + case["zero_timed_telemetry"]:
                check(grade["flag"] == 0 and grade["recomputed_true_rel_residual"] <= tau,
                      f"AB/BA solver/residual failure {key}")
                check(grade["boundary_maxabs"] <= 1e-14, f"AB/BA boundary failure {key}")
        summary = pair["summary"]
        check(summary["case_clustered_bootstrap_reps"] == expected["boot"],
              f"wrong AB/BA clustered bootstrap {key}")
        check(summary["arm_outlier_count"] >= 0 and summary["zero_outlier_count"] >= 0,
              f"missing AB/BA outlier counts {key}")

        prod = prod_blocks[key]
        check(prod["methods"] == CONTROLS, f"wrong production methods {key}")
        check(prod["exact_position_and_pairwise_precedence_balance"] is True,
              f"production balance absent {key}")
        check(prod["same_invocation_cost_accuracy_work"] is True,
              f"production same-invocation contract absent {key}")
        check(len(prod["cases"]) == expected["n_time"], f"wrong production cases {key}")
        for method in CONTROLS:
            check(len(prod["all_s"][method]) == expected["n_time"],
                  f"wrong production case count {method}/{key}")
            check(all(len(case) == 10 for case in prod["all_s"][method]),
                  f"wrong production raw reps {method}/{key}")
            check(finite_nested(prod["all_s"][method]), f"nonfinite production time {method}/{key}")
            check(prod["summaries"][method]["case_clustered_bootstrap_reps"] == expected["boot"],
                  f"wrong production clustered bootstrap {method}/{key}")
            eligibility = prod["eligibility"][method]
            check(eligibility["boundary_maxabs"] <= 1e-14,
                  f"production boundary failure {method}/{key}")
            if method == "zero_cg" or method.startswith("spectral_q"):
                check(eligibility["eligible"] is True,
                      f"required iterative control ineligible {method}/{key}")
        position_counts = prod["balance_audit"]["position_counts"]
        check(all(counts == [2, 2, 2, 2, 2] for counts in position_counts.values()),
              f"production position counts wrong {key}")
        check(all(item["a_before_b"] == 5 and item["b_before_a"] == 5
                  for item in prod["balance_audit"]["precedence_counts"].values()),
              f"production precedence counts wrong {key}")

    check(len(data["rows"]) == 9, "expected three arms by three tolerances")
    for row in data["rows"]:
        check(row["boundary_contract_maxabs"] <= 1e-14, "guess hard-BC failure")
        check(row["final_true_rel_residual_max"] <= row["fom_tau"],
              "timed row true-residual failure")
        if row["arm"] == "lmtrmean_c64_q0":
            check(row["balanced_pair_authoritative"] is True,
                  "learned row lacks AB/BA authority")
        else:
            check(row["balanced_control_authoritative"] is True,
                  "spectral row lacks balanced-control authority")

    return dict(
        audit_pass=True,
        mode=expected_kind,
        source_json=path,
        job_id=prov["slurm_job_id"],
        gpu_kind=prov["gpu_kind"],
        commit=prov["commit"],
        source_sha256=prov["source_sha256"]["feasibility.py"],
        seed=cfg["test_seed"],
        peak_bytes=peak,
        limit_bytes=limit,
        peak_fraction=peak / limit,
        learned_raw_records=3 * expected["n_time"] * expected["reps"] * 2,
        production_raw_records=3 * expected["n_time"] * 10 * len(CONTROLS),
    )


def main():
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("smoke-license", "final"):
        raise SystemExit(__doc__)
    result = audit(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        with open(sys.argv[3], "w") as fh:
            json.dump(result, fh, indent=1)
            fh.write("\n")
    print("N2048 AUDIT PASS", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
