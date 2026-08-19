"""Fail closed on the structural and numerical contract of the balanced confirmation."""
from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_balanced_pair.py RUN.json")
    with open(sys.argv[1]) as fh:
        run = json.load(fh)

    cfg = run["config"]
    prov = run["provenance"]
    assert run["complete"] is True
    assert prov["jax_backend"] == "gpu"
    assert prov["x64"] is True
    assert prov["matmul_precision"] == "highest"
    assert cfg["dtype"] == "f64"
    assert cfg["test_seed"] == 20260820
    assert cfg["ns"] == [512, 1024]
    assert cfg["fom_taus"] == [1e-6, 1e-8, 1e-10]
    assert cfg["n_time"] == 8
    assert cfg["time_reps"] == 12
    assert cfg["balanced_pair_arm"] == "lmtrmean_c64_q0"
    assert cfg["pairwise_diagnostic"] is False
    assert len(run["rows"]) == 6

    for row in run["rows"]:
        tau = row["fom_tau"]
        assert row["arm"] == "lmtrmean_c64_q0"
        assert row["balanced_pair_authoritative"] is True
        assert row["joint_timing_authoritative"] is False
        assert row["boundary_contract_maxabs"] == 0.0
        assert row["final_true_rel_residual_max"] <= tau
        assert row["baseline_true_rel_residual_max"] <= tau
        assert len(row["balanced_pair_cases"]) == 8
        assert len(row["balanced_pair_timed_telemetry"]) == 8
        assert len(row["balanced_pair_baseline_timed_telemetry"]) == 8
        for grades in (row["balanced_pair_timed_telemetry"],
                       row["balanced_pair_baseline_timed_telemetry"]):
            assert all(len(case) == 12 for case in grades)
            assert all(g["flag"] == 0 for case in grades for g in case)
            assert all(g["recomputed_true_rel_residual"] <= tau
                       for case in grades for g in case)

    for mesh in run["mesh_checks"]:
        blocks = mesh["balanced_pair_timing"]
        assert set(blocks) == {"1e-06", "1e-08", "1e-10"}
        for tau_text, block in blocks.items():
            tau = float(tau_text)
            assert block["exact_equal_position_asserted"] is True
            assert block["same_invocation_cost_accuracy_work"] is True
            assert len(block["cases"]) == 8
            for case in block["cases"]:
                assert case["arm_positions"] == ["first", "second"] * 6
                assert case["zero_positions"] == ["second", "first"] * 6
                assert len(case["arm_all_s"]) == 12
                assert len(case["zero_all_s"]) == 12
                assert len(case["arm_timed_telemetry"]) == 12
                assert len(case["zero_timed_telemetry"]) == 12
                assert len(case["burn_iterations"]) == 12
                assert all(g["flag"] == 0 for g in case["arm_timed_telemetry"])
                assert all(g["flag"] == 0 for g in case["zero_timed_telemetry"])
                assert all(g["recomputed_true_rel_residual"] <= tau
                           for g in case["arm_timed_telemetry"])
                assert all(g["recomputed_true_rel_residual"] <= tau
                           for g in case["zero_timed_telemetry"])

    print(
        f"balanced_pair_audit=PASS job={prov['slurm_job_id']} "
        f"gpu={prov['gpu_kind']} rows={len(run['rows'])}"
    )


if __name__ == "__main__":
    main()
