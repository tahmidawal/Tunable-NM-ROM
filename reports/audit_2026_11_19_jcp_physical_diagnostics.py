"""Independent coordinator audit of saved follow-up JSON evidence.

An integrity pass is deliberately separate from a scientific qualification pass.
No PDE solves or changes to source lanes are performed.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/2026-11-19-jcp-physical-diagnostics-assets"
PHASE = ROOT / "worktrees/2026-11-19-jcp-phase-confirm/experiments/jcp-phase-confirm"
WIDTH = ROOT / "worktrees/2026-11-19-jcp-width-deploy/experiments/jcp-width-deploy"
COST = ROOT / "worktrees/2026-11-19-jcp-cost-confirm/experiments/jcp-cost-confirm"


def load(path):
    return json.loads(path.read_text())


def main():
    phase = load(PHASE / "runs/c01/output/summary.json")
    paudit = load(PHASE / "CONFIRMATION_AUDIT.json")
    parameters = load(COST / "runs/p01/output/parameters.json")
    cost = load(COST / "runs/p01/output/paired/result.json")
    qualification = load(COST / "runs/p01/output/numerical_qualification.json")
    refs = load(COST / "runs/p01/output/references.json")
    rows = [json.loads(line) for line in (COST / "runs/p01/output/paired/invocations.jsonl").read_text().splitlines()]
    assert phase["parameter_sha256"] == parameters["sha256"]
    assert phase["config"]["eval_seed"] == parameters["seed"] == 2026111901
    assert [c["index"] for c in phase["cases"]] == list(range(16))
    assert len(qualification) == 64 and len(refs) == 16
    assert len(rows) == 516
    counts = Counter(row["block"] for row in rows)
    assert dict(counts) == {"sentinel_before": 18, "fresh_confirmation": 480, "sentinel_after": 18}
    assert len({(r["block"], r["subject"], r["case_id"], r["repetition"]) for r in rows}) == len(rows)
    failed = [r for r in rows if r["status"] != "ok"]
    assert len(failed) == 10
    assert {r["case_id"] for r in failed} == {15}
    assert {r["subject"] for r in failed} == {"learned_head", "learned_head_m292_dt0.02"}
    for name, result in cost["confirmation"].items():
        for key, case in result["cases"].items():
            raw = [r for r in rows if r["block"] == "fresh_confirmation" and
                   r["subject"] == name and r["case_id"] == int(key) and r["status"] == "ok"]
            assert len(raw) == len(case["repetitions_ms"]) == 5
            times = [r["elapsed_ms"] for r in raw]
            assert np.allclose(times, case["repetitions_ms"], rtol=0, atol=0)
            assert float(np.median(times)) == case["median_ms"]
            assert max(r["evolved_max_error"] for r in raw) == case["evolved_max_error"]
    cross = []
    for case in phase["cases"]:
        assert np.array_equal(case["parameters"], parameters["parameters"][case["index"]])
        assert len(case["runs"]) == 6
        for run in case["runs"]:
            assert run["rows"] == 584 and run["dt"] == .01
            assert len(run["metrics"]) == 21
            records = load(PHASE / f"runs/c01/output/case_{case['index']:03d}" / run["records_file"])
            assert run["all_steps_stationary"] and run["termination_counts"] == {"stationarity": 20}
            assert all(rec["reason"] == "stationarity" for rec in records["records"])
        run = next(r for r in case["runs"] if r["arm"] == "head_common" and r["gtol"] == 1e-7)
        target = next(r for r in qualification if r["subject"] == "learned_head" and r["case"] == case["index"])
        shared = np.array([m["relative_error"] for m in run["metrics"]])[::4]
        queried = np.array(target["nominal_metrics"]["relative_errors"])
        cross.append({"case": case["index"], "max_absolute_difference_in_normalized_error": float(np.max(np.abs(shared - queried)))})
    assert max(r["max_absolute_difference_in_normalized_error"] for r in cross) < 1e-7
    assert not phase["qualification"]["passed"] and not phase["qualification"]["ic_fit_passed"]
    for key in ("all_stationary", "all_tight_tighter", "all_action_checks", "common_ic", "operator_passed", "reference_passed"):
        assert phase["qualification"][key]
    assert paudit["initial_fit_failed_case_indices"] == [8, 10]
    assert not any(v["qualified_fixed_threshold"] or v["qualified_fresh_head_match"]
                   for group in cost["comparisons"].values() for v in group.values())
    width = load(WIDTH / "checks/fresh/analysis.json")
    assert width["case_count"] == 8 and width["all_qualified"] and width["all_sources_same_cases"]
    derived_width = {}
    for name, arm in width["arms"].items():
        assert len(arm["cases"]) == 8 and all(c["qualified"] for c in arm["cases"])
        maxima = [max(c["error"]) for c in arm["cases"]]
        assert np.isclose(np.median(maxima), arm["median_max_error"], rtol=1e-13)
        assert max(maxima) == arm["worst_max_error"]
        derived_width[name] = maxima
    ratios = np.array(derived_width["R512-M8192"]) / np.array(derived_width["R128-M2048"])
    assert np.allclose(ratios, width["primary_confirmation"]["casewise_selected_over_baseline"], rtol=1e-13)
    assert np.all(ratios < 1) and width["primary_confirmation"]["gate_passed"]
    bridge = load(WIDTH / "runs/b01/output/summary.json")
    assert bridge["completed"] and not bridge["fresh_data_used"] and not bridge["admitted_rules"]
    assert not bridge["further_rollout_gate_open"]
    for rule in bridge["rules"].values():
        assert rule["actual_positive_weights"] == rule["sample_count"]
        assert rule["unique_stencil_interior_sites"] == rule["dense_interior_sites"] == 65025
        assert not rule["admitted"] and not rule["gates"]["site_economy"]
    checked = [PHASE / "runs/c01/output/summary.json", COST / "runs/p01/output/paired/result.json",
               COST / "runs/p01/output/paired/invocations.jsonl", COST / "runs/p01/output/numerical_qualification.json",
               WIDTH / "checks/fresh/analysis.json", WIDTH / "runs/b01/output/summary.json"]
    output = {"integrity_passed": True,
        "scope": "independent JSON arithmetic, cohort identity, failure retention, per-time cross-implementation error agreement; lane raw-array/source audits are additionally retained",
        "phase_scientific_confirmation_passed": False, "phase_initial_fit_failed_cases": [8, 10],
        "phase_all_trajectory_and_reference_checks_passed": True,
        "cost_scientific_comparisons_passed": 0, "cost_comparisons": 6,
        "timed_invocation_count": len(rows), "failed_timed_invocations": len(failed),
        "cost_qualification_count": len(qualification),
        "cross_implementation_error_checks": cross,
        "cross_implementation_max_absolute_error_difference": max(r["max_absolute_difference_in_normalized_error"] for r in cross),
        "burgers_fresh_confirmation_passed": True, "burgers_case_count": width["case_count"],
        "burgers_selected_over_baseline_case_maxima": ratios.tolist(),
        "sparse_rules_admitted": 0,
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in checked]}
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "coordinator-audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in output.items() if not isinstance(v, (dict, list))}))


if __name__ == "__main__":
    main()
