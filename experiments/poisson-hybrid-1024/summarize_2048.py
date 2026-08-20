#!/usr/bin/env python3
"""Independently verify and summarize the locked Poisson N=2048 result.

This file deliberately recomputes the timing estimators, case-clustered bootstrap
intervals, solver gates, balance, and outlier counts from retained raw arrays.  It
does not import the scientific driver or trust its precomputed summaries.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
import subprocess
import sys

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "runs" / "n2048final1" / "out" / "n2048final1.json"
DEFAULT_AUDIT = HERE / "runs" / "n2048final1" / "INDEPENDENT-AUDIT.json"
DEFAULT_REPORT = HERE / "N2048.generated.md"
METHODS = [
    "zero_cg",
    "dense_dst_direct",
    "fft_dst_direct",
    "spectral_q1024",
    "spectral_q2048",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"N2048 INDEPENDENT AUDIT FAIL: {message}")


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-14):
        raise SystemExit(
            f"N2048 INDEPENDENT AUDIT FAIL: {label}: "
            f"recomputed={actual!r}, stored={expected!r}"
        )


def quantiles(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_provenance(input_path: pathlib.Path, prov: dict) -> dict:
    """Bind pulled bytes -> staged manifest -> immutable git commit."""
    run_dir = input_path.parent.parent
    local_checksums = (run_dir / "LOCAL.sha256").read_text()
    remote_checksums = (run_dir / "REMOTE.sha256").read_text()
    require(local_checksums == remote_checksums, "local/remote pull checksums differ")
    manifest_text = (run_dir / "MANIFEST.sha256").read_text()
    expected_source = (
        "fa39fa6dd8e9d9ea82ae8d26bc533904c2504114206ed3cc3f601d7aa20a3c27"
    )
    require(
        f"{expected_source}  ./code/poisson-hybrid-1024/feasibility.py" in manifest_text,
        "staged manifest does not bind the scientific source",
    )
    require(hash_file(HERE / "feasibility.py") == expected_source,
            "worktree scientific source differs from staged source")
    committed = subprocess.check_output(
        [
            "git", "show",
            f"{prov['commit']}:experiments/poisson-hybrid-1024/feasibility.py",
        ],
        cwd=HERE,
    )
    require(hashlib.sha256(committed).hexdigest() == expected_source,
            "scientific commit does not contain staged source")
    run_script = (run_dir / "run.sbatch").read_text()
    require(f"export WSF_COMMIT={prov['commit']}" in run_script,
            "batch script does not bind scientific commit")
    require("export JAX_DEFAULT_MATMUL_PRECISION=highest" in run_script,
            "batch script does not bind highest precision")
    require("POISSON_2048_KIND=final" in run_script and "TEST_SEED=20260826" in run_script,
            "batch script does not bind locked final mode/seed")
    return {
        "remote_local_checksums_match": True,
        "manifest_source_sha256": expected_source,
        "worktree_source_sha256": expected_source,
        "commit_source_sha256": expected_source,
        "batch_commit_bound": True,
    }


def verify_pair(block: dict, tau: float, cfg: dict) -> dict:
    arm = np.asarray([case["arm_all_s"] for case in block["cases"]], dtype=float)
    zero = np.asarray([case["zero_all_s"] for case in block["cases"]], dtype=float)
    require(arm.shape == (8, 12) and zero.shape == (8, 12), f"raw pair shape tau={tau}")
    require(np.all(np.isfinite(arm)) and np.all(np.isfinite(zero)), f"finite pair tau={tau}")
    expected_positions = ["first", "second"] * 6
    require(
        all(case["arm_positions"] == expected_positions for case in block["cases"]),
        f"AB/BA arm positions tau={tau}",
    )
    require(
        all(case["zero_positions"] == ["second" if p == "first" else "first"
                                       for p in expected_positions]
            for case in block["cases"]),
        f"AB/BA zero positions tau={tau}",
    )
    require(
        all(len(case["burn_iterations"]) == 12 and min(case["burn_iterations"]) > 0
            for case in block["cases"]),
        f"immediate pair burns tau={tau}",
    )

    arm_case = np.median(arm, axis=1)
    zero_case = np.median(zero, axis=1)
    delta = arm - zero
    delta_case = np.median(delta, axis=1)
    arm_median = float(np.median(arm_case))
    zero_median = float(np.median(zero_case))
    speedup = zero_median / arm_median
    delta_median = float(np.median(delta_case))

    seed = cfg["bootstrap_seed"] + 1009 * 2048 + int(round(-math.log10(tau)))
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, 8, size=(cfg["bootstrap_reps"], 8))
    boot_arm = np.median(arm_case[samples], axis=1)
    boot_zero = np.median(zero_case[samples], axis=1)
    boot_delta = np.median(delta_case[samples], axis=1)
    speed_ci = quantiles(boot_zero / boot_arm)
    delta_ci = quantiles(boot_delta)

    stored = block["summary"]
    close(arm_median, stored["arm_median_across_case_medians_s"], f"arm median tau={tau}")
    close(zero_median, stored["zero_median_across_case_medians_s"], f"zero median tau={tau}")
    close(speedup, stored["speedup_zero_over_arm"], f"speedup tau={tau}")
    close(delta_median, stored["paired_delta_arm_minus_zero_median_across_cases_s"],
          f"delta tau={tau}")
    for got, want in zip(speed_ci, stored["speedup_case_clustered_bootstrap_ci95"]):
        close(got, want, f"speed CI tau={tau}")
    for got, want in zip(delta_ci, stored["paired_delta_bootstrap_ci95_s"]):
        close(got, want, f"delta CI tau={tau}")

    arm_grades = [grade for case in block["cases"] for grade in case["arm_timed_telemetry"]]
    zero_grades = [grade for case in block["cases"] for grade in case["zero_timed_telemetry"]]
    max_arm_residual = max(float(grade["recomputed_true_rel_residual"]) for grade in arm_grades)
    max_zero_residual = max(float(grade["recomputed_true_rel_residual"]) for grade in zero_grades)
    require(max(int(grade["flag"]) for grade in arm_grades + zero_grades) == 0,
            f"pair solver flags tau={tau}")
    require(max_arm_residual <= tau and max_zero_residual <= tau,
            f"pair residual gate tau={tau}")
    require(max(float(grade["boundary_maxabs"]) for grade in arm_grades + zero_grades) <= 1e-14,
            f"pair boundary gate tau={tau}")

    arm_outliers = int(np.sum(arm > 1.5 * arm_case[:, None]))
    zero_outliers = int(np.sum(zero > 1.5 * zero_case[:, None]))
    require(arm_outliers == stored["arm_outlier_count"], f"arm outliers tau={tau}")
    require(zero_outliers == stored["zero_outlier_count"], f"zero outliers tau={tau}")
    signs = {
        "learned_faster": int(np.sum(delta_case < 0)),
        "zero_faster": int(np.sum(delta_case > 0)),
        "tie": int(np.sum(delta_case == 0)),
    }
    require(
        signs == {
            "learned_faster": stored["paired_case_sign_counts"]["arm_faster"],
            "zero_faster": stored["paired_case_sign_counts"]["zero_faster"],
            "tie": stored["paired_case_sign_counts"]["exact_tie"],
        },
        f"case signs tau={tau}",
    )
    supported = bool(speed_ci[0] > 1.0 and delta_ci[1] < 0.0)
    return {
        "tau": tau,
        "learned_median_ms": arm_median * 1e3,
        "zero_median_ms": zero_median * 1e3,
        "speedup": speedup,
        "speedup_ci95": speed_ci,
        "paired_delta_ms": delta_median * 1e3,
        "paired_delta_ci95_ms": [value * 1e3 for value in delta_ci],
        "supported": supported,
        "case_signs": signs,
        "learned_outliers": arm_outliers,
        "zero_outliers": zero_outliers,
        "learned_true_residual_max": max_arm_residual,
        "zero_true_residual_max": max_zero_residual,
        "learned_iterations_mean": float(np.mean([grade["iterations"] for grade in arm_grades])),
        "zero_iterations_mean": float(np.mean([grade["iterations"] for grade in zero_grades])),
    }


def verify_controls(block: dict, tau: float, cfg: dict) -> list[dict]:
    require(block["methods"] == METHODS, f"control method order tau={tau}")
    orders = block["cases"][0]["orders"]
    require(len(orders) == 10, f"control order count tau={tau}")
    for method in METHODS:
        positions = [order.index(method) for order in orders]
        require([positions.count(pos) for pos in range(5)] == [2] * 5,
                f"control position balance {method}/tau={tau}")
    for first, second in itertools.combinations(METHODS, 2):
        n_first = sum(order.index(first) < order.index(second) for order in orders)
        require(n_first == 5, f"control precedence {first}/{second}/tau={tau}")
    require(all(case["orders"] == orders for case in block["cases"]),
            f"control order consistency tau={tau}")
    require(all(len(case["burn_iterations"]) == 10 and min(case["burn_iterations"]) > 0
                for case in block["cases"]),
            f"immediate control burns tau={tau}")

    arrays = {method: np.asarray(block["all_s"][method], dtype=float) for method in METHODS}
    require(all(values.shape == (8, 10) for values in arrays.values()),
            f"control raw shape tau={tau}")
    require(all(np.all(np.isfinite(values)) for values in arrays.values()),
            f"finite controls tau={tau}")
    zero_case = np.median(arrays["zero_cg"], axis=1)
    summary_seed = cfg["bootstrap_seed"] + 1009 * 2048 + int(round(-math.log10(tau)))
    rows = []
    for index, method in enumerate(METHODS):
        values = arrays[method]
        case_medians = np.median(values, axis=1)
        median = float(np.median(case_medians))
        rng = np.random.default_rng(summary_seed + 37 * index)
        samples = rng.integers(0, 8, size=(cfg["bootstrap_reps"], 8))
        boot = np.median(case_medians[samples], axis=1)
        ci = quantiles(boot)
        stored = block["summaries"][method]
        close(median, stored["median_across_case_medians_s"], f"control median {method}/{tau}")
        for got, want in zip(ci, stored["bootstrap_ci95_s"]):
            close(got, want, f"control CI {method}/{tau}")
        outliers = int(np.sum(values > 1.5 * case_medians[:, None]))
        require(outliers == stored["outlier_count"], f"control outliers {method}/{tau}")

        grades = [grade for case in block["timed_telemetry"][method] for grade in case]
        residual = max(float(grade["recomputed_true_rel_residual"]) for grade in grades)
        flag = max(int(grade["flag"]) for grade in grades)
        boundary = max(float(grade["boundary_maxabs"]) for grade in grades)
        eligible = bool(flag == 0 and residual <= tau and boundary <= 1e-14)
        recorded = block["eligibility"][method]
        require(eligible == recorded["eligible"], f"control eligibility {method}/{tau}")
        close(residual, recorded["true_rel_residual_max"], f"control residual {method}/{tau}")

        # Independently quantify the production speedup against the paired zero-case cohort.
        speedup = float(np.median(zero_case) / median)
        rng = np.random.default_rng(summary_seed + 50000 + 79 * index)
        samples = rng.integers(0, 8, size=(cfg["bootstrap_reps"], 8))
        boot_speedup = (
            np.median(zero_case[samples], axis=1)
            / np.median(case_medians[samples], axis=1)
        )
        rows.append({
            "tau": tau,
            "method": method,
            "median_ms": median * 1e3,
            "ci95_ms": [value * 1e3 for value in ci],
            "speedup_vs_zero": speedup,
            "speedup_vs_zero_independent_ci95": quantiles(boot_speedup),
            "eligible": eligible,
            "true_residual_max": residual,
            "iterations_mean": float(np.mean([grade["iterations"] for grade in grades])),
            "outliers": outliers,
        })
    return rows


def verify_cohort(mesh: dict, seed: int, size: int) -> float:
    """Recreate the locked parameter draw without importing the staged driver."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, size)
    cy = rng.uniform(0.15, 0.85, size)
    width = np.exp(rng.uniform(np.log(0.02), np.log(0.1), size))
    amplitude = rng.uniform(0.5, 2.0, size)
    expected = np.stack(
        [
            (cx - 0.5) / 0.35,
            (cy - 0.5) / 0.35,
            (np.log(width) - np.log(0.045)) / 0.8,
            (amplitude - 1.25) / 0.75,
        ],
        axis=1,
    )
    recorded = np.asarray(mesh["test_parameters"], dtype=float)
    require(recorded.shape == expected.shape, "persisted cohort shape changed")
    discrepancy = float(np.max(np.abs(recorded - expected)))
    require(discrepancy == 0.0, f"seed-regenerated cohort mismatch {discrepancy:.3e}")
    return discrepancy


def fmt_tau(value: float) -> str:
    return f"{value:.0e}"


def make_report(audit: dict) -> str:
    learned = audit["learned_pair"]
    controls = audit["production_controls"]
    lines = [
        "# Poisson hybrid and structured controls at N=2048",
        "",
        "These are final results from the frozen resolution-only extension of the audited K=8 "
        "NM-ROM warm start. All numbers below are generated from retained raw timing and solver "
        "telemetry; the independent recomputation and integrity audit pass.",
        "",
        "## Result",
        "",
        "| FOM tolerance | K8 hybrid (ms) | Zero CG (ms) | speedup | clustered 95% CI | paired saving (ms) | supported | case signs K8/zero | outliers K8/zero |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in learned:
        ci = row["speedup_ci95"]
        saving = -row["paired_delta_ms"]
        signs = row["case_signs"]
        lines.append(
            f"| {fmt_tau(row['tau'])} | {row['learned_median_ms']:.3f} | "
            f"{row['zero_median_ms']:.3f} | {row['speedup']:.4f}x | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {saving:.3f} | "
            f"{'yes' if row['supported'] else 'no'} | "
            f"{signs['learned_faster']}/{signs['zero_faster']} | "
            f"{row['learned_outliers']}/{row['zero_outliers']} |"
        )
    lines.extend([
        "",
        "A supported crossover requires both the whole-case speedup interval above one and the "
        "paired K8-minus-zero time interval below zero, in addition to solver, residual, and "
        "boundary gates. Timing includes NM-ROM construction and the finishing counting-CG solve.",
        "",
        "## Same-job structured controls",
        "",
        "| FOM tolerance | method | median (ms) | clustered 95% CI (ms) | speedup vs zero | independent 95% CI | eligible | max true residual | outliers |",
        "|---:|---|---:|---:|---:|---:|:---:|---:|---:|",
    ])
    for row in controls:
        ci = row["ci95_ms"]
        sci = row["speedup_vs_zero_independent_ci95"]
        lines.append(
            f"| {fmt_tau(row['tau'])} | `{row['method']}` | {row['median_ms']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {row['speedup_vs_zero']:.1f}x | "
            f"[{sci[0]:.1f}, {sci[1]:.1f}] | "
            f"{'yes' if row['eligible'] else 'no'} | {row['true_residual_max']:.3e} | "
            f"{row['outliers']} |"
        )
    prov = audit["provenance"]
    mem = audit["memory"]
    cohort = audit["cohort"]
    quadrature = audit["quadrature"]
    lines.extend([
        "",
        "The structured methods are production controls, not learned components. "
        "`spectral_q1024` is a fixed partial sine expansion followed by counting CG; "
        "`spectral_q2048` clamps to all 2046 interior sine modes. Direct rows remain visible "
        "when their measured residual makes them ineligible.",
        "",
        "## Audit and scope",
        "",
        f"- Job `{prov['job_id']}` ran on `{prov['gpu_kind']}` with GPU backend, f64/x64, and highest matmul precision.",
        f"- Scientific commit `{prov['commit']}` and staged `feasibility.py` SHA-256 `{prov['source_sha256']}` are bound to git and the pulled manifest.",
        f"- Peak allocator use was {mem['peak_fraction']:.3%} ({mem['peak_bytes']} / {mem['limit_bytes']} bytes), below the locked 80% gate.",
        f"- The {cohort['size']}-case parameter cohort was independently regenerated from seed {cohort['seed']} with maximum discrepancy {cohort['seed_regenerated_maxabs']:.1e}.",
        f"- Decoder-output quadrature was refit at N={quadrature['grid_N']} with M={quadrature['M']}, m={quadrature['m']}, support {quadrature['support']}, and relative fit {quadrature['rel_fit']:.3e}.",
        f"- Retained records: {audit['record_counts']['learned']} learned/zero timed invocations and {audit['record_counts']['production']} structured-control timed invocations; all observations, including {audit['record_counts']['outliers']} diagnosed outliers, remain retained.",
        "- This confirms only N=2048 on seed 20260826. It does not reopen architecture, rank, objective, quadrature, or stopping-policy selection.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    input_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    audit_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_AUDIT
    report_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_REPORT
    data = json.loads(input_path.read_text())
    cfg = data["config"]
    prov = data["provenance"]
    require(data.get("complete") is True, "run is incomplete")
    require(cfg["poisson_2048_kind"] == "final", "not the locked final mode")
    require(cfg["ns"] == [2048] and cfg["test_seed"] == 20260826, "mesh/seed changed")
    require(cfg["n_test"] == 16 and cfg["n_time"] == 8, "cohort changed")
    require(cfg["time_reps"] == 12 and cfg["balanced_control_reps"] == 10,
            "timing contract changed")
    require(cfg["time_warm"] == 3 and cfg["burn_s"] == 3.0, "warm/burn changed")
    require(cfg["dtype"] == "f64" and cfg["matmul_precision"] == "highest",
            "precision changed")
    require(prov["jax_backend"] == "gpu" and prov["x64"], "backend/x64 changed")
    require(prov["commit"] == "a98b1e8d77f27fd76c6bd008d38e6064a591f6fb",
            "scientific commit changed")
    require(prov["source_sha256"]["feasibility.py"] == "fa39fa6dd8e9d9ea",
            "runtime source hash changed")
    require(prov["slurm_job_id"] == "2670159", "job id changed")
    require("A100" in prov["gpu_kind"] and "80GB" in prov["gpu_kind"], "GPU class changed")
    provenance_audit = verify_provenance(input_path, prov)

    mesh = data["mesh_checks"][0]
    cohort_maxabs = verify_cohort(mesh, cfg["test_seed"], cfg["n_test"])
    eq = mesh["eq_info"]
    require(eq["grid_N"] == 2048 and eq["M"] == 64 and eq["m"] == 256,
            "N-specific decoder-output EQ was not refit at locked M/m")
    require(eq["implementation"] == "streamed_separable_exact_equivalent"
            and not eq["full_grid_mode_table_materialized"],
            "unexpected EQ implementation")
    learned = [
        verify_pair(mesh["balanced_pair_timing"][str(tau)], tau, cfg)
        for tau in cfg["fom_taus"]
    ]
    controls = [
        row
        for tau in cfg["fom_taus"]
        for row in verify_controls(mesh["balanced_production_controls"][str(tau)], tau, cfg)
    ]
    peak = int(mesh["device_memory_gate"]["peak_bytes"])
    limit = int(mesh["device_memory_gate"]["limit_bytes"])
    require(mesh["device_memory_gate"]["passed"] and peak / limit <= 0.80,
            "device-memory gate failed")
    final_rows = [row for row in data["rows"] if row["N"] == 2048]
    require(len(final_rows) == 9, "wrong scientific row count")
    require(all(row["final_true_rel_residual_max"] <= row["fom_tau"] for row in final_rows),
            "row residual gate failed")
    require(all(row["boundary_contract_maxabs"] <= 1e-14 for row in final_rows),
            "guess boundary gate failed")

    result = {
        "audit_pass": True,
        "source_json": str(input_path),
        "source_json_sha256": hash_file(input_path),
        "provenance": {
            "job_id": prov["slurm_job_id"],
            "gpu_kind": prov["gpu_kind"],
            "commit": prov["commit"],
            "source_sha256": "fa39fa6dd8e9d9ea82ae8d26bc533904c2504114206ed3cc3f601d7aa20a3c27",
            "seed": cfg["test_seed"],
            "binding": provenance_audit,
        },
        "memory": {"peak_bytes": peak, "limit_bytes": limit, "peak_fraction": peak / limit},
        "cohort": {
            "seed": cfg["test_seed"],
            "size": cfg["n_test"],
            "seed_regenerated_maxabs": cohort_maxabs,
        },
        "quadrature": {
            "grid_N": eq["grid_N"],
            "M": eq["M"],
            "m": eq["m"],
            "support": eq["support"],
            "rel_fit": eq["rel_fit"],
            "implementation": eq["implementation"],
        },
        "record_counts": {
            "learned": 3 * 8 * 12 * 2,
            "production": 3 * 8 * 10 * 5,
            "outliers": sum(row["learned_outliers"] + row["zero_outliers"] for row in learned)
                        + sum(row["outliers"] for row in controls),
        },
        "learned_pair": learned,
        "production_controls": controls,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=1) + "\n")
    report_path.write_text(make_report(result))
    print(f"N2048 INDEPENDENT AUDIT PASS: {audit_path}")
    print(f"generated: {report_path}")


if __name__ == "__main__":
    main()
