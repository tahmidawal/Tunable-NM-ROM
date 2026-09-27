"""Lock the fastest eligible cubic/Helmholtz FOM against solution error.

One job calibrates N=256/512/1024 on the preregistered seed-20260822 cohort.
Accuracy, work, residuals, and final timing are returned by the same invocation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import b10_common as c
HERE = os.path.dirname(os.path.abspath(__file__))
HYBRID = os.path.abspath(os.path.join(HERE, "..", "burgers-hybrid-1024"))
if os.path.isfile(os.path.join(HYBRID, "bh_common.py")):
    sys.path.insert(0, HYBRID)
import bh_common as bc

OUTPUT = sys.argv[1]
NS = tuple(int(value) for value in os.environ.get("NS", "256,512,1024").split(","))
SEED = int(os.environ.get("CAL_SEED", "20260822"))
DRAW_COUNT = int(os.environ.get("CAL_DRAW_COUNT", "32"))
N_CASES = int(os.environ.get("CAL_CASES", "4"))
TIME_REPS = int(os.environ.get("TIME_REPS", "14"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_SECONDS = float(os.environ.get("BURN_SECONDS", "3"))
OUTER_TOLS = (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6)
INNER_BY_OUTER = {
    3e-3: (3e-1, 1e-1),
    1e-3: (3e-1, 1e-1),
    3e-4: (3e-1, 1e-1),
    1e-4: (1e-1, 3e-2),
    3e-5: (1e-1, 3e-2),
    1e-5: (3e-2, 1e-2),
    1e-6: (3e-2, 1e-2),
}


def key(outer, inner):
    return f"outer={outer:.0e}:inner={inner:.0e}"


def grade(output, truth, elapsed=None):
    fields, newton, linear, breakdowns, flags, residuals = [
        np.asarray(value) for value in output
    ]
    result = {
        "trajectory_relative_l2": float(
            np.linalg.norm(fields - truth[1:]) / np.linalg.norm(truth[1:])
        ),
        "max_returned_relative_residual": float(np.max(residuals)),
        "newton_total": int(np.sum(newton)),
        "linear_total": int(np.sum(linear)),
        "newton_per_step": newton.tolist(),
        "linear_per_step": linear.tolist(),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "finite": bool(np.all(np.isfinite(fields)) and np.all(np.isfinite(residuals))),
    }
    if elapsed is not None:
        result["elapsed_s"] = float(elapsed)
    return result


def summarize_records(records, outer):
    errors = np.asarray([item["trajectory_relative_l2"] for item in records])
    residuals = np.asarray([item["max_returned_relative_residual"] for item in records])
    elapsed = np.asarray([item["elapsed_s"] for item in records])
    healthy = all(
        item["finite"] and item["breakdowns"] == 0 and item["flags_nonzero"] == 0
        for item in records
    )
    return {
        "trajectory_error_mean": float(np.mean(errors)),
        "trajectory_error_worst": float(np.max(errors)),
        "max_returned_relative_residual": float(np.max(residuals)),
        "median_elapsed_s": float(np.median(elapsed)),
        "mean_newton_total": float(np.mean([item["newton_total"] for item in records])),
        "mean_linear_total": float(np.mean([item["linear_total"] for item in records])),
        "healthy": healthy,
        "accuracy_eligible": bool(
            healthy and np.max(residuals) <= outer
            and np.mean(errors) <= 1e-3 and np.max(errors) <= 3e-3
        ),
    }


def cluster_ci(case_medians, seed):
    values = np.asarray(case_medians, np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(10000, values.size))]
    return [float(value) for value in np.quantile(np.median(sample, axis=1), (0.025, 0.975))]


def main():
    c.require_gpu_highest()
    if TIME_REPS % 2:
        raise SystemExit("TIME_REPS must be even for forward/reverse balance")
    report = {
        "stage": "fastest eligible FOM calibration",
        "status": "running",
        "config": {
            "ns": list(NS), "seed": SEED, "draw_count": DRAW_COUNT,
            "indices": list(range(N_CASES)), "outer_tolerances": list(OUTER_TOLS),
            "inner_candidates": {f"{outer:.0e}": list(INNER_BY_OUTER[outer])
                                 for outer in OUTER_TOLS},
            "history": "cubic", "preconditioner": "exact Helmholtz DST-I",
            "time_reps": TIME_REPS, "time_warm": TIME_WARM,
            "burn_seconds": BURN_SECONDS, "f64": True,
            "solution_accuracy_gate": {"mean": 1e-3, "worst": 3e-3},
            "reference_numerical_gate": 1e-4,
            "confirmation_touched": False,
        },
        "provenance": c.provenance(),
        "meshes": {},
    }
    c.save_json(OUTPUT, report)
    dummy_by_n = {}
    for n in NS:
        c.log("FOM calibration N", n)
        fields, parameters, reference_health = c.generate_population(
            n, SEED, DRAW_COUNT, np.arange(N_CASES), chunk=1
        )
        if reference_health["independent_max_relative_residual"] > 1e-8:
            raise SystemExit(f"N={n} unhealthy reference")
        dummy = jnp.zeros((c.NUM_STEPS, n * n), jnp.float64)
        dummy_by_n[n] = dummy

        # Independent tight counting-chain equivalence bounds reference numerical error.
        reference_chain, _ = bc.make_chain(
            n, 1e-11, lin_tol=1e-6, preconditioner="helmholtz"
        )
        tight_differences = []
        tight_residual = []
        for case in range(N_CASES):
            output = reference_chain(
                jnp.asarray(fields[case, 0]), parameters["nu"][case], dummy,
                jnp.int32(5)
            )
            item = grade(output, fields[case])
            tight_differences.append(item["trajectory_relative_l2"])
            tight_residual.append(item["max_returned_relative_residual"])
        reference_numerical_error = max(tight_differences)
        if reference_numerical_error > 1e-4:
            raise SystemExit(
                f"N={n} reference numerical gate failed: {reference_numerical_error:.3e}"
            )

        calls = {}
        build = {}
        preliminary = {}
        for outer in OUTER_TOLS:
            candidates = []
            for inner in INNER_BY_OUTER[outer]:
                label = key(outer, inner)
                build_start = time.perf_counter()
                chain, _ = bc.make_chain(
                    n, outer, lin_tol=inner, preconditioner="helmholtz"
                )
                build_s = time.perf_counter() - build_start
                build[label] = {"python_build_s": build_s}
                calls[label] = []
                preliminary[label] = []
                first_call_s = None
                for case in range(N_CASES):
                    def call(case_index=case, fn=chain):
                        return fn(
                            jnp.asarray(fields[case_index, 0]),
                            parameters["nu"][case_index], dummy, jnp.int32(5)
                        )

                    calls[label].append(call)
                    started = time.perf_counter()
                    output = call()
                    jax.block_until_ready(output)
                    elapsed = time.perf_counter() - started
                    if first_call_s is None:
                        first_call_s = elapsed
                    preliminary[label].append(grade(output, fields[case]))
                build[label]["first_call_compile_and_solve_s"] = first_call_s
                healthy = all(
                    item["finite"] and item["breakdowns"] == 0
                    and item["flags_nonzero"] == 0
                    and item["max_returned_relative_residual"] <= outer
                    for item in preliminary[label]
                )
                mean_linear = float(np.mean([
                    item["linear_total"] for item in preliminary[label]
                ]))
                candidates.append((not healthy, mean_linear, -inner, label))
            candidates.sort()
            # Time only the work-best healthy inner setting for this outer tolerance.
            selected = candidates[0]
            if selected[0]:
                raise SystemExit(f"N={n} outer={outer}: no healthy inner candidate")
            build[selected[3]]["selected_by"] = "minimum mean BiCGStab work among healthy candidates"

        selected_labels = [
            min(
                (label for label in preliminary if label.startswith(f"outer={outer:.0e}:")),
                key=lambda label: (
                    any(
                        not item["finite"] or item["breakdowns"] or item["flags_nonzero"]
                        or item["max_returned_relative_residual"] > outer
                        for item in preliminary[label]
                    ),
                    np.mean([item["linear_total"] for item in preliminary[label]]),
                ),
            ) for outer in OUTER_TOLS
        ]
        if len(set(selected_labels)) != len(OUTER_TOLS):
            raise SystemExit("selected FOM labels are not unique")
        for label in selected_labels:
            for call in calls[label]:
                jax.block_until_ready(call())
        burn_count = c.gpu_burn(BURN_SECONDS)
        for warm in range(TIME_WARM):
            order = selected_labels[warm:] + selected_labels[:warm]
            for case in range(N_CASES):
                for label in order:
                    jax.block_until_ready(calls[label][case]())

        timed = {label: [] for label in selected_labels}
        timing_orders = []
        half = TIME_REPS // 2
        for repetition in range(TIME_REPS):
            offset = repetition % len(selected_labels)
            order = selected_labels[offset:] + selected_labels[:offset]
            if repetition >= half:
                order = list(reversed(order))
            timing_orders.append(order)
            case_order = list(range(N_CASES))
            case_order = case_order[repetition % N_CASES:] + case_order[:repetition % N_CASES]
            for case in case_order:
                for label in order:
                    started = time.perf_counter()
                    output = calls[label][case]()
                    jax.block_until_ready(output)
                    elapsed = time.perf_counter() - started
                    item = grade(output, fields[case], elapsed)
                    item.update(case_index=case, repetition=repetition)
                    timed[label].append(item)

        rows = {}
        eligible = []
        for outer, label in zip(OUTER_TOLS, selected_labels):
            summary = summarize_records(timed[label], outer)
            case_medians = []
            for case in range(N_CASES):
                values = [item["elapsed_s"] for item in timed[label]
                          if item["case_index"] == case]
                if len(values) != TIME_REPS:
                    raise SystemExit("unbalanced timing records")
                case_medians.append(float(np.median(values)))
            summary["per_case_median_elapsed_s"] = case_medians
            summary["clustered_median_elapsed_ci_s"] = cluster_ci(
                case_medians, 20260822 + n + int(abs(np.log10(outer)))
            )
            rows[label] = {
                "outer_tolerance": outer,
                "inner_tolerance": float(label.split("inner=")[1]),
                "setup": build[label],
                "preliminary_same_invocation": preliminary[label],
                "timed_records": timed[label],
                "summary": summary,
            }
            if summary["accuracy_eligible"]:
                eligible.append((summary["median_elapsed_s"], label))
        if not eligible:
            raise SystemExit(f"N={n}: no FOM configuration satisfies solution accuracy")
        eligible.sort()
        selected_fom = eligible[0][1]
        report["meshes"][str(n)] = {
            "reference_health": reference_health,
            "tight_reference_counting_difference": {
                "trajectory_relative_l2_all": tight_differences,
                "worst": reference_numerical_error,
                "max_returned_residual": max(tight_residual),
            },
            "burn_count": burn_count,
            "timing_orders": timing_orders,
            "all_inner_preliminary": preliminary,
            "rows": rows,
            "selected_fastest_eligible": selected_fom,
        }
        c.save_json(OUTPUT, report)
        c.log("N", n, "selected", selected_fom, rows[selected_fom]["summary"])

    report["status"] = "complete"
    c.save_json(OUTPUT, report)
    c.log(json.dumps({
        "status": report["status"],
        "selected": {n: report["meshes"][str(n)]["selected_fastest_eligible"]
                     for n in NS},
    }, indent=1), "\nALL-DONE")


if __name__ == "__main__":
    main()
