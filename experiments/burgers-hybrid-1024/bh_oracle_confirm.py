"""Audited dynamic warm-start oracle against previous and extrapolated states.

Unlike the preliminary gate, each oracle guess is built from the live FOM
history inside the scan:

    g_t = g_extrap(live history) + q * (u_reference,t+1 - g_extrap).

The exact next field makes this nondeployable.  It measures the work and total
wall-time budget available to a real predictor at a named quality.  Every timed
call returns and is graded for fields, residuals, and work from that same call.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import bh_common as bc

OUT = sys.argv[1]
NS = [int(v) for v in os.environ.get("NS", "256").split(",")]
FOM_TAUS = [float(v) for v in os.environ.get("FOM_TAUS", "1e-6").split(",")]
QUALITIES = [float(v) for v in os.environ.get("QUALITIES", ".5,.75,.9,.99,1").split(",")]
LINEAR_TOL = float(os.environ.get("LINEAR_TOL", "1e-4"))
PRECONDITIONER = os.environ.get("PRECONDITIONER", "none")
TEST_SEED = int(os.environ.get("TEST_SEED", str(bc.bf.SEED + 1)))
TEST_DRAW_COUNT = int(os.environ.get("TEST_DRAW_COUNT", "16"))
TEST_START = int(os.environ.get("TEST_START", "0"))
N_TEST_TRAJ = int(os.environ.get("N_TEST_TRAJ", "4"))
TIME_REPS = int(os.environ.get("TIME_REPS", "21"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_S = float(os.environ.get("BURN_S", "3"))
TIGHTEST_FINAL_TAU = float(os.environ.get("TIGHTEST_FINAL_TAU", "1e-10"))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def outlier_count(values):
    values = np.asarray(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    return int(np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))


def bootstrap_ci(values, seed):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(4000, values.size))]
    return [float(v) for v in np.quantile(np.median(sample, axis=1), [0.025, 0.975])]


def live_guess_stream(U_solved, u0, exact_next, kind, quality):
    guesses = np.empty_like(U_solved)
    for step in range(bc.T):
        previous = u0 if step == 0 else U_solved[step - 1]
        previous2 = u0 if step <= 1 else U_solved[step - 2]
        if kind == "prev":
            guesses[step] = previous
        else:
            extrap = previous if step == 0 else 2.0 * previous - previous2
            guesses[step] = extrap if kind == "extrap" else (
                extrap + quality * (exact_next[step] - extrap)
            )
    return guesses


def grade(output, trajectory, elapsed, kind, quality, residual):
    U, newton, linear, breakdowns, flags, rel_res = [np.asarray(v) for v in output]
    target = trajectory["U"][1:]
    guesses = live_guess_stream(
        U, trajectory["U"][0], target, kind, quality if quality is not None else 0.0
    )
    guess_l2, guess_res = bc.guess_diagnostics(
        trajectory["U"], guesses, residual, trajectory["nu"]
    )
    return {
        "elapsed_s": float(elapsed),
        "trajectory_rel_l2": float(np.linalg.norm(U - target) / np.linalg.norm(target)),
        "max_final_rel_residual": float(np.max(rel_res)),
        "guess_rel_l2_mean": float(np.mean(guess_l2)),
        "guess_rel_l2_max": float(np.max(guess_l2)),
        "guess_rel_residual_mean": float(np.mean(guess_res)),
        "guess_rel_residual_max": float(np.max(guess_res)),
        "newton_total": int(np.sum(newton)),
        "linear_total": int(np.sum(linear)),
        "newton_per_step": newton.tolist(),
        "linear_per_step": linear.tolist(),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "finite": bool(np.all(np.isfinite(U)) and np.all(np.isfinite(rel_res))),
    }


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing oracle confirmation")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    reference_gate = min(min(FOM_TAUS), TIGHTEST_FINAL_TAU) / 10.0
    specs = [("prev", None), ("extrap", None)] + [("oracle", q) for q in QUALITIES]
    arm_keys = [kind if q is None else f"oracle:q={q:g}" for kind, q in specs]
    report = {
        "config": {
            "purpose": "audited live-history nondeployable oracle break-even curve",
            "ns": NS,
            "fom_taus": FOM_TAUS,
            "linear_tol": LINEAR_TOL,
            "preconditioner": PRECONDITIONER,
            "qualities": QUALITIES,
            "quality_definition": "dynamic_extrap + q*(exact_next-dynamic_extrap)",
            "oracle_non_deployable": True,
            "test_seed": TEST_SEED,
            "canonical_draw_count": TEST_DRAW_COUNT,
            "selected_test_indices": list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            "time_reps": TIME_REPS,
            "same_invocation_cost_accuracy_work": True,
            "reference_residual_gate": reference_gate,
            "tightest_final_tau": TIGHTEST_FINAL_TAU,
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "equivalence": {},
        "complete": False,
    }
    save(report)

    for n in NS:
        trajectories = bc.generate_reference(
            n,
            list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            TEST_SEED,
            draw_count=TEST_DRAW_COUNT,
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > reference_gate:
            raise SystemExit(
                f"N={n}: reference {worst_reference:.3e} exceeds gate {reference_gate:.1e}"
            )
        exact = {t["index"]: jnp.asarray(t["U"][1:]) for t in trajectories}
        dummy = jnp.zeros((bc.T, n * n), jnp.float64)
        for tau in FOM_TAUS:
            calls = {}
            metadata = {}
            residual_for_grade = None
            baseline_chain = None
            for kind, quality in specs:
                chain, residual = bc.make_chain(
                    n,
                    tau,
                    lin_tol=LINEAR_TOL,
                    preconditioner=PRECONDITIONER,
                    oracle_quality=0.0 if quality is None else quality,
                )
                if residual_for_grade is None:
                    residual_for_grade = residual
                if kind == "prev":
                    mode, guesses = jnp.int32(0), None
                elif kind == "extrap":
                    mode, guesses = jnp.int32(1), None
                    baseline_chain = chain
                else:
                    mode, guesses = jnp.int32(7), "exact"
                key = kind if quality is None else f"oracle:q={quality:g}"
                metadata[key] = (kind, quality)
                for trajectory in trajectories:
                    supplied = exact[trajectory["index"]] if guesses == "exact" else dummy
                    calls[(key, trajectory["index"])] = (
                        lambda chain_fn=chain, traj=trajectory, supplied_guesses=supplied,
                        mode_value=mode: chain_fn(
                            jnp.asarray(traj["U"][0]), traj["nu"], supplied_guesses, mode_value
                        )
                    )

            for call in calls.values():
                jax.block_until_ready(call())
            burn_count = bc.gpu_burn(
                lambda: jax.block_until_ready(calls[("extrap", trajectories[0]["index"])]()),
                BURN_S,
            )
            for warm in range(TIME_WARM):
                order = arm_keys[warm % len(arm_keys):] + arm_keys[:warm % len(arm_keys)]
                for trajectory in trajectories:
                    for key in order:
                        jax.block_until_ready(calls[(key, trajectory["index"])]())

            records = {key: [] for key in arm_keys}
            timing_orders = []
            for repetition in range(TIME_REPS):
                offset = repetition % len(arm_keys)
                order = arm_keys[offset:] + arm_keys[:offset]
                if (repetition // len(arm_keys)) % 2:
                    order = list(reversed(order))
                timing_orders.append(order)
                trajectory_order = trajectories[repetition % N_TEST_TRAJ:] + trajectories[:repetition % N_TEST_TRAJ]
                for trajectory in trajectory_order:
                    for key in order:
                        start = time.perf_counter()
                        output = calls[(key, trajectory["index"])]()
                        jax.block_until_ready(output)
                        elapsed = time.perf_counter() - start
                        kind, quality = metadata[key]
                        record = grade(
                            output, trajectory, elapsed, kind, quality, residual_for_grade
                        )
                        record.update(
                            trajectory_index=trajectory["index"], repetition=repetition
                        )
                        if (
                            not record["finite"]
                            or record["breakdowns"]
                            or record["flags_nonzero"]
                            or record["max_final_rel_residual"] > tau
                        ):
                            raise SystemExit(
                                f"N={n} tau={tau} {key}: unhealthy timed solve {record}"
                            )
                        records[key].append(record)

            extrap_pairs = {
                (r["trajectory_index"], r["repetition"]): r["elapsed_s"]
                for r in records["extrap"]
            }
            prev_pairs = {
                (r["trajectory_index"], r["repetition"]): r["elapsed_s"]
                for r in records["prev"]
            }
            for arm_index, key in enumerate(arm_keys):
                elapsed = [r["elapsed_s"] for r in records[key]]
                delta_extrap = [
                    extrap_pairs[(r["trajectory_index"], r["repetition"])] - r["elapsed_s"]
                    for r in records[key]
                ]
                delta_prev = [
                    prev_pairs[(r["trajectory_index"], r["repetition"])] - r["elapsed_s"]
                    for r in records[key]
                ]
                row = {
                    "N": n,
                    "n_dof": n * n,
                    "fom_tau": tau,
                    "candidate": key,
                    "quality": metadata[key][1],
                    "oracle_non_deployable": metadata[key][0] == "oracle",
                    "reference_max_residual": worst_reference,
                    "timed_records": records[key],
                    "timing_repetitions_s": elapsed,
                    "timing_shape": [N_TEST_TRAJ, TIME_REPS],
                    "timing_median_ms": float(np.median(elapsed) * 1e3),
                    "timing_outlier_count_tukey": outlier_count(elapsed),
                    "paired_delta_vs_extrap_s": delta_extrap,
                    "paired_delta_vs_extrap_median_ms": float(np.median(delta_extrap) * 1e3),
                    "paired_delta_vs_extrap_median_95ci_ms": [
                        1e3 * v for v in bootstrap_ci(
                            delta_extrap, 20260819 + n + arm_index
                        )
                    ],
                    "paired_delta_vs_prev_s": delta_prev,
                    "paired_delta_vs_prev_median_ms": float(np.median(delta_prev) * 1e3),
                    "newton_total_median": float(np.median([
                        r["newton_total"] for r in records[key]
                    ])),
                    "linear_total_median": float(np.median([
                        r["linear_total"] for r in records[key]
                    ])),
                    "guess_rel_l2_mean": float(np.mean([
                        r["guess_rel_l2_mean"] for r in records[key]
                    ])),
                    "max_timed_outer_residual": float(max(
                        r["max_final_rel_residual"] for r in records[key]
                    )),
                    "max_timed_trajectory_rel_l2": float(max(
                        r["trajectory_rel_l2"] for r in records[key]
                    )),
                    "gpu_burn_iterations": burn_count,
                    "paired_timing_orders": timing_orders,
                }
                report["rows"].append(row)
                bc.log(
                    f"N={n} tau={tau:.0e} {key:14s}: "
                    f"{row['timing_median_ms']:.3f}ms "
                    f"delta_extrap={row['paired_delta_vs_extrap_median_ms']:.3f}ms "
                    f"Newton={row['newton_total_median']:.0f} "
                    f"BiCG={row['linear_total_median']:.0f}"
                )
            report["equivalence"][f"N={n}:tau={tau:.0e}"] = bc.reference_equivalence(
                n,
                trajectories,
                baseline_chain,
                lin_tol=LINEAR_TOL,
                preconditioner=PRECONDITIONER,
            )
            save(report)

    report["complete"] = True
    save(report)


if __name__ == "__main__":
    main()
