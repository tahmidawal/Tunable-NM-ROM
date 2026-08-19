"""Paired calibration gate for cheap classical Burgers predictors.

Every timed invocation returns the solved field and all solver telemetry.  The
returned object is blocked inside the timer and graded afterward, so cost,
accuracy, outer residuals, and work counts cannot be assembled from different
calls.  This is a calibration-only seed-2 gate; final seed-1 confirmation is a
separate locked job.
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
NS = [int(v) for v in os.environ.get("NS", "64,256").split(",")]
FOM_TAUS = [float(v) for v in os.environ.get("FOM_TAUS", "1e-6").split(",")]
LINEAR_TOL = float(os.environ.get("LINEAR_TOL", "1e-4"))
PRECONDITIONER = os.environ.get("PRECONDITIONER", "none")
CALIB_SEED = int(os.environ.get("CALIB_SEED", "2"))
CALIB_DRAW_COUNT = int(os.environ.get("CALIB_DRAW_COUNT", "8"))
N_TRAJ = int(os.environ.get("N_CALIB_TRAJ", "4"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_S = float(os.environ.get("BURN_S", "3"))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def outliers(values):
    values = np.asarray(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    width = q3 - q1
    return int(np.sum((values < q1 - 1.5 * width) | (values > q3 + 1.5 * width)))


def paired_ci(values, seed):
    """Deterministic percentile bootstrap CI for a paired median."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(2000, values.size))
    medians = np.median(values[indices], axis=1)
    return [float(v) for v in np.quantile(medians, [0.025, 0.975])]


def grade(output, trajectory, elapsed):
    U, newton, linear, breakdowns, flags, rel_res = [np.asarray(v) for v in output]
    target = trajectory["U"][1:]
    return {
        "elapsed_s": float(elapsed),
        "trajectory_rel_l2": float(np.linalg.norm(U - target) / np.linalg.norm(target)),
        "max_final_rel_residual": float(np.max(rel_res)),
        "newton_total": int(np.sum(newton)),
        "linear_total": int(np.sum(linear)),
        "newton_per_step": newton.tolist(),
        "linear_per_step": linear.tolist(),
        "breakdowns": int(np.sum(breakdowns)),
        "flags_nonzero": int(np.sum(flags != 0)),
        "finite": bool(np.all(np.isfinite(U)) and np.all(np.isfinite(rel_res))),
    }


def diagnostic_guess(arm, U, nu, n):
    if arm == "linear":
        return bc.extrapolated_guesses(U)
    predictor = bc.make_physics_predictor(n, arm)
    history = [jnp.asarray(U[0])] * 4
    guesses = []
    for step in range(bc.T):
        guess = predictor(*history, nu, jnp.int32(step))
        guesses.append(np.asarray(guess))
        history = [jnp.asarray(U[step + 1]), history[0], history[1], history[2]]
    return np.stack(guesses)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing physics gate")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    arm_names = ["linear", "explicit_euler", "imex_euler", "imex_ab2"]
    report = {
        "config": {
            "purpose": "calibration-only charged classical physics predictor gate",
            "classification": "classical controls; no learned or NM-ROM claim",
            "ns": NS,
            "fom_taus": FOM_TAUS,
            "linear_tol": LINEAR_TOL,
            "preconditioner": PRECONDITIONER,
            "calibration_seed": CALIB_SEED,
            "canonical_draw_count": CALIB_DRAW_COUNT,
            "trajectory_indices": list(range(N_TRAJ)),
            "test_population_touched": False,
            "arms": arm_names,
            "time_reps": TIME_REPS,
            "time_warm": TIME_WARM,
            "burn_seconds": BURN_S,
            "same_invocation_cost_accuracy_work": True,
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
            n, list(range(N_TRAJ)), CALIB_SEED, draw_count=CALIB_DRAW_COUNT
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > 1e-8:
            raise SystemExit(f"N={n}: bad reference residual {worst_reference:.3e}")
        dummy = jnp.zeros((bc.T, n * n), jnp.float64)

        for tau in FOM_TAUS:
            chains = {}
            modes = {}
            for arm in arm_names:
                predictor = None if arm == "linear" else bc.make_physics_predictor(n, arm)
                chains[arm], _ = bc.make_chain(
                    n,
                    tau,
                    predictor=predictor,
                    lin_tol=LINEAR_TOL,
                    preconditioner=PRECONDITIONER,
                )
                modes[arm] = jnp.int32(1 if arm == "linear" else 3)
            calls = {
                (arm, trajectory["index"]): (
                    lambda chain=chains[arm], traj=trajectory, mode=modes[arm]: chain(
                        jnp.asarray(traj["U"][0]), traj["nu"], dummy, mode
                    )
                )
                for arm in arm_names
                for trajectory in trajectories
            }
            # Compile all variants, then a common post-compile burn and rotated
            # warm block.  No result from these calls enters the table.
            for call in calls.values():
                jax.block_until_ready(call())
            burn_count = bc.gpu_burn(
                lambda: jax.block_until_ready(calls[("linear", trajectories[0]["index"])]()),
                BURN_S,
            )
            for warm in range(TIME_WARM):
                order = arm_names[warm % len(arm_names):] + arm_names[:warm % len(arm_names)]
                for trajectory in trajectories:
                    for arm in order:
                        jax.block_until_ready(calls[(arm, trajectory["index"])]())

            records = {arm: [] for arm in arm_names}
            timing_orders = []
            for repetition in range(TIME_REPS):
                offset = repetition % len(arm_names)
                order = arm_names[offset:] + arm_names[:offset]
                if (repetition // len(arm_names)) % 2:
                    order = list(reversed(order))
                timing_orders.append(order)
                trajectory_order = trajectories[repetition % N_TRAJ:] + trajectories[:repetition % N_TRAJ]
                for trajectory in trajectory_order:
                    for arm in order:
                        start = time.perf_counter()
                        output = calls[(arm, trajectory["index"])]()
                        jax.block_until_ready(output)
                        elapsed = time.perf_counter() - start
                        record = grade(output, trajectory, elapsed)
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
                                f"N={n} tau={tau} arm={arm}: unhealthy timed solve {record}"
                            )
                        records[arm].append(record)

            baseline_by_pair = {
                (r["trajectory_index"], r["repetition"]): r["elapsed_s"]
                for r in records["linear"]
            }
            for arm_index, arm in enumerate(arm_names):
                elapsed = [r["elapsed_s"] for r in records[arm]]
                paired_delta = [
                    baseline_by_pair[(r["trajectory_index"], r["repetition"])]
                    - r["elapsed_s"]
                    for r in records[arm]
                ]
                diagnostic_errors = []
                for trajectory in trajectories:
                    guesses = diagnostic_guess(arm, trajectory["U"], trajectory["nu"], n)
                    diagnostic_errors.append(float(
                        np.linalg.norm(guesses - trajectory["U"][1:])
                        / np.linalg.norm(trajectory["U"][1:])
                    ))
                row = {
                    "N": n,
                    "n_dof": n * n,
                    "fom_tau": tau,
                    "arm": arm,
                    "reference_max_residual": worst_reference,
                    "guess_rel_l2_per_trajectory_reference_history_diagnostic": diagnostic_errors,
                    "timed_records": records[arm],
                    "timing_repetitions_s": elapsed,
                    "timing_shape": [N_TRAJ, TIME_REPS],
                    "timing_median_ms": float(np.median(elapsed) * 1e3),
                    "timing_outlier_count_tukey": outliers(elapsed),
                    "paired_saving_vs_linear_s": paired_delta,
                    "paired_saving_vs_linear_median_ms": float(np.median(paired_delta) * 1e3),
                    "paired_saving_vs_linear_median_95ci_ms": [
                        v * 1e3 for v in paired_ci(paired_delta, 20260819 + n + arm_index)
                    ],
                    "newton_total_median": float(np.median([
                        r["newton_total"] for r in records[arm]
                    ])),
                    "linear_total_median": float(np.median([
                        r["linear_total"] for r in records[arm]
                    ])),
                    "max_timed_outer_residual": float(max(
                        r["max_final_rel_residual"] for r in records[arm]
                    )),
                    "max_timed_trajectory_rel_l2": float(max(
                        r["trajectory_rel_l2"] for r in records[arm]
                    )),
                    "gpu_burn_iterations": burn_count,
                    "paired_timing_orders": timing_orders,
                }
                report["rows"].append(row)
                bc.log(
                    f"N={n} tau={tau:.0e} {arm:14s}: "
                    f"{row['timing_median_ms']:.3f} ms, "
                    f"save={row['paired_saving_vs_linear_median_ms']:.3f} ms, "
                    f"Newton={row['newton_total_median']:.0f}, "
                    f"BiCG={row['linear_total_median']:.0f}"
                )
            report["equivalence"][f"N={n}:tau={tau:.0e}"] = bc.reference_equivalence(
                n,
                trajectories,
                chains["linear"],
                lin_tol=LINEAR_TOL,
                preconditioner=PRECONDITIONER,
            )
            save(report)

    report["complete"] = True
    save(report)


if __name__ == "__main__":
    main()
