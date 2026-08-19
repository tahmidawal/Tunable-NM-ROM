"""Calibration-only inexact-Newton inner-tolerance ladder.

This uses seed 2, disjoint from both training (seed 0) and final test (seed 1).
It selects a linear tolerance for each named outer tolerance before any final
warm-start comparison. The selected tolerance is then locked and shared by all
arms; it is never selected on the final test population.
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
NS = [int(value) for value in os.environ.get("NS", "64,256").split(",")]
FOM_TAUS = [float(value) for value in os.environ.get(
    "FOM_TAUS", "1e-6,1e-8,1e-10"
).split(",")]
LINEAR_TOLS = [float(value) for value in os.environ.get(
    "LINEAR_TOLS", "1e-4,1e-6,1e-8,1e-10"
).split(",")]
CALIB_SEED = int(os.environ.get("CALIB_SEED", "2"))
CALIB_DRAW_COUNT = int(os.environ.get("CALIB_DRAW_COUNT", "8"))
N_CALIB_TRAJ = int(os.environ.get("N_CALIB_TRAJ", "4"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_S = float(os.environ.get("BURN_S", "3"))
ARMS = (("linear", 1), ("quadratic", 4), ("cubic", 5))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing calibration")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    report = {
        "config": {
            "pde": "burgers2d",
            "purpose": "calibration-only inner linear tolerance selection",
            "ns": NS,
            "fom_taus": FOM_TAUS,
            "linear_tols": LINEAR_TOLS,
            "arms": [name for name, _ in ARMS],
            "calibration_seed": CALIB_SEED,
            "calibration_draw_count": CALIB_DRAW_COUNT,
            "n_calibration_trajectories": N_CALIB_TRAJ,
            "calibration_indices": list(range(N_CALIB_TRAJ)),
            "test_population_touched": False,
            "time_reps_per_trajectory": TIME_REPS,
            "time_warm": TIME_WARM,
            "burn_seconds": BURN_S,
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "selection": {},
        "complete": False,
    }
    save(report)

    for n in NS:
        trajectories = bc.generate_reference(
            n, list(range(N_CALIB_TRAJ)), CALIB_SEED,
            draw_count=CALIB_DRAW_COUNT,
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > 1e-8:
            raise SystemExit(f"N={n}: invalid calibration reference {worst_reference:.3e}")
        dummy = jnp.zeros((bc.T, n * n), dtype=jnp.float64)

        for outer_tau in FOM_TAUS:
            calls = {}
            metrics = {}
            for linear_tol in LINEAR_TOLS:
                chain, _ = bc.make_chain(n, outer_tau, lin_tol=linear_tol)
                for arm, mode in ARMS:
                    key = f"lin={linear_tol:g}:{arm}"
                    per_trajectory = []
                    for trajectory in trajectories:
                        output = chain(
                            jnp.asarray(trajectory["U"][0]), trajectory["nu"], dummy,
                            jnp.int32(mode)
                        )
                        U, newton, linear, breakdowns, flags, rel_res = [
                            np.asarray(value) for value in output
                        ]
                        if (
                            not np.all(np.isfinite(rel_res))
                            or float(np.max(rel_res)) > outer_tau
                            or int(np.sum(breakdowns))
                            or int(np.sum(flags != 0))
                        ):
                            raise SystemExit(
                                f"N={n} outer={outer_tau} lin={linear_tol} arm={arm}: "
                                f"health failure maxres={np.max(rel_res):.3e} "
                                f"breakdowns={np.sum(breakdowns)} flags={np.sum(flags != 0)}"
                            )
                        per_trajectory.append({
                            "trajectory_index": trajectory["index"],
                            "newton_total": int(np.sum(newton)),
                            "linear_total": int(np.sum(linear)),
                            "newton_per_step": newton.tolist(),
                            "linear_per_step": linear.tolist(),
                            "max_final_rel_residual": float(np.max(rel_res)),
                            "final_trajectory_rel_l2": float(
                                np.linalg.norm(U - trajectory["U"][1:])
                                / np.linalg.norm(trajectory["U"][1:])
                            ),
                        })
                    metrics[key] = {
                        "N": n,
                        "n_dof": n * n,
                        "fom_tau": outer_tau,
                        "linear_tol": linear_tol,
                        "arm": arm,
                        "newton_total_mean": float(np.mean([
                            item["newton_total"] for item in per_trajectory
                        ])),
                        "linear_total_mean": float(np.mean([
                            item["linear_total"] for item in per_trajectory
                        ])),
                        "max_final_rel_residual": float(max(
                            item["max_final_rel_residual"] for item in per_trajectory
                        )),
                        "reference_max_newton_residual": worst_reference,
                        "per_trajectory": per_trajectory,
                    }
                    for trajectory in trajectories:
                        trajectory_index = trajectory["index"]
                        calls[(key, trajectory_index)] = (
                            lambda chain_fn=chain, traj=trajectory, mode_value=jnp.int32(mode):
                            chain_fn(
                                jnp.asarray(traj["U"][0]), traj["nu"], dummy, mode_value
                            )[0].block_until_ready()
                        )

            # Compile/warm all calls, burn once, then pair by trajectory and
            # repetition while rotating the (linear tolerance, arm) order.
            keys = list(metrics)
            for call in calls.values():
                call()
            anchor = calls[(f"lin={LINEAR_TOLS[-1]:g}:linear", 0)]
            burn_count = bc.gpu_burn(anchor, BURN_S)
            for warm in range(TIME_WARM):
                offset = warm % len(keys)
                for trajectory in trajectories:
                    for key in keys[offset:] + keys[:offset]:
                        calls[(key, trajectory["index"])]()
            samples = {key: [] for key in keys}
            timing_orders = []
            for repetition in range(TIME_REPS):
                offset = repetition % len(keys)
                order = keys[offset:] + keys[:offset]
                if (repetition // len(keys)) % 2:
                    order = list(reversed(order))
                timing_orders.append(order)
                for trajectory in trajectories:
                    for key in order:
                        start = time.perf_counter()
                        calls[(key, trajectory["index"])]()
                        samples[key].append(float(time.perf_counter() - start))

            for key, row in metrics.items():
                row.update(
                    time_median_over_population_ms=float(np.median(samples[key]) * 1e3),
                    timing_repetitions_s=samples[key],
                    timing_shape=[N_CALIB_TRAJ, TIME_REPS],
                    paired_timing_orders=timing_orders,
                    gpu_burn_iterations=burn_count,
                )
                report["rows"].append(row)
                bc.log(
                    f"N={n} outer={outer_tau:.0e} lin={row['linear_tol']:.0e} "
                    f"{row['arm']:9s}: Newton={row['newton_total_mean']:.1f} "
                    f"BiCG={row['linear_total_mean']:.0f} "
                    f"population median={row['time_median_over_population_ms']:.2f}ms"
                )
            save(report)

    # Pre-registered selection: for each outer tolerance, pick one shared inner
    # tolerance that minimises the median time pooled over N for the fastest
    # polynomial arm. Ties within 1% choose the tighter tolerance. No test data.
    for outer_tau in FOM_TAUS:
        candidates = []
        for linear_tol in LINEAR_TOLS:
            relevant = [row for row in report["rows"] if (
                row["fom_tau"] == outer_tau and row["linear_tol"] == linear_tol
            )]
            by_arm = {}
            for arm, _ in ARMS:
                values = [row["time_median_over_population_ms"] for row in relevant
                          if row["arm"] == arm]
                by_arm[arm] = float(np.median(values))
            best_arm = min(by_arm, key=by_arm.get)
            candidates.append({
                "linear_tol": linear_tol,
                "best_polynomial_arm": best_arm,
                "pooled_mesh_median_ms": by_arm[best_arm],
                "arm_times_ms": by_arm,
            })
        best_value = min(item["pooled_mesh_median_ms"] for item in candidates)
        near = [item for item in candidates
                if item["pooled_mesh_median_ms"] <= 1.01 * best_value]
        selected = min(near, key=lambda item: item["linear_tol"])
        report["selection"][f"{outer_tau:.0e}"] = {
            "selected_linear_tol": selected["linear_tol"],
            "selected_on_arm": selected["best_polynomial_arm"],
            "rule": "fastest pooled-N polynomial; within 1% choose tighter",
            "candidates": candidates,
        }
    report["complete"] = True
    save(report)
    bc.log("SELECTION", report["selection"])


if __name__ == "__main__":
    main()

