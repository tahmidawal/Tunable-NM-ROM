"""Zero-training polynomial history baselines for Burgers warm starts."""
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
NS = [int(value) for value in os.environ.get("NS", "32,64,128,256").split(",")]
FOM_TAUS = [float(value) for value in os.environ.get("FOM_TAUS", "1e-6").split(",")]
N_TEST_TRAJ = int(os.environ.get("N_TEST_TRAJ", "4"))
TEST_START = int(os.environ.get("TEST_START", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", str(bc.bf.SEED + 1)))
TEST_DRAW_COUNT = int(os.environ.get("TEST_DRAW_COUNT", "16"))
TIME_REPS = int(os.environ.get("TIME_REPS", "21"))
TIME_WARM = int(os.environ.get("TIME_WARM", "3"))
BURN_S = float(os.environ.get("BURN_S", "3"))
RUN_ROLE = os.environ.get("RUN_ROLE", "panel")
ARMS = (("prev", 0), ("linear", 1), ("quadratic", 4), ("cubic", 5))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


def median_ci(values, seed=20260819, draws=10000):
    """Deterministic paired bootstrap 95% interval for the sample median."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def diagnostic_guesses(U, name):
    if name == "prev":
        return U[:-1]
    if name == "linear":
        return bc.extrapolated_guesses(U)
    if name == "quadratic":
        return bc.polynomial_guesses(U, 2)
    if name == "cubic":
        return bc.polynomial_guesses(U, 3)
    raise ValueError(name)


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit("jax_backend is not gpu; refusing to produce results")
    provenance = bc.provenance()
    if provenance["matmul_precision"] != "highest":
        raise SystemExit("JAX_DEFAULT_MATMUL_PRECISION must be highest")
    report = {
        "config": {
            "pde": "burgers2d",
            "dt": bc.bf.DT,
            "num_steps": bc.T,
            "ns": NS,
            "fom_taus": FOM_TAUS,
            "arms": [name for name, _ in ARMS],
            "startup": "previous at step 1; linear at step 2; quadratic at step 3",
            "classification": "zero-training classical history controls",
            "n_test_trajectories": N_TEST_TRAJ,
            "test_indices": list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            "test_seed": TEST_SEED,
            "test_draw_count": TEST_DRAW_COUNT,
            "time_reps": TIME_REPS,
            "time_warm": TIME_WARM,
            "burn_seconds": BURN_S,
            "run_role": RUN_ROLE,
            "f64": True,
        },
        "provenance": provenance,
        "rows": [],
        "complete": False,
    }
    save(report)

    for n in NS:
        started = time.time()
        trajectories = bc.generate_reference(
            n, list(range(TEST_START, TEST_START + N_TEST_TRAJ)), TEST_SEED,
            draw_count=TEST_DRAW_COUNT,
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > 1e-8:
            raise SystemExit(f"N={n}: invalid reference residual {worst_reference:.3e}")
        dummy = jnp.zeros((bc.T, n * n), dtype=jnp.float64)

        for tau in FOM_TAUS:
            chain, residual = bc.make_chain(n, tau)
            first = trajectories[0]
            for _, mode in ARMS:
                chain(jnp.asarray(first["U"][0]), first["nu"], dummy,
                      jnp.int32(mode))[0].block_until_ready()
            burn_count = bc.gpu_burn(
                lambda: chain(jnp.asarray(first["U"][0]), first["nu"], dummy,
                              jnp.int32(1))[0].block_until_ready(),
                BURN_S,
            )

            arm_data = {}
            timed_calls = {}
            for name, mode in ARMS:
                per_trajectory = []
                for trajectory in trajectories:
                    U_final, newton, linear, breakdowns, flags, rel_res = chain(
                        jnp.asarray(trajectory["U"][0]), trajectory["nu"], dummy,
                        jnp.int32(mode)
                    )
                    U_final = np.asarray(U_final)
                    newton, linear = np.asarray(newton), np.asarray(linear)
                    breakdowns, flags, rel_res = (
                        np.asarray(breakdowns), np.asarray(flags), np.asarray(rel_res)
                    )
                    if not np.all(np.isfinite(rel_res)) or float(np.max(rel_res)) > tau:
                        raise SystemExit(
                            f"N={n} tau={tau} {name}: finish residual {np.max(rel_res):.3e}"
                        )
                    if int(np.sum(breakdowns)) or int(np.sum(flags != 0)):
                        raise SystemExit(f"N={n} tau={tau} {name}: solver health failure")
                    guesses = diagnostic_guesses(trajectory["U"], name)
                    guess_error, guess_residual = bc.guess_diagnostics(
                        trajectory["U"], guesses, residual, trajectory["nu"]
                    )
                    per_trajectory.append({
                        "trajectory_index": trajectory["index"],
                        "parameters": trajectory["parameters"],
                        "newton_total": int(np.sum(newton)),
                        "linear_total": int(np.sum(linear)),
                        "newton_per_step": newton.tolist(),
                        "linear_per_step": linear.tolist(),
                        "guess_rel_l2_mean": float(np.mean(guess_error)),
                        "guess_rel_l2_max": float(np.max(guess_error)),
                        "guess_rel_residual_mean": float(np.mean(guess_residual)),
                        "guess_rel_residual_max": float(np.max(guess_residual)),
                        "max_final_rel_residual": float(np.max(rel_res)),
                        "final_trajectory_rel_l2": float(
                            np.linalg.norm(U_final - trajectory["U"][1:])
                            / np.linalg.norm(trajectory["U"][1:])
                        ),
                    })
                mode_j = jnp.int32(mode)
                timed_calls[name] = lambda mode_value=mode_j: chain(
                    jnp.asarray(first["U"][0]), first["nu"], dummy, mode_value
                )[0].block_until_ready()
                arm_data[name] = {
                    "newton_total_mean": float(np.mean([
                        item["newton_total"] for item in per_trajectory
                    ])),
                    "linear_total_mean": float(np.mean([
                        item["linear_total"] for item in per_trajectory
                    ])),
                    "guess_rel_l2_mean": float(np.mean([
                        item["guess_rel_l2_mean"] for item in per_trajectory
                    ])),
                    "guess_rel_residual_mean": float(np.mean([
                        item["guess_rel_residual_mean"] for item in per_trajectory
                    ])),
                    "per_trajectory": per_trajectory,
                }

            keys = list(timed_calls)
            for warm in range(TIME_WARM):
                offset = warm % len(keys)
                for key in keys[offset:] + keys[:offset]:
                    timed_calls[key]()
            samples = {key: [] for key in keys}
            timing_orders = []
            for repetition in range(TIME_REPS):
                offset = repetition % len(keys)
                order = keys[offset:] + keys[:offset]
                if (repetition // len(keys)) % 2:
                    order = list(reversed(order))
                timing_orders.append(order)
                for key in order:
                    start_timing = time.perf_counter()
                    timed_calls[key]()
                    samples[key].append(float(time.perf_counter() - start_timing))

            linear_samples = np.asarray(samples["linear"])
            for arm_index, (name, _) in enumerate(ARMS):
                arm_samples = np.asarray(samples[name])
                paired = linear_samples - arm_samples
                total_ms = float(np.median(arm_samples) * 1e3)
                row = {
                    "N": n,
                    "n_dof": n * n,
                    "fom_tau": tau,
                    "arm": name,
                    "total_ms": total_ms,
                    "timing_repetitions_s": samples[name],
                    "paired_timing_orders": timing_orders,
                    "paired_delta_vs_linear_s": paired.tolist(),
                    "paired_delta_vs_linear_median_ms": float(np.median(paired) * 1e3),
                    "paired_delta_vs_linear_bootstrap95_ms": [
                        value * 1e3 for value in median_ci(
                            paired, seed=20260819 + n + arm_index
                        )
                    ],
                    "speedup_vs_linear": float(
                        np.median(linear_samples) / np.median(arm_samples)
                    ),
                    "reference_max_newton_residual": worst_reference,
                    "gpu_burn_iterations": burn_count,
                    "run_role": RUN_ROLE,
                    **arm_data[name],
                }
                report["rows"].append(row)
                bc.log(
                    f"N={n} tau={tau:.0e} {name:9s}: "
                    f"guessL2={row['guess_rel_l2_mean']:.2e} "
                    f"Newton={row['newton_total_mean']:.1f} "
                    f"BiCG={row['linear_total_mean']:.0f} total={total_ms:.2f}ms "
                    f"speedup/linear={row['speedup_vs_linear']:.3f}x"
                )
            save(report)
        bc.log(f"N={n} complete in {time.time() - started:.1f}s")

    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
