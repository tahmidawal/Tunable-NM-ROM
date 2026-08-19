"""Deployable weak-LSPG history-manifold warm start versus free baselines."""
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
MODE_COUNT = int(os.environ.get("MODE_COUNT", "16"))
REDUCED_ITERS = int(os.environ.get("REDUCED_ITERS", "2"))
TRUST_RADIUS = float(os.environ.get("TRUST_RADIUS", "0.5"))
N_TEST_TRAJ = int(os.environ.get("N_TEST_TRAJ", "4"))
TEST_START = int(os.environ.get("TEST_START", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", str(bc.bf.SEED + 1)))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
TIME_WARM = int(os.environ.get("TIME_WARM", "2"))
BURN_S = float(os.environ.get("BURN_S", "3"))
RUN_ROLE = os.environ.get("RUN_ROLE", "panel")
REFERENCE_RESIDUAL_GATE = float(os.environ.get("REFERENCE_RESIDUAL_GATE", "1e-8"))
ARMS = (("prev", 0), ("extrap", 1), ("history_weak", 3))


def save(report):
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)


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
            "candidate": "history-line weak-LSPG classical control",
            "classification": "history-only classical control; not a learned NM-ROM",
            "manifold": "u(alpha)=u_n+alpha*(u_n-u_{n-1})",
            "mode_count": MODE_COUNT,
            "reduced_iters": REDUCED_ITERS,
            "trust_radius": TRUST_RADIUS,
            "full_grid_weak_projection": True,
            "fom_exact_upwind_in_weak_advection": True,
            "n_test_trajectories": N_TEST_TRAJ,
            "test_indices": list(range(TEST_START, TEST_START + N_TEST_TRAJ)),
            "test_seed": TEST_SEED,
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
    bc.log(
        f"backend={jax.default_backend()} gpu={provenance['gpu_kind']} NS={NS} "
        f"taus={FOM_TAUS} M={MODE_COUNT} reduced_iters={REDUCED_ITERS}"
    )

    for n in NS:
        started = time.time()
        trajectories = bc.generate_reference(
            n, list(range(TEST_START, TEST_START + N_TEST_TRAJ)), TEST_SEED
        )
        worst_reference = max(t["max_reference_newton_residual"] for t in trajectories)
        if not np.isfinite(worst_reference) or worst_reference > REFERENCE_RESIDUAL_GATE:
            raise SystemExit(f"N={n}: invalid reference residual {worst_reference:.3e}")
        predictor = bc.make_full_weak_history_predictor(
            n,
            mode_count=MODE_COUNT,
            reduced_iters=REDUCED_ITERS,
            trust_radius=TRUST_RADIUS,
        )
        dummy = jnp.zeros((bc.T, n * n), dtype=jnp.float64)

        for tau in FOM_TAUS:
            chain, _ = bc.make_chain(n, tau, predictor=predictor)
            first = trajectories[0]
            # Compile every traced branch before burn-in and timing.
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
                    U, newton, linear, breakdowns, flags, rel_res = chain(
                        jnp.asarray(trajectory["U"][0]), trajectory["nu"], dummy,
                        jnp.int32(mode)
                    )
                    U = np.asarray(U)
                    newton = np.asarray(newton)
                    linear = np.asarray(linear)
                    breakdowns = np.asarray(breakdowns)
                    flags = np.asarray(flags)
                    rel_res = np.asarray(rel_res)
                    if not np.all(np.isfinite(rel_res)) or float(np.max(rel_res)) > tau:
                        raise SystemExit(
                            f"N={n} tau={tau} arm={name}: FOM finish failed, "
                            f"max residual={np.max(rel_res):.3e}"
                        )
                    if int(np.sum(breakdowns)) or int(np.sum(flags != 0)):
                        raise SystemExit(
                            f"N={n} tau={tau} arm={name}: linear/Newton failure"
                        )
                    final_error = float(
                        np.linalg.norm(U - trajectory["U"][1:])
                        / np.linalg.norm(trajectory["U"][1:])
                    )
                    item = {
                            "trajectory_index": trajectory["index"],
                            "parameters": trajectory["parameters"],
                            "newton_total": int(np.sum(newton)),
                            "linear_total": int(np.sum(linear)),
                            "newton_per_step": newton.tolist(),
                            "linear_per_step": linear.tolist(),
                            "max_final_rel_residual": float(np.max(rel_res)),
                            "final_trajectory_rel_l2": final_error,
                    }
                    if name == "history_weak":
                        truth = jnp.asarray(trajectory["U"])
                        previous2 = jnp.concatenate([truth[:1], truth[:-2]], axis=0)
                        alpha, weak_initial, weak_final = jax.vmap(
                            lambda up, up2: predictor.diagnostics(
                                up, up2, trajectory["nu"]
                            )
                        )(truth[:-1], previous2)
                        alpha_np = np.asarray(alpha)
                        reduction = np.asarray(weak_final) / np.maximum(
                            np.asarray(weak_initial), 1e-300
                        )
                        item.update(
                            alpha_per_step=alpha_np.tolist(),
                            alpha_mean=float(np.mean(alpha_np)),
                            alpha_min=float(np.min(alpha_np)),
                            alpha_max=float(np.max(alpha_np)),
                            weak_residual_ratio_per_step=reduction.tolist(),
                            weak_residual_ratio_mean=float(np.mean(reduction)),
                        )
                    per_trajectory.append(item)
                mode_j = jnp.int32(mode)
                timed_calls[name] = lambda mode_value=mode_j: chain(
                    jnp.asarray(first["U"][0]), first["nu"], dummy, mode_value
                )[0].block_until_ready()
                arm_data[name] = {
                    "newton_total_mean": float(
                        np.mean([p["newton_total"] for p in per_trajectory])
                    ),
                    "linear_total_mean": float(
                        np.mean([p["linear_total"] for p in per_trajectory])
                    ),
                    "per_trajectory": per_trajectory,
                }
                if name == "history_weak":
                    arm_data[name].update(
                        alpha_mean=float(np.mean([p["alpha_mean"] for p in per_trajectory])),
                        alpha_min=float(np.min([p["alpha_min"] for p in per_trajectory])),
                        alpha_max=float(np.max([p["alpha_max"] for p in per_trajectory])),
                        weak_residual_ratio_mean=float(np.mean([
                            p["weak_residual_ratio_mean"] for p in per_trajectory
                        ])),
                    )

            # Paired/interleaved repetitions, rotating arm order every repetition.
            keys = list(timed_calls)
            for warm in range(TIME_WARM):
                order = keys[warm % len(keys):] + keys[:warm % len(keys)]
                for key in order:
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

            extrap_ms = float(np.median(samples["extrap"]) * 1e3)
            prev_ms = float(np.median(samples["prev"]) * 1e3)
            extrap_samples = np.asarray(samples["extrap"])
            prev_samples = np.asarray(samples["prev"])
            for name, _ in ARMS:
                total_ms = float(np.median(samples[name]) * 1e3)
                arm_samples = np.asarray(samples[name])
                row = {
                    "N": n,
                    "n_dof": n * n,
                    "fom_tau": tau,
                    "arm": name,
                    "total_ms": total_ms,
                    "timing_repetitions_s": samples[name],
                    "paired_timing_orders": timing_orders,
                    "paired_delta_vs_extrap_s": (
                        extrap_samples - arm_samples
                    ).tolist(),
                    "paired_delta_vs_prev_s": (prev_samples - arm_samples).tolist(),
                    "paired_delta_vs_extrap_median_ms": float(
                        np.median(extrap_samples - arm_samples) * 1e3
                    ),
                    "speedup_vs_extrap": extrap_ms / total_ms,
                    "speedup_vs_prev": prev_ms / total_ms,
                    "time_delta_vs_extrap_ms": extrap_ms - total_ms,
                    "reference_max_newton_residual": worst_reference,
                    "gpu_burn_iterations": burn_count,
                    "run_role": RUN_ROLE,
                    **arm_data[name],
                }
                report["rows"].append(row)
                bc.log(
                    f"N={n} tau={tau:.0e} {name:12s}: "
                    f"Newton={row['newton_total_mean']:.1f} "
                    f"BiCG={row['linear_total_mean']:.0f} total={total_ms:.2f}ms "
                    f"speedup/extrap={row['speedup_vs_extrap']:.3f}x"
                )
            save(report)
        bc.log(f"N={n} complete in {time.time() - started:.1f}s")

    report["complete"] = True
    save(report)
    bc.log("DONE")


if __name__ == "__main__":
    main()
